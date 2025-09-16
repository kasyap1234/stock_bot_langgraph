

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import threading

import pandas as pd
import requests
from alpha_vantage.timeseries import TimeSeries
from newsapi import NewsApiClient
from fredapi import Fred
from bs4 import BeautifulSoup

from config.config import (
    ALPHA_VANTAGE_API_KEY, NEWS_API_KEY, FRED_API_KEY,
    API_RATE_LIMIT_DELAY, REQUEST_TIMEOUT
)
from .models import StockData, create_stock_data, validate_stock_data
from utils.scraping_utils import rate_limited_get, extract_numeric_value, safe_extract_text

logger = logging.getLogger(__name__)

_data_cache: Dict[str, Dict[str, Any]] = {}
_cache_lock = threading.Lock()

_last_api_call = {}
_rate_limit_lock = threading.Lock()


def _is_rate_limited(source: str, delay: float = API_RATE_LIMIT_DELAY) -> bool:
    
    with _rate_limit_lock:
        now = time.time()
        if source in _last_api_call:
            if now - _last_api_call[source] < delay:
                return True
        _last_api_call[source] = now
        return False


def _cache_data(symbol: str, data: Dict[str, Any], ttl: int = 300) -> None:
    
    with _cache_lock:
        _data_cache[symbol] = {
            'data': data,
            'timestamp': time.time(),
            'ttl': ttl
        }


def _get_cached_data(symbol: str) -> Optional[Dict[str, Any]]:
    
    with _cache_lock:
        if symbol in _data_cache:
            cached = _data_cache[symbol]
            if time.time() - cached['timestamp'] < cached['ttl']:
                return cached['data']
            else:
                del _data_cache[symbol]
        return None


def _validate_and_clean_data(data: Dict[str, Any]) -> Dict[str, Any]:
    
    # Remove outliers (price changes > 20%)
    if 'price' in data and 'previous_price' in data:
        change = abs(data['price'] - data['previous_price']) / data['previous_price']
        if change > 0.2:  # 20% change
            logger.warning(f"Outlier detected for {data.get('symbol', 'unknown')}: {change:.2%} change")
            return {}

    # Ensure required fields
    required_fields = ['symbol', 'price', 'timestamp']
    for field in required_fields:
        if field not in data:
            logger.error(f"Missing required field: {field}")
            return {}

    return data


class RealTimeDataManager:
    

    def __init__(self):
        self.sources = {
            'yahoo': self._yahoo_realtime,
            'alpha_vantage': self._alpha_vantage_realtime,
            'newsapi': self._newsapi_realtime,
            'fred': self._fred_realtime,
            'moneycontrol': self._moneycontrol_scrape,
            'bse': self._bse_scrape
        }
        self.active_streams: Dict[str, asyncio.Task] = {}
        self.callbacks: List[Callable] = []

    def add_callback(self, callback: Callable) -> None:
        
        self.callbacks.append(callback)

    def remove_callback(self, callback: Callable) -> None:
        
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    async def start_streaming(self, symbols: List[str], sources: List[str] = None,
                            interval: int = 60) -> None:
        
        if sources is None:
            sources = ['yahoo', 'alpha_vantage']

        for symbol in symbols:
            task_name = f"{symbol}_stream"
            if task_name not in self.active_streams:
                task = asyncio.create_task(
                    self._stream_with_fallback(symbol, sources, interval)
                )
                self.active_streams[task_name] = task
                logger.info(f"Started streaming {symbol} with fallback support")

    async def stop_streaming(self, symbols: List[str] = None) -> None:
        
        if symbols is None:
            tasks_to_cancel = list(self.active_streams.values())
            self.active_streams.clear()
        else:
            tasks_to_cancel = []
            for symbol in symbols:
                task_name = f"{symbol}_stream"
                if task_name in self.active_streams:
                    tasks_to_cancel.append(self.active_streams[task_name])
                    del self.active_streams[task_name]

        for task in tasks_to_cancel:
            task.cancel()

        await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        logger.info("Stopped streaming tasks")

    async def _get_data_with_fallback(self, symbol: str, primary_source: str,
                                     fallback_sources: List[str] = None) -> Dict[str, Any]:
        
        if fallback_sources is None:
            fallback_sources = ['yahoo', 'alpha_vantage', 'moneycontrol', 'bse']

        # Try primary source first
        if primary_source in self.sources:
            try:
                data = await self.sources[primary_source](symbol)
                if data and 'error' not in data:
                    return data
            except Exception as e:
                logger.warning(f"Primary source {primary_source} failed for {symbol}: {e}")

        # Try fallback sources
        for source in fallback_sources:
            if source != primary_source and source in self.sources:
                try:
                    data = await self.sources[source](symbol)
                    if data and 'error' not in data:
                        logger.info(f"Using fallback source {source} for {symbol}")
                        return data
                except Exception as e:
                    logger.warning(f"Fallback source {source} failed for {symbol}: {e}")
                    continue

        logger.error(f"All data sources failed for {symbol}")
        return {"symbol": symbol, "error": "All data sources failed"}

    async def _stream_with_fallback(self, symbol: str, sources: List[str], interval: int) -> None:
        
        while True:
            try:
                # Try to get data from highest priority source with fallbacks
                data = await self._get_data_with_fallback(symbol, sources[0], sources[1:])

                if data and 'error' not in data:
                    # Validate and clean data
                    cleaned_data = _validate_and_clean_data(data)
                    if cleaned_data:
                        # Cache data
                        _cache_data(symbol, cleaned_data)
                        # Notify callbacks
                        for callback in self.callbacks:
                            try:
                                callback(cleaned_data)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")

                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Streaming error for {symbol}: {e}")
                await asyncio.sleep(interval * 2)  # Back off on error

    async def _yahoo_realtime(self, symbol: str) -> Dict[str, Any]:
        
        try:
            from yahooquery import Ticker
            ticker = Ticker(symbol)

            # Get current price
            price_data = ticker.price
            if not price_data or symbol not in price_data:
                return {}

            data = price_data[symbol]
            return {
                'symbol': symbol,
                'price': data.get('regularMarketPrice', 0),
                'previous_price': data.get('regularMarketPreviousClose', 0),
                'volume': data.get('regularMarketVolume', 0),
                'timestamp': datetime.now().isoformat(),
                'source': 'yahoo'
            }

        except Exception as e:
            logger.error(f"Yahoo Finance error for {symbol}: {e}")
            return {}

    async def _alpha_vantage_realtime(self, symbol: str) -> Dict[str, Any]:
        
        if not ALPHA_VANTAGE_API_KEY:
            return {}

        try:
            ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='json')
            data, _ = ts.get_quote_endpoint(symbol=symbol)

            if not data:
                return {}

            quote = data.get(symbol, {})
            return {
                'symbol': symbol,
                'price': float(quote.get('05. price', 0)),
                'previous_price': float(quote.get('08. previous close', 0)),
                'volume': int(quote.get('06. volume', 0)),
                'timestamp': datetime.now().isoformat(),
                'source': 'alpha_vantage'
            }

        except Exception as e:
            logger.error(f"Alpha Vantage error for {symbol}: {e}")
            return {}

    async def _newsapi_realtime(self, symbol: str) -> Dict[str, Any]:
        
        if not NEWS_API_KEY:
            return {}

        try:
            newsapi = NewsApiClient(api_key=NEWS_API_KEY)
            base_symbol = symbol.split('.')[0]

            articles = newsapi.get_everything(
                q=f'"{base_symbol}" stock',
                language='en',
                sort_by='publishedAt',
                page_size=5
            )

            if articles.get('status') != 'ok':
                return {}

            latest_article = articles.get('articles', [])[0] if articles.get('articles') else {}

            return {
                'symbol': symbol,
                'news_title': latest_article.get('title', ''),
                'news_url': latest_article.get('url', ''),
                'news_published': latest_article.get('publishedAt', ''),
                'timestamp': datetime.now().isoformat(),
                'source': 'newsapi'
            }

        except Exception as e:
            logger.error(f"NewsAPI error for {symbol}: {e}")
            return {}

    async def _fred_realtime(self, symbol: str) -> Dict[str, Any]:
        
        if not FRED_API_KEY:
            return {}

        try:
            fred = Fred(api_key=FRED_API_KEY)

            # Map common symbols to FRED series
            fred_mapping = {
                'NSEI': 'INDUNEMP',  # India Unemployment
                'BSESN': 'GDPC1',    # US GDP as proxy
            }

            fred_symbol = fred_mapping.get(symbol, 'INDUNEMP')
            series = fred.get_series_latest_release(fred_symbol)

            if series.empty:
                return {}

            latest_value = series.iloc[-1]

            return {
                'symbol': symbol,
                'macro_indicator': fred_symbol,
                'value': float(latest_value),
                'timestamp': datetime.now().isoformat(),
                'source': 'fred'
            }

        except Exception as e:
            logger.error(f"FRED error for {symbol}: {e}")
            return {}

    async def _moneycontrol_scrape(self, symbol: str) -> Dict[str, Any]:
        
        try:
            base_symbol = symbol.split('.')[0]
            url = f"https://www.moneycontrol.com/india/stockpricequote/{base_symbol}"

            response = await asyncio.get_event_loop().run_in_executor(
                None, rate_limited_get, url
            )

            if response.status_code != 200:
                return {}

            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract price
            price_elem = soup.find('div', {'class': 'inprice1'})
            if price_elem:
                price_text = safe_extract_text(price_elem)
                price = extract_numeric_value(price_text)

                return {
                    'symbol': symbol,
                    'price': price,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'moneycontrol'
                }

        except Exception as e:
            logger.error(f"Moneycontrol scrape error for {symbol}: {e}")

        return {}

    async def _bse_scrape(self, symbol: str) -> Dict[str, Any]:
        
        try:
            base_symbol = symbol.split('.')[0]
            url = f"https://www.bseindia.com/stock-share-price/{base_symbol}"

            response = await asyncio.get_event_loop().run_in_executor(
                None, rate_limited_get, url
            )

            if response.status_code != 200:
                return {}

            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract price
            price_elem = soup.find('span', {'class': 'strong'})
            if price_elem:
                price_text = safe_extract_text(price_elem)
                price = extract_numeric_value(price_text)

                return {
                    'symbol': symbol,
                    'price': price,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'bse'
                }

        except Exception as e:
            logger.error(f"BSE scrape error for {symbol}: {e}")

        return {}


real_time_manager = RealTimeDataManager()
async def get_real_time_data(symbols: List[str]) -> Dict[str, Any]:
    
    data = {}
    for symbol in symbols:
        # Try to get cached data first
        cached = _get_cached_data(symbol)
        if cached:
            data[symbol] = cached
        else:
            # Try to get fresh data
            try:
                fresh_data = await real_time_manager._get_data_with_fallback(symbol, 'yahoo')
                if fresh_data and 'error' not in fresh_data:
                    data[symbol] = fresh_data
                    _cache_data(symbol, fresh_data)
            except Exception as e:
                logger.error(f"Failed to get real-time data for {symbol}: {e}")

    return data