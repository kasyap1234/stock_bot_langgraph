

import time
import logging
import re
from typing import List, Dict, Any, Optional, Callable

import pandas as pd
import requests
from bs4 import BeautifulSoup
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.timeseries import TimeSeries
from yahooquery import Ticker

from config.config import ALPHA_VANTAGE_API_KEY, FRED_API_KEY, NEWS_API_KEY
from utils.error_handling import DataFetchingError, APIKeyMissingError
from .models import StockData, HistoricalData, FundamentalsData, create_stock_data
from .quality_validator import DataQualityValidator, validate_data_quality, validate_data, InsufficientDataError, ConstantPriceError

logger = logging.getLogger(__name__)

RATE_LIMIT_DELAY = 60 / 5  # 5 calls per minute for Alpha Vantage

# --- Unified Data Fetching with Fallbacks ---

class UnifiedDataFetcher:
    """
    A class to fetch financial data from multiple sources with fallbacks.
    """
    def __init__(self, symbol: str):
        self.symbol = symbol
        self._validate_symbol()

    def _validate_symbol(self):
        if not re.match(r'^[A-Z0-9\.\-\^]+$', self.symbol):
            raise ValueError(f"Invalid symbol format: {self.symbol}")

    def _fetch_with_retry(self, func: Callable, source_name: str, max_retries: int = 3, base_delay: float = 1.0):
        last_exception = None
        for attempt in range(max_retries):
            try:
                data = func()
                if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                    raise DataFetchingError("No data returned", source=source_name)
                logger.info(f"Successfully fetched data from {source_name} for {self.symbol}")
                return data
            except (requests.exceptions.RequestException, DataFetchingError, ValueError) as e:
                last_exception = e
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed for {source_name} on symbol {self.symbol}: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                time.sleep(delay)
        
        logger.error(f"All retries failed for {source_name} on symbol {self.symbol}. Last error: {last_exception}")
        raise DataFetchingError(f"Failed to fetch data from {source_name}", source=source_name) from last_exception

    def _fetch_yahooquery_history(self, period: str, interval: str) -> pd.DataFrame:
        """Primary source for historical data."""
        if self.symbol.endswith('.NS'):
            ticker_symbol = f"NSE:{self.symbol.split('.')[0]}"
        else:
            ticker_symbol = self.symbol
        ticker = Ticker(ticker_symbol, asynchronous=False)
        df = ticker.history(period=period, interval=interval)
        
        # Data validation
        if df.empty or 'close' not in df.columns or df['close'].isnull().all():
            raise DataFetchingError("Yahooquery returned empty or invalid dataframe.", "yahooquery")
        
        df.reset_index(inplace=True)
        df.rename(columns={
            'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        }, inplace=True)
        
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Check for all-zero values
        if (df['Close'] == 0).all() or (df['Volume'] == 0).all():
            raise DataFetchingError("All Close or Volume values are zero.", "yahooquery")
        
        return df

    def _fetch_alpha_vantage_history(self) -> pd.DataFrame:
        """Fallback source for historical data."""
        # FIXED: Raise ValueError if key not set
        if not settings.alpha_vantage_api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY is required but not set")

        ts = TimeSeries(key=settings.alpha_vantage_api_key, output_format='pandas')
        
        # Use a slightly longer delay to be safe
        time.sleep(RATE_LIMIT_DELAY)
        
        data, _ = ts.get_daily_adjusted(symbol=self.symbol, outputsize='full')
        
        if data.empty:
            raise DataFetchingError("Alpha Vantage returned no data.", "alpha_vantage")
        
        data.rename(columns={
            '1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close', '6. volume': 'Volume'
        }, inplace=True)
        
        data.index = pd.to_datetime(data.index)
        data.sort_index(ascending=True, inplace=True)
        
        df = data.reset_index().rename(columns={'index': 'Date'})
        
        # Check for all-zero values
        if (df['Close'] == 0).all() or (df['Volume'] == 0).all():
            raise DataFetchingError("All Close or Volume values are zero.", "alpha_vantage")
        
        return df

    def get_historical_data(self, period: str = "1y", interval: str = "1d") -> HistoricalData:
        """
        Fetches historical stock data with a primary (Yahooquery) and fallback (Alpha Vantage) source.
        """
        fetch_sources = [
            (lambda: self._fetch_yahooquery_history(period, interval), "yahooquery"),
            (self._fetch_alpha_vantage_history, "alpha_vantage")
        ]

        df = None
        for fetch_func, source_name in fetch_sources:
            try:
                df = self._fetch_with_retry(fetch_func, source_name=source_name)
                if df is not None and not df.empty:
                    break  # Success
            except (DataFetchingError, APIKeyMissingError) as e:
                logger.warning(f"Could not fetch data from {source_name} for {self.symbol}: {e}")
        
        if df is None or df.empty:
            logger.error(f"Failed to fetch historical data for {self.symbol} from all sources.")
            return []

        # Standardize and convert to list of StockData
        historical_data = []
        for _, row in df.iterrows():
            if pd.isna(row['Date']):
                continue
            
            try:
                stock_data = create_stock_data(
                    symbol=self.symbol,
                    date=pd.to_datetime(row['Date']).strftime("%Y-%m-%d"),
                    open_price=round(float(row["Open"]), 2),
                    high=round(float(row["High"]), 2),
                    low=round(float(row["Low"]), 2),
                    close=round(float(row["Close"]), 2),
                    volume=int(row["Volume"])
                )
                historical_data.append(stock_data)
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid data row for {self.symbol}: {row}. Error: {e}")

        if not historical_data:
            raise DataFetchingError("No valid historical data points after processing.", "processing")

        # Quality validation
        try:
            validator = DataQualityValidator()
            report = validate_data_quality(historical_data, self.symbol, validator)
            logger.info(f"Quality report for {self.symbol}: score={report.overall_quality_score}, issues={len(report.issues)}")
            
            if report.overall_quality_score < 0.85:
                raise DataFetchingError(f"Low data quality score: {report.overall_quality_score}", "quality_validation")
            
            validate_data(historical_data)
        except (InsufficientDataError, ConstantPriceError, DataFetchingError) as e:
            logger.warning(f"Quality validation failed for {self.symbol}: {e}")
            raise DataFetchingError(f"Data quality validation failed: {str(e)}", "quality_validation") from e

        logger.info(f"Successfully processed {len(historical_data)} data points for {self.symbol}")
        return historical_data


def get_stock_history(symbol: str, period: str = "1y", interval: str = "1d") -> HistoricalData:
    """
    High-level function to fetch historical data using the unified fetcher.
    """
    try:
        fetcher = UnifiedDataFetcher(symbol)
        return fetcher.get_historical_data(period=period, interval=interval)
    except (DataFetchingError, ValueError) as e:
        logger.error(f"Critical failure in fetching historical data for {symbol}: {e}")
        return []


def _retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0, source_name: str = "unknown"):
    last_exception = None
    for attempt in range(max_retries):
        try:
            result = func()
            if result is None or (isinstance(result, (dict, list)) and not result):
                 raise DataFetchingError("API returned no data", source=source_name)
            return result
        except (requests.exceptions.RequestException, DataFetchingError, APIKeyMissingError) as e:
            last_exception = e
            delay = base_delay * (2 ** attempt)
            logger.warning(
                f"Attempt {attempt + 1}/{max_retries} failed for {source_name}: {e}. Retrying in {delay:.2f}s..."
            )
            time.sleep(delay)
    
    logger.error(f"All retries failed for {source_name}. Last error: {last_exception}")
    raise DataFetchingError(f"Failed to fetch data from {source_name}", source=source_name) from last_exception

def get_fundamentals(symbol: str) -> FundamentalsData:
    """Fetches fundamental data for a given stock symbol from Alpha Vantage with retries."""
    def _fetch_alpha_vantage_fundamentals():
        # FIXED: Raise ValueError if key not set
        if not settings.alpha_vantage_api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY is required but not set")

        base_symbol = symbol.split('.')[0]
        fd = FundamentalData(key=settings.alpha_vantage_api_key, output_format='json')
        
        # Sequentially fetch data with delays
        time.sleep(RATE_LIMIT_DELAY)
        overview, _ = fd.get_company_overview(symbol=base_symbol)
        
        time.sleep(RATE_LIMIT_DELAY)
        income_stmt, _ = fd.get_income_statement_annual(symbol=base_symbol)

        time.sleep(RATE_LIMIT_DELAY)
        balance_sheet, _ = fd.get_balance_sheet_annual(symbol=base_symbol)

        time.sleep(RATE_LIMIT_DELAY)
        cash_flow, _ = fd.get_cash_flow_annual(symbol=base_symbol)

        if not overview and not income_stmt and not balance_sheet and not cash_flow:
            raise DataFetchingError("All fundamental data endpoints returned empty.", "alpha_vantage")

        return {
            "symbol": symbol,
            "overview": overview or {},
            "income_statement": income_stmt.get('annualReports', [])[:1] if income_stmt else [],
            "balance_sheet": balance_sheet.get('annualReports', [])[:1] if balance_sheet else [],
            "cash_flow": cash_flow.get('annualReports', [])[:1] if cash_flow else [],
        }

    try:
        return _retry_with_backoff(_fetch_alpha_vantage_fundamentals, max_retries=2, source_name="alpha_vantage")
    except DataFetchingError as e:
        logger.error(f"Failed to fetch fundamentals for {symbol}: {e}")
        return {"symbol": symbol, "error": str(e)}


def get_stock_info(symbol: str) -> Dict[str, Any]:
    """
    Fetches key stock information from Yahoo Finance with retries.
    """
    def _fetch_yahoo_info():
        ticker = Ticker(symbol)
        info = ticker.summary_detail.get(symbol, {})
        
        if not info:
            raise DataFetchingError("summary_detail returned no data", "yahooquery")

        # Basic validation
        if not info.get("longName") or not info.get("marketCap"):
             raise DataFetchingError("Incomplete data from yahooquery", "yahooquery")

        return {
            "symbol": symbol,
            "name": info.get("longName", "Unknown"),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "dividend_yield": info.get("dividendYield"),
            "beta": info.get("beta"),
            "currency": info.get("currency", "INR"),
        }

    try:
        return _retry_with_backoff(_fetch_yahoo_info, source_name="yahooquery")
    except DataFetchingError as e:
        logger.error(f"Failed to fetch stock info for {symbol}: {e}")
        return {"symbol": symbol, "error": str(e)}


def get_news_articles(symbol: str, max_articles: int = 10) -> List[Dict[str, Any]]:
    """
    Fetches news articles from NewsAPI with retries.
    """
    def _fetch_news_api_articles():
        if not NEWS_API_KEY or NEWS_API_KEY == "YOUR_NEWS_API_KEY_HERE":
            raise APIKeyMissingError("NewsAPI key not configured.", "newsapi")

        from newsapi import NewsApiClient
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        
        base_symbol = symbol.split('.')[0]
        query = f'"{base_symbol}" stock OR shares OR company'
        
        all_articles = newsapi.get_everything(
            q=query,
            language='en',
            sort_by='relevancy',
            page_size=min(max_articles, 100)
        )

        if all_articles.get('status') != 'ok':
            raise DataFetchingError(f"NewsAPI error: {all_articles.get('message', 'Unknown')}", "newsapi")
        
        articles = all_articles.get('articles', [])
        if not articles:
            # This is not an error, just no news
            logger.info(f"No news articles found for {symbol} on NewsAPI.")
            return []

        return [
            {
                'title': article.get('title', ''),
                'url': article.get('url', ''),
                'published_date': article.get('publishedAt', '')[:10] if article.get('publishedAt') else '',
                'source': article.get('source', {}).get('name', 'NewsAPI'),
                'description': article.get('description', ''),
                'content': article.get('content', '')
            } for article in articles[:max_articles]
        ]

    try:
        return _retry_with_backoff(_fetch_news_api_articles, source_name="newsapi")
    except DataFetchingError as e:
        logger.error(f"Failed to fetch news for {symbol}: {e}")
        return []


def get_fred_macro_data() -> Dict[str, Dict[str, Any]]:
    """
    Fetches macroeconomic data from FRED, with fallbacks for Indian data.
    """
    def _fetch_fred_data():
        if not FRED_API_KEY or FRED_API_KEY == "YOUR_FRED_API_KEY_HERE":
            raise APIKeyMissingError("FRED API key not configured.", "fred")

        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)
        macro_data = {}

        # FRED series map (India-focused with US fallbacks)
        series_map = {
            'INDUNEMP': ('Unemployment Rate', 'UNRATE'),  # India Unemployment -> US Unemployment
            'INDCPIALLMINMEI': ('Inflation Rate', 'CPIAUCSL'), # India CPI -> US CPI
            'SPASTT01INM661N': ('GDP Growth Rate YoY', 'GDPC1'), # India GDP Growth -> US GDP
        }

        for series_id, (indicator_name, fallback_id) in series_map.items():
            try:
                series = fred.get_series_latest_release(series_id)
                if series.empty:
                    raise DataFetchingError(f"Series '{series_id}' is empty.", "fred")
                
                latest_value = series.iloc[-1]
                
                # For GDP, calculate YoY growth if not already a growth rate
                if 'GDP' in indicator_name and 'Growth' not in indicator_name:
                    if len(series) >= 5: # Quarterly data needs ~5 points for YoY
                        prev_year_value = series.iloc[-5]
                        growth = ((latest_value - prev_year_value) / prev_year_value) * 100 if prev_year_value != 0 else 0
                        latest_value = growth
                    else:
                        continue # Not enough data for YoY

                macro_data[indicator_name] = {
                    'value': round(float(latest_value), 2),
                    'date': str(series.index[-1].date())
                }
                logger.info(f"Fetched {indicator_name} from FRED (Series: {series_id})")

            except Exception as e:
                logger.warning(f"Failed to fetch '{indicator_name}' from FRED (Series: {series_id}): {e}. Trying fallback '{fallback_id}'.")
                try:
                    series = fred.get_series_latest_release(fallback_id)
                    if series.empty:
                        raise DataFetchingError(f"Fallback series '{fallback_id}' is empty.", "fred")

                    latest_value = series.iloc[-1]
                    macro_data[indicator_name] = {
                        'value': round(float(latest_value), 2),
                        'date': str(series.index[-1].date())
                    }
                    logger.info(f"Fetched fallback {indicator_name} from FRED (Series: {fallback_id})")
                except Exception as e2:
                    logger.error(f"Failed to fetch fallback for {indicator_name} (Series: {fallback_id}): {e2}")

        return macro_data

    try:
        return _retry_with_backoff(_fetch_fred_data, source_name="fred")
    except DataFetchingError:
        return {} # Return empty if FRED fails entirely




def get_macro_data() -> Dict[str, Dict[str, Any]]:
    """
    Fetches and combines macro data from FRED, providing sensible defaults for missing Indian data.
    """
    logger.info("Fetching macro data from available sources...")
    
    macro_data = {}

    # Attempt to fetch data from FRED
    try:
        fred_data = get_fred_macro_data()
        macro_data.update(fred_data)
    except Exception as e:
        logger.error(f"Could not fetch any data from FRED: {e}. Proceeding with defaults.")

    # --- Provide sensible defaults for key Indian indicators if they are missing ---
    # These defaults are based on recent, typical values for the Indian economy.
    
    # RBI Repo Rate (Central Bank Interest Rate)
    if 'RBI Repo Rate' not in macro_data:
        # Scrape from a reliable source as a fallback if FRED fails. This is a placeholder for a real scraper.
        # In a real-world scenario, you'd use a library like BeautifulSoup here.
        # For now, we use a default.
        macro_data['RBI Repo Rate'] = {'value': 6.5, 'date': 'default', 'source': 'default'}
        logger.info("Using default for RBI Repo Rate (6.5%).")

    # Unemployment Rate
    if 'Unemployment Rate' not in macro_data:
        macro_data['Unemployment Rate'] = {'value': 7.5, 'date': 'default', 'source': 'default'}
        logger.info("Using default for Unemployment Rate (7.5%).")

    # GDP Growth Rate YoY
    if 'GDP Growth Rate YoY' not in macro_data:
        macro_data['GDP Growth Rate YoY'] = {'value': 6.8, 'date': 'default', 'source': 'default'}
        logger.info("Using default for GDP Growth Rate (6.8%).")
        
    # Inflation Rate
    if 'Inflation Rate' not in macro_data:
        macro_data['Inflation Rate'] = {'value': 5.0, 'date': 'default', 'source': 'default'}
        logger.info("Using default for Inflation Rate (5.0%).")

    logger.info(f"Final macro data collected for: {list(macro_data.keys())}")
    return macro_data