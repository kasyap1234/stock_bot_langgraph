

import time
import logging
import re
from typing import List, Dict, Any, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.timeseries import TimeSeries

from config.config import ALPHA_VANTAGE_API_KEY, FRED_API_KEY, NEWS_API_KEY
from utils.scraping_utils import rate_limited_get, extract_numeric_value, safe_extract_text, add_request_delay
from .models import StockData, HistoricalData, FundamentalsData, create_stock_data

logger = logging.getLogger(__name__)

RATE_LIMIT_DELAY = 60 / 5  # 5 calls per minute


def _retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0):
    
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e

            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error(f"All {max_retries + 1} attempts failed. Last error: {e}")
                raise last_exception

    raise last_exception


def get_stock_history(
    symbol: str,
    period: str = "1y",
    interval: str = "1d"
) -> HistoricalData:
    
    def _fetch_data():
        from yahooquery import Ticker
        ticker = Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            raise ValueError(f"No data found for symbol: {symbol}")

        logger.info(f"Fetched real historical data for {symbol}: {len(df)} rows")

        # Convert to list of StockData dictionaries
        historical_data = []
        for index, row in df.iterrows():
            stock_data = create_stock_data(
                symbol=symbol,
                date=index.strftime("%Y-%m-%d"),
                open_price=round(float(row["Open"]), 2),
                high=round(float(row["High"]), 2),
                low=round(float(row["Low"]), 2),
                close=round(float(row["Close"]), 2),
                volume=int(row["Volume"])
            )
            historical_data.append(stock_data)

        return historical_data

    try:
        return _retry_with_backoff(_fetch_data, max_retries=3)
    except Exception as e:
        logger.error(f"Failed to fetch historical data for {symbol}: {e}")
        return []


def get_fundamentals(symbol: str) -> FundamentalsData:
    
    if not ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY == "YOUR_ALPHA_VANTAGE_KEY_HERE":
        logger.warning("Alpha Vantage API key not configured. Skipping fundamentals fetch.")
        return {"error": "API key not configured"}

    # Alpha Vantage typically uses US symbols, so convert NSE to base symbol if needed
    base_symbol = symbol.split('.')[0] if '.' in symbol else symbol

    def _fetch_fundamentals():
        # Rate limit alpha vantage calls
        time.sleep(RATE_LIMIT_DELAY)

        fd = FundamentalData(key=ALPHA_VANTAGE_API_KEY, output_format='json')

        # Try to get company overview
        try:
            overview, _ = fd.get_company_overview(symbol=base_symbol)
        except Exception as e:
            logger.warning(f"Could not get company overview for {base_symbol}: {e}")
            overview = {}

        # Get income statement (annual)
        try:
            time.sleep(RATE_LIMIT_DELAY)
            income_stmt, _ = fd.get_income_statement_annual(symbol=base_symbol)
        except Exception as e:
            logger.warning(f"Could not get income statement for {base_symbol}: {e}")
            income_stmt = {}

        # Get balance sheet (annual)
        try:
            time.sleep(RATE_LIMIT_DELAY)
            balance_sheet, _ = fd.get_balance_sheet_annual(symbol=base_symbol)
        except Exception as e:
            logger.warning(f"Could not get balance sheet for {base_symbol}: {e}")
            balance_sheet = {}

        # Get cash flow (annual)
        try:
            time.sleep(RATE_LIMIT_DELAY)
            cash_flow, _ = fd.get_cash_flow_annual(symbol=base_symbol)
        except Exception as e:
            logger.warning(f"Could not get cash flow for {base_symbol}: {e}")
            cash_flow = {}

        return {
            "symbol": symbol,
            "overview": overview,
            "income_statement": income_stmt.get('annualReports', [])[:1],  # Latest year only
            "balance_sheet": balance_sheet.get('annualReports', [])[:1],
            "cash_flow": cash_flow.get('annualReports', [])[:1],
        }

    try:
        return _retry_with_backoff(_fetch_fundamentals, max_retries=2)
    except Exception as e:
        logger.error(f"Failed to fetch fundamentals for {symbol}: {e}")
        return {"symbol": symbol, "error": str(e)}


def get_stock_info(symbol: str) -> Dict[str, Any]:
    
    def _fetch_info():
        from yahooquery import Ticker
        ticker = Ticker(symbol)
        info = ticker.summary_detail

        if not info:
            raise ValueError(f"No information found for symbol: {symbol}")

        logger.info(f"Fetched real stock info for {symbol}")

        # Extract key information
        stock_info = {
            "symbol": symbol,
            "name": info.get("longName", "Unknown"),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "market_cap": info.get("marketCap", None),
            "pe_ratio": info.get("trailingPE", None),
            "dividend_yield": info.get("dividendYield", None),
            "beta": info.get("beta", None),
            "currency": info.get("currency", "INR"),
        }

        return stock_info

    try:
        return _retry_with_backoff(_fetch_info, max_retries=3)
    except Exception as e:
        logger.error(f"Failed to fetch stock info for {symbol}: {e}")
        return {"symbol": symbol, "error": str(e)}


def get_news_articles(symbol: str, max_articles: int = 10) -> List[Dict[str, Any]]:
    
    if not NEWS_API_KEY or NEWS_API_KEY == "":
        logger.warning("NewsAPI key not configured. Skipping news fetch.")
        return []

    try:
        from newsapi import NewsApiClient
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)

        # Clean symbol for search (remove .NS suffix)
        base_symbol = symbol.split('.')[0] if '.' in symbol else symbol

        # Search for news articles
        query = f'"{base_symbol}" stock OR shares OR company'
        all_articles = newsapi.get_everything(
            q=query,
            language='en',
            sort_by='relevancy',
            page_size=min(max_articles, 100)  # NewsAPI max is 100
        )

        if all_articles.get('status') != 'ok':
            logger.warning(f"NewsAPI error: {all_articles.get('message', 'Unknown error')}")
            return []

        articles = all_articles.get('articles', [])[:max_articles]

        # Format articles to match existing NewsItem structure
        news_data = []
        for article in articles:
            news_item = {
                'title': article.get('title', ''),
                'url': article.get('url', ''),
                'published_date': article.get('publishedAt', '')[:10] if article.get('publishedAt') else '',
                'source': article.get('source', {}).get('name', 'NewsAPI'),
                'description': article.get('description', ''),
                'content': article.get('content', '')
            }
            news_data.append(news_item)

        logger.info(f"Fetched {len(news_data)} news articles for {symbol} using NewsAPI")
        return news_data

    except Exception as e:
        logger.error(f"Failed to fetch news for {symbol}: {e}")
        return []


def get_fred_macro_data() -> Dict[str, Dict[str, Any]]:
    
    if not FRED_API_KEY or FRED_API_KEY == "":
        logger.warning("FRED API key not configured. Skipping FRED macro data fetch.")
        return {}

    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)

        macro_data = {}

        # FRED series for macro indicators (India preferred, US proxies as fallback)
        series_map = {
            'INDUNEMP': 'Unemployment Rate',  # India Unemployment Rate
            'GDPC1': 'Real Gross Domestic Product',  # US Real GDP
            'INDCPIALLMINMEI': 'Inflation Rate',  # India CPI
        }

        for series_id, indicator_name in series_map.items():
            try:
                series = fred.get_series_latest_release(series_id)
                if not series.empty:
                    if indicator_name == 'Real Gross Domestic Product':
                        # Calculate YoY growth for GDP
                        if len(series) >= 5:  # Need at least a few quarters
                            latest_value = series.iloc[-1]
                            prev_year_value = series.iloc[-5] if len(series) >= 5 else series.iloc[0]  # Approx 1 year back
                            growth = ((latest_value - prev_year_value) / prev_year_value) * 100 if prev_year_value != 0 else 0
                            latest_value = growth
                    else:
                        latest_value = series.iloc[-1]

                    macro_data[indicator_name] = {
                        'value': float(latest_value),
                        'date': str(series.index[-1].date())
                    }
                    logger.info(f"Fetched {indicator_name}: {latest_value} from FRED")
            except Exception as e:
                if indicator_name == 'Unemployment Rate':
                    logger.warning(f"Failed to fetch India {indicator_name} (series: {series_id}): {e}. Trying US proxy 'UNRATE'.")
                    try:
                        us_series = fred.get_series_latest_release('UNRATE')
                        if not us_series.empty:
                            latest_value = us_series.iloc[-1]
                            macro_data[indicator_name] = {
                                'value': float(latest_value),
                                'date': str(us_series.index[-1].date())
                            }
                            logger.info(f"Fetched US {indicator_name} (UNRATE): {latest_value} from FRED")
                        else:
                            logger.warning("US Unemployment series 'UNRATE' is empty.")
                    except Exception as e2:
                        logger.error(f"Failed to fetch US Unemployment 'UNRATE': {e2}")
                else:
                    logger.warning(f"Failed to fetch {indicator_name} from FRED (series: {series_id}): {e}")

        return macro_data

    except Exception as e:
        logger.error(f"Failed to fetch macro data from FRED: {e}")
        return {}




def get_macro_data() -> Dict[str, Dict[str, Any]]:
    
    logger.info("Fetching macro data from free APIs and web sources")

    macro_data = {}

    # Try FRED API first
    fred_data = get_fred_macro_data()
    macro_data.update(fred_data)

    # If still missing key indicators, provide defaults based on recent Indian economic data
    if 'RBI Repo Rate' not in macro_data:
        # India's RBI repo rate is typically around 6.5%
        macro_data['RBI Repo Rate'] = {'value': 6.5, 'date': 'default'}

    if 'Unemployment Rate' not in macro_data:
        # India's unemployment rate is typically around 6-8%
        macro_data['Unemployment Rate'] = {'value': 7.0, 'date': 'default'}

    if 'Real Gross Domestic Product' not in macro_data:
        # India's GDP growth rate is typically around 6-7%
        macro_data['Real Gross Domestic Product'] = {'value': 6.5, 'date': 'default'}

    logger.info(f"Collected macro data: {list(macro_data.keys())}")
    return macro_data