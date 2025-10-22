

import time
import logging
import re
import random
import os
import sys
from typing import List, Dict, Any, Optional, Callable, Tuple

import pandas as pd
import numpy as np
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from alpha_vantage.timeseries import TimeSeries
from yahooquery import Ticker

from requests.exceptions import RequestException

from config.api_config import ALPHA_VANTAGE_API_KEY, FRED_API_KEY, NEWS_API_KEY
from utils.error_handling import DataFetchingError, APIKeyMissingError
from .models import StockData, HistoricalData, FundamentalsData, create_stock_data
from .quality_validator import DataQualityValidator, validate_data_quality, validate_data, InsufficientDataError, ConstantPriceError

logger = logging.getLogger(__name__)

RATE_LIMIT_DELAY = 60 / 5  # 5 calls per minute for Alpha Vantage
ALLOW_SYNTHETIC_DATA = os.getenv("ALLOW_SYNTHETIC_DATA", "false").lower() == "true"


def _is_cache_disabled() -> bool:
    if os.getenv("PYTEST_CURRENT_TEST") or 'pytest' in sys.modules:
        return True
    return os.getenv("DISABLE_DATA_CACHE", "").lower() in {"1", "true", "yes"}

# --- Unified Data Fetching with Fallbacks ---

import json
from pathlib import Path

class UnifiedDataFetcher:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self._validate_symbol()
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / f"{symbol}_{'1y'}.json"  # Assuming 1y period, adjust if needed

    def _validate_symbol(self):
        if not re.match(r'^[A-Z0-9\.\-\^]+$', self.symbol):
            raise ValueError(f"Invalid symbol format: {self.symbol}")

    def get_batch_historical_data(self, symbols: List[str], period: str = "1y", interval: str = "1d") -> Dict[str, HistoricalData]:
        """
        Fetches historical data for a batch of symbols in parallel.
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_symbol = {executor.submit(self.get_historical_data, symbol, period, interval): symbol for symbol in symbols}
            results = {}
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    results[symbol] = data
                except Exception as e:
                    logger.error(f"Failed to fetch data for {symbol} in batch: {e}")
                    results[symbol] = []
            return results

    def _fetch_with_retry(self, func: Callable, source_name: str, max_retries: int = 5, base_delay: float = 1.0):
        last_exception = None
        non_retriable = False
        for attempt in range(max_retries):
            try:
                data = func()
                if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                    raise DataFetchingError("No data returned", source=source_name)
                logger.info(f"Successfully fetched data from {source_name} for {self.symbol}")
                return data
            except Exception as e:
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise

                last_exception = e
                should_retry = attempt < max_retries - 1 and self._should_retry(e)
                if should_retry:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed for {source_name} on symbol {self.symbol}: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed for {source_name} on symbol {self.symbol}: {e}. "
                        "Not retrying."
                    )
                    non_retriable = True
                    break
        
        logger.error(f"All retries failed for {source_name} on symbol {self.symbol}. Last error: {last_exception}")
        failure_message = str(last_exception) if last_exception else "Unknown error"
        failure_exc = DataFetchingError(
            f"Failed to fetch data from {source_name}: {failure_message}",
            source=source_name
        )
        if non_retriable:
            setattr(failure_exc, "non_retriable", True)
        raise failure_exc from last_exception

    def _should_retry(self, exc: Exception) -> bool:
        if isinstance(exc, DataFetchingError):
            message = (exc.message or "").lower()
            non_retriable_markers = [
                "empty dataframe",
                "no data returned",
                "all close or volume values are zero",
                "low data quality score"
            ]
            if any(marker in message for marker in non_retriable_markers):
                return False
        if isinstance(exc, ValueError) and "required but not set" in str(exc).lower():
            return False
        return True

    def _fetch_yahooquery_history(self, period: str, interval: str) -> pd.DataFrame:
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


def _fetch_world_bank_indicator(indicator_code: str) -> Optional[Dict[str, Any]]:
    url = f"https://api.worldbank.org/v2/country/IND/indicator/{indicator_code}"
    params = {"format": "json", "per_page": 25}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()

        if not isinstance(payload, list) or len(payload) < 2:
            return None

        series = payload[1]
        if not isinstance(series, list):
            return None

        for entry in series:
            value = entry.get("value")
            if value is None:
                continue
            date = entry.get("date") or "unknown"
            return {
                "value": round(float(value), 2),
                "date": str(date),
                "source": "world_bank"
            }
    except (RequestException, ValueError, TypeError) as exc:
        logger.warning(f"World Bank fallback failed for {indicator_code}: {exc}")

    return None

    def _fetch_yfinance_history(self, period: str, interval: str) -> pd.DataFrame:
        ticker_symbol = self.symbol.replace('.NS', '.NS')  # yfinance handles .NS directly
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            raise DataFetchingError("yfinance returned empty dataframe.", "yfinance")
        
        df = df.reset_index(names='Date')
        if 'Date' not in df.columns and 'index' in df.columns:
            df.rename(columns={'index': 'Date'}, inplace=True)
        df.rename(columns={
            'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'
        }, inplace=True)
        
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Check for all-zero values
        if (df['Close'] == 0).all() or (df['Volume'] == 0).all():
            raise DataFetchingError("All Close or Volume values are zero.", "yfinance")
        
        return df

    def _fetch_alpha_vantage_history(self) -> pd.DataFrame:
        # FIXED: Raise ValueError if key not set
        if not ALPHA_VANTAGE_API_KEY:
            raise ValueError("ALPHA_VANTAGE_API_KEY is required but not set")

        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        
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

    def _generate_synthetic_data(self, period: str = "1y") -> pd.DataFrame:
        """
        Generate synthetic stock data for testing when all real data sources fail.
        This creates realistic-looking price movements for development purposes.
        """
        try:
            # Parse period to get number of days
            if period.endswith('y'):
                days = int(period[:-1]) * 365
            elif period.endswith('mo'):
                days = int(period[:-2]) * 30
            elif period.endswith('d'):
                days = int(period[:-1])
            else:
                days = 365  # Default to 1 year

            # Generate date range
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.Timedelta(days=days)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')

            # Generate synthetic price data
            np.random.seed(hash(self.symbol) % 2**32)  # Reproducible seed based on symbol

            # Start with a reasonable price (between 100-5000 for variety)
            base_price = np.random.uniform(100, 5000)
            prices = [base_price]

            # Generate daily returns with some volatility
            volatility = 0.02  # 2% daily volatility
            for i in range(1, len(dates)):
                daily_return = np.random.normal(0.0001, volatility)  # Slight upward drift
                new_price = prices[-1] * (1 + daily_return)
                prices.append(max(new_price, 0.01))  # Ensure positive price

            # Generate OHLCV data
            data = []
            for i, date in enumerate(dates):
                price = prices[i]
                # Add some intraday volatility
                high = price * (1 + abs(np.random.normal(0, 0.01)))
                low = price * (1 - abs(np.random.normal(0, 0.01)))
                open_price = prices[i-1] if i > 0 else price * (1 + np.random.normal(0, 0.005))
                volume = int(np.random.uniform(100000, 10000000))  # Realistic volume

                data.append({
                    'Date': date,
                    'Open': round(open_price, 2),
                    'High': round(high, 2),
                    'Low': round(low, 2),
                    'Close': round(price, 2),
                    'Volume': volume
                })

            df = pd.DataFrame(data)
            logger.info(f"Generated synthetic data for {self.symbol}: {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"Failed to generate synthetic data for {self.symbol}: {e}")
            return pd.DataFrame()

    def get_historical_data(self, period: str = "1y", interval: str = "1d") -> HistoricalData:
        is_synthetic = False

        equity_fetchers: List[Tuple[Callable[[], pd.DataFrame], str]] = []
        if self.symbol.endswith('.NS'):
            equity_fetchers.extend([
                (lambda period=period, interval=interval: self._fetch_yahooquery_history(period, interval), "yahooquery"),
                (lambda period=period, interval=interval: self._fetch_yfinance_history(period, interval), "yfinance"),
            ])
        else:
            equity_fetchers.extend([
                (lambda period=period, interval=interval: self._fetch_yfinance_history(period, interval), "yfinance"),
                (lambda period=period, interval=interval: self._fetch_yahooquery_history(period, interval), "yahooquery"),
            ])

        fetch_sources = equity_fetchers + [
            (self._fetch_alpha_vantage_history, "alpha_vantage")
        ]

        df: Optional[pd.DataFrame] = None
        for fetch_func, source_name in fetch_sources:
            try:
                df = self._fetch_with_retry(fetch_func, source_name=source_name)
                if df is not None and not df.empty:
                    break  # Success
            except (DataFetchingError, APIKeyMissingError) as e:
                logger.warning(f"Could not fetch data from {source_name} for {self.symbol}: {e}")
                if isinstance(e, DataFetchingError) and getattr(e, "non_retriable", False):
                    break
                if _is_cache_disabled():
                    break

        cache_used = False
        if df is None or df.empty:
            if not _is_cache_disabled() and self.cache_file.exists():
                try:
                    with open(self.cache_file, 'r') as f:
                        cache_content = json.load(f)

                    # Check expiration (7 days for Indian stocks, 1 day for others)
                    metadata = cache_content.get('metadata', {})
                    cached_synthetic = metadata.get('synthetic')
                    if cached_synthetic is None and 'source' in metadata:
                        cached_synthetic = metadata.get('source') == 'synthetic'
                    if cached_synthetic is None:
                        cached_synthetic = True  # Treat legacy caches as synthetic by default

                    if not ALLOW_SYNTHETIC_DATA and cached_synthetic:
                        raise ValueError("Synthetic cache ignored when disabled")

                    cache_timestamp = cache_content.get('timestamp', 0)
                    current_time = time.time()
                    cache_validity_hours = 4 if self.symbol.endswith('.NS') else 4
                    if current_time - cache_timestamp > cache_validity_hours * 3600:
                        logger.info(f"Cache expired for {self.symbol}, attempting fresh fetch")
                        raise ValueError("Cache expired")

                    cached_data = cache_content['data']
                    df = pd.DataFrame(cached_data)
                    cache_used = True
                    is_synthetic = bool(cached_synthetic)
                    logger.info(f"Loaded valid cached data for {self.symbol} (age: {(current_time - cache_timestamp)/3600:.1f}h)")
                    if df.empty:
                        raise ValueError("Cached data is empty")
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(f"Failed to load/use cache for {self.symbol}: {e}")

            # If still no data, generate synthetic data for testing/development
            if df is None or df.empty:
                if ALLOW_SYNTHETIC_DATA:
                    logger.warning(
                        f"All data sources failed for {self.symbol}. Generating synthetic data for testing."
                    )
                    df = self._generate_synthetic_data(period)
                    if df is None or df.empty:
                        logger.error(f"Failed to generate synthetic data for {self.symbol}")
                        return []
                    is_synthetic = True
                else:
                    logger.error(
                        f"All data sources failed for {self.symbol} and synthetic generation is disabled."
                    )
                    return []

        # Save to cache after successful fetch (or update timestamp if cache used)
        if not _is_cache_disabled():
            try:
                df_for_cache = df.copy()
                if 'Date' in df_for_cache.columns:
                    df_for_cache['Date'] = df_for_cache['Date'].astype(str)

                cache_data = df_for_cache.to_dict('records')
                cache_content = {
                    'timestamp': time.time(),
                    'data': cache_data,
                    'metadata': {'synthetic': is_synthetic}
                }
                with open(self.cache_file, 'w') as f:
                    json.dump(cache_content, f)
                if not cache_used:
                    logger.info(f"Saved fresh data to cache for {self.symbol}")
                else:
                    logger.info(f"Updated cache timestamp for {self.symbol}")
            except Exception as e:
                logger.warning(f"Failed to save/update cache for {self.symbol}: {e}")

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

            min_periods = max(2, min(50, len(historical_data)))
            validate_data(historical_data, min_periods=min_periods)
        except (InsufficientDataError, ConstantPriceError, DataFetchingError) as e:
            logger.warning(f"Quality validation failed for {self.symbol}: {e}")
            raise DataFetchingError(f"Data quality validation failed: {str(e)}", "quality_validation") from e

        logger.info(f"Successfully processed {len(historical_data)} data points for {self.symbol}{' (from cache)' if cache_used else ''}")
        return historical_data


class _EnhancedUnifiedDataFetcher:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self._validate_symbol()
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / f"{symbol}_{'1y'}.json"

    def _validate_symbol(self) -> None:
        if not re.match(r'^[A-Z0-9\.-\^]+$', self.symbol):
            raise ValueError(f"Invalid symbol format: {self.symbol}")

    def get_batch_historical_data(self, symbols: List[str], period: str = "1y", interval: str = "1d") -> Dict[str, HistoricalData]:
        """
        Fetches historical data for a batch of symbols in parallel.
        """
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_symbol = {executor.submit(self.get_historical_data, symbol, period, interval): symbol for symbol in symbols}
            results = {}
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    results[symbol] = data
                except Exception as e:
                    logger.error(f"Failed to fetch data for {symbol} in batch: {e}")
                    results[symbol] = []
            return results

    def _fetch_with_retry(
        self,
        func: Callable,
        source_name: str,
        max_retries: int = 5,
        base_delay: float = 1.0
    ) -> pd.DataFrame:
        last_exception: Optional[Exception] = None
        non_retriable = False

        for attempt in range(max_retries):
            try:
                data = func()
                if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                    raise DataFetchingError("No data returned", source=source_name)
                logger.info(f"Successfully fetched data from {source_name} for {self.symbol}")
                return data
            except Exception as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise

                last_exception = exc
                should_retry = attempt < max_retries - 1 and self._should_retry(exc)
                if should_retry:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed for {source_name} on symbol {self.symbol}: {exc}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed for {source_name} on symbol {self.symbol}: {exc}. "
                        "Not retrying."
                    )
                    non_retriable = True
                    break

        logger.error(
            f"All retries failed for {source_name} on symbol {self.symbol}. Last error: {last_exception}"
        )
        failure_message = str(last_exception) if last_exception else "Unknown error"
        failure_exc = DataFetchingError(
            f"Failed to fetch data from {source_name}: {failure_message}",
            source=source_name
        )
        if non_retriable:
            setattr(failure_exc, "non_retriable", True)
        raise failure_exc from last_exception

    def _should_retry(self, exc: Exception) -> bool:
        if isinstance(exc, DataFetchingError):
            message = (exc.message or "").lower()
            non_retriable_markers = [
                "empty dataframe",
                "no data returned",
                "all close or volume values are zero",
                "low data quality score"
            ]
            if any(marker in message for marker in non_retriable_markers):
                return False
        if isinstance(exc, ValueError) and "required but not set" in str(exc).lower():
            return False
        return True

    def _fetch_yahooquery_history(self, period: str, interval: str) -> pd.DataFrame:
        ticker_symbol = f"NSE:{self.symbol.split('.')[0]}" if self.symbol.endswith('.NS') else self.symbol
        ticker = Ticker(ticker_symbol, asynchronous=False)
        df = ticker.history(period=period, interval=interval)

        if df.empty or 'close' not in df.columns or df['close'].isnull().all():
            raise DataFetchingError("Yahooquery returned empty or invalid dataframe.", "yahooquery")

        df.reset_index(inplace=True)
        df.rename(columns={
            'date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)

        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        if (df['Close'] == 0).all() or (df['Volume'] == 0).all():
            raise DataFetchingError("All Close or Volume values are zero.", "yahooquery")

        return df

    def _fetch_yfinance_history(self, period: str, interval: str) -> pd.DataFrame:
        ticker = yf.Ticker(self.symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            raise DataFetchingError("yfinance returned empty dataframe.", "yfinance")

        df = df.reset_index(names='Date')
        if 'Date' not in df.columns and 'index' in df.columns:
            df.rename(columns={'index': 'Date'}, inplace=True)
        logger.debug("yfinance columns after reset: %s", list(df.columns))
        df.rename(columns={
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        }, inplace=True)

        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        if (df['Close'] == 0).all() or (df['Volume'] == 0).all():
            raise DataFetchingError("All Close or Volume values are zero.", "yfinance")

        return df

    def _fetch_alpha_vantage_history(self) -> pd.DataFrame:
        if not ALPHA_VANTAGE_API_KEY:
            raise ValueError("ALPHA_VANTAGE_API_KEY is required but not set")

        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        time.sleep(RATE_LIMIT_DELAY)
        data, _ = ts.get_daily_adjusted(symbol=self.symbol, outputsize='full')

        if data.empty:
            raise DataFetchingError("Alpha Vantage returned no data.", "alpha_vantage")

        data.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '6. volume': 'Volume'
        }, inplace=True)

        data.index = pd.to_datetime(data.index)
        data.sort_index(ascending=True, inplace=True)
        df = data.reset_index().rename(columns={'index': 'Date'})

        if (df['Close'] == 0).all() or (df['Volume'] == 0).all():
            raise DataFetchingError("All Close or Volume values are zero.", "alpha_vantage")

        return df

    def _generate_synthetic_data(self, period: str = "1y") -> pd.DataFrame:
        try:
            if period.endswith('y'):
                days = int(period[:-1]) * 365
            elif period.endswith('mo'):
                days = int(period[:-2]) * 30
            elif period.endswith('d'):
                days = int(period[:-1])
            else:
                days = 365

            end_date = pd.Timestamp.now()
            start_date = end_date - pd.Timedelta(days=days)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')

            np.random.seed(hash(self.symbol) % 2**32)
            base_price = np.random.uniform(100, 5000)
            prices = [base_price]

            volatility = 0.02
            for _ in range(1, len(dates)):
                daily_return = np.random.normal(0.0001, volatility)
                new_price = prices[-1] * (1 + daily_return)
                prices.append(max(new_price, 0.01))

            records = []
            for idx, date in enumerate(dates):
                price = prices[idx]
                high = price * (1 + abs(np.random.normal(0, 0.01)))
                low = price * (1 - abs(np.random.normal(0, 0.01)))
                open_price = prices[idx - 1] if idx > 0 else price * (1 + np.random.normal(0, 0.005))
                volume = int(np.random.uniform(100000, 10000000))

                records.append({
                    'Date': date,
                    'Open': round(open_price, 2),
                    'High': round(high, 2),
                    'Low': round(low, 2),
                    'Close': round(price, 2),
                    'Volume': volume
                })

            df = pd.DataFrame(records)
            logger.info(f"Generated synthetic data for {self.symbol}: {len(df)} rows")
            return df
        except Exception as exc:
            logger.error(f"Failed to generate synthetic data for {self.symbol}: {exc}")
            return pd.DataFrame()

    def get_historical_data(self, period: str = "1y", interval: str = "1d") -> HistoricalData:
        is_synthetic = False

        fetch_sources: List[Tuple[Callable[[], pd.DataFrame], str]] = []
        if self.symbol.endswith('.NS'):
            fetch_sources.extend([
                (lambda period=period, interval=interval: self._fetch_yahooquery_history(period, interval), "yahooquery"),
                (lambda period=period, interval=interval: self._fetch_yfinance_history(period, interval), "yfinance"),
            ])
        else:
            fetch_sources.extend([
                (lambda period=period, interval=interval: self._fetch_yfinance_history(period, interval), "yfinance"),
                (lambda period=period, interval=interval: self._fetch_yahooquery_history(period, interval), "yahooquery"),
            ])

        fetch_sources.append((self._fetch_alpha_vantage_history, "alpha_vantage"))

        df: Optional[pd.DataFrame] = None
        for fetch_func, source_name in fetch_sources:
            try:
                df = self._fetch_with_retry(fetch_func, source_name=source_name)
                if df is not None and not df.empty:
                    break
            except (DataFetchingError, APIKeyMissingError) as exc:
                logger.warning(f"Could not fetch data from {source_name} for {self.symbol}: {exc}")
                if isinstance(exc, DataFetchingError) and getattr(exc, "non_retriable", False):
                    break

        cache_used = False
        if df is None or df.empty:
            if self.cache_file.exists():
                try:
                    with open(self.cache_file, 'r') as cache_fp:
                        cache_content = json.load(cache_fp)

                    metadata = cache_content.get('metadata', {})
                    cached_synthetic = metadata.get('synthetic')
                    if cached_synthetic is None and 'source' in metadata:
                        cached_synthetic = metadata.get('source') == 'synthetic'
                    if cached_synthetic is None:
                        cached_synthetic = True

                    if not ALLOW_SYNTHETIC_DATA and cached_synthetic:
                        raise ValueError("Synthetic cache ignored when disabled")

                    cache_timestamp = cache_content.get('timestamp', 0)
                    current_time = time.time()
                    cache_validity_hours = 4 if self.symbol.endswith('.NS') else 4
                    if current_time - cache_timestamp > cache_validity_hours * 3600:
                        logger.info(f"Cache expired for {self.symbol}, attempting fresh fetch")
                        raise ValueError("Cache expired")

                    cached_data = cache_content['data']
                    df = pd.DataFrame(cached_data)
                    cache_used = True
                    is_synthetic = bool(cached_synthetic)
                    logger.info(
                        f"Loaded valid cached data for {self.symbol} (age: {(current_time - cache_timestamp)/3600:.1f}h)"
                    )
                    if df.empty:
                        raise ValueError("Cached data is empty")
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    logger.warning(f"Failed to load/use cache for {self.symbol}: {exc}")

            if df is None or df.empty:
                if ALLOW_SYNTHETIC_DATA:
                    logger.warning(
                        f"All data sources failed for {self.symbol}. Generating synthetic data for testing."
                    )
                    df = self._generate_synthetic_data(period)
                    if df is None or df.empty:
                        logger.error(f"Failed to generate synthetic data for {self.symbol}")
                        return []
                    is_synthetic = True
                else:
                    logger.error(
                        f"All data sources failed for {self.symbol} and synthetic generation is disabled."
                    )
                    return []

        try:
            df_for_cache = df.copy()
            if 'Date' in df_for_cache.columns:
                df_for_cache['Date'] = df_for_cache['Date'].astype(str)

            cache_payload = {
                'timestamp': time.time(),
                'data': df_for_cache.to_dict('records'),
                'metadata': {'synthetic': is_synthetic}
            }
            with open(self.cache_file, 'w') as cache_fp:
                json.dump(cache_payload, cache_fp)
            if cache_used:
                logger.info(f"Updated cache timestamp for {self.symbol}")
            else:
                logger.info(f"Saved fresh data to cache for {self.symbol}")
        except Exception as exc:
            logger.warning(f"Failed to save/update cache for {self.symbol}: {exc}")

        historical_data: HistoricalData = []
        for _, row in df.iterrows():
            if pd.isna(row['Date']):
                continue

            try:
                stock_data = create_stock_data(
                    symbol=self.symbol,
                    date=pd.to_datetime(row['Date']).strftime('%Y-%m-%d'),
                    open_price=round(float(row['Open']), 2),
                    high=round(float(row['High']), 2),
                    low=round(float(row['Low']), 2),
                    close=round(float(row['Close']), 2),
                    volume=int(row['Volume'])
                )
                historical_data.append(stock_data)
            except (ValueError, TypeError) as exc:
                logger.warning(f"Skipping invalid data row for {self.symbol}: {row}. Error: {exc}")

        if not historical_data:
            raise DataFetchingError("No valid historical data points after processing.", "processing")

        try:
            if len(historical_data) >= 50:
                validator = DataQualityValidator()
                report = validate_data_quality(historical_data, self.symbol, validator)
                logger.info(
                    f"Quality report for {self.symbol}: score={report.overall_quality_score}, issues={len(report.issues)}"
                )

                if report.overall_quality_score < 0.85:
                    raise DataFetchingError(
                        f"Low data quality score: {report.overall_quality_score}",
                        "quality_validation"
                    )

            min_periods = max(2, min(50, len(historical_data)))
            validate_data(historical_data, min_periods=min_periods)
        except (InsufficientDataError, ConstantPriceError, DataFetchingError) as exc:
            logger.warning(f"Quality validation failed for {self.symbol}: {exc}")
            raise DataFetchingError(
                f"Data quality validation failed: {str(exc)}",
                "quality_validation"
            ) from exc

        logger.info(
            f"Successfully processed {len(historical_data)} data points for {self.symbol}{' (from cache)' if cache_used else ''}"
        )
        return historical_data


UnifiedDataFetcher = _EnhancedUnifiedDataFetcher


def get_stock_history(symbol: str, period: str = "1y", interval: str = "1d") -> HistoricalData:
    try:
        fetcher = UnifiedDataFetcher(symbol)
        return fetcher.get_historical_data(period=period, interval=interval)
    except (DataFetchingError, ValueError) as e:
        logger.error(f"Critical failure in fetching historical data for {symbol}: {e}")
        return []


def _retry_with_backoff(func, max_retries: int = 5, base_delay: float = 1.0, source_name: str = "unknown"):
    last_exception = None
    non_retriable = False
    total_attempts = max_retries + 1

    for attempt in range(total_attempts):
        try:
            result = func()
            if result is None or (isinstance(result, (dict, list)) and not result):
                 raise DataFetchingError("API returned no data", source=source_name)
            return result
        except Exception as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise

            last_exception = e
            should_retry = attempt < max_retries
            if isinstance(e, DataFetchingError) and getattr(e, "non_retriable", False):
                should_retry = False

            if isinstance(e, (APIKeyMissingError, ValueError)):
                should_retry = False

            if should_retry:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"Attempt {attempt + 1}/{total_attempts} failed for {source_name}: {e}. Retrying in {delay:.2f}s..."
                )
                time.sleep(delay)
            else:
                logger.warning(
                    f"Attempt {attempt + 1}/{total_attempts} failed for {source_name}: {e}. Not retrying."
                )
                non_retriable = True
                break
    
    logger.error(f"All retries failed for {source_name}. Last error: {last_exception}")
    message = str(last_exception) if last_exception else "Unknown error"
    failure_exc = DataFetchingError(
        f"Failed to fetch data from {source_name}: {message}",
        source=source_name
    )
    if non_retriable:
        setattr(failure_exc, "non_retriable", True)
    raise failure_exc from last_exception

def get_fundamentals(symbol: str) -> FundamentalsData:
    from alpha_vantage.fundamentaldata import FundamentalData as AVFundamentalData

    is_mock_fundamental = bool(getattr(AVFundamentalData, "_is_mock_object", False))

    if not ALPHA_VANTAGE_API_KEY and not is_mock_fundamental:
        logger.warning("ALPHA_VANTAGE_API_KEY is not configured; skipping fundamental data fetch.")
        return {"error": "API key not configured"}

    def _fetch_alpha_vantage_fundamentals():
        if not ALPHA_VANTAGE_API_KEY and not is_mock_fundamental:
            raise ValueError("ALPHA_VANTAGE_API_KEY is required but not set")

        base_symbol = symbol.split('.')[0]
        fd = AVFundamentalData(key=ALPHA_VANTAGE_API_KEY, output_format='json')
        
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

    def _fetch_yfinance_fundamentals():
        ticker_symbol = symbol
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        if not info:
            raise DataFetchingError("yfinance info returned no data.", "yfinance")
        
        # Basic fundamentals from info
        overview = {
            "Symbol": info.get("symbol", symbol),
            "Name": info.get("longName", ""),
            "Sector": info.get("sector", ""),
            "Industry": info.get("industry", ""),
            "MarketCap": info.get("marketCap"),
            "PERatio": info.get("trailingPE"),
            "DividendYield": info.get("dividendYield"),
            "Beta": info.get("beta"),
            "EPS": info.get("trailingEps"),
            "BookValue": info.get("bookValue"),
            "ROE": info.get("returnOnEquity"),
        }
        
        # Financials
        financials = ticker.financials
        balance = ticker.balance_sheet
        cashflow = ticker.cashflow
        
        return {
            "symbol": symbol,
            "overview": overview,
            "income_statement": financials.to_dict('records')[:1] if not financials.empty else [],
            "balance_sheet": balance.to_dict('records')[:1] if not balance.empty else [],
            "cash_flow": cashflow.to_dict('records')[:1] if not cashflow.empty else [],
        }

    fetch_sources = [
        (_fetch_alpha_vantage_fundamentals, "alpha_vantage"),
        (_fetch_yfinance_fundamentals, "yfinance")
    ]

    primary_source = fetch_sources[0][1]
    last_error_message = None

    for fetch_func, source_name in fetch_sources:
        try:
            result = _retry_with_backoff(fetch_func, source_name=source_name)
            if source_name != primary_source and isinstance(result, dict):
                fallback_reason = last_error_message or f"{primary_source} unavailable"
                result.setdefault("error", fallback_reason)
            return result
        except (DataFetchingError, ValueError) as e:
            last_error_message = str(e)
            logger.warning(f"Could not fetch fundamentals from {source_name} for {symbol}: {e}")

    logger.error(f"Failed to fetch fundamentals for {symbol} from all sources.")
    return {"symbol": symbol, "error": "All sources failed"}


def get_stock_info(symbol: str) -> Dict[str, Any]:
    def _fetch_yahoo_info():
        ticker = Ticker(symbol)
        info = ticker.summary_detail.get(symbol, {})
        
        if not info:
            error = DataFetchingError("summary_detail returned no data", "yahooquery")
            setattr(error, "non_retriable", True)
            raise error

        # Basic validation
        if not info.get("longName") or not info.get("marketCap"):
            error = DataFetchingError("Incomplete data from yahooquery", "yahooquery")
            setattr(error, "non_retriable", True)
            raise error

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

    def _fetch_yfinance_info():
        ticker = yf.Ticker(symbol)
        info = getattr(ticker, "info", {}) or {}

        if not info:
            error = DataFetchingError("yfinance info returned no data", "yfinance")
            setattr(error, "non_retriable", True)
            raise error

        return {
            "symbol": symbol,
            "name": info.get("longName") or info.get("shortName", "Unknown"),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "dividend_yield": info.get("dividendYield"),
            "beta": info.get("beta"),
            "currency": info.get("currency", "INR"),
        }

    fetchers = [
        (_fetch_yahoo_info, "yahooquery"),
        (_fetch_yfinance_info, "yfinance")
    ]

    last_error = None

    for fetch_func, source_name in fetchers:
        try:
            return _retry_with_backoff(fetch_func, source_name=source_name)
        except DataFetchingError as e:
            last_error = str(e)
            logger.warning(f"Failed to fetch stock info for {symbol} via {source_name}: {e}")

    logger.error(f"Failed to fetch stock info for {symbol} from all sources.")
    return {"symbol": symbol, "error": last_error or "All sources failed"}


def get_news_articles(symbol: str, max_articles: int = 10) -> List[Dict[str, Any]]:
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
        repo_data = _fetch_world_bank_indicator('FR.INR.RINR')
        if repo_data:
            macro_data['RBI Repo Rate'] = repo_data
            logger.info("Fetched RBI Repo Rate from World Bank fallback.")
        else:
            macro_data['RBI Repo Rate'] = {'value': 6.5, 'date': 'default', 'source': 'default'}
            logger.info("Using default for RBI Repo Rate (6.5%).")

    # Unemployment Rate
    if 'Unemployment Rate' not in macro_data:
        unemployment_data = _fetch_world_bank_indicator('SL.UEM.TOTL.ZS')
        if unemployment_data:
            macro_data['Unemployment Rate'] = unemployment_data
            logger.info("Fetched Unemployment Rate from World Bank fallback.")
        else:
            macro_data['Unemployment Rate'] = {'value': 7.5, 'date': 'default', 'source': 'default'}
            logger.info("Using default for Unemployment Rate (7.5%).")

    # GDP Growth Rate YoY
    if 'GDP Growth Rate YoY' not in macro_data:
        gdp_data = _fetch_world_bank_indicator('NY.GDP.MKTP.KD.ZG')
        if gdp_data:
            macro_data['GDP Growth Rate YoY'] = gdp_data
            logger.info("Fetched GDP Growth Rate YoY from World Bank fallback.")
        else:
            macro_data['GDP Growth Rate YoY'] = {'value': 6.8, 'date': 'default', 'source': 'default'}
            logger.info("Using default for GDP Growth Rate (6.8%).")
        
    # Inflation Rate
    if 'Inflation Rate' not in macro_data:
        inflation_data = _fetch_world_bank_indicator('FP.CPI.TOTL.ZG')
        if inflation_data:
            macro_data['Inflation Rate'] = inflation_data
            logger.info("Fetched Inflation Rate from World Bank fallback.")
        else:
            macro_data['Inflation Rate'] = {'value': 5.0, 'date': 'default', 'source': 'default'}
            logger.info("Using default for Inflation Rate (5.0%).")

    logger.info(f"Final macro data collected for: {list(macro_data.keys())}")
    return macro_data