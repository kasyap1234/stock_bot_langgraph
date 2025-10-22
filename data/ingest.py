

import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import re

from .models import StockData, HistoricalData, NewsData, NewsItem, validate_stock_data
from .quality_validator import validate_data, InsufficientDataError, ConstantPriceError, validate_data_quality, DataQualityValidator

logger = logging.getLogger(__name__)


def clean_stock_data(stocks_data: List[StockData]) -> HistoricalData:
    
    cleaned_data = []

    for stock_data in stocks_data:
        # Validate basic structure
        if not validate_stock_data(stock_data):
            logger.warning(f"Invalid stock data structure: {stock_data}")
            continue

        # Clean individual record
        cleaned_record = clean_single_stock_record(stock_data)

        if cleaned_record:
            cleaned_data.append(cleaned_record)
        else:
            logger.warning(f"Failed to clean stock record: {stock_data}")

    try:
        validate_data(cleaned_data)
    except (InsufficientDataError, ConstantPriceError) as e:
        logger.warning(f"Data validation failed during ingestion: {e}")
        return []  # Skip invalid data by returning empty list
    
    # Full quality validation
    if cleaned_data:
        validator = DataQualityValidator()
        report = validate_data_quality(cleaned_data, 'unknown', validator)
        logger.info(f"Quality report: score={report.overall_quality_score}, issues={len(report.issues)}")
        
        # Check for critical issues
        critical_issues = [issue for issue in report.issues if issue.severity in ['high', 'critical']]
        if report.overall_quality_score < 0.85 or len(critical_issues) > len(cleaned_data) * 0.1:
            logger.warning(f"Critical quality issues in data: score={report.overall_quality_score}, critical={len(critical_issues)}")
            return []
        
        # Filter excessive zero close/volume
        zero_close_count = sum(1 for record in cleaned_data if record.get('close', 0) == 0)
        zero_volume_count = sum(1 for record in cleaned_data if record.get('volume', 0) == 0)
        total_records = len(cleaned_data)
        
        if zero_close_count / total_records > 0.1:
            logger.warning(f"Filtering {zero_close_count} zero-close records (>10%)")
            cleaned_data = [r for r in cleaned_data if r.get('close', 0) != 0]
        
        if zero_volume_count / total_records > 0.1:
            logger.warning(f"Filtering {zero_volume_count} zero-volume records (>10%)")
            cleaned_data = [r for r in cleaned_data if r.get('volume', 0) != 0]
        
        if len(cleaned_data) < total_records * 0.8:  # If >20% filtered, consider invalid
            logger.warning("Too many records filtered due to zero values")
            return []

    return cleaned_data


def clean_single_stock_record(stock_data: StockData) -> Optional[StockData]:
    
    try:
        cleaned = stock_data.copy()

        # Clean and validate symbol
        cleaned['symbol'] = clean_symbol(cleaned['symbol'])

        # Clean and validate date
        cleaned['date'] = clean_date(cleaned.get('date'))

        # Clean numeric fields
        numeric_fields = ['open', 'high', 'low', 'close']
        for field in numeric_fields:
            cleaned[field] = clean_price(cleaned[field])

        # Clean volume
        cleaned['volume'] = clean_volume(cleaned['volume'])

        # Remove rows with invalid prices (NaN, negative, zero for OHLC)
        if any(cleaned[field] <= 0 for field in numeric_fields if cleaned[field] is not None):
            return None

        return cleaned

    except Exception as e:
        logger.warning(f"Error cleaning stock record: {e}")
        return None


def clean_symbol(symbol: str) -> str:
    
    if not isinstance(symbol, str):
        symbol = str(symbol)

    # Remove extra whitespace and convert to uppercase
    symbol = symbol.strip().upper()

    # Validate symbol format (should contain letters, numbers, dots, hyphens)
    if not re.match(r'^[A-Z0-9.-]+$', symbol):
        logger.warning(f"Invalid symbol format: {symbol}")
        return ""

    # FIXED: Check for NSE or BSE suffix, otherwise leave as is
    if not symbol.endswith(('.NS', '.BO')):
        logger.info(f"Symbol {symbol} does not have NSE/BSE suffix, leaving unchanged")
    else:
        logger.debug(f"Symbol {symbol} has valid NSE/BSE suffix")

    return symbol


def clean_date(date_str: Union[str, Any]) -> str:
    
    if isinstance(date_str, str):
        date_str = date_str.strip()

        # If already in YYYY-MM-DD format, validate it
        if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
            try:
                datetime.strptime(date_str, '%Y-%m-%d')
                return date_str
            except ValueError:
                pass

        # Handle MM/DD/YYYY format
        elif re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', date_str):
            try:
                dt = datetime.strptime(date_str, '%m/%d/%Y')
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                pass

        # Handle DD-MM-YYYY format
        elif re.match(r'^\d{1,2}-\d{1,2}-\d{4}$', date_str):
            try:
                dt = datetime.strptime(date_str, '%d-%m-%Y')
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                pass

        # Handle other formats by extracting date components
        try:
            # Look for patterns with year, month, day
            match = re.search(r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', date_str)
            if match:
                year, month, day = match.groups()
                dt = datetime(int(year), int(month), int(day))
                return dt.strftime('%Y-%m-%d')
        except ValueError:
            pass

    # If we can't parse, return today's date as fallback
    logger.warning(f"Could not parse date: {date_str}, using today")
    return datetime.now().strftime('%Y-%m-%d')


def clean_price(price: Union[float, int, str, None]) -> Optional[float]:
    
    try:
        if price is None or str(price).lower() in ['nan', 'null', '', 'n/a']:
            return None

        # Convert to float
        if isinstance(price, str):
            # Remove currency symbols and commas
            price = re.sub(r'[₹$€£,]', '', price)

        price_float = float(price)

        # Check for reasonable price range (Indian stocks typically 1-100,000 rupees)
        if not (0.01 <= price_float <= 100000):
            logger.warning(f"Price outside reasonable range: {price_float}")
            return None

        # Round to 2 decimal places
        return round(price_float, 2)

    except (ValueError, TypeError):
        logger.warning(f"Invalid price value: {price}")
        return None


def clean_volume(volume: Union[int, float, str, None]) -> Optional[int]:
    
    try:
        if volume is None or str(volume).lower() in ['nan', 'null', '', 'n/a']:
            return None

        # Convert to int
        if isinstance(volume, str):
            # Remove commas
            volume = re.sub(r',', '', volume)

        volume_int = int(float(volume))  # Handle cases where volume comes as float

        # Volume should be positive
        if volume_int < 0:
            logger.warning(f"Negative volume: {volume_int}")
            return None

        return volume_int

    except (ValueError, TypeError):
        logger.warning(f"Invalid volume value: {volume}")
        return None


def clean_news_data(news_data: NewsData) -> NewsData:
    
    cleaned_news = []

    for news_item in news_data:
        cleaned_item = clean_single_news_item(news_item)

        if cleaned_item:
            cleaned_news.append(cleaned_item)
        else:
            logger.warning(f"Failed to clean news item: {news_item}")

    return cleaned_news


def clean_single_news_item(news_item: NewsItem) -> Optional[NewsItem]:
    
    try:
        cleaned = news_item.copy()

        # Validate required fields
        if not all(key in cleaned for key in ['title', 'url', 'published_date', 'source']):
            return None

        # Clean title (remove extra whitespace)
        cleaned['title'] = cleaned['title'].strip()
        if not cleaned['title']:
            return None

        # Clean URL
        cleaned['url'] = cleaned['url'].strip()
        if not cleaned['url'] or not cleaned['url'].startswith('http'):
            return None

        # Clean date
        cleaned['published_date'] = clean_date(cleaned['published_date'])

        # Clean source
        cleaned['source'] = cleaned['source'].strip()

        # Clean summary if present
        if 'summary' in cleaned and cleaned['summary']:
            cleaned['summary'] = cleaned['summary'].strip()

        return cleaned

    except Exception as e:
        logger.warning(f"Error cleaning news item: {e}")
        return None


def fill_missing_data(stocks_data: HistoricalData, method: str = 'forward_fill') -> HistoricalData:
    
    if not stocks_data:
        return stocks_data

    # Sort by date
    stocks_data.sort(key=lambda x: x['date'])

    filled_data = []

    for i, stock_data in enumerate(stocks_data):
        filled = stock_data.copy()

        # Fill missing OHLC values
        if method == 'forward_fill' and i > 0:
            prev_data = stocks_data[i-1]
            for field in ['open', 'high', 'low', 'close']:
                if filled[field] is None:
                    filled[field] = prev_data.get(field)

        elif method == 'backward_fill' and i < len(stocks_data) - 1:
            next_data = stocks_data[i+1]
            for field in ['open', 'high', 'low', 'close']:
                if filled[field] is None:
                    filled[field] = next_data.get(field)

        # Fill missing volume with 0
        if filled['volume'] is None:
            filled['volume'] = 0

        filled_data.append(filled)

    return filled_data


def detect_outliers(stocks_data: HistoricalData, threshold: float = 3.0) -> List[bool]:
    
    if len(stocks_data) < 2:
        return [False] * len(stocks_data)

    # Extract closing prices
    prices = [data['close'] for data in stocks_data if data['close'] is not None]

    if not prices:
        return [False] * len(stocks_data)

    # Calculate mean and standard deviation
    mean_price = sum(prices) / len(prices)
    std_price = (sum((p - mean_price) ** 2 for p in prices) / len(prices)) ** 0.5

    outliers = []
    price_idx = 0

    for data in stocks_data:
        is_outlier = False
        if data['close'] is not None:
            z_score = abs(data['close'] - mean_price) / std_price if std_price > 0 else 0
            is_outlier = z_score > threshold
            price_idx += 1

        outliers.append(is_outlier)

    return outliers


def remove_duplicates(stocks_data: HistoricalData) -> HistoricalData:
    
    seen = set()
    unique_data = []

    for stock_data in stocks_data:
        key = (stock_data['symbol'], stock_data['date'])

        if key not in seen:
            seen.add(key)
            unique_data.append(stock_data)
        else:
            logger.info(f"Removed duplicate: {key}")

    return unique_data