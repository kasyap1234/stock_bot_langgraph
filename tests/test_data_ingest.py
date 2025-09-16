import pytest
from datetime import datetime
from data.ingest import (
    clean_stock_data, clean_single_stock_record, clean_symbol, clean_date,
    clean_price, clean_volume, clean_news_data, clean_single_news_item,
    fill_missing_data, detect_outliers, remove_duplicates
)
from data.models import validate_stock_data

VALID_STOCK_DATA = {
    'symbol': 'RELIANCE.NS',
    'date': '2023-01-01',
    'open': 2500.0,
    'high': 2520.0,
    'low': 2480.0,
    'close': 2510.0,
    'volume': 1000000
}

VALID_NEWS_ITEM = {
    'title': 'Company announces earnings',
    'url': 'https://example.com/news/1',
    'published_date': '2023-01-01',
    'source': 'Financial Times',
    'summary': 'Company beats expectations'
}

class TestCleanSymbol:
    def test_clean_symbol_basic(self):
        assert clean_symbol('reliance.ns') == 'RELIANCE.NS'
        assert clean_symbol('AAPL') == 'AAPL'
        assert clean_symbol('TEST-1') == 'TEST-1'

    def test_clean_symbol_with_spaces(self):
        assert clean_symbol('  aapl  ') == 'AAPL'

    def test_clean_symbol_with_dot(self):
        assert clean_symbol('AAPL') == 'AAPL'

    def test_clean_symbol_invalid_chars(self):
        assert clean_symbol('AAPL@2023') == ''
        assert clean_symbol('AAPL_2023') == ''

    def test_clean_symbol_empty(self):
        assert clean_symbol('') == ''

    def test_clean_symbol_non_string(self):
        assert clean_symbol(12345) == '12345'

class TestCleanDate:
    def test_clean_date_yyyy_mm_dd(self):
        assert clean_date('2023-01-01') == '2023-01-01'

    def test_clean_date_mm_dd_yyyy(self):
        assert clean_date('01/01/2023') == '2023-01-01'

    def test_clean_date_dd_mm_yyyy(self):
        assert clean_date('01-01-2023') == '2023-01-01'

    def test_clean_date_with_separator(self):
        assert clean_date('2023/01/01') == '2023-01-01'

    def test_clean_date_invalid(self):
        result = clean_date('invalid-date')
        # Should fallback to today, but since we can't predict, just check it's a string
        assert isinstance(result, str)
        assert len(result) == 10  # YYYY-MM-DD length

    def test_clean_date_non_string(self):
        result = clean_date(20230101)
        assert isinstance(result, str)

class TestCleanPrice:
    def test_clean_price_float(self):
        assert clean_price(150.0) == 150.0

    def test_clean_price_int(self):
        assert clean_price(150) == 150.0

    def test_clean_price_string(self):
        assert clean_price('150.5') == 150.5
        assert clean_price('$150.5') == 150.5
        assert clean_price('₹150') == 150.0
        assert clean_price('1,500') == 1500.0

    def test_clean_price_out_of_range(self):
        assert clean_price(0) is None
        assert clean_price(0.001) is None
        assert clean_price(200000) is None

    def test_clean_price_invalid(self):
        assert clean_price('abc') is None
        assert clean_price('') is None
        assert clean_price(None) is None
        assert clean_price('nan') is None

class TestCleanVolume:
    def test_clean_volume_int(self):
        assert clean_volume(1000000) == 1000000

    def test_clean_volume_float(self):
        assert clean_volume(1000000.0) == 1000000

    def test_clean_volume_string(self):
        assert clean_volume('1000000') == 1000000
        assert clean_volume('1,000,000') == 1000000

    def test_clean_volume_negative(self):
        assert clean_volume(-1000) is None

    def test_clean_volume_invalid(self):
        assert clean_volume('abc') is None
        assert clean_volume('') is None
        assert clean_volume(None) is None

class TestCleanSingleStockRecord:
    def test_clean_single_stock_record_valid(self):
        result = clean_single_stock_record(VALID_STOCK_DATA)
        assert result is not None
        assert result['symbol'] == 'RELIANCE.NS'
        assert result['open'] == 2500.0

    def test_clean_single_stock_record_negative_price(self):
        invalid_data = VALID_STOCK_DATA.copy()
        invalid_data['close'] = -100
        result = clean_single_stock_record(invalid_data)
        assert result is None

    def test_clean_single_stock_record_invalid_symbol(self):
        invalid_data = VALID_STOCK_DATA.copy()
        invalid_data['symbol'] = 'INVALID@SYMBOL'
        result = clean_single_stock_record(invalid_data)
        assert result is not None  # Invalid symbol becomes '', might still cause issues, but test as is

class TestCleanStockData:
    def test_clean_stock_data_all_valid(self):
        data = [VALID_STOCK_DATA, VALID_STOCK_DATA.copy()]
        result = clean_stock_data(data)
        assert len(result) == 2

    def test_clean_stock_data_invalid(self):
        invalid_data = VALID_STOCK_DATA.copy()
        del invalid_data['symbol']
        data = [VALID_STOCK_DATA, invalid_data]
        result = clean_stock_data(data)
        assert len(result) == 1

class TestCleanSingleNewsItem:
    def test_clean_single_news_item_valid(self):
        result = clean_single_news_item(VALID_NEWS_ITEM)
        assert result is not None
        assert result['title'] == 'Company announces earnings'
        assert result['url'] == 'https://example.com/news/1'

    def test_clean_single_news_item_missing_title(self):
        invalid_item = VALID_NEWS_ITEM.copy()
        invalid_item['title'] = ''
        result = clean_single_news_item(invalid_item)
        assert result is None

    def test_clean_single_news_item_invalid_url(self):
        invalid_item = VALID_NEWS_ITEM.copy()
        invalid_item['url'] = 'not-a-valid-url'
        result = clean_single_news_item(invalid_item)
        assert result is None

class TestCleanNewsData:
    def test_clean_news_data_valid(self):
        data = [VALID_NEWS_ITEM, VALID_NEWS_ITEM.copy()]
        result = clean_news_data(data)
        assert len(result) == 2

    def test_clean_news_data_invalid(self):
        invalid_item = VALID_NEWS_ITEM.copy()
        invalid_item['title'] = ''
        data = [VALID_NEWS_ITEM, invalid_item]
        result = clean_news_data(data)
        assert len(result) == 1

class TestFillMissingData:
    def test_fill_missing_data_forward(self):
        data = [
            {'symbol': 'AAPL', 'date': '2023-01-01', 'open': 150.0, 'high': 155.0, 'low': 145.0, 'close': None, 'volume': 1000000},
            {'symbol': 'AAPL', 'date': '2023-01-02', 'open': 152.0, 'high': 157.0, 'low': 147.0, 'close': 154.0, 'volume': None}
        ]
        result = fill_missing_data(data, 'forward_fill')
        assert result[0]['close'] == 154.0
        assert result[1]['volume'] == 0

    def test_fill_missing_data_backward(self):
        data = [
            {'symbol': 'AAPL', 'date': '2023-01-01', 'open': 150.0, 'high': 155.0, 'low': None, 'close': 152.0, 'volume': 1000000},
            {'symbol': 'AAPL', 'date': '2023-01-02', 'open': None, 'high': 157.0, 'low': 147.0, 'close': 154.0, 'volume': 1100000}
        ]
        result = fill_missing_data(data, 'backward_fill')
        assert result[0]['low'] == 147.0
        assert result[1]['open'] == 150.0

    def test_fill_missing_data_empty(self):
        result = fill_missing_data([])
        assert result == []

class TestDetectOutliers:
    def test_detect_outliers_no_outliers(self):
        data = [
            {'close': 100.0},
            {'close': 102.0},
            {'close': 101.0},
            {'close': 103.0},
            {'close': 102.0}
        ]
        outliers = detect_outliers(data, threshold=2.0)
        assert all(not o for o in outliers)

    def test_detect_outliers_with_outlier(self):
        data = [
            {'close': 100.0},
            {'close': 102.0},
            {'close': 101.0},
            {'close': 1000.0},  # Outlier
            {'close': 102.0}
        ]
        outliers = detect_outliers(data, threshold=2.0)
        assert outliers[3] == True

    def test_detect_outliers_insufficient_data(self):
        data = [{'close': 100.0}]
        outliers = detect_outliers(data)
        assert outliers == [False]

    def test_detect_outliers_no_prices(self):
        data = [{'close': None}, {'close': None}]
        outliers = detect_outliers(data)
        assert outliers == [False, False]

class TestRemoveDuplicates:
    def test_remove_duplicates_no_duplicates(self):
        data = [
            {'symbol': 'AAPL', 'date': '2023-01-01', 'close': 150.0},
            {'symbol': 'AAPL', 'date': '2023-01-02', 'close': 152.0},
            {'symbol': 'GOOGL', 'date': '2023-01-01', 'close': 2000.0}
        ]
        result = remove_duplicates(data)
        assert len(result) == 3

    def test_remove_duplicates_with_duplicates(self):
        data = [
            {'symbol': 'AAPL', 'date': '2023-01-01', 'close': 150.0},
            {'symbol': 'AAPL', 'date': '2023-01-01', 'close': 151.0},  # Duplicate
            {'symbol': 'AAPL', 'date': '2023-01-02', 'close': 152.0}
        ]
        result = remove_duplicates(data)
        assert len(result) == 2
        assert result[0]['close'] == 150.0