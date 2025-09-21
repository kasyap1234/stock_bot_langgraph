import pytest
from unittest.mock import Mock, patch
import pandas as pd
from data.apis import get_stock_history, get_fundamentals, get_stock_info, _retry_with_backoff
from config.config import ALPHA_VANTAGE_API_KEY

MOCK_HISTORY_DATAFRAME = pd.DataFrame({
    'Open': [150.0, 152.0, 151.5],
    'High': [155.0, 156.0, 155.5],
    'Low': [145.0, 147.0, 146.5],
    'Close': [152.5, 155.0, 154.5],
    'Volume': [1000000, 1200000, 1100000]
}, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']))

MOCK_STOCK_INFO = {
    'longName': 'Apple Inc.',
    'sector': 'Technology',
    'industry': 'Consumer Electronics',
    'marketCap': 2900000000000,
    'trailingPE': 28.5,
    'dividendYield': 0.005,
    'beta': 1.2,
    'currency': 'USD'
}

MOCK_FUNDAMENTALS_OVERVIEW = {
    'Symbol': 'AAPL',
    'Name': 'Apple Inc.',
    'Exchange': 'NASDAQ'
}

MOCK_FUNDAMENTALS_INCOME = {'annualReports': [{'fiscalDateEnding': '2022-09-30'}]}
MOCK_FUNDAMENTALS_BALANCE = {'annualReports': [{'fiscalDateEnding': '2022-09-30'}]}
MOCK_FUNDAMENTALS_CASH = {'annualReports': [{'fiscalDateEnding': '2022-09-30'}]}

def test_get_stock_history_success():
    with patch('yfinance.Ticker') as MockTicker:
        mock_ticker = MockTicker.return_value
        mock_ticker.history.return_value = MOCK_HISTORY_DATAFRAME
        
        result = get_stock_history('AAPL', '1mo', '1d')
        
        assert len(result) == 3
        assert result[0]['symbol'] == 'AAPL'
        assert result[0]['date'] == '2023-01-01'
        assert result[0]['open'] == 150.0
        assert result[0]['close'] == 152.5
        assert result[0]['volume'] == 1000000

def test_get_stock_history_empty():
    empty_df = pd.DataFrame()
    with patch('yfinance.Ticker') as MockTicker:
        mock_ticker = MockTicker.return_value
        mock_ticker.history.return_value = empty_df
        
        result = get_stock_history('INVALID', '1mo', '1d')
        
        # Should return empty list on error
        assert result == []

def test_get_stock_history_general_exception():
    with patch('yfinance.Ticker') as MockTicker:
        mock_ticker = MockTicker.return_value
        mock_ticker.history.side_effect = Exception("Network error")
        
        result = get_stock_history('AAPL', '1mo', '1d')
        
        assert result == []

def test_get_fundamentals_no_api_key():
    with patch('data.apis.ALPHA_VANTAGE_API_KEY', ''):
        result = get_fundamentals('AAPL')
        
        assert result == {"error": "API key not configured"}

def test_get_fundamentals_success():
    with patch('alpha_vantage.fundamentaldata.FundamentalData') as MockFD, \
         patch('time.sleep'), \
         patch('data.apis.RATE_LIMIT_DELAY', 0.1):
        
        mock_fd = MockFD.return_value
        mock_fd.get_company_overview.return_value = (MOCK_FUNDAMENTALS_OVERVIEW, None)
        mock_fd.get_income_statement_annual.return_value = (MOCK_FUNDAMENTALS_INCOME, None)
        mock_fd.get_balance_sheet_annual.return_value = (MOCK_FUNDAMENTALS_BALANCE, None)
        mock_fd.get_cash_flow_annual.return_value = (MOCK_FUNDAMENTALS_CASH, None)
        
        result = get_fundamentals('AAPL')
        
        assert result['symbol'] == 'AAPL'
        assert 'overview' in result
        assert 'income_statement' in result
        assert 'balance_sheet' in result
        assert 'cash_flow' in result

def test_get_fundamentals_exception():
    with patch('alpha_vantage.fundamentaldata.FundamentalData') as MockFD, \
         patch('time.sleep'), \
         patch('data.apis.RATE_LIMIT_DELAY', 0.1):
        
        mock_fd = MockFD.return_value
        mock_fd.get_company_overview.side_effect = Exception("API error")
        
        result = get_fundamentals('AAPL')
        
        assert result['symbol'] == 'AAPL'
        assert 'error' in result

def test_get_stock_info_success():
    with patch('yfinance.Ticker') as MockTicker:
        mock_ticker = MockTicker.return_value
        mock_ticker.info = MOCK_STOCK_INFO
        
        result = get_stock_info('AAPL')
        
        assert result['symbol'] == 'AAPL'
        assert result['name'] == 'Apple Inc.'
        assert result['sector'] == 'Technology'
        assert result['market_cap'] == 2900000000000

def test_get_stock_info_no_info():
    with patch('yfinance.Ticker') as MockTicker:
        mock_ticker = MockTicker.return_value
        mock_ticker.info = {}
        
        result = get_stock_info('INVALID')
        
        # Should return error dict
        assert result['symbol'] == 'INVALID'
        assert 'error' in result

def test_get_stock_info_exception():
    with patch('yfinance.Ticker') as MockTicker:
        mock_ticker = MockTicker.return_value
        mock_ticker.info = None
        
        result = get_stock_info('AAPL')
        
        assert result['symbol'] == 'AAPL'
        assert 'error' in result

def test_get_stock_info_with_short_name():
    mock_info = {'shortName': 'Apple'}
    
    with patch('yfinance.Ticker') as MockTicker:
        mock_ticker = MockTicker.return_value
        mock_ticker.info = mock_info
        
        result = get_stock_info('AAPL')
        
        assert result['name'] == 'Apple'

def test_retry_with_backoff_success():
    func = Mock(return_value="success")
    
    with patch('time.sleep'):
        result = _retry_with_backoff(func, max_retries=3)
        
        assert result == "success"
        func.assert_called_once()

def test_retry_with_backoff_one_failure():
    func = Mock(side_effect=[Exception("fail"), "success"])
    
    with patch('time.sleep'):
        result = _retry_with_backoff(func, max_retries=2)
        
        assert result == "success"
        assert func.call_count == 2

def test_retry_with_backoff_all_fail():
    func = Mock(side_effect=Exception("fail"))
    
    with patch('time.sleep'):
        with pytest.raises(Exception):
            _retry_with_backoff(func, max_retries=1)
            
        assert func.call_count == 2

def test_retry_with_backoff_base_delay(mocker):
    spy = mocker.spy(time, 'sleep')
    func = Mock(side_effect=[Exception("fail"), Exception("fail"), "success"])
    
    _retry_with_backoff(func, max_retries=2, base_delay=2.0)
    
    # Check sleep delays: 2**0 = 1 -> 2*1=2s, 2**1=2 -> 2*2=4s
    assert spy.call_count == 2
    spy.assert_any_call(2.0)
    spy.assert_any_call(4.0)


def test_get_stock_history_nse_success():
    """Test successful NSE fetch with non-zero data."""
    mock_df = pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=5),
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [95, 96, 97, 98, 99],
        'close': [102, 103, 104, 105, 106],
        'volume': [1000000, 1100000, 1200000, 1300000, 1400000]
    })
    
    with patch('yahooquery.Ticker') as MockTicker:
        mock_ticker = MockTicker.return_value
        mock_ticker.history.return_value = mock_df
        
        result = get_stock_history('RELIANCE.NS', '1mo', '1d')
        
        assert len(result) == 5
        assert all(r['close'] > 0 for r in result)
        assert all(r['volume'] > 0 for r in result)
        assert result[0]['symbol'] == 'RELIANCE.NS'


def test_get_stock_history_all_zero_raises():
    """Test that all-zero data raises DataFetchingError."""
    mock_df = pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=5),
        'open': [0] * 5,
        'high': [0] * 5,
        'low': [0] * 5,
        'close': [0] * 5,
        'volume': [0] * 5
    })
    
    with patch('yahooquery.Ticker') as MockTicker:
        mock_ticker = MockTicker.return_value
        mock_ticker.history.return_value = mock_df
        
        with pytest.raises(DataFetchingError):
            get_stock_history('TATAMOTORS.NS', '1mo', '1d')


def test_get_stock_history_low_quality_raises(mocker):
    """Test that low quality score raises DataFetchingError."""
    # Mock successful fetch
    mock_df = pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=5),
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [95, 96, 97, 98, 99],
        'close': [102, 0, 104, 0, 106],  # Some zeros to lower score
        'volume': [1000000, 0, 1200000, 0, 1400000]
    })
    
    with patch('yahooquery.Ticker') as MockTicker:
        mock_ticker = MockTicker.return_value
        mock_ticker.history.return_value = mock_df
        
        # Mock validate_data_quality to return low score
        mocker.patch('data.apis.validate_data_quality', return_value=Mock(spec=DataQualityReport, overall_quality_score=0.7))
        
        with pytest.raises(DataFetchingError):
            get_stock_history('RELIANCE.NS', '1mo', '1d')