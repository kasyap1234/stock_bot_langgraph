import pytest
from data.models import create_stock_data, validate_stock_data

def test_create_stock_data():
    data = create_stock_data(
        symbol="AAPL",
        date="2023-01-01",
        open_price=150.0,
        high=155.0,
        low=145.0,
        close=152.0,
        volume=1000000
    )
    expected = {
        'symbol': 'AAPL',
        'date': '2023-01-01',
        'open': 150.0,
        'high': 155.0,
        'low': 145.0,
        'close': 152.0,
        'volume': 1000000
    }
    assert data == expected

def test_create_stock_data_with_additional_fields():
    data = create_stock_data(
        symbol="AAPL",
        date="2023-01-01",
        open_price=150.0,
        high=155.0,
        low=145.0,
        close=152.0,
        volume=1000000,
        pe_ratio=25.0
    )
    assert 'pe_ratio' in data
    assert data['pe_ratio'] == 25.0

def test_create_stock_data_with_sector():
    data = create_stock_data(
        symbol="RELIANCE.NS",
        date="2023-01-01",
        open_price=2500.0,
        high=2520.0,
        low=2480.0,
        close=2510.0,
        volume=2000000,
        sector="Energy"
    )
    assert data['sector'] == "Energy"
    assert data['symbol'] == "RELIANCE.NS"

def test_validate_stock_data_valid():
    data = create_stock_data(
        symbol="AAPL",
        date="2023-01-01",
        open_price=150.0,
        high=155.0,
        low=145.0,
        close=152.0,
        volume=1000000
    )
    assert validate_stock_data(data) is True

def test_validate_stock_data_missing_symbol():
    data = {
        'date': '2023-01-01',
        'open': 150.0,
        'high': 155.0,
        'low': 145.0,
        'close': 152.0,
        'volume': 1000000
    }
    assert validate_stock_data(data) is False

def test_validate_stock_data_missing_date():
    data = create_stock_data(
        symbol="AAPL",
        date="2023-01-01",
        open_price=150.0,
        high=155.0,
        low=145.0,
        close=152.0,
        volume=1000000
    )
    del data['date']
    assert validate_stock_data(data) is False

def test_validate_stock_data_missing_open():
    data = create_stock_data(
        symbol="AAPL",
        date="2023-01-01",
        open_price=150.0,
        high=155.0,
        low=145.0,
        close=152.0,
        volume=1000000
    )
    del data['open']
    assert validate_stock_data(data) is False

def test_validate_stock_data_missing_high():
    data = create_stock_data(
        symbol="AAPL",
        date="2023-01-01",
        open_price=150.0,
        high=155.0,
        low=145.0,
        close=152.0,
        volume=1000000
    )
    del data['high']
    assert validate_stock_data(data) is False

def test_validate_stock_data_missing_low():
    data = create_stock_data(
        symbol="AAPL",
        date="2023-01-01",
        open_price=150.0,
        high=155.0,
        low=145.0,
        close=152.0,
        volume=1000000
    )
    del data['low']
    assert validate_stock_data(data) is False

def test_validate_stock_data_missing_close():
    data = create_stock_data(
        symbol="AAPL",
        date="2023-01-01",
        open_price=150.0,
        high=155.0,
        low=145.0,
        close=152.0,
        volume=1000000
    )
    del data['close']
    assert validate_stock_data(data) is False

def test_validate_stock_data_missing_volume():
    data = create_stock_data(
        symbol="AAPL",
        date="2023-01-01",
        open_price=150.0,
        high=155.0,
        low=145.0,
        close=152.0,
        volume=1000000
    )
    del data['volume']
    assert validate_stock_data(data) is False

def test_validate_stock_data_string_open():
    data = create_stock_data(
        symbol="AAPL",
        date="2023-01-01",
        open_price="150.0",
        high=155.0,
        low=145.0,
        close=152.0,
        volume=1000000
    )
    # Since isinstance("150.0", (int, float)) is False, should be False
    assert validate_stock_data(data) is False

def test_validate_stock_data_string_high():
    data = create_stock_data(
        symbol="AAPL",
        date="2023-01-01",
        open_price=150.0,
        high="155.0",
        low=145.0,
        close=152.0,
        volume=1000000
    )
    assert validate_stock_data(data) is False

def test_validate_stock_data_string_low():
    data = create_stock_data(
        symbol="AAPL",
        date="2023-01-01",
        open_price=150.0,
        high=155.0,
        low="145.0",
        close=152.0,
        volume=1000000
    )
    assert validate_stock_data(data) is False

def test_validate_stock_data_string_close():
    data = create_stock_data(
        symbol="AAPL",
        date="2023-01-01",
        open_price=150.0,
        high=155.0,
        low=145.0,
        close="152.0",
        volume=1000000
    )
    assert validate_stock_data(data) is False

def test_validate_stock_data_string_volume():
    data = create_stock_data(
        symbol="AAPL",
        date="2023-01-01",
        open_price=150.0,
        high=155.0,
        low=145.0,
        close=152.0,
        volume="1000000"
    )
    # isinstance("1000000", int) is False, so False
    assert validate_stock_data(data) is False

def test_validate_stock_data_zero_volume():
    data = create_stock_data(
        symbol="AAPL",
        date="2023-01-01",
        open_price=150.0,
        high=155.0,
        low=145.0,
        close=152.0,
        volume=0
    )
    # Volume can be 0, since check is only not in dict and isinstance, no restriction to positive
    assert validate_stock_data(data) is True