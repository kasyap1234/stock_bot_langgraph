"""
Data models and state management for stock bot - JSON/dict based structures for stock data and LangGraph state.
"""

from typing import Any, Dict, List, Optional, Union, Annotated
from datetime import datetime

from typing_extensions import TypedDict as ExtTypedDict


# StockData structure for OHLCV data
StockData = Dict[
    str,  # key
    Union[
        str,        # symbol, date
        float,      # open, high, low, close
        int,        # volume
        Dict,       # additional data
    ]
]

# Historical data structure (list of StockData)
HistoricalData = List[StockData]

# Fundamentals data structure
FundamentalsData = Dict[str, Union[str, float, int, Dict]]

# News data structure
NewsItem = Dict[str, Union[str, datetime]]
NewsData = List[NewsItem]

# Complete data structure combining all data types
CompleteStockData = Dict[str, Dict[str, Union[HistoricalData, FundamentalsData, NewsData]]]


def create_stock_data(
    symbol: str,
    date: str,
    open_price: float,
    high: float,
    low: float,
    close: float,
    volume: int,
    **additional_fields
) -> StockData:
    """
    Create a StockData dictionary with the standard OHLCV format.

    Args:
        symbol: Stock symbol (e.g., "RELIANCE.NS")
        date: Date in YYYY-MM-DD format
        open_price: Opening price
        high: High price
        low: Low price
        close: Closing price
        volume: Trading volume
        **additional_fields: Any additional data fields

    Returns:
        StockData dictionary
    """
    stock_data = {
        "symbol": symbol,
        "date": date,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }

    if additional_fields:
        stock_data.update(additional_fields)

    return stock_data


def create_news_item(
    title: str,
    url: str,
    published_date: str,
    source: str,
    summary: Optional[str] = None
) -> NewsItem:
    """
    Create a NewsItem dictionary.

    Args:
        title: News headline/title
        url: Source URL
        published_date: Publication date
        source: News source
        summary: Brief summary (optional)

    Returns:
        NewsItem dictionary
    """
    news_item = {
        "title": title,
        "url": url,
        "published_date": published_date,
        "source": source,
    }

    if summary:
        news_item["summary"] = summary

    return news_item


def validate_stock_data(stock_data: StockData) -> bool:
    """
    Validate that a StockData dictionary has all required fields.

    Args:
        stock_data: StockData dictionary to validate

    Returns:
        True if valid, False otherwise
    """
    required_fields = ["symbol", "date", "open", "high", "low", "close", "volume"]

    for field in required_fields:
        if field not in stock_data:
            return False

        if field not in ['symbol', 'date'] and field != "volume" and not isinstance(stock_data[field], (int, float)):
            return False

        if field == "volume" and not isinstance(stock_data[field], int):
            return False

    return True


def merge_analysis_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two analysis dicts for the same symbol."""
    result = a.copy() if a else {}
    if b:
        for symbol, data in b.items():
            if symbol in result:
                if isinstance(result[symbol], dict) and isinstance(data, dict):
                    result[symbol].update(data)
                else:
                    # If not both dicts, prefer the new data to avoid type mismatch
                    result[symbol] = data
            else:
                result[symbol] = data
    return result


# LangGraph State class for the trading bot workflow
class State(ExtTypedDict):
    """State definition for LangGraph workflow"""
    stock_data: Dict[str, Any]
    technical_signals: Annotated[Dict[str, Dict[str, str]], merge_analysis_dicts]
    fundamental_analysis: Annotated[Dict[str, Dict[str, Union[float, str]]], merge_analysis_dicts]
    sentiment_scores: Annotated[Dict[str, Dict[str, float]], merge_analysis_dicts]
    macro_scores: Dict[str, float]
    risk_metrics: Annotated[Dict[str, Dict[str, Union[float, bool]]], merge_analysis_dicts]
    final_recommendation: Dict[str, Dict[str, str]]
    simulation_results: Dict[str, Any]
    performance_analysis: Dict[str, Any]
    backtest_results: Dict[str, Any]
    # Multi-stock analysis keys
    top_buy_candidate: Dict[str, Dict[str, str]]
    buy_ranking: List[Dict[str, Any]]
    ranking_reasoning: str
    # Additional reference data
    all_recommendations: Dict[str, Dict[str, Any]]
    llm_reasoning: Dict[str, str]
    failed_stocks: List[Dict[str, str]]
    # Real-time data
    real_time_data: Dict[str, List[Dict[str, Any]]]
    real_time_active: bool
    failed_stocks: List[Dict[str, str]]