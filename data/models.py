

from typing import Any, Dict, List, Optional, Union, Annotated
from datetime import datetime

from typing_extensions import TypedDict as ExtTypedDict


StockData = Dict[
    str,  # key
    Union[
        str,        # symbol, date
        float,      # open, high, low, close
        int,        # volume
        Dict,       # additional data
    ]
]

HistoricalData = List[StockData]

FundamentalsData = Dict[str, Union[str, float, int, Dict]]

NewsItem = Dict[str, Union[str, datetime]]
NewsData = List[NewsItem]

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


class State(ExtTypedDict):
    
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
    data_valid: bool
    validation_errors: List[str]
    backtest: bool
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