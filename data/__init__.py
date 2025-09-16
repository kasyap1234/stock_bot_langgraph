"""
Data fetching and processing package for stock bot.
Provides unified interface for historical data, fundamentals, and news scraping.
"""

# Import models
from .models import (
    StockData,
    HistoricalData,
    FundamentalsData,
    NewsData,
    NewsItem,
    CompleteStockData,
    create_stock_data,
    create_news_item,
    validate_stock_data,
)

# Import API functions
from .apis import (
    get_stock_history,
    get_fundamentals,
    get_stock_info,
    get_news_articles,
    get_macro_data,
)

# Import scraper functions
from .scraper import (
    scrape_moneycontrol_news,
)

# Import ingest functions
from .ingest import (
    clean_stock_data,
    clean_news_data,
    fill_missing_data,
    detect_outliers,
    remove_duplicates,
)

__all__ = [
    # Models
    'StockData',
    'HistoricalData',
    'FundamentalsData',
    'NewsData',
    'NewsItem',
    'CompleteStockData',
    'create_stock_data',
    'create_news_item',
    'validate_stock_data',

    # API functions
    'get_stock_history',
    'get_fundamentals',
    'get_stock_info',
    'get_news_articles',
    'get_macro_data',

    # Scraper functions
    'scrape_moneycontrol_news',

    # Ingest functions
    'clean_stock_data',
    'clean_news_data',
    'fill_missing_data',
    'detect_outliers',
    'remove_duplicates',
]