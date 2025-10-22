import pandas as pd
from datetime import datetime, timedelta
from data.scraper import scrape_news

symbol = 'RELIANCE.NS'
news = scrape_news(symbol, 5)
print(f'Fetched {len(news)} articles')

if len(news) == 0:
    print("Scraping failed, using mock news for sentiment test")
    mock_news = [
        {'title': 'Reliance reports strong Q2 earnings', 'url': 'mock1.com', 'published_date': '2025-09-24', 'source': 'Moneycontrol'},
        {'title': 'Reliance stock surges on positive news', 'url': 'mock2.com', 'published_date': '2025-09-24', 'source': 'Business Standard'},
        {'title': 'Reliance faces regulatory scrutiny', 'url': 'mock3.com', 'published_date': '2025-09-24', 'source': 'Livemint'},
        {'title': 'Reliance dividend announcement expected', 'url': 'mock4.com', 'published_date': '2025-09-24', 'source': 'The Hindu'},
        {'title': 'Reliance market share grows in Q3', 'url': 'mock5.com', 'published_date': '2025-09-24', 'source': 'Financial Express'}
    ]
    news = mock_news
    print(f'Using mock news: {len(news)} articles')

for n in news[:5]:
    print(f'- {n["title"]} ({n["source"]})')