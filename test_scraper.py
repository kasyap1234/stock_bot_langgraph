from data.scraper import scrape_news

symbol = 'RELIANCE.NS'
news = scrape_news(symbol, 5)
print(f'Fetched {len(news)} articles')
for n in news[:5]:
    print(f'- {n["title"]} ({n["source"]})')