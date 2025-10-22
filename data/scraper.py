

import time
import logging
import random
import re
from typing import List, Dict, Optional
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup
import warnings
from bs4 import XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
from urllib.parse import urljoin, urlparse
from dateutil import parser as date_parser

from .models import NewsData, NewsItem, create_news_item

logger = logging.getLogger(__name__)

class ScrapingError(Exception):
    pass

class ValidationError(Exception):
    pass

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (iPad; CPU OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (Android 13; Mobile; rv:121.0) Gecko/121.0 Firefox/121.0',
    'Mozilla/5.0 (Linux; Android 10; SM-G973F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
]

REQUEST_DELAY = 2  # seconds between requests
BATCH_DELAY = 10    # seconds between batches
MAX_REQUESTS_PER_BATCH = 5

session = requests.Session()


def parse_date(date_text: str) -> str:
    if not date_text or not date_text.strip():
        return datetime.now().strftime("%Y-%m-%d")
    date_text = date_text.strip()
    try:
        parsed = date_parser.parse(date_text, fuzzy=True)
        return parsed.strftime("%Y-%m-%d")
    except Exception:
        # Fallback: try to extract date-like string
        date_match = re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', date_text)
        if date_match:
            try:
                return date_parser.parse(date_match.group()).strftime("%Y-%m-%d")
            except:
                pass
        # Another fallback: if starts with digits, take first 10
        if date_text[0].isdigit() and len(date_text) >= 10:
            return date_text[:10]
        return datetime.now().strftime("%Y-%m-%d")


def validate_news_item(title: str, url: str) -> None:
    if not title or len(title.strip()) < 5:
        raise ValidationError(f"Title too short or empty: '{title}'")
    if not url or not url.startswith('http'):
        raise ValidationError(f"Invalid URL: '{url}'")
    # Check if URL is valid
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValidationError(f"Malformed URL: '{url}'")
    except Exception as e:
        raise ValidationError(f"URL validation error: {e}")


def get_article_snippet(url: str) -> str:
    try:
        response = _rate_limited_get(url, timeout=10)  # Shorter timeout for snippets
        if not response:
            return ""
        soup = BeautifulSoup(response.content, 'html.parser')
        # Remove scripts and styles
        for script in soup(["script", "style"]):
            script.decompose()
        # Common content selectors
        selectors = [
            'div.article-content',
            'div.content',
            'article',
            'div.story-body',
            'div.main-content',
            'div.article-body',
            'p'  # Fallback to first p
        ]
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(separator=' ', strip=True)
                if len(text) > 50:  # Ensure some content
                    return text[:300] + "..." if len(text) > 300 else text
        # Fallback: get all p tags
        paragraphs = soup.find_all('p')
        if paragraphs:
            text = ' '.join(p.get_text(strip=True) for p in paragraphs[:3])
            return text[:300] + "..." if len(text) > 300 else text
        return ""
    except Exception as e:
        logger.warning(f"Failed to extract snippet from {url}: {e}")
        return ""


def _get_random_user_agent() -> str:
    
    return random.choice(USER_AGENTS)


def _rate_limited_get(url: str, timeout: int = 30, retries: int = 3) -> Optional[requests.Response]:
    
    # Try different user agents on retries to avoid blocks
    for attempt in range(retries + 1):
        # Get a fresh user agent for each attempt
        user_agent = _get_random_user_agent()

        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        }

        # Add referer for subsequent attempts
        if attempt > 0:
            parsed_url = urlparse(url)
            headers['Referer'] = f"{parsed_url.scheme}://{parsed_url.netloc}/"

        try:
            response = session.get(
                url,
                headers=headers,
                timeout=timeout,
                allow_redirects=True
            )

            if response.status_code == 200:
                return response
            elif response.status_code == 403:  # Forbidden - try different user agent
                logger.warning(f"HTTP 403 for {url} (attempt {attempt + 1}) - trying different user agent")
                if attempt < retries:
                    time.sleep(REQUEST_DELAY * (attempt + 1))
                    continue
                else:
                    return None
            elif response.status_code == 429:  # Rate limited
                wait_time = 60 + random.uniform(0, 30)  # Wait 60-90 seconds
                logger.warning(f"Rate limited. Waiting {wait_time:.1f}s before retry...")
                time.sleep(wait_time)
                continue
            elif response.status_code >= 400:
                logger.warning(f"HTTP {response.status_code} for {url}")
                return None

        except requests.RequestException as e:
            logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
            if attempt < retries:
                time.sleep(REQUEST_DELAY * (attempt + 1))  # Progressive delay

    return None


def scrape_moneycontrol_news(symbol: str, max_articles: int = 10) -> NewsData:
    
    base_symbol = symbol.split('.')[0] if '.' in symbol else symbol

    news_data = []

    # Moneycontrol search URL
    search_url = f"https://www.moneycontrol.com/stocks/company_info/stock_news.php?sc_id={base_symbol}"

    articles_found = 0

    logger.info(f"Scraping Moneycontrol: {search_url}")

    time.sleep(REQUEST_DELAY)

    response = _rate_limited_get(search_url)
    if not response:
        return news_data

    soup = BeautifulSoup(response.content, 'html.parser')

    # Find news articles
    articles = (
        soup.find_all('li', class_='clearfix') +
        soup.find_all('div', class_='MT15') +
        soup.find_all('article') +
        soup.find_all('div', class_=re.compile(r'article|news|story', re.I))
    )

    for article in articles[:max_articles]:
        try:
            # Extract title and link
            link_elem = article.find('a')
            if not link_elem:
                continue

            title = link_elem.get_text(strip=True)
            url = link_elem.get('href')

            if not url:
                continue

            # Ensure absolute URL
            if not url.startswith('http'):
                url = urljoin('https://www.moneycontrol.com', url)

            try:
                validate_news_item(title, url)
            except ValidationError as e:
                logger.warning(f"Validation failed for Moneycontrol: {e}")
                continue

            # Extract date if available
            date_elem = article.find('span', class_='date') or article.find('em') or article.find('time') or article.find('span', class_=re.compile(r'date', re.I))
            date_text = date_elem.get_text(strip=True) if date_elem else ""
            published_date = parse_date(date_text)

            snippet = get_article_snippet(url)

            news_item = create_news_item(
                title=title,
                url=url,
                published_date=published_date,
                source="Moneycontrol",
                summary=snippet
            )
            news_data.append(news_item)
            articles_found += 1

            if articles_found >= max_articles:
                break

        except ValidationError:
            continue
        except Exception as e:
            logger.warning(f"Error parsing Moneycontrol article: {e}")
            continue

    return news_data


def scrape_business_standard_news(symbol: str, max_articles: int = 10) -> NewsData:
    
    base_symbol = symbol.split('.')[0] if '.' in symbol else symbol

    news_data = []

    # Business Standard search URL
    search_url = f"https://www.business-standard.com/search?q={base_symbol}"

    articles_found = 0

    logger.info(f"Scraping Business Standard: {search_url}")

    time.sleep(REQUEST_DELAY)

    response = _rate_limited_get(search_url)
    if not response:
        return news_data

    soup = BeautifulSoup(response.content, 'html.parser')

    # Find news articles
    articles = (
        soup.find_all('div', class_='listing') +
        soup.find_all('article') +
        soup.find_all('div', class_=re.compile(r'article|news|story', re.I))
    )

    for article in articles[:max_articles]:
        try:
            # Extract title and link
            link_elem = article.find('a')
            if not link_elem:
                continue

            title = link_elem.get_text(strip=True)
            url = link_elem.get('href')

            if not url:
                continue

            # Ensure absolute URL
            if not url.startswith('http'):
                url = urljoin('https://www.business-standard.com', url)

            try:
                validate_news_item(title, url)
            except ValidationError as e:
                logger.warning(f"Validation failed for Business Standard: {e}")
                continue

            # Extract date if available
            date_elem = article.find('span', class_='date') or article.find('time') or article.find('span', class_=re.compile(r'date', re.I))
            date_text = date_elem.get_text(strip=True) if date_elem else ""
            published_date = parse_date(date_text)

            snippet = get_article_snippet(url)

            news_item = create_news_item(
                title=title,
                url=url,
                published_date=published_date,
                source="Business Standard",
                summary=snippet
            )
            news_data.append(news_item)
            articles_found += 1

            if articles_found >= max_articles:
                break

        except ValidationError:
            continue
        except Exception as e:
            logger.warning(f"Error parsing Business Standard article: {e}")
            continue

    return news_data


def scrape_livemint_news(symbol: str, max_articles: int = 10) -> NewsData:
    
    base_symbol = symbol.split('.')[0] if '.' in symbol else symbol

    news_data = []

    # Livemint search URL
    search_url = f"https://www.livemint.com/search?q={base_symbol}"

    articles_found = 0

    logger.info(f"Scraping Livemint: {search_url}")

    time.sleep(REQUEST_DELAY)

    response = _rate_limited_get(search_url)
    if not response:
        return news_data

    # Suppress XML parsing warning and use html.parser for RSS
    import warnings
    from bs4 import XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

    soup = BeautifulSoup(response.content, 'html.parser')

    # Find news articles
    articles = (
        soup.find_all('div', class_='headlineSec') +
        soup.find_all('article') +
        soup.find_all('div', class_=re.compile(r'article|news|story', re.I))
    )

    for article in articles[:max_articles]:
        try:
            # Extract title and link
            link_elem = article.find('a')
            if not link_elem:
                continue

            title = link_elem.get_text(strip=True)
            url = link_elem.get('href')

            if not url:
                continue

            # Ensure absolute URL
            if not url.startswith('http'):
                url = urljoin('https://www.livemint.com', url)

            try:
                validate_news_item(title, url)
            except ValidationError as e:
                logger.warning(f"Validation failed for Livemint: {e}")
                continue

            # Extract date if available
            date_elem = article.find('span', class_='date') or article.find('time') or article.find('span', class_=re.compile(r'date', re.I))
            date_text = date_elem.get_text(strip=True) if date_elem else ""
            published_date = parse_date(date_text)

            snippet = get_article_snippet(url)

            news_item = create_news_item(
                title=title,
                url=url,
                published_date=published_date,
                source="Livemint",
                summary=snippet
            )
            news_data.append(news_item)
            articles_found += 1

            if articles_found >= max_articles:
                break

        except ValidationError:
            continue
        except Exception as e:
            logger.warning(f"Error parsing Livemint article: {e}")
            continue

    return news_data


def scrape_the_hindu_news(symbol: str, max_articles: int = 10) -> NewsData:
    
    base_symbol = symbol.split('.')[0] if '.' in symbol else symbol

    news_data = []

    # The Hindu search URL
    search_url = f"https://www.thehindu.com/search/?q={base_symbol}"

    articles_found = 0

    logger.info(f"Scraping The Hindu: {search_url}")

    time.sleep(REQUEST_DELAY)

    response = _rate_limited_get(search_url)
    if not response:
        return news_data

    soup = BeautifulSoup(response.content, 'html.parser')

    # Find news articles
    articles = (
        soup.find_all('div', class_='story-card') +
        soup.find_all('article') +
        soup.find_all('div', class_=re.compile(r'article|news|story', re.I))
    )

    for article in articles[:max_articles]:
        try:
            # Extract title and link
            link_elem = article.find('a')
            if not link_elem:
                continue

            title = link_elem.get_text(strip=True)
            url = link_elem.get('href')

            if not url:
                continue

            # Ensure absolute URL
            if not url.startswith('http'):
                url = urljoin('https://www.thehindu.com', url)

            try:
                validate_news_item(title, url)
            except ValidationError as e:
                logger.warning(f"Validation failed for The Hindu: {e}")
                continue

            # Extract date if available
            date_elem = article.find('span', class_='date') or article.find('time') or article.find('span', class_=re.compile(r'date', re.I))
            date_text = date_elem.get_text(strip=True) if date_elem else ""
            published_date = parse_date(date_text)

            snippet = get_article_snippet(url)

            news_item = create_news_item(
                title=title,
                url=url,
                published_date=published_date,
                source="The Hindu",
                summary=snippet
            )
            news_data.append(news_item)
            articles_found += 1

            if articles_found >= max_articles:
                break

        except ValidationError:
            continue
        except Exception as e:
            logger.warning(f"Error parsing The Hindu article: {e}")
            continue

    return news_data


def scrape_financial_express_news(symbol: str, max_articles: int = 10) -> NewsData:
    
    base_symbol = symbol.split('.')[0] if '.' in symbol else symbol

    news_data = []

    # Financial Express search URL
    search_url = f"https://www.financialexpress.com/search/{base_symbol}/"

    articles_found = 0

    logger.info(f"Scraping Financial Express: {search_url}")

    time.sleep(REQUEST_DELAY)

    response = _rate_limited_get(search_url)
    if not response:
        return news_data

    soup = BeautifulSoup(response.content, 'html.parser')

    # Find news articles
    articles = (
        soup.find_all('div', class_='entry') +
        soup.find_all('article') +
        soup.find_all('div', class_=re.compile(r'article|news|story', re.I))
    )

    for article in articles[:max_articles]:
        try:
            # Extract title and link
            link_elem = article.find('a')
            if not link_elem:
                continue

            title = link_elem.get_text(strip=True)
            url = link_elem.get('href')

            if not url:
                continue

            # Ensure absolute URL
            if not url.startswith('http'):
                url = urljoin('https://www.financialexpress.com', url)

            try:
                validate_news_item(title, url)
            except ValidationError as e:
                logger.warning(f"Validation failed for Financial Express: {e}")
                continue

            # Extract date if available
            date_elem = article.find('span', class_='date') or article.find('time') or article.find('span', class_=re.compile(r'date', re.I))
            date_text = date_elem.get_text(strip=True) if date_elem else ""
            published_date = parse_date(date_text)

            snippet = get_article_snippet(url)

            news_item = create_news_item(
                title=title,
                url=url,
                published_date=published_date,
                source="Financial Express",
                summary=snippet
            )
            news_data.append(news_item)
            articles_found += 1

            if articles_found >= max_articles:
                break

        except ValidationError:
            continue
        except Exception as e:
            logger.warning(f"Error parsing Financial Express article: {e}")
            continue

    return news_data


def scrape_business_today_news(symbol: str, max_articles: int = 10) -> NewsData:
    
    base_symbol = symbol.split('.')[0] if '.' in symbol else symbol

    news_data = []

    # Business Today search URL
    search_url = f"https://www.businesstoday.in/search/{base_symbol}"

    articles_found = 0

    logger.info(f"Scraping Business Today: {search_url}")

    time.sleep(REQUEST_DELAY)

    response = _rate_limited_get(search_url)
    if not response:
        return news_data

    soup = BeautifulSoup(response.content, 'html.parser')

    # Find news articles
    articles = (
        soup.find_all('div', class_='search-result') +
        soup.find_all('article') +
        soup.find_all('div', class_=re.compile(r'article|news|story', re.I))
    )

    for article in articles[:max_articles]:
        try:
            # Extract title and link
            link_elem = article.find('a')
            if not link_elem:
                continue

            title = link_elem.get_text(strip=True)
            url = link_elem.get('href')

            if not url:
                continue

            # Ensure absolute URL
            if not url.startswith('http'):
                url = urljoin('https://www.businesstoday.in', url)

            try:
                validate_news_item(title, url)
            except ValidationError as e:
                logger.warning(f"Validation failed for Business Today: {e}")
                continue

            # Extract date if available
            date_elem = article.find('span', class_='date') or article.find('time') or article.find('span', class_=re.compile(r'date', re.I))
            date_text = date_elem.get_text(strip=True) if date_elem else ""
            published_date = parse_date(date_text)

            snippet = get_article_snippet(url)

            news_item = create_news_item(
                title=title,
                url=url,
                published_date=published_date,
                source="Business Today",
                summary=snippet
            )
            news_data.append(news_item)
            articles_found += 1

            if articles_found >= max_articles:
                break

        except ValidationError:
            continue
        except Exception as e:
            logger.warning(f"Error parsing Business Today article: {e}")
            continue

    return news_data


def scrape_google_news_fallback(symbol: str, max_articles: int = 10) -> NewsData:
    
    base_symbol = symbol.split('.')[0] if '.' in symbol else symbol

    news_data = []

    # Google News search URL
    search_url = f"https://news.google.com/rss/search?q={base_symbol}+stock+OR+shares+when:7d"

    articles_found = 0

    logger.info(f"Scraping Google News fallback: {search_url}")

    time.sleep(REQUEST_DELAY)

    response = _rate_limited_get(search_url)
    if not response:
        return news_data

    soup = BeautifulSoup(response.content, 'html.parser')

    # Parse RSS feed
    items = soup.find_all('item')

    for item in items[:max_articles]:
        try:
            # Extract title
            title_elem = item.find('title')
            if not title_elem:
                continue

            title = title_elem.get_text(strip=True)

            # Extract URL
            link_elem = item.find('link')
            url = link_elem.get_text(strip=True) if link_elem else None

            if not url:
                continue

            try:
                validate_news_item(title, url)
            except ValidationError as e:
                logger.warning(f"Validation failed for Google News: {e}")
                continue

            # Extract date
            date_elem = item.find('pubdate')
            date_text = date_elem.get_text(strip=True) if date_elem else ""
            published_date = parse_date(date_text)

            snippet = get_article_snippet(url)

            news_item = create_news_item(
                title=title,
                url=url,
                published_date=published_date,
                source="Google News",
                summary=snippet
            )
            news_data.append(news_item)
            articles_found += 1

            if articles_found >= max_articles:
                break

        except ValidationError:
            continue
        except Exception as e:
            logger.warning(f"Error parsing Google News item: {e}")
            continue

    return news_data


def scrape_news(symbol: str, max_articles: int = 15) -> NewsData:
    
    logger.info(f"Fetching news for {symbol}")

    all_news = []

    # Scrape from multiple sources with smaller batches
    sources = [
        (scrape_moneycontrol_news, max_articles // 6),
        (scrape_business_standard_news, max_articles // 6),
        (scrape_livemint_news, max_articles // 6),
        (scrape_the_hindu_news, max_articles // 6),
        (scrape_financial_express_news, max_articles // 6),
        (scrape_business_today_news, max_articles // 6)
    ]

    successful_sources = 0
    for scrape_func, limit in sources:
        try:
            news = scrape_func(symbol, limit)
            if news:  # Only count if we got articles
                all_news.extend(news)
                successful_sources += 1
        except Exception as e:
            logger.warning(f"Failed to scrape from {scrape_func.__name__}: {e}")

    # Remove duplicates based on URL
    seen_urls = set()
    unique_news = []

    for news_item in all_news:
        if news_item['url'] not in seen_urls:
            seen_urls.add(news_item['url'])
            unique_news.append(news_item)

    logger.info(f"Found {len(unique_news)} unique news articles for {symbol} from {successful_sources} successful sources")

    # If no articles found from primary sources or very few successful sources, try fallback
    if not unique_news or successful_sources < 2:
        logger.info(f"Insufficient articles from primary sources ({len(unique_news)} articles, {successful_sources} sources), trying Google News fallback")
        fallback_news = scrape_google_news_fallback(symbol, max_articles)
        # Add fallback articles, avoiding duplicates
        for item in fallback_news:
            if item['url'] not in seen_urls:
                seen_urls.add(item['url'])
                unique_news.append(item)
        logger.info(f"Fallback added {len(fallback_news)} articles")

    return unique_news[:max_articles]  # Return only up to max_articles