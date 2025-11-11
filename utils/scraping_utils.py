

import time
import random
import logging
import re
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

ENHANCED_USER_AGENTS = [
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
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
]

session = requests.Session()

def get_random_user_agent() -> str:
    
    return random.choice(ENHANCED_USER_AGENTS)

def add_request_delay(base_delay: float = 1.0) -> None:
    
    delay = base_delay + random.uniform(0.5, 2.0)
    logger.debug(f"Adding delay: {delay:.2f}s")
    time.sleep(delay)

def create_realistic_headers(user_agent: str, referer: Optional[str] = None) -> Dict[str, str]:
    
    headers = {
        'User-Agent': user_agent,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'en-US,en;q=0.9,en-GB;q=0.8,en;q=0.7',
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

    if referer:
        headers['Referer'] = referer

    return headers

def rate_limited_get(url: str, timeout: int = 30, retries: int = 3, base_delay: float = 1.0) -> Optional[requests.Response]:
    
    for attempt in range(retries + 1):
        # Get fresh user agent for each attempt
        user_agent = get_random_user_agent()
        headers = create_realistic_headers(user_agent)

        # Add referer for subsequent attempts
        if attempt > 0:
            parsed_url = urlparse(url)
            headers['Referer'] = f"{parsed_url.scheme}://{parsed_url.netloc}/"

        try:
            logger.debug(f"Request attempt {attempt + 1} for {url}")

            response = session.get(
                url,
                headers=headers,
                timeout=timeout,
                allow_redirects=True
            )

            if response.status_code == 200:
                logger.debug(f"Successful request to {url}")
                return response
            elif response.status_code == 403:
                logger.warning(f"HTTP 403 for {url} (attempt {attempt + 1}) - trying different user agent")
                if attempt < retries:
                    add_request_delay(base_delay * 2)
                    continue
                else:
                    return None
            elif response.status_code == 429:
                wait_time = 60 + random.uniform(0, 30)
                logger.warning(f"Rate limited. Waiting {wait_time:.1f}s before retry...")
                time.sleep(wait_time)
                continue
            elif response.status_code in [404, 410, 418]:
                logger.warning(f"HTTP {response.status_code} for {url} - resource not available")
                return None
            elif response.status_code >= 500:
                logger.warning(f"Server error {response.status_code} for {url}")
                if attempt < retries:
                    add_request_delay(base_delay * 3)
                    continue
                else:
                    return None
            else:
                logger.warning(f"HTTP {response.status_code} for {url}")
                return None

        except requests.RequestException as e:
            logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
            if attempt < retries:
                add_request_delay(base_delay * (attempt + 1))

    return None

def extract_numeric_value(text: str, default: Optional[float] = None) -> Optional[float]:
    
    if not text or not isinstance(text, str):
        return default

    # Clean the text
    text = text.strip().replace(',', '').replace('₹', '').replace('$', '')

    # Try different number patterns
    patterns = [
        r'(\d+\.?\d+)',  # Standard decimal
        r'(\d{1,3}(?:,\d{3})*\.?\d*)',  # With commas
        r'(\d+\.?\d*|\.\d+)',  # Decimal variations
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                value = float(match.group(1).replace(',', ''))
                logger.debug(f"Extracted numeric value: {value} from '{text}'")
                return value
            except ValueError:
                continue

    logger.debug(f"Could not extract numeric value from '{text}'")
    return default

def safe_extract_text(element, selectors: List[str], default: str = "") -> str:
    
    if not element:
        return default

    # If no selectors provided, extract text directly from the element
    if not selectors:
        text = element.get_text(strip=True)
        if text:
            logger.debug(f"Extracted text directly from element: '{text}'")
            return text
        return default

    for selector in selectors:
        try:
            if selector.startswith('.'):
                # Class selector
                found = element.find(class_=selector[1:])
            elif selector.startswith('#'):
                # ID selector
                found = element.find(id=selector[1:])
            else:
                # Tag selector
                found = element.find(selector)

            if found:
                text = found.get_text(strip=True)
                if text:
                    logger.debug(f"Extracted text using selector '{selector}': '{text}'")
                    return text
        except Exception as e:
            logger.debug(f"Failed to extract with selector '{selector}': {e}")
            continue

    return default

def find_element_by_multiple_selectors(soup: BeautifulSoup, selectors: List[str]) -> Optional[Any]:
    
    for selector in selectors:
        try:
            if selector.startswith('.'):
                # Class selector
                element = soup.find(class_=selector[1:])
            elif selector.startswith('#'):
                # ID selector
                element = soup.find(id=selector[1:])
            else:
                # Tag selector or complex selector
                element = soup.select_one(selector)

            if element:
                logger.debug(f"Found element using selector: {selector}")
                return element
        except Exception as e:
            logger.debug(f"Failed with selector '{selector}': {e}")
            continue

    return None

def create_fallback_data_structure() -> Dict[str, Any]:
    
    return {
        'PERatio': None,
        'EPS': None,
        'DebtEquityRatio': None,
        'MarketCapitalization': None,
        'DividendYield': None,
        'Beta': None,
        'RBI_Repo_Rate': None,
        'Unemployment_Rate': None,
        'GDP_Growth': None
    }