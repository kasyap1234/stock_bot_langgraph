"""
Fundamental analysis agent for stock data.
Analyzes company financial fundamentals using Alpha Vantage and other sources.
"""

import logging
from functools import lru_cache
from typing import Dict, Union, Optional

import requests

from config.config import ALPHA_VANTAGE_API_KEY, API_RATE_LIMIT_DELAY
from utils.scraping_utils import rate_limited_get, extract_numeric_value, safe_extract_text, add_request_delay, find_element_by_multiple_selectors
from data.models import State

# Configure logging
logger = logging.getLogger(__name__)

# Indian market sector-specific PE benchmarks (min, max)
SECTOR_PE_BENCHMARKS = {
    'PSU': (10, 15),  # Public Sector Undertakings
    'FMCG': (50, 70),  # Fast Moving Consumer Goods
    'Pharma': (30, 50),  # Pharmaceuticals
    'IT': (25, 35),  # Information Technology
    'Banking': (15, 25),  # Banking and Financial Services
    'Auto': (20, 30),  # Automobile
    'Energy': (12, 18),  # Energy and Oil & Gas
    'Cement': (20, 30),  # Cement and Construction
    'Telecom': (15, 25),  # Telecom
    'Default': (20, 30)  # Default for unknown sectors
}

# Mapping of major Indian stocks to sectors
STOCK_SECTORS = {
    'RELIANCE.NS': 'Energy',
    'TCS.NS': 'IT',
    'HDFCBANK.NS': 'Banking',
    'ICICIBANK.NS': 'Banking',
    'INFY.NS': 'IT',
    'HINDUNILVR.NS': 'FMCG',
    'ITC.NS': 'FMCG',
    'KOTAKBANK.NS': 'Banking',
    'LT.NS': 'Energy',
    'AXISBANK.NS': 'Banking',
    'MARUTI.NS': 'Auto',
    'BAJFINANCE.NS': 'Banking',
    'BHARTIARTL.NS': 'Telecom',
    'HCLTECH.NS': 'IT',
    'WIPRO.NS': 'IT',
    'ULTRACEMCO.NS': 'Cement',
    'NESTLEIND.NS': 'FMCG',
    'POWERGRID.NS': 'PSU',
    'NTPC.NS': 'PSU',
    'ONGC.NS': 'PSU',
    'COALINDIA.NS': 'PSU',
    'GRASIM.NS': 'Cement',
    'JSWSTEEL.NS': 'Energy',
    'TATASTEEL.NS': 'Energy',
    'ADANIPORTS.NS': 'Energy',
    'SHREECEM.NS': 'Cement',
    'BAJAJ-AUTO.NS': 'Auto',
    'TITAN.NS': 'FMCG',
    'HEROMOTOCO.NS': 'Auto',
    'DRREDDY.NS': 'Pharma',
    'SUNPHARMA.NS': 'Pharma',
    'CIPLA.NS': 'Pharma',
    'DIVISLAB.NS': 'Pharma',
    'APOLLOHOSP.NS': 'Pharma',
    'INDUSINDBK.NS': 'Banking',
    'HDFCLIFE.NS': 'Banking',
    'SBILIFE.NS': 'Banking',
    'BRITANNIA.NS': 'FMCG',
    'TECHM.NS': 'IT',
    'EICHERMOT.NS': 'Auto',
    'BPCL.NS': 'PSU',
    'UPL.NS': 'FMCG',
    'M&M.NS': 'Auto',
    'TATACONSUM.NS': 'FMCG',
    'ASIANPAINT.NS': 'FMCG',
    'PIDILITIND.NS': 'FMCG',
    'NMDC.NS': 'PSU',
    'GAIL.NS': 'PSU',
    'VEDL.NS': 'Energy'
}


def get_stock_sector(symbol: str) -> str:
    """
    Determine the sector for a given stock symbol.

    Args:
        symbol: Stock symbol (e.g., "RELIANCE.NS")

    Returns:
        Sector name or 'Default' if unknown
    """
    # Try exact match
    if symbol in STOCK_SECTORS:
        return STOCK_SECTORS[symbol]

    # Try base symbol match
    base_symbol = symbol.split('.')[0] if '.' in symbol else symbol
    for key, sector in STOCK_SECTORS.items():
        if key.startswith(base_symbol + '.'):
            return sector

    return 'Default'


@lru_cache(maxsize=128)
def get_historical_pe_analysis(symbol: str) -> Dict[str, float]:
    """
    Fetch historical PE analysis for a stock symbol.

    Args:
        symbol: Stock symbol

    Returns:
        Dictionary with historical PE metrics
    """
    try:
        from yahooquery import Ticker
        ticker = Ticker(symbol)

        # Get historical prices (last 5 years, monthly)
        history = ticker.history(period='5y', interval='1mo')
        if history is not None and not history.empty:
            # Get current EPS
            key_stats = ticker.key_stats
            current_eps = None
            if isinstance(key_stats, dict):
                if symbol in key_stats:
                    current_eps = key_stats[symbol].get('trailingEps')
                else:
                    current_eps = key_stats.get('trailingEps')
            elif hasattr(key_stats, 'empty') and not key_stats.empty:
                if 'trailingEps' in key_stats.columns:
                    current_eps = key_stats['trailingEps'].iloc[0]

            if current_eps and current_eps > 0:
                # Approximate historical PE = close price / current EPS
                pe_values = history['close'] / current_eps
                return {
                    'avg_historical_pe': pe_values.mean(),
                    'min_historical_pe': pe_values.min(),
                    'max_historical_pe': pe_values.max()
                }
    except Exception as e:
        logger.warning(f"Error fetching historical PE for {symbol}: {e}")

    return {}


@lru_cache(maxsize=128)
def _get_fundamental_data(symbol: str) -> Dict[str, Union[float, str]]:
    """
    Fetch fundamental data for a stock symbol using multiple free sources.

    Args:
        symbol: Stock symbol (e.g., "RELIANCE.NS")

    Returns:
        Dictionary containing fundamental metrics
    """
    # Convert NSE symbol to base symbol
    base_symbol = symbol.split('.')[0] if '.' in symbol else symbol

    # Try yahooquery first (free and works well with Indian stocks)
    try:
        from yahooquery import Ticker
        ticker = Ticker(symbol)

        # Get key statistics
        key_stats = ticker.key_stats
        logger.info(f"YahooQuery key_stats type: {type(key_stats)}, value: {key_stats}")
        if key_stats:
            fundamentals = {}

            # Handle both DataFrame and dict formats
            if isinstance(key_stats, dict):
                # Extract the inner dict for the symbol
                if symbol in key_stats:
                    stats_dict = key_stats[symbol]
                else:
                    stats_dict = key_stats
                # Extract key metrics from dict
                fundamentals['PERatio'] = stats_dict.get('trailingPE') or stats_dict.get('forwardPE')
                fundamentals['EPS'] = stats_dict.get('trailingEps')
                fundamentals['DebtEquityRatio'] = stats_dict.get('debtToEquity')
            elif hasattr(key_stats, 'empty') and not key_stats.empty:
                # Extract key metrics from DataFrame
                if 'trailingPE' in key_stats.columns:
                    fundamentals['PERatio'] = key_stats['trailingPE'].iloc[0]
                elif 'forwardPE' in key_stats.columns:
                    fundamentals['PERatio'] = key_stats['forwardPE'].iloc[0]
                if 'trailingEps' in key_stats.columns:
                    fundamentals['EPS'] = key_stats['trailingEps'].iloc[0]
                if 'debtToEquity' in key_stats.columns:
                    fundamentals['DebtEquityRatio'] = key_stats['debtToEquity'].iloc[0]
            else:
                # Skip if neither dict nor valid DataFrame
                fundamentals = {}
                logger.warning("key_stats is neither dict nor valid DataFrame")

            # Get summary detail for more metrics
            summary = ticker.summary_detail
            if summary:
                fundamentals['MarketCapitalization'] = summary.get('marketCap')
                fundamentals['DividendYield'] = summary.get('dividendYield')
                fundamentals['Beta'] = summary.get('beta')

            # Get financial data
            financials = ticker.financial_data
            if financials:
                fundamentals['TotalRevenue'] = financials.get('totalRevenue')
                fundamentals['NetIncome'] = financials.get('netIncome')

            logger.info(f"Successfully fetched fundamentals for {symbol} using yahooquery")
            return fundamentals

    except Exception as e:
        logger.warning(f"YahooQuery failed for {symbol}: {e}")

    # Fallback to Alpha Vantage if yahooquery fails
    if ALPHA_VANTAGE_API_KEY and ALPHA_VANTAGE_API_KEY not in ["demo", "9YM9MF6IN0GJMMCO"]:
        try:
            import time
            time.sleep(API_RATE_LIMIT_DELAY)  # Rate limiting

            # Overview endpoint
            url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={base_symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                overview = response.json()
                if "Error Message" not in overview and "Note" not in overview:
                    logger.info(f"Successfully fetched fundamentals for {symbol} using Alpha Vantage")
                    return overview

        except Exception as e:
            logger.warning(f"Alpha Vantage failed for {symbol}: {e}")

    # If both fail, try web scraping from Moneycontrol or other Indian sources
    try:
        fundamentals = _scrape_indian_fundamentals(symbol)
        if fundamentals:
            logger.info(f"Successfully scraped fundamentals for {symbol}")
            return fundamentals
    except Exception as e:
        logger.warning(f"Web scraping failed for {symbol}: {e}")

    logger.warning(f"All fundamental data sources failed for {symbol}")
    return {"error": "All data sources unavailable"}


def _scrape_indian_fundamentals(symbol: str) -> Dict[str, Union[float, str]]:
    """
    Scrape fundamental data from multiple Indian financial websites.

    Args:
        symbol: Stock symbol (e.g., "RELIANCE.NS")

    Returns:
        Dictionary with fundamental metrics
    """
    from bs4 import BeautifulSoup

    try:
        base_symbol = symbol.split('.')[0] if '.' in symbol else symbol
        fundamentals = {}

        # 1. Try Screener.in for Indian stock fundamentals (primary source)
        try:
            screener_url = f"https://www.screener.in/company/{base_symbol}/"
            response = rate_limited_get(screener_url)

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Extract PE ratio from Screener.in
                pe_elements = soup.find_all(string=re.compile(r'PE\s*ratio', re.IGNORECASE))
                if pe_elements:
                    for elem in pe_elements:
                        parent = elem.parent
                        if parent:
                            value_elem = parent.find_next('span', class_='number')
                            if value_elem:
                                pe_text = value_elem.get_text(strip=True)
                                pe_match = re.search(r'(\d+\.?\d*)', pe_text)
                                if pe_match and 'PERatio' not in fundamentals:
                                    fundamentals['PERatio'] = float(pe_match.group(1))
                                    logger.info(f"Extracted PE Ratio from Screener.in: {fundamentals['PERatio']} for {symbol}")
                                    break

                # Extract EPS from Screener.in
                eps_elements = soup.find_all(string=re.compile(r'EPS', re.IGNORECASE))
                if eps_elements:
                    for elem in eps_elements:
                        parent = elem.parent
                        if parent:
                            value_elem = parent.find_next('span', class_='number')
                            if value_elem:
                                eps_text = value_elem.get_text(strip=True)
                                eps_match = re.search(r'(\d+\.?\d*)', eps_text)
                                if eps_match and 'EPS' not in fundamentals:
                                    fundamentals['EPS'] = float(eps_match.group(1))
                                    logger.info(f"Extracted EPS from Screener.in: {fundamentals['EPS']} for {symbol}")
                                    break

                # Extract Debt to Equity from Screener.in
                de_elements = soup.find_all(string=re.compile(r'Debt.*equity', re.IGNORECASE))
                if de_elements:
                    for elem in de_elements:
                        parent = elem.parent
                        if parent:
                            value_elem = parent.find_next('span', class_='number')
                            if value_elem:
                                de_text = value_elem.get_text(strip=True)
                                de_match = re.search(r'(\d+\.?\d*)', de_text)
                                if de_match and 'DebtEquityRatio' not in fundamentals:
                                    fundamentals['DebtEquityRatio'] = float(de_match.group(1))
                                    logger.info(f"Extracted Debt/Equity from Screener.in: {fundamentals['DebtEquityRatio']} for {symbol}")
                                    break
                   
                                    # Extract ROE
                                    roe_elements = soup.find_all(string=re.compile(r'ROE', re.IGNORECASE))
                                    if roe_elements:
                                        for elem in roe_elements:
                                            parent = elem.parent
                                            if parent:
                                                value_elem = parent.find_next('span', class_='number')
                                                if value_elem:
                                                    roe_text = value_elem.get_text(strip=True)
                                                    roe_match = re.search(r'(\d+\.?\d*)', roe_text)
                                                    if roe_match and 'ROE' not in fundamentals:
                                                        fundamentals['ROE'] = float(roe_match.group(1))
                                                        logger.info(f"Extracted ROE from Screener.in: {fundamentals['ROE']} for {symbol}")
                                                        break
                   
                                    # Extract ROA
                                    roa_elements = soup.find_all(string=re.compile(r'ROA', re.IGNORECASE))
                                    if roa_elements:
                                        for elem in roa_elements:
                                            parent = elem.parent
                                            if parent:
                                                value_elem = parent.find_next('span', class_='number')
                                                if value_elem:
                                                    roa_text = value_elem.get_text(strip=True)
                                                    roa_match = re.search(r'(\d+\.?\d*)', roa_text)
                                                    if roa_match and 'ROA' not in fundamentals:
                                                        fundamentals['ROA'] = float(roa_match.group(1))
                                                        logger.info(f"Extracted ROA from Screener.in: {fundamentals['ROA']} for {symbol}")
                                                        break
                   
                                    # Extract PB Ratio
                                    pb_elements = soup.find_all(string=re.compile(r'Price.*book', re.IGNORECASE))
                                    if pb_elements:
                                        for elem in pb_elements:
                                            parent = elem.parent
                                            if parent:
                                                value_elem = parent.find_next('span', class_='number')
                                                if value_elem:
                                                    pb_text = value_elem.get_text(strip=True)
                                                    pb_match = re.search(r'(\d+\.?\d*)', pb_text)
                                                    if pb_match and 'PB' not in fundamentals:
                                                        fundamentals['PB'] = float(pb_match.group(1))
                                                        logger.info(f"Extracted PB Ratio from Screener.in: {fundamentals['PB']} for {symbol}")
                                                        break
                   
                                    logger.info(f"Successfully scraped fundamentals from Screener.in: {fundamentals}")

        except Exception as e:
            logger.warning(f"Screener.in scraping failed: {e}")

        # 2. Try Screener.in for additional fundamental data
        try:
            screener_url = f"https://www.screener.in/company/{base_symbol}/"
            response = rate_limited_get(screener_url)

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Extract PE ratio from Screener.in
                pe_elements = soup.find_all(string=re.compile(r'PE\s*ratio', re.IGNORECASE))
                if pe_elements:
                    for elem in pe_elements:
                        parent = elem.parent
                        if parent:
                            value_elem = parent.find_next('span', class_='number')
                            if value_elem:
                                pe_text = value_elem.get_text(strip=True)
                                pe_match = re.search(r'(\d+\.?\d*)', pe_text)
                                if pe_match and 'PERatio' not in fundamentals:
                                    fundamentals['PERatio'] = float(pe_match.group(1))
                                    logger.info(f"Extracted PE Ratio from Screener.in: {fundamentals['PERatio']} for {symbol}")
                                    break

                # Extract Debt to Equity from Screener.in
                de_elements = soup.find_all(string=re.compile(r'Debt.*equity', re.IGNORECASE))
                if de_elements:
                    for elem in de_elements:
                        parent = elem.parent
                        if parent:
                            value_elem = parent.find_next('span', class_='number')
                            if value_elem:
                                de_text = value_elem.get_text(strip=True)
                                de_match = re.search(r'(\d+\.?\d*)', de_text)
                                if de_match and 'DebtEquityRatio' not in fundamentals:
                                    fundamentals['DebtEquityRatio'] = float(de_match.group(1))
                                    logger.info(f"Extracted Debt/Equity from Screener.in: {fundamentals['DebtEquityRatio']} for {symbol}")
                                    break

                logger.info(f"Successfully scraped additional fundamentals from Screener.in: {fundamentals}")

        except Exception as e:
            logger.warning(f"Screener.in scraping failed: {e}")


        # 4. Try Yahoo Finance India for additional data
        try:
            yahoo_url = f"https://finance.yahoo.com/quote/{symbol}"
            response = rate_limited_get(yahoo_url)

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Extract PE ratio from Yahoo Finance
                pe_labels = soup.find_all('td', string=re.compile(r'PE\s*Ratio', re.IGNORECASE))
                if pe_labels:
                    for label in pe_labels:
                        value_elem = label.find_next('td')
                        if value_elem:
                            pe_text = value_elem.get_text(strip=True)
                            pe_match = re.search(r'(\d+\.?\d*)', pe_text)
                            if pe_match and 'PERatio' not in fundamentals:
                                fundamentals['PERatio'] = float(pe_match.group(1))
                                logger.info(f"Extracted PE Ratio from Yahoo Finance: {fundamentals['PERatio']} for {symbol}")
                                break

                logger.info(f"Successfully scraped fundamentals from Yahoo Finance: {fundamentals}")

        except Exception as e:
            logger.warning(f"Yahoo Finance scraping failed: {e}")

        # If we got some data, return it
        if fundamentals:
            return fundamentals

        # Fallback: return basic structure with None values
        return {
            'PERatio': None,
            'EPS': None,
            'DebtEquityRatio': None,
            'MarketCapitalization': None,
            'DividendYield': None,
            'Beta': None
        }

    except Exception as e:
        logger.error(f"Error scraping Indian fundamentals for {symbol}: {e}")
        return {"error": "Scraping failed"}


def _analyze_fundamentals(fund_data: Dict, symbol: str) -> Dict[str, Union[float, str]]:
    """
    Analyze fundamental data and extract key metrics with valuations.

    Args:
        fund_data: Raw fundamental data from API
        symbol: Stock symbol for sector and historical analysis

    Returns:
        Dictionary with analyzed fundamentals and valuations
    """
    try:
        fundamentals = {}

        # Extract key metrics
        pe_ratio = float(fund_data.get("PERatio", 0)) if fund_data.get("PERatio") not in [None, "None", ""] else None
        eps = float(fund_data.get("EPS", 0)) if fund_data.get("EPS") not in [None, "None", ""] else None
        de_ratio = float(fund_data.get("DebtEquityRatio", fund_data.get("DebtEquityRatio", None))) if fund_data.get("DebtEquityRatio", fund_data.get("DebtEquityRatio")) else None
        roe = float(fund_data.get("ROE", 0)) if fund_data.get("ROE") not in [None, "None", ""] else None
        roa = float(fund_data.get("ROA", 0)) if fund_data.get("ROA") not in [None, "None", ""] else None
        pb = float(fund_data.get("PB", 0)) if fund_data.get("PB") not in [None, "None", ""] else None

        fundamentals["PE"] = pe_ratio
        fundamentals["EPS"] = eps
        fundamentals["DE"] = de_ratio
        fundamentals["ROE"] = roe
        fundamentals["ROA"] = roa
        fundamentals["PB"] = pb

        # Calculate PEG Ratio if both PE and EPS available
        if pe_ratio and eps and eps > 0:
            fundamentals["PEG"] = pe_ratio / eps

        # Get sector and historical analysis
        sector = get_stock_sector(symbol)
        historical_pe = get_historical_pe_analysis(symbol)

        fundamentals["sector"] = sector
        fundamentals.update(historical_pe)

        # ROE calculation from income statement if not from scraping
        if roe is None and "income_statement" in fund_data and fund_data["income_statement"]:
            try:
                revenue = float(fund_data["income_statement"][0].get("totalRevenue", 0))
                net_income = float(fund_data["income_statement"][0].get("netIncome", 0))

                if revenue > 0:
                    fundamentals["ROE"] = (net_income / revenue) * 100
            except (ValueError, IndexError) as e:
                logger.warning(f"Error calculating ROE: {e}")

        # Valuation assessments
        # General valuation (conservative NSE average)
        industry_avg_pe = 20
        if pe_ratio:
            if pe_ratio < industry_avg_pe * 0.8:
                fundamentals["general_valuation"] = "undervalued"
            elif pe_ratio > industry_avg_pe * 1.2:
                fundamentals["general_valuation"] = "overvalued"
            else:
                fundamentals["general_valuation"] = "fairly valued"
        else:
            fundamentals["general_valuation"] = "unknown"

        # Sector-specific valuation
        sector_pe_min, sector_pe_max = SECTOR_PE_BENCHMARKS.get(sector, SECTOR_PE_BENCHMARKS['Default'])
        if pe_ratio:
            if pe_ratio < sector_pe_min:
                fundamentals["sector_valuation"] = "undervalued"
            elif pe_ratio > sector_pe_max:
                fundamentals["sector_valuation"] = "overvalued"
            else:
                fundamentals["sector_valuation"] = "fairly valued"
        else:
            fundamentals["sector_valuation"] = "unknown"

        # Historical valuation
        avg_hist_pe = historical_pe.get('avg_historical_pe')
        if avg_hist_pe and pe_ratio:
            if pe_ratio < avg_hist_pe * 0.9:
                fundamentals["historical_valuation"] = "potentially undervalued"
            elif pe_ratio > avg_hist_pe * 1.1:
                fundamentals["historical_valuation"] = "potentially overvalued"
            else:
                fundamentals["historical_valuation"] = "in line with history"
        else:
            fundamentals["historical_valuation"] = "historical data unavailable"

        return fundamentals

    except Exception as e:
        logger.error(f"Error analyzing fundamentals: {e}")
        return {"error": "Analysis failed"}


def fundamental_analysis_agent(state: State) -> State:
    """
    Fundamental analysis agent for the LangGraph workflow.
    Analyzes company fundamentals and valuation metrics.

    Args:
        state: Current workflow state

    Returns:
        Updated state with fundamental analysis
    """
    logging.info("Starting fundamental analysis agent")

    stock_data = state.get("stock_data", {})
    fundamental_analysis = {}

    for symbol in stock_data.keys():
        try:
            # Fetch fundamental data
            raw_fundamentals = _get_fundamental_data(symbol)

            if "error" not in raw_fundamentals:
                # Analyze and structure the data
                analyzed_fundamentals = _analyze_fundamentals(raw_fundamentals, symbol)
                fundamental_analysis[symbol] = analyzed_fundamentals
                logger.info(f"Completed fundamental analysis for {symbol}: {analyzed_fundamentals}")
            else:
                fundamental_analysis[symbol] = raw_fundamentals
                logger.warning(f"Fundamental analysis failed for {symbol}: {raw_fundamentals}")

        except Exception as e:
            logger.error(f"Error in fundamental analysis for {symbol}: {e}")
            fundamental_analysis[symbol] = {"error": "Fundamental analysis failed"}

    return {"fundamental_analysis": fundamental_analysis}