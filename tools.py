import yfinance as yf
import os
import nltk
from ddgs import DDGS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from data.models import State
from agents import data_fetcher_agent, technical_analysis_agent, sentiment_analysis_agent
from config.constants import DEFAULT_STOCKS
from data.scraper import scrape_news

# Ensure NLTK VADER lexicon is available
try:
    SentimentIntensityAnalyzer()
except LookupError:
    nltk.download('vader_lexicon')

def fetch_stock_price(symbol: str, period: str = "1d") -> dict:
    if not symbol.endswith(".NS"):
        return {"error": "Only Indian stocks (.NS) are supported"}
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period)
    if data.empty:
        return {"error": "No data found for symbol"}
    latest = data.iloc[-1]
    return {
        "symbol": symbol,
        "current_price": round(latest["Close"], 2),
        "open": round(latest["Open"], 2),
        "high": round(latest["High"], 2),
        "low": round(latest["Low"], 2),
        "volume": int(latest["Volume"]),
        "date": latest.name.strftime("%Y-%m-%d")
    }

def compute_technical_indicators(symbol: str, indicators: list = ["RSI", "MACD"]) -> dict:
    state = State()
    state["stock_data"] = {symbol: {}}
    temp_state = data_fetcher_agent(state, [symbol], period="1y")
    if symbol not in temp_state["stock_data"]:
        return {"error": "Failed to fetch data for technical analysis"}
    analysis_state = technical_analysis_agent(temp_state)
    technical = analysis_state.get("technical_signals", {}).get(symbol, {})
    results = {}
    for ind in indicators:
        if ind in technical:
            results[ind] = technical[ind]
    return {"symbol": symbol, "indicators": results}

def web_search(query: str, max_results: int = 5) -> list[dict]:
    try:
        results = DDGS().text(query, max_results=max_results)
        formatted_results = []
        for r in results:
            formatted_results.append({
                'title': r.get('title', ''),
                'url': r.get('href', ''),
                'content': r.get('body', '')
            })
        return formatted_results
    except Exception as e:
        return [{"error": str(e)}]

def stock_search(query: str) -> list[str]:
    query_upper = query.upper()
    matches = [stock for stock in DEFAULT_STOCKS if query_upper in stock.upper()]
    if not matches:
        return []
    return matches

def get_sentiment(symbol: str, use_web: bool = True) -> dict:
    if not symbol.endswith(".NS"):
        return {"error": "Only Indian stocks (.NS) are supported"}
    if use_web:
        news_items = scrape_news(symbol, max_articles=5)
        if not news_items:
            return {"error": "No news found for analysis"}
        sia = SentimentIntensityAnalyzer()
        polarities = []
        sources = []
        for item in news_items:
            content = item.get('title', '') + ' ' + (item.get('summary', '') or '')
            if content:
                polarity = sia.polarity_scores(content)['compound']
                polarities.append(polarity)
                sources.append(item.get('url', ''))
        if polarities:
            avg_polarity = sum(polarities) / len(polarities)
            if avg_polarity > 0.05:
                sentiment = 'positive'
            elif avg_polarity < -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            return {
                'polarity': round(avg_polarity, 4),
                'sentiment': sentiment,
                'sources': sources
            }
        else:
            return {"error": "No content found for analysis"}
    else:
        state = State()
        state["stock_data"] = {symbol: {}}
        temp_state = data_fetcher_agent(state, [symbol], period="1y")
        if symbol not in temp_state["stock_data"]:
            return {"error": "Failed to fetch data for sentiment analysis"}
        analysis_state = sentiment_analysis_agent(temp_state)
        sentiment = analysis_state.get("sentiment_scores", {}).get(symbol, {})
        return {"symbol": symbol, "sentiment": sentiment}

def validate_indian_stock(symbol: str) -> dict:
    if symbol.endswith(".NS"):
        return {"valid": True, "symbol": symbol}
    else:
        return {"valid": False, "error": "Only Indian stocks (.NS) are supported"}

def get_tool_schemas():
    return [
        {
            "type": "function",
            "function": {
                "name": "fetch_stock_price",
                "description": "Use this tool to get the latest or historical stock price for a specific Indian stock. The symbol must end with '.NS'. This provides real-time data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "The official NSE stock symbol, ending in '.NS' (e.g., 'RELIANCE.NS')."
                        },
                        "period": {
                            "type": "string",
                            "description": "The data period to fetch (e.g., '1d', '5d', '1mo', '1y'). Defaults to '1d'.",
                            "default": "1d"
                        }
                    },
                    "required": ["symbol"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "compute_technical_indicators",
                "description": "Calculate key technical indicators like RSI and MACD for an Indian stock symbol. This is useful for analyzing momentum and trend.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "The official NSE stock symbol, ending in '.NS' (e.g., 'INFY.NS')."
                        },
                        "indicators": {
                            "type": "array",
                            "description": "A list of indicators to compute. Supported: 'RSI', 'MACD'.",
                            "items": {
                                "type": "string"
                            },
                            "default": ["RSI", "MACD"]
                        }
                    },
                    "required": ["symbol"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_sentiment",
                "description": "Analyzes the market sentiment for an Indian stock by searching for recent news and applying sentiment analysis. Essential for understanding market mood.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "The official NSE stock symbol, ending in '.NS' (e.g., 'HDFCBANK.NS')."
                        },
                        "use_web": {
                            "type": "boolean",
                            "description": "Set to True to perform a web search for the latest news. Defaults to True.",
                            "default": True
                        }
                    },
                    "required": ["symbol"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "validate_indian_stock",
                "description": "Verify if a given stock symbol is a valid Indian stock by checking if it ends with the '.NS' suffix. Use this for validation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "The stock symbol to validate (e.g., 'SBIN.NS' or 'AAPL')."
                        }
                    },
                    "required": ["symbol"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Use this tool to search the web for any information, such as latest news, financial reports, or market analysis for a given query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query (e.g., 'latest news on the Indian stock market')."
                        },
                        "max_results": {
                            "type": "integer",
                            "default": 5,
                            "description": "The maximum number of search results to return."
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "stock_search",
                "description": "Search for official Indian stock symbols (.NS) based on a company name or keyword. Useful if you don't know the exact symbol.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The company name or keyword to search for (e.g., 'Tata')."
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]