

import logging
from functools import lru_cache
from typing import Dict, List, Any

from config.constants import DEFAULT_STOCKS
from config.api_config import TWITTER_BEARER_TOKEN
from data.models import State
from data.apis import get_news_articles

logger = logging.getLogger(__name__)

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    logger.warning("VADER not available, using basic sentiment analysis")
    VADER_AVAILABLE = False


@lru_cache(maxsize=128)
def _get_news_sentiment(symbol: str, max_articles: int = 15) -> Dict[str, float]:
    
    try:
        # Use NewsAPI for news data
        news_data = get_news_articles(symbol, max_articles=max_articles)

        if not news_data:
            logger.warning(f"No news data found for {symbol}")
            return {
                "positive": 0.0,
                "negative": 0.0,
                "compound": 0.0,
                "articles_analyzed": 0
            }

        # Extract headlines for sentiment analysis
        headlines = [article.get('title', '') for article in news_data if article.get('title')]

        if not headlines:
            return {
                "positive": 0.0,
                "negative": 0.0,
                "compound": 0.0,
                "articles_analyzed": 0
            }

        # Perform sentiment analysis
        # Twitter sentiment if key available
        twitter_compound = 0.0
        twitter_analyzed = 0
        if TWITTER_BEARER_TOKEN and TWITTER_BEARER_TOKEN != "":
            try:
                from tweepy import Client
                client = Client(bearer_token=TWITTER_BEARER_TOKEN)
                tweets = client.search_recent_tweets(
                    query=f"{symbol} stock OR shares lang:en -is:retweet",
                    max_results=100,
                    tweet_fields=['created_at']
                )
                
                if tweets.data:
                    tweet_texts = [tweet.text for tweet in tweets.data]
                    if VADER_AVAILABLE:
                        analyzer = SentimentIntensityAnalyzer()
                        tweet_sentiments = [analyzer.polarity_scores(text) for text in tweet_texts]
                        twitter_compound = sum(s['compound'] for s in tweet_sentiments) / len(tweet_sentiments)
                        twitter_analyzed = len(tweet_texts)
                        logger.info(f"Analyzed {twitter_analyzed} tweets for {symbol}")
            except Exception as e:
                logger.warning(f"Twitter fetch failed for {symbol}: {e}")
        
        if VADER_AVAILABLE:
            analyzer = SentimentIntensityAnalyzer()
            sentiments = [analyzer.polarity_scores(headline) for headline in headlines]
    
            # Calculate averages
            avg_positive = sum(s['pos'] for s in sentiments) / len(sentiments) if sentiments else 0
            avg_negative = sum(s['neg'] for s in sentiments) / len(sentiments) if sentiments else 0
            avg_compound = sum(s['compound'] for s in sentiments) / len(sentiments) if sentiments else 0
    
            # Average with Twitter if available
            if twitter_analyzed > 0:
                total_compound = (avg_compound * len(headlines) + twitter_compound * twitter_analyzed) / (len(headlines) + twitter_analyzed)
                avg_compound = total_compound
                total_analyzed = len(headlines) + twitter_analyzed
            else:
                total_analyzed = len(headlines)
    
            scores = {
                "positive": avg_positive,
                "negative": avg_negative,
                "compound": avg_compound,
                "articles_analyzed": total_analyzed,
                "twitter_analyzed": twitter_analyzed
            }
        else:
            # Basic sentiment analysis without VADER
            scores = _basic_sentiment_analysis(headlines)

        logger.info(f"Analyzed {len(headlines)} articles for {symbol}")
        return scores

    except Exception as e:
        logger.error(f"Error getting sentiment for {symbol}: {e}")
        return {"error": "Sentiment analysis failed"}
    
    if not TWITTER_BEARER_TOKEN or TWITTER_BEARER_TOKEN == "":
        logger.info("Twitter API key not configured. Skipping Twitter sentiment.")


def _basic_sentiment_analysis(headlines: List[str]) -> Dict[str, float]:
    
    positive_words = {
        'profit', 'surge', 'rise', 'gain', 'growth', 'up', 'increase', 'boost', 'strong',
        'bull', 'buy', 'rally', 'jump', 'beat', 'exceed', 'positive', 'good', 'excellent'
    }

    negative_words = {
        'loss', 'fall', 'drop', 'decline', 'down', 'decrease', 'weak', 'bear', 'sell',
        'crash', 'plunge', 'slump', 'tumble', 'miss', 'below', 'negative', 'bad', 'terrible'
    }

    total_score = 0
    total_words = 0

    for headline in headlines:
        words = headline.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)

        # Net sentiment for this headline
        headline_score = (positive_count - negative_count) / max(len(words), 1)
        total_score += headline_score
        total_words += len(words)

    if headlines:
        avg_compound = total_score / len(headlines)
    else:
        avg_compound = 0

    # Convert to positive/negative scales
    avg_positive = max(0, avg_compound) * 0.5 + 0.1  # Scale and bias
    avg_negative = max(0, -avg_compound) * 0.5 + 0.1

    return {
        "positive": min(avg_positive, 1.0),
        "negative": min(avg_negative, 1.0),
        "compound": avg_compound,
        "articles_analyzed": len(headlines)
    }


def sentiment_analysis_agent(state: State) -> State:
    
    logging.info("Starting sentiment analysis agent")

    stock_data = state.get("stock_data", {})
    sentiment_scores = {}

    for symbol in stock_data.keys():
        try:
            # Get news sentiment
            news_sentiment = _get_news_sentiment(symbol)

            # Store sentiment data
            if "error" not in news_sentiment:
                sentiment_scores[symbol] = news_sentiment
                logger.info(f"Completed sentiment analysis for {symbol}")
            else:
                sentiment_scores[symbol] = news_sentiment

        except Exception as e:
            logger.error(f"Error in sentiment analysis for {symbol}: {e}")
            sentiment_scores[symbol] = {"error": "Sentiment analysis failed"}

    return {"sentiment_scores": sentiment_scores}