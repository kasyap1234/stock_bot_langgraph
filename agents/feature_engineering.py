

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import yfinance as yf

from config.config import (
    RSI_OVERBOUGHT, RSI_OVERSOLD,
    FIB_LEVELS, SUPPORT_RESISTANCE_PERIODS
)
from data.models import State

logger = logging.getLogger(__name__)

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    logger.warning("TA-Lib not available, using basic technical analysis")
    TALIB_AVAILABLE = False


class FeatureEngineer:
    

    def __init__(self):
        self.sector_mapping = self._load_sector_mapping()
        self.market_cap_cache = {}

    def _load_sector_mapping(self) -> Dict[str, str]:
        
        # Basic sector mapping for NIFTY 50 stocks
        return {
            'RELIANCE.NS': 'Energy',
            'TCS.NS': 'Technology',
            'HDFCBANK.NS': 'Financial Services',
            'ICICIBANK.NS': 'Financial Services',
            'INFY.NS': 'Technology',
            'HINDUNILVR.NS': 'Consumer Goods',
            'ITC.NS': 'Consumer Goods',
            'KOTAKBANK.NS': 'Financial Services',
            'LT.NS': 'Industrials',
            'AXISBANK.NS': 'Financial Services',
            'MARUTI.NS': 'Consumer Goods',
            'BAJFINANCE.NS': 'Financial Services',
            'BHARTIARTL.NS': 'Telecommunications',
            'HCLTECH.NS': 'Technology',
            'WIPRO.NS': 'Technology',
            'ULTRACEMCO.NS': 'Materials',
            'NESTLEIND.NS': 'Consumer Goods',
            'POWERGRID.NS': 'Utilities',
            'NTPC.NS': 'Utilities',
            'ONGC.NS': 'Energy',
            'COALINDIA.NS': 'Energy',
            'GRASIM.NS': 'Materials',
            'JSWSTEEL.NS': 'Materials',
            'TATASTEEL.NS': 'Materials',
            'ADANIPORTS.NS': 'Industrials',
            'SHREECEM.NS': 'Materials',
            'BAJAJ-AUTO.NS': 'Consumer Goods',
            'TITAN.NS': 'Consumer Goods',
            'HEROMOTOCO.NS': 'Consumer Goods',
            'DRREDDY.NS': 'Healthcare',
            'SUNPHARMA.NS': 'Healthcare',
            'CIPLA.NS': 'Healthcare',
            'DIVISLAB.NS': 'Healthcare',
            'APOLLOHOSP.NS': 'Healthcare',
            'INDUSINDBK.NS': 'Financial Services',
            'HDFCLIFE.NS': 'Financial Services',
            'SBILIFE.NS': 'Financial Services',
            'BRITANNIA.NS': 'Consumer Goods',
            'TECHM.NS': 'Technology',
            'EICHERMOT.NS': 'Consumer Goods',
            'BPCL.NS': 'Energy',
            'UPL.NS': 'Materials',
            'M&M.NS': 'Consumer Goods',
            'TATACONSUM.NS': 'Consumer Goods',
            'ASIANPAINT.NS': 'Materials',
            'PIDILITIND.NS': 'Materials',
            'NMDC.NS': 'Materials',
            'GAIL.NS': 'Utilities',
            'VEDL.NS': 'Materials'
        }

    def get_market_cap(self, symbol: str) -> float:
        
        if symbol in self.market_cap_cache:
            return self.market_cap_cache[symbol]

        try:
            ticker = yf.Ticker(symbol)
            market_cap = ticker.info.get('marketCap', 0)
            if market_cap:
                self.market_cap_cache[symbol] = market_cap
                return market_cap
        except Exception as e:
            logger.warning(f"Could not fetch market cap for {symbol}: {e}")

        # Default market caps for major stocks
        defaults = {
            'RELIANCE.NS': 1800000,  # ~18 lakh crores
            'TCS.NS': 1400000,
            'HDFCBANK.NS': 800000,
            'ICICIBANK.NS': 600000,
            'INFY.NS': 700000,
        }
        self.market_cap_cache[symbol] = defaults.get(symbol, 100000)  # Default 1 lakh crores
        return self.market_cap_cache[symbol]

    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        features = pd.DataFrame(index=df.index)

        # Ensure consistent column names
        if 'close' in df.columns:
            df = df.rename(columns={
                'close': 'Close', 'high': 'High', 'low': 'Low',
                'open': 'Open', 'volume': 'Volume'
            })

        # Ensure data types are correct for TA-Lib (float64)
        df = df.astype({
            'Open': 'float64',
            'High': 'float64',
            'Low': 'float64',
            'Close': 'float64',
            'Volume': 'float64'
        })

        # Price-based features
        features['close'] = df['Close']
        features['high'] = df['High']
        features['low'] = df['Low']
        features['open'] = df['Open']
        features['volume'] = df['Volume']

        # Returns
        features['returns'] = df['Close'].pct_change()
        features['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # Volatility
        features['volatility_20'] = df['Close'].pct_change().rolling(20).std()
        features['volatility_60'] = df['Close'].pct_change().rolling(60).std()

        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            features[f'sma_{period}'] = df['Close'].rolling(period).mean()
            features[f'ema_{period}'] = df['Close'].ewm(span=period).mean()

        # RSI
        if TALIB_AVAILABLE:
            features['rsi'] = talib.RSI(df['Close'].values, timeperiod=14)
        else:
            # Basic RSI calculation
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        if TALIB_AVAILABLE:
            macd, macdsignal, macdhist = talib.MACD(df['Close'].values)
            features['macd'] = macd
            features['macd_signal'] = macdsignal
            features['macd_hist'] = macdhist
        else:
            ema12 = df['Close'].ewm(span=12).mean()
            ema26 = df['Close'].ewm(span=26).mean()
            features['macd'] = ema12 - ema26
            features['macd_signal'] = features['macd'].ewm(span=9).mean()

        # Bollinger Bands
        if TALIB_AVAILABLE:
            upper, middle, lower = talib.BBANDS(df['Close'].values)
            features['bb_upper'] = upper
            features['bb_middle'] = middle
            features['bb_lower'] = lower
        else:
            sma20 = df['Close'].rolling(20).mean()
            std20 = df['Close'].rolling(20).std()
            features['bb_upper'] = sma20 + (std20 * 2)
            features['bb_middle'] = sma20
            features['bb_lower'] = sma20 - (std20 * 2)

        # Bollinger Band position and width
        features['bb_position'] = (df['Close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']

        # Stochastic Oscillator
        if TALIB_AVAILABLE:
            slowk, slowd = talib.STOCH(df['High'].values, df['Low'].values, df['Close'].values)
            features['stoch_k'] = slowk
            features['stoch_d'] = slowd
        else:
            high_14 = df['High'].rolling(14).max()
            low_14 = df['Low'].rolling(14).min()
            features['stoch_k'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
            features['stoch_d'] = features['stoch_k'].rolling(3).mean()

        # Williams %R
        if TALIB_AVAILABLE:
            features['williams_r'] = talib.WILLR(df['High'].values, df['Low'].values, df['Close'].values)
        else:
            high_14 = df['High'].rolling(14).max()
            low_14 = df['Low'].rolling(14).min()
            features['williams_r'] = -100 * ((high_14 - df['Close']) / (high_14 - low_14))

        # Commodity Channel Index
        if TALIB_AVAILABLE:
            features['cci'] = talib.CCI(df['High'].values, df['Low'].values, df['Close'].values)
        else:
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            sma_tp = tp.rolling(20).mean()
            mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
            features['cci'] = (tp - sma_tp) / (0.015 * mad)

        # Average True Range
        if TALIB_AVAILABLE:
            features['atr'] = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values)
        else:
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift(1))
            low_close = np.abs(df['Low'] - df['Close'].shift(1))
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            features['atr'] = tr.rolling(14).mean()

        # On Balance Volume
        if TALIB_AVAILABLE:
            features['obv'] = talib.OBV(df['Close'].values, df['Volume'].values)
        else:
            obv = pd.Series(0, index=df.index)
            for i in range(1, len(df)):
                if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + df['Volume'].iloc[i]
                elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - df['Volume'].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            features['obv'] = obv

        # Volume indicators
        features['volume_sma_20'] = df['Volume'].rolling(20).mean()
        features['volume_ratio'] = df['Volume'] / features['volume_sma_20']

        # Momentum indicators
        for period in [1, 3, 5, 10]:
            features[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1

        # Rate of Change
        for period in [5, 10, 20]:
            features[f'roc_{period}'] = ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)) * 100

        return features.dropna()

    def create_sentiment_features(self, sentiment_data: Dict[str, Any]) -> pd.Series:
        
        features = pd.Series(dtype=float)

        if not sentiment_data or 'error' in sentiment_data:
            # Return neutral sentiment features
            features = pd.Series({
                'sentiment_positive': 0.5,
                'sentiment_negative': 0.5,
                'sentiment_compound': 0.0,
                'sentiment_articles': 0,
                'sentiment_twitter': 0
            })
        else:
            features = pd.Series({
                'sentiment_positive': sentiment_data.get('positive', 0.5),
                'sentiment_negative': sentiment_data.get('negative', 0.5),
                'sentiment_compound': sentiment_data.get('compound', 0.0),
                'sentiment_articles': sentiment_data.get('articles_analyzed', 0),
                'sentiment_twitter': sentiment_data.get('twitter_analyzed', 0)
            })

        return features

    def create_macro_features(self, macro_data: Dict[str, float]) -> pd.Series:
        
        features = pd.Series({
            'macro_rbi_repo': macro_data.get('RBI_REPO', 0.0),
            'macro_unemployment': macro_data.get('INDIA_UNRATE', 0.0),
            'macro_gdp_growth': macro_data.get('INDIA_GDP', 0.0),
            'macro_composite': macro_data.get('composite', 0.0)
        })

        return features

    def create_cross_sectional_features(self, symbol: str) -> pd.Series:
        
        sector = self.sector_mapping.get(symbol, 'Unknown')
        market_cap = self.get_market_cap(symbol)

        # One-hot encode sector
        sector_features = {}
        all_sectors = set(self.sector_mapping.values())
        for sec in all_sectors:
            sector_features[f'sector_{sec.lower().replace(" ", "_")}'] = 1 if sector == sec else 0

        # Market cap features
        market_cap_log = np.log(market_cap) if market_cap > 0 else 0

        features = pd.Series({
            'market_cap': market_cap,
            'market_cap_log': market_cap_log,
            **sector_features
        })

        return features

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        features = pd.DataFrame(index=df.index)

        if isinstance(df.index, pd.DatetimeIndex):
            features['day_of_week'] = df.index.dayofweek
            features['month'] = df.index.month
            features['quarter'] = df.index.quarter
            features['day_of_year'] = df.index.dayofyear
            features['week_of_year'] = df.index.isocalendar().week.astype(int)

            # Cyclical encoding for day of week and month
            features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
            features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
            features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
            features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        else:
            # If not datetime index, create neutral features
            features['day_of_week'] = 0
            features['month'] = 1
            features['quarter'] = 1
            features['day_of_year'] = 1
            features['week_of_year'] = 1
            features['day_sin'] = 0
            features['day_cos'] = 1
            features['month_sin'] = 0
            features['month_cos'] = 1

        return features

    def create_fibonacci_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        features = pd.DataFrame(index=df.index)

        if len(df) < 50:
            return features

        # Calculate recent high/low for Fibonacci levels
        lookback = min(50, len(df))
        recent_high = df['High'].tail(lookback).max()
        recent_low = df['Low'].tail(lookback).min()
        diff = recent_high - recent_low

        for level in FIB_LEVELS:
            fib_level = recent_high - (diff * level)
            features[f'fib_{level}'] = fib_level
            # Distance from current price to fib level
            features[f'fib_{level}_distance'] = (df['Close'] - fib_level) / df['Close']

        return features

    def create_support_resistance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        features = pd.DataFrame(index=df.index)

        for period_name, period in SUPPORT_RESISTANCE_PERIODS.items():
            if len(df) < period:
                continue

            recent_high = df['High'].tail(period).max()
            recent_low = df['Low'].tail(period).min()

            features[f'support_{period_name}'] = recent_low
            features[f'resistance_{period_name}'] = recent_high

            # Distance from current price to S/R levels
            features[f'support_{period_name}_distance'] = (df['Close'] - recent_low) / df['Close']
            features[f'resistance_{period_name}_distance'] = (df['Close'] - recent_high) / df['Close']

        return features

    def create_all_features(self, state: State, symbol: str) -> pd.DataFrame:
        
        stock_data = state.get('stock_data', {})
        sentiment_scores = state.get('sentiment_scores', {})
        macro_scores = state.get('macro_scores', {})

        if symbol not in stock_data:
            logger.warning(f"No stock data found for {symbol}")
            return pd.DataFrame()

        df = stock_data[symbol]
        if df.empty:
            logger.warning(f"Empty stock data for {symbol}")
            return pd.DataFrame()

        # Technical features
        technical_features = self.create_technical_features(df)

        # Sentiment features
        sentiment_data = sentiment_scores.get(symbol, {})
        sentiment_features = self.create_sentiment_features(sentiment_data)

        # Macro features
        macro_features = self.create_macro_features(macro_scores)

        # Cross-sectional features
        cross_features = self.create_cross_sectional_features(symbol)

        # Temporal features
        temporal_features = self.create_temporal_features(df)

        # Fibonacci features
        fib_features = self.create_fibonacci_features(df)

        # Support/Resistance features
        sr_features = self.create_support_resistance_features(df)

        # Combine all features
        all_features = [technical_features]

        # Add scalar features to each row
        for feature_df in [temporal_features, fib_features, sr_features]:
            if not feature_df.empty:
                all_features.append(feature_df)

        # Concatenate DataFrames
        combined_features = pd.concat(all_features, axis=1)

        # Add scalar features as constant columns
        for name, value in sentiment_features.items():
            combined_features[name] = value

        for name, value in macro_features.items():
            combined_features[name] = value

        for name, value in cross_features.items():
            combined_features[name] = value

        # Fill any remaining NaN values
        combined_features = combined_features.fillna(0)

        logger.info(f"Created {len(combined_features.columns)} features for {symbol}")
        return combined_features

    def prepare_training_data(self, features: pd.DataFrame, target_horizon: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
        
        if len(features) < target_horizon + 10:
            logger.warning("Insufficient data for training")
            return pd.DataFrame(), pd.Series()

        # Create target: future price movement
        future_close = features['close'].shift(-target_horizon)
        target = (future_close > features['close']).astype(int)  # 1 for up, 0 for down

        # Remove rows with NaN target
        valid_idx = target.dropna().index
        X = features.loc[valid_idx]
        y = target.loc[valid_idx]

        # Remove target from features
        if 'close' in X.columns:
            X = X.drop('close', axis=1)

        return X, y


def feature_engineering_agent(state: State) -> State:
    
    logging.info("Starting feature engineering agent")

    stock_data = state.get("stock_data", {})
    sentiment_scores = state.get("sentiment_scores", {})
    macro_scores = state.get("macro_scores", {})

    if not stock_data:
        logger.warning("No stock data available for feature engineering")
        return state

    engineer = FeatureEngineer()
    engineered_features = {}

    for symbol in stock_data.keys():
        try:
            features = engineer.create_all_features(state, symbol)
            if not features.empty:
                engineered_features[symbol] = features
                logger.info(f"Engineered {len(features.columns)} features for {symbol}")
            else:
                logger.warning(f"No features created for {symbol}")

        except Exception as e:
            logger.error(f"Error in feature engineering for {symbol}: {e}")
            continue

    state["engineered_features"] = engineered_features
    logger.info(f"Completed feature engineering for {len(engineered_features)} symbols")

    return state