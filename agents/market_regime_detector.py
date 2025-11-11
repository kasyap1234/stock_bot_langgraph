import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from data.models import State
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MarketRegimeDetector:
    """
    Detects market regimes (e.g., bull, bear, sideways, high volatility) based on price and volume data.
    """
    
    def __init__(self, lookback_period: int = 60, volatility_window: int = 20):
        self.lookback_period = lookback_period
        self.volatility_window = volatility_window
        self.regime_model = KMeans(n_clusters=4, random_state=42)
        self.scaler = StandardScaler()
        
    def detect_regime(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Detect the current market regime for a given stock.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with regime information
        """
        if df.empty or len(df) < self.lookback_period:
            logger.warning(f"Insufficient data for regime detection. Required: {self.lookback_period}, Got: {len(df)}")
            return {
                'regime': 'insufficient_data',
                'regime_score': 0.0,
                'volatility_regime': 'unknown',
                'trend_regime': 'unknown',
                'confidence': 0.0
            }
        
        # Calculate features for regime detection
        features = self._calculate_regime_features(df)
        
        # Determine regime based on features
        regime_info = self._classify_regime(features, df)
        
        return regime_info
    
    def _calculate_regime_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate features used for regime detection."""
        df = df.copy()
        
        # Ensure correct column names
        if 'close' in df.columns:
            df = df.rename(columns={'close': 'Close', 'high': 'High', 'low': 'Low', 'open': 'Open', 'volume': 'Volume'})
        
        # Calculate returns
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Volatility features
        df['volatility_20'] = df['returns'].rolling(window=self.volatility_window).std() * np.sqrt(252)
        df['volatility_60'] = df['returns'].rolling(window=60).std() * np.sqrt(252)
        
        # Trend features
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        df['price_position'] = (df['Close'] - df['sma_20']) / df['sma_20']
        
        # Momentum features
        df['rsi'] = self._calculate_rsi(df['Close'], 14)
        df['macd'], df['macd_signal'], _ = self._calculate_macd(df['Close'])
        
        # Volume features
        df['volume_sma'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        # Recent statistics
        recent_data = df.tail(self.lookback_period)
        
        features = {
            'avg_volatility': recent_data['volatility_20'].mean(),
            'volatility_trend': recent_data['volatility_20'].tail(10).mean() - recent_data['volatility_20'].head(10).mean(),
            'avg_returns': recent_data['returns'].mean() * 252,  # Annualized
            'trend_strength': recent_data['price_position'].mean(),
            'momentum': recent_data['rsi'].mean(),
            'volume_activity': recent_data['volume_ratio'].mean(),
            'price_acceleration': recent_data['returns'].diff().mean()
        }
        
        return features
    
    def _classify_regime(self, features: Dict[str, float], df: pd.DataFrame) -> Dict[str, any]:
        """Classify the market regime based on calculated features."""
        avg_volatility = features['avg_volatility']
        avg_returns = features['avg_returns']
        trend_strength = features['trend_strength']
        momentum = features['momentum']
        
        # Define regime classification rules
        if avg_returns > 0.15 and trend_strength > 0.02 and momentum > 50:
            regime = 'bull_strong'
            confidence = 0.9
        elif avg_returns > 0.05 and trend_strength > 0.05:
            regime = 'bull_moderate'
            confidence = 0.8
        elif avg_returns < -0.10 and trend_strength < -0.02 and momentum < 40:
            regime = 'bear_strong'
            confidence = 0.9
        elif avg_returns < -0.02 and trend_strength < -0.005:
            regime = 'bear_moderate'
            confidence = 0.8
        elif avg_volatility > 0.4:
            regime = 'high_volatility'
            confidence = 0.85
        elif abs(avg_returns) < 0.03 and abs(trend_strength) < 0.01:
            regime = 'sideways'
            confidence = 0.7
        else:
            regime = 'moderate'
            confidence = 0.6
        
        # Determine volatility and trend sub-regimes
        if avg_volatility > 0.3:
            volatility_regime = 'high'
        elif avg_volatility > 0.15:
            volatility_regime = 'medium'
        else:
            volatility_regime = 'low'
            
        if trend_strength > 0.02:
            trend_regime = 'strong_up'
        elif trend_strength > 0.005:
            trend_regime = 'moderate_up'
        elif trend_strength < -0.02:
            trend_regime = 'strong_down'
        elif trend_strength < -0.05:
            trend_regime = 'moderate_down'
        else:
            trend_regime = 'sideways'
        
        return {
            'regime': regime,
            'regime_score': avg_returns,  # Use annualized return as a score
            'volatility_regime': volatility_regime,
            'trend_regime': trend_regime,
            'confidence': confidence,
            'features': features
        }
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram


def market_regime_detection_agent(state: State) -> State:
    """
    Agent that detects market regimes for all stocks in the state.
    """
    logger.info("Starting market regime detection agent")
    
    stock_data = state.get("stock_data", {})
    regime_results = {}
    
    detector = MarketRegimeDetector()
    
    for symbol, df in stock_data.items():
        try:
            regime_info = detector.detect_regime(df)
            regime_results[symbol] = regime_info
            
            logger.info(f"Detected regime for {symbol}: {regime_info['regime']} "
                       f"(confidence: {regime_info['confidence']:.2f})")
        except Exception as e:
            logger.error(f"Error detecting regime for {symbol}: {e}")
            regime_results[symbol] = {
                'regime': 'error',
                'regime_score': 0.0,
                'volatility_regime': 'unknown',
                'trend_regime': 'unknown',
                'confidence': 0.0,
                'error': str(e)
            }
    
    # Update state with regime information
    state["market_regimes"] = regime_results
    
    # Also update individual regime dictionaries for backward compatibility
    volatility_regime = {symbol: info['volatility_regime'] for symbol, info in regime_results.items()}
    trend_regime = {symbol: info['trend_regime'] for symbol, info in regime_results.items()}
    market_sentiment_regime = {symbol: 'bullish' if info['regime_score'] > 0 else 'bearish' 
                               for symbol, info in regime_results.items()}
    correlation_regime = {symbol: 'high' if info['features'].get('avg_volatility', 0) > 0.2 else 'low' 
                          for symbol, info in regime_results.items() if 'features' in info}
    volume_regime = {symbol: info['features'].get('volume_activity', 'normal') 
                     for symbol, info in regime_results.items() if 'features' in info}
    
    state["volatility_regime"] = volatility_regime
    state["trend_regime"] = trend_regime
    state["market_sentiment_regime"] = market_sentiment_regime
    state["correlation_regime"] = correlation_regime
    state["volume_regime"] = volume_regime
    
    logger.info(f"Completed market regime detection for {len(regime_results)} symbols")
    return state