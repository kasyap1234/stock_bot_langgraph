

import logging
from typing import Dict, List, Any
import pandas as pd
import os
import numpy as np

from config.config import (
    RSI_OVERBOUGHT, RSI_OVERSOLD, CONFIRMATION_THRESHOLD,
    ENSEMBLE_THRESHOLD, TREND_STRENGTH_THRESHOLD,
    PROBABILITY_THRESHOLD, BACKTEST_VALIDATION_THRESHOLD,
    ENABLE_ADVANCED_TECH,
    ICHIMOKU_PERIODS, FIB_LEVELS, SUPPORT_RESISTANCE_PERIODS,
    ML_MODEL_PARAMS, ADAPTIVE_THRESHOLDS, GRID_SEARCH_PARAMS,
    MONTE_CARLO_SIMULATIONS, MONTE_CARLO_HORIZON,
    VPVR_BINS, VISIBLE_RANGE, VOLUME_PERCENTILE,
    HA_CONFLUENCE_BARS, HA_TFS, GARCH_P, GARCH_Q, FORECAST_HORIZON,
    HARMONIC_PATTERNS, TOLERANCE, LOOKBACK, MIN_CONFIDENCE,
    HMM_STATES, HMM_ITER, LSTM_EPOCHS, LSTM_BATCH, LSTM_WINDOW, LSTM_FEATURES,
    MC_VA_PATHS, VA_CONFIDENCE, STRESS_SCENARIOS,
    INDIA_SPECIFIC_PARAMS
)
from data.models import State
from utils.error_handling import retry_indicator_calculation
from data.quality_validator import validate_data, InsufficientDataError, ConstantPriceError

logger = logging.getLogger(__name__)

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    logger.warning("TA-Lib not available, using basic technical analysis")
    TALIB_AVAILABLE = False
try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    logger.warning("scipy not available, harmonic pattern detection disabled")
    SCIPY_AVAILABLE = False

try:
    from hmmlearn import hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    logger.warning("hmmlearn not available, HMM regime detection disabled")
    HMMLEARN_AVAILABLE = False

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    import numpy as np
    TENSORFLOW_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow or scikit-learn not available, LSTM predictor disabled")
    TENSORFLOW_AVAILABLE = False


class TradingSetup:
    
    def __init__(self, setup_type: str, direction: str, entry_price: float, stop_loss: float, take_profit: float, confidence: float = 0.5, risk_reward_ratio: float = 1.0, description: str = ""):
        self.setup_type = setup_type
        self.direction = direction
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.confidence = confidence
        self.risk_reward_ratio = risk_reward_ratio
        self.description = description


class MultiTimeframeAnalyzer:
    

    def __init__(self):
        self.timeframes = ['daily', 'weekly', 'monthly']

    def resample_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        
        if timeframe == 'weekly':
            return df.resample('W').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        elif timeframe == 'monthly':
            return df.resample('ME').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        else:
            return df

    def analyze_multi_timeframe(self, df: pd.DataFrame, signals_func) -> Dict[str, str]:
        
        combined_signals = {}

        # Ensure consistent column names
        if 'close' in df.columns:
            df = df.rename(columns={
                'close': 'Close',
                'high': 'High',
                'low': 'Low',
                'open': 'Open',
                'volume': 'Volume'
            })

        for tf in self.timeframes:
            resampled_df = self.resample_data(df, tf)
            if not resampled_df.empty and len(resampled_df) >= 20:
                # Ensure resampled df has consistent names
                if 'close' in resampled_df.columns:
                    resampled_df = resampled_df.rename(columns={
                        'close': 'Close',
                        'high': 'High',
                        'low': 'Low',
                        'open': 'Open',
                        'volume': 'Volume'
                    })
                tf_signals = signals_func(resampled_df)
                for indicator, signal in tf_signals.items():
                    key = f"{indicator}_{tf}"
                    combined_signals[key] = signal

        # Combine signals: if multiple timeframes agree, strengthen signal
        final_signals = {}
        for indicator in ['RSI', 'MACD', 'SMA', 'EMA', 'Bollinger', 'Stochastic', 'WilliamsR', 'CCI']:
            tf_signals = [combined_signals.get(f"{indicator}_{tf}", "neutral") for tf in self.timeframes]
            buy_count = tf_signals.count("buy")
            sell_count = tf_signals.count("sell")

            if buy_count >= 2:
                final_signals[indicator] = "buy"
            elif sell_count >= 2:
                final_signals[indicator] = "sell"
            else:
                final_signals[indicator] = "neutral"

        return final_signals


class SignalConfirmer:
    

    def __init__(self, confirmation_threshold: int = CONFIRMATION_THRESHOLD):
        self.confirmation_threshold = confirmation_threshold

    def confirm_signals(self, signals: Dict[str, str]) -> Dict[str, str]:
        
        confirmed_signals = {}
        buy_indicators = [k for k, v in signals.items() if v == "buy"]
        sell_indicators = [k for k, v in signals.items() if v == "sell"]

        if len(buy_indicators) >= self.confirmation_threshold:
            for indicator in buy_indicators:
                confirmed_signals[indicator] = "buy_confirmed"
        elif len(sell_indicators) >= self.confirmation_threshold:
            for indicator in sell_indicators:
                confirmed_signals[indicator] = "sell_confirmed"
        else:
            for indicator in signals:
                confirmed_signals[indicator] = signals[indicator]

        return confirmed_signals


class AdaptiveParameterCalculator:
    

    def __init__(self):
        pass

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        
        # Ensure consistent column names
        if 'close' in df.columns:
            df = df.rename(columns={
                'close': 'Close',
                'high': 'High',
                'low': 'Low',
                'open': 'Open',
                'volume': 'Volume'
            })
        high = df['High']
        low = df['Low']
        close = df['Close'].shift(1)
        tr = pd.concat([high - low, abs(high - close), abs(low - close)], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def adaptive_rsi_period(self, df: pd.DataFrame) -> int:
        
        atr = self.calculate_atr(df)
        if len(atr) == 0 or pd.isna(atr.iloc[-1]):
            return 14
        volatility = atr.iloc[-1] / df['Close'].iloc[-1]
        if volatility > ADAPTIVE_THRESHOLDS['high_volatility']:  # High volatility
            return 9
        elif volatility > ADAPTIVE_THRESHOLDS['medium_volatility']:  # Medium
            return 14
        else:  # Low
            return 21

    def adaptive_macd_periods(self, df: pd.DataFrame) -> Dict[str, int]:
        
        atr = self.calculate_atr(df)
        if len(atr) == 0 or pd.isna(atr.iloc[-1]):
            return {'fast': 12, 'slow': 26, 'signal': 9}

        volatility = atr.iloc[-1] / df['Close'].iloc[-1]

        if volatility > ADAPTIVE_THRESHOLDS['high_volatility']:
            return {'fast': 8, 'slow': 17, 'signal': 6}  # Shorter periods for high volatility
        elif volatility > ADAPTIVE_THRESHOLDS['medium_volatility']:
            return {'fast': 12, 'slow': 26, 'signal': 9}  # Standard periods
        else:
            return {'fast': 17, 'slow': 35, 'signal': 12}  # Longer periods for low volatility

    def adaptive_stochastic_periods(self, df: pd.DataFrame) -> Dict[str, int]:
        
        atr = self.calculate_atr(df)
        if len(atr) == 0 or pd.isna(atr.iloc[-1]):
            return {'k': 14, 'd': 3}

        volatility = atr.iloc[-1] / df['Close'].iloc[-1]

        if volatility > ADAPTIVE_THRESHOLDS['high_volatility']:
            return {'k': 5, 'd': 3}  # Shorter periods for high volatility
        elif volatility > ADAPTIVE_THRESHOLDS['medium_volatility']:
            return {'k': 14, 'd': 3}  # Standard periods
        else:
            return {'k': 21, 'd': 5}  # Longer periods for low volatility

    def adaptive_ichimoku_periods(self, df: pd.DataFrame) -> Dict[str, int]:
        
        atr = self.calculate_atr(df)
        if len(atr) == 0 or pd.isna(atr.iloc[-1]):
            return ICHIMOKU_PERIODS

        volatility = atr.iloc[-1] / df['Close'].iloc[-1]

        if volatility > ADAPTIVE_THRESHOLDS['high_volatility']:
            return {
                'tenkan_sen': 6,    # Shorter for high volatility
                'kijun_sen': 17,
                'senkou_span_b': 26,
                'chikou_span': 17
            }
        elif volatility > ADAPTIVE_THRESHOLDS['medium_volatility']:
            return ICHIMOKU_PERIODS  # Standard periods
        else:
            return {
                'tenkan_sen': 12,   # Longer for low volatility
                'kijun_sen': 30,
                'senkou_span_b': 60,
                'chikou_span': 30
            }


class RiskAdjuster:
    

    def __init__(self):
        self.adaptive_calc = AdaptiveParameterCalculator()

    def calculate_stop_loss(self, df: pd.DataFrame, entry_price: float, direction: str) -> float:
        
        atr = self.adaptive_calc.calculate_atr(df)
        if len(atr) == 0 or pd.isna(atr.iloc[-1]):
            atr_value = entry_price * 0.02  # 2% default
        else:
            atr_value = atr.iloc[-1]

        if direction == "buy":
            return entry_price - (atr_value * 2)
        else:
            return entry_price + (atr_value * 2)

    def calculate_position_size(self, capital: float, risk_per_trade: float, stop_loss_distance: float) -> int:
        
        risk_amount = capital * risk_per_trade
        return int(risk_amount / stop_loss_distance)


class EnsembleSignalGenerator:
    """
    Generates a unified signal by combining weighted outputs from multiple indicators.
    This simplified version focuses on a core set of reliable indicators to reduce noise.
    """
    def __init__(self):
        # Simplified, more robust set of base weights
        self.base_weights = {
            'RSI': 0.15,
            'MACD': 0.15,
            'Bollinger': 0.10,
            'Stochastic': 0.10,
            'TrendStrength': 0.10, # ADX-based
            'Ichimoku': 0.15,
            'VolumeProfile': 0.10, # VPVR-based signal
            'Regime': 0.15, # HMM-based regime
        }
        self.performance_history = {} # For future self-adaptive weighting

    def _update_weights_dynamically(self, df: pd.DataFrame, signals: Dict[str, str]) -> Dict[str, float]:
        """
        Adjusts indicator weights based on market conditions like trend and volatility.
        """
        if len(df) < 20:
            return self.base_weights.copy()

        weights = self.base_weights.copy()
        
        # 1. Trend Adjustment
        trend_scorer = TrendStrengthScorer()
        trend_score = trend_scorer.score_trend_strength(df) # 0 to 1
        
        if trend_score > 0.7:  # Strong trending market
            weights['MACD'] *= 1.2
            weights['TrendStrength'] *= 1.2
            weights['RSI'] *= 0.8 # Reduce weight of oscillators
        elif trend_score < 0.3:  # Ranging market
            weights['RSI'] *= 1.2
            weights['Bollinger'] *= 1.2
            weights['Stochastic'] *= 1.2
            weights['MACD'] *= 0.8 # Reduce weight of trend followers

        # 2. Volatility Adjustment
        adaptive_calc = AdaptiveParameterCalculator()
        atr_normalized = adaptive_calc.calculate_atr(df, period=14).iloc[-1] / df['Close'].iloc[-1]
        
        if atr_normalized > ADAPTIVE_THRESHOLDS.get('high_volatility', 0.03): # High volatility
            # In high volatility, prefer signals that provide clear levels
            weights['Ichimoku'] *= 1.1
            weights['VolumeProfile'] *= 1.1
            weights['MACD'] *= 0.9 # MACD can be choppy

        # 3. Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            return {k: v / total_weight for k, v in weights.items()}
        return self.base_weights

    def generate_ensemble_signal(self, signals: Dict[str, str], df: pd.DataFrame, risk_adjustment: float = 0.0) -> Dict[str, Any]:
        """
        Calculates the final weighted score and generates a buy/sell/neutral signal.
        """
        weights = self._update_weights_dynamically(df, signals)
        
        score = 0.0
        contributing_signals = {}

        for indicator, signal in signals.items():
            if indicator in weights:
                weight = weights[indicator]
                if signal == "buy":
                    score += weight
                    contributing_signals[indicator] = weight
                elif signal == "sell":
                    score -= weight
                    contributing_signals[indicator] = -weight
        
        # Apply risk adjustment (e.g., from VaR model)
        # A negative adjustment makes a 'buy' less likely and 'sell' more likely
        score += risk_adjustment

        # Determine final signal based on the composite score
        if score > ENSEMBLE_THRESHOLD:
            final_signal = "buy"
        elif score < -ENSEMBLE_THRESHOLD:
            final_signal = "sell"
        else:
            final_signal = "neutral"
            
        return {
            "signal": final_signal,
            "score": score,
            "contributing_signals": contributing_signals,
            "weights": weights
        }


class TrendStrengthScorer:
    

    def __init__(self):
        pass

    def score_trend_strength(self, df: pd.DataFrame) -> float:
        
        if len(df) < 20:
            return 0.5

        # ADX
        if TALIB_AVAILABLE:
            try:
                adx = talib.ADX(df['High'].values, df['Low'].values, df['Close'].values)
                adx_score = adx[-1] / 100 if not pd.isna(adx[-1]) else 0.5
            except:
                adx_score = 0.5
        else:
            adx_score = 0.5

        # Moving average slope
        sma20 = df['Close'].rolling(20).mean()
        if len(sma20) >= 2:
            slope = (sma20.iloc[-1] - sma20.iloc[-2]) / sma20.iloc[-2]
            slope_score = min(max(slope * 10 + 0.5, 0), 1)  # Normalize
        else:
            slope_score = 0.5

        # Volume trend
        volume_sma = df['Volume'].rolling(20).mean()
        if len(volume_sma) >= 2:
            vol_trend = (volume_sma.iloc[-1] - volume_sma.iloc[-2]) / volume_sma.iloc[-2]
            vol_score = min(max(vol_trend + 0.5, 0), 1)
        else:
            vol_score = 0.5

        return (adx_score + slope_score + vol_score) / 3


class VolatilityAdjuster:
    

    def __init__(self):
        self.adaptive_calc = AdaptiveParameterCalculator()

    def normalize_by_volatility(self, indicator_value: float, df: pd.DataFrame) -> float:
        
        atr = self.adaptive_calc.calculate_atr(df)
        if len(atr) == 0 or pd.isna(atr.iloc[-1]) or atr.iloc[-1] == 0:
            return indicator_value
        return indicator_value / atr.iloc[-1]


class IchimokuCloud:
    

    def __init__(self, periods: Dict[str, int] = ICHIMOKU_PERIODS):
        self.tenkan_period = periods['tenkan_sen']
        self.kijun_period = periods['kijun_sen']
        self.senkou_period = periods['senkou_span_b']
        self.chikou_period = periods['chikou_span']

    def calculate_ichimoku(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        
        if len(df) < self.senkou_period:
            return {}

        # Tenkan-sen (Conversion Line)
        tenkan_high = df['High'].rolling(self.tenkan_period).max()
        tenkan_low = df['Low'].rolling(self.tenkan_period).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2

        # Kijun-sen (Base Line)
        kijun_high = df['High'].rolling(self.kijun_period).max()
        kijun_low = df['Low'].rolling(self.kijun_period).min()
        kijun_sen = (kijun_high + kijun_low) / 2

        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(self.kijun_period)

        # Senkou Span B (Leading Span B)
        senkou_high = df['High'].rolling(self.senkou_period).max()
        senkou_low = df['Low'].rolling(self.senkou_period).min()
        senkou_span_b = ((senkou_high + senkou_low) / 2).shift(self.kijun_period)

        # Chikou Span (Lagging Span)
        chikou_span = df['Close'].shift(-self.chikou_period)

        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }

    def get_ichimoku_signal(self, df: pd.DataFrame) -> str:
        
        ichimoku = self.calculate_ichimoku(df)
        if not ichimoku:
            return "neutral"

        close = df['Close'].iloc[-1]
        tenkan = ichimoku['tenkan_sen'].iloc[-1] if not pd.isna(ichimoku['tenkan_sen'].iloc[-1]) else None
        kijun = ichimoku['kijun_sen'].iloc[-1] if not pd.isna(ichimoku['kijun_sen'].iloc[-1]) else None
        senkou_a = ichimoku['senkou_span_a'].iloc[-1] if not pd.isna(ichimoku['senkou_span_a'].iloc[-1]) else None
        senkou_b = ichimoku['senkou_span_b'].iloc[-1] if not pd.isna(ichimoku['senkou_span_b'].iloc[-1]) else None

        if None in [tenkan, kijun, senkou_a, senkou_b]:
            return "neutral"

        # Bullish signals
        if (close > senkou_a and close > senkou_b and
            tenkan > kijun and tenkan > senkou_a and tenkan > senkou_b):
            return "buy"

        # Bearish signals
        if (close < senkou_a and close < senkou_b and
            tenkan < kijun and tenkan < senkou_a and tenkan < senkou_b):
            return "sell"

        return "neutral"


class FibonacciRetracement:
    

    def __init__(self, levels: List[float] = FIB_LEVELS):
        self.levels = levels

    def calculate_fib_levels(self, df: pd.DataFrame, lookback: int = 50) -> Dict[str, float]:
        
        if len(df) < lookback:
            return {}

        recent_df = df.tail(lookback)
        high = recent_df['High'].max()
        low = recent_df['Low'].min()
        diff = high - low

        fib_levels = {}
        for level in self.levels:
            fib_levels[f'fib_{level}'] = high - (diff * level)

        fib_levels['fib_high'] = high
        fib_levels['fib_low'] = low

        return fib_levels

    def get_fib_signal(self, df: pd.DataFrame) -> str:
        
        fib_levels = self.calculate_fib_levels(df)
        if not fib_levels:
            return "neutral"

        close = df['Close'].iloc[-1]
        high = fib_levels['fib_high']
        low = fib_levels['fib_low']

        # Check if price is near key Fibonacci levels
        for level_name, level_value in fib_levels.items():
            if level_name.startswith('fib_') and level_name not in ['fib_high', 'fib_low']:
                # Check if close is within 1% of the level
                if abs(close - level_value) / level_value < 0.01:
                    # Determine if it's a support or resistance level
                    if level_value < (high + low) / 2:  # Below midpoint = potential support
                        return "buy"
                    else:  # Above midpoint = potential resistance
                        return "sell"

        return "neutral"


class SupportResistanceCalculator:
    

    def __init__(self, periods: Dict[str, int] = SUPPORT_RESISTANCE_PERIODS):
        self.short_period = periods['short_term']
        self.medium_period = periods['medium_term']
        self.long_period = periods['long_term']

    def calculate_support_resistance(self, df: pd.DataFrame) -> Dict[str, float]:
        
        if len(df) < self.long_period:
            return {}

        levels = {}

        # Short-term levels
        short_high = df['High'].tail(self.short_period).max()
        short_low = df['Low'].tail(self.short_period).min()
        levels['short_resistance'] = short_high
        levels['short_support'] = short_low

        # Medium-term levels
        medium_high = df['High'].tail(self.medium_period).max()
        medium_low = df['Low'].tail(self.medium_period).min()
        levels['medium_resistance'] = medium_high
        levels['medium_support'] = medium_low

        # Long-term levels
        long_high = df['High'].tail(self.long_period).max()
        long_low = df['Low'].tail(self.long_period).min()
        levels['long_resistance'] = long_high
        levels['long_support'] = long_low

        return levels

    def get_sr_signal(self, df: pd.DataFrame) -> str:
        
        sr_levels = self.calculate_support_resistance(df)
        if not sr_levels:
            return "neutral"

        close = df['Close'].iloc[-1]

        # Check proximity to support levels (potential buy)
        support_levels = [sr_levels.get('short_support'), sr_levels.get('medium_support'), sr_levels.get('long_support')]
        for support in support_levels:
            if support and abs(close - support) / support < 0.02:  # Within 2% of support
                return "buy"

        # Check proximity to resistance levels (potential sell)
        resistance_levels = [sr_levels.get('short_resistance'), sr_levels.get('medium_resistance'), sr_levels.get('long_resistance')]
        for resistance in resistance_levels:
            if resistance and abs(close - resistance) / resistance < 0.02:  # Within 2% of resistance
                return "sell"

        return "neutral"


class VWAPAnalyzer:
    

    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        
        if len(df) < 2:
            return pd.Series(dtype=float)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
        return vwap

    def get_vwap_signal(self, df: pd.DataFrame) -> str:
        
        vwap = self.calculate_vwap(df)
        if len(vwap) == 0 or pd.isna(vwap.iloc[-1]):
            return "neutral"
        close = df['Close'].iloc[-1]
        deviation = abs(close - vwap.iloc[-1]) / vwap.iloc[-1]
        if deviation > 0.01:  # 1% deviation
            return "buy" if close > vwap.iloc[-1] else "sell"
        return "neutral"


class PivotPointsAnalyzer:
    

    def calculate_pivot_points(self, df: pd.DataFrame) -> Dict[str, float]:
        
        if len(df) < 2:
            return {}
        prev_high = df['High'].iloc[-2]
        prev_low = df['Low'].iloc[-2]
        prev_close = df['Close'].iloc[-2]
        pp = (prev_high + prev_low + prev_close) / 3
        r1 = 2 * pp - prev_low
        s1 = 2 * pp - prev_high
        r2 = pp + (prev_high - prev_low)
        s2 = pp - (prev_high - prev_low)
        return {
            'PP': pp,
            'R1': r1,
            'S1': s1,
            'R2': r2,
            'S2': s2
        }

    def get_pivot_signal(self, df: pd.DataFrame) -> str:
        
        pivots = self.calculate_pivot_points(df)
        if not pivots:
            return "neutral"
        close = df['Close'].iloc[-1]
        if close > pivots['R1']:
            return "buy"
        elif close < pivots['S1']:
            return "sell"
        elif pivots['S1'] < close < pivots['R1']:
            return "hold"
        return "neutral"


class MLSignalPredictor:
    

    def __init__(self, model_params: Dict[str, Any] = ML_MODEL_PARAMS):
        self.model_params = model_params
        self.model = None
        self.feature_columns = [
            'RSI', 'MACD', 'SMA', 'EMA', 'Bollinger', 'Stochastic',
            'WilliamsR', 'CCI', 'TrendStrength', 'Volume'
        ]
        self._initialize_model()

    def _initialize_model(self):
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(**self.model_params)
        except ImportError:
            logger.warning("scikit-learn not available, ML predictor disabled")
            self.model = None

    def prepare_features(self, df: pd.DataFrame, signals: Dict[str, str]) -> pd.DataFrame:
        
        features = pd.DataFrame(index=df.index)

        # Add price-based features
        features['close'] = df['Close']
        features['high'] = df['High']
        features['low'] = df['Low']
        features['volume'] = df['Volume']

        # Add technical indicator features
        for indicator in self.feature_columns:
            if indicator in signals:
                # Convert signal to numeric
                signal_value = 0
                if signals[indicator] == "buy":
                    signal_value = 1
                elif signals[indicator] == "sell":
                    signal_value = -1
                features[indicator] = signal_value
            else:
                features[indicator] = 0

        # Add lagged features
        for col in ['close', 'volume']:
            for lag in [1, 2, 3]:
                features[f'{col}_lag_{lag}'] = features[col].shift(lag)

        # Add rolling statistics
        features['close_ma_5'] = features['close'].rolling(5).mean()
        features['close_ma_20'] = features['close'].rolling(20).mean()
        features['volume_ma_5'] = features['volume'].rolling(5).mean()

        # Drop NaN values
        features = features.dropna()

        return features

    def train_model(self, df: pd.DataFrame, signals: Dict[str, str], target_horizon: int = 5):
        
        if self.model is None:
            return False

        features = self.prepare_features(df, signals)
        if len(features) < target_horizon + 10:
            return False

        # Create target: future price movement
        future_close = df['Close'].shift(-target_horizon)
        target = (future_close > df['Close']).astype(int)  # 1 for up, 0 for down

        # Align features and target
        common_index = features.index.intersection(target.index)
        X = features.loc[common_index]
        y = target.loc[common_index]

        if len(X) < 10:
            return False

        try:
            self.model.fit(X, y)
            return True
        except Exception as e:
            logger.error(f"Error training ML model: {e}")
            return False

    def predict_signal(self, df: pd.DataFrame, signals: Dict[str, str]) -> str:
        
        if self.model is None:
            return "neutral"

        features = self.prepare_features(df, signals)
        if len(features) == 0:
            return "neutral"

        try:
            # Use latest available features
            latest_features = features.iloc[-1:].values
            prediction = self.model.predict(latest_features)[0]
            probability = self.model.predict_proba(latest_features)[0]

            # Convert prediction to signal
            if prediction == 1 and probability[1] > 0.6:  # Confident up prediction
                return "buy"
            elif prediction == 0 and probability[0] > 0.6:  # Confident down prediction
                return "sell"
            else:
                return "neutral"
        except Exception as e:
            logger.error(f"Error predicting with ML model: {e}")
            return "neutral"

    def get_feature_importance(self) -> Dict[str, float]:
        
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return {}

        importance_dict = {}
        for i, col in enumerate(self.feature_columns + ['close', 'high', 'low', 'volume']):
            if i < len(self.model.feature_importances_):
                importance_dict[col] = self.model.feature_importances_[i]

        return importance_dict


class ProbabilityScorer:
    

    def __init__(self):
        self.historical_performance = {}  # Could be loaded from file

    def score_probability(self, signals: Dict[str, str]) -> float:
        
        # Simple implementation: average win rate
        win_rates = {
            'RSI': 0.55,
            'MACD': 0.52,
            'SMA': 0.53,
            'EMA': 0.54,
            'Bollinger': 0.51,
            'Stochastic': 0.56,
            'WilliamsR': 0.57,
            'CCI': 0.53
        }

        total_prob = 0.0
        count = 0

        for indicator, signal in signals.items():
            if indicator in win_rates and signal in ['buy', 'sell']:
                total_prob += win_rates[indicator]
                count += 1

        return total_prob / count if count > 0 else 0.5


class BacktestValidator:
    

    def __init__(self):
        from simulation.backtesting_engine import BacktestingEngine
        self.engine = BacktestingEngine(initial_capital=10000, commission_rate=0.001)
        self.walk_forward_window = 100  # Training window size
        self.walk_forward_step = 20    # Step size for walk-forward
        self.monte_carlo_sims = MONTE_CARLO_SIMULATIONS

    def validate_signal(self, df: pd.DataFrame, signal: str) -> float:
        
        if len(df) < 50:
            return 0.5

        # Use recent data for validation
        recent_df = df.tail(50)

        # Mock recommendations
        recommendations = {list(df.columns)[0]: {'action': signal.upper()}}

        try:
            results = self.engine.run_backtest(
                recommendations=recommendations,
                stock_data={list(df.columns)[0]: recent_df},
                start_date=recent_df.index[0],
                end_date=recent_df.index[-1]
            )
            return results.get('win_rate', 0.5)
        except:
            return 0.5

    def walk_forward_analysis(self, df: pd.DataFrame, signals: Dict[str, str]) -> Dict[str, float]:
        
        if len(df) < self.walk_forward_window + self.walk_forward_step:
            return {'stability_score': 0.5, 'avg_performance': 0.5}

        performances = []
        stability_scores = []

        # Perform walk-forward validation
        for i in range(0, len(df) - self.walk_forward_window, self.walk_forward_step):
            train_end = i + self.walk_forward_window
            test_end = min(train_end + self.walk_forward_step, len(df))

            if test_end >= len(df):
                break

            train_df = df.iloc[i:train_end]
            test_df = df.iloc[train_end:test_end]

            # Test signal performance on out-of-sample data
            test_signals = self._generate_test_signals(test_df, signals)
            performance = self._evaluate_signal_performance(test_df, test_signals)
            performances.append(performance)

            # Calculate stability (consistency of signals)
            stability = self._calculate_signal_stability(signals, test_signals)
            stability_scores.append(stability)

        avg_performance = sum(performances) / len(performances) if performances else 0.5
        avg_stability = sum(stability_scores) / len(stability_scores) if stability_scores else 0.5

        return {
            'stability_score': avg_stability,
            'avg_performance': avg_performance,
            'performance_std': pd.Series(performances).std() if performances else 0.0
        }

    def monte_carlo_simulation(self, df: pd.DataFrame, signals: Dict[str, str]) -> Dict[str, float]:
        
        if len(df) < 30:
            return {'expected_return': 0.0, 'var_95': 0.0, 'max_drawdown': 0.0}

        try:
            from scipy import stats
            import numpy as np
        except ImportError:
            logger.warning("scipy not available for Monte Carlo simulation")
            return {'expected_return': 0.0, 'var_95': 0.0, 'max_drawdown': 0.0}

        # Calculate historical returns
        returns = df['Close'].pct_change().dropna()
        if len(returns) < 10:
            return {'expected_return': 0.0, 'var_95': 0.0, 'max_drawdown': 0.0}

        # Fit distribution to returns
        try:
            mu, sigma = stats.norm.fit(returns)
        except:
            mu, sigma = returns.mean(), returns.std()

        # Run Monte Carlo simulations
        simulated_returns = []
        max_drawdowns = []

        for _ in range(self.monte_carlo_sims):
            # Generate random returns
            sim_returns = np.random.normal(mu, sigma, MONTE_CARLO_HORIZON)
            sim_prices = [df['Close'].iloc[-1]]

            # Simulate price path
            for ret in sim_returns:
                new_price = sim_prices[-1] * (1 + ret)
                sim_prices.append(new_price)

            # Calculate metrics
            total_return = (sim_prices[-1] - sim_prices[0]) / sim_prices[0]
            simulated_returns.append(total_return)

            # Calculate maximum drawdown
            peak = sim_prices[0]
            max_dd = 0
            for price in sim_prices:
                if price > peak:
                    peak = price
                dd = (peak - price) / peak
                max_dd = max(max_dd, dd)
            max_drawdowns.append(max_dd)

        # Calculate statistics
        expected_return = np.mean(simulated_returns)
        var_95 = np.percentile(simulated_returns, 5)  # 95% VaR
        avg_max_drawdown = np.mean(max_drawdowns)

        return {
            'expected_return': expected_return,
            'var_95': var_95,
            'max_drawdown': avg_max_drawdown,
            'return_std': np.std(simulated_returns)
        }

    def _generate_test_signals(self, df: pd.DataFrame, base_signals: Dict[str, str]) -> Dict[str, str]:
        
        # Simplified signal generation for testing
        signals = {}
        for indicator in base_signals.keys():
            # Use simple logic for testing
            if len(df) > 10:
                recent_trend = df['Close'].iloc[-1] > df['Close'].iloc[-10]
                signals[indicator] = "buy" if recent_trend else "sell"
            else:
                signals[indicator] = "neutral"
        return signals

    def _evaluate_signal_performance(self, df: pd.DataFrame, signals: Dict[str, str]) -> float:
        
        if not signals:
            return 0.5

        # Simple performance metric: consistency with price movement
        buy_signals = sum(1 for s in signals.values() if s == "buy")
        sell_signals = sum(1 for s in signals.values() if s == "sell")

        if len(df) > 5:
            recent_return = (df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5]
            if recent_return > 0 and buy_signals > sell_signals:
                return 0.7
            elif recent_return < 0 and sell_signals > buy_signals:
                return 0.7
            else:
                return 0.3

        return 0.5

    def _calculate_signal_stability(self, base_signals: Dict[str, str], test_signals: Dict[str, str]) -> float:
        
        if not base_signals or not test_signals:
            return 0.5

        matching_signals = 0
        total_signals = 0

        for indicator in base_signals:
            if indicator in test_signals:
                if base_signals[indicator] == test_signals[indicator]:
                    matching_signals += 1
                total_signals += 1

        return matching_signals / total_signals if total_signals > 0 else 0.5
class VPVRProfile:
    

    def __init__(self, bins: int = VPVR_BINS, visible_range: int = VISIBLE_RANGE, volume_percentile: float = VOLUME_PERCENTILE):
        self.bins = bins
        self.visible_range = visible_range
        self.volume_percentile = volume_percentile

    def calculate_vpvr(self, df: pd.DataFrame) -> Dict[str, float]:
        
        if len(df) < self.visible_range:
            return {}

        # Use recent data for VPVR calculation
        recent_df = df.tail(self.visible_range)

        # Create price bins
        price_min = recent_df['Low'].min()
        price_max = recent_df['High'].max()
        price_bins = pd.cut(recent_df['Close'], bins=self.bins, retbins=True)[1]

        # Calculate volume profile
        volume_profile = {}
        for i in range(len(price_bins) - 1):
            bin_start = price_bins[i]
            bin_end = price_bins[i + 1]
            mask = (recent_df['Close'] >= bin_start) & (recent_df['Close'] < bin_end)
            volume_profile[(bin_start + bin_end) / 2] = recent_df.loc[mask, 'Volume'].sum()

        if not volume_profile:
            return {}

        # Find Point of Control (POC) - highest volume level
        poc_price = max(volume_profile, key=volume_profile.get)
        max_volume = volume_profile[poc_price]

        # Find Value Area (VA) - 70% of total volume around POC
        total_volume = sum(volume_profile.values())
        target_volume = total_volume * self.volume_percentile

        # Sort prices by volume
        sorted_prices = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)

        va_volume = 0
        va_prices = []
        for price, volume in sorted_prices:
            va_prices.append(price)
            va_volume += volume
            if va_volume >= target_volume:
                break

        vah = max(va_prices)  # Value Area High
        val = min(va_prices)  # Value Area Low

        return {
            'poc': poc_price,
            'vah': vah,
            'val': val,
            'total_volume': total_volume,
            'va_volume': va_volume
        }

    def get_vpvr_signal(self, df: pd.DataFrame) -> str:
        
        vpvr_levels = self.calculate_vpvr(df)
        if not vpvr_levels:
            return "neutral"

        current_price = df['Close'].iloc[-1]
        vah = vpvr_levels['vah']
        val = vpvr_levels['val']

        # Check proximity to key levels (within 1% of current price)
        vah_proximity = abs(current_price - vah) / current_price
        val_proximity = abs(current_price - val) / current_price

        if vah_proximity <= 0.01:  # Within 1% of VAH - potential sell
            return "sell"
        elif val_proximity <= 0.01:  # Within 1% of VAL - potential buy
            return "buy"
        else:
            return "neutral"

    def merge_with_support_resistance(self, vpvr_levels: Dict[str, float], sr_calculator: 'SupportResistanceCalculator') -> Dict[str, float]:
        
        if not vpvr_levels:
            return sr_calculator.calculate_support_resistance(sr_calculator.sr_df if hasattr(sr_calculator, 'sr_df') else pd.DataFrame())

        merged_levels = vpvr_levels.copy()

        # Add VPVR levels as additional S/R levels
        merged_levels['vpvr_resistance'] = vpvr_levels['vah']
        merged_levels['vpvr_support'] = vpvr_levels['val']

        return merged_levels


class HeikinAshiTransformer:
    

    def __init__(self, confluence_bars: int = HA_CONFLUENCE_BARS, timeframes: List[str] = HA_TFS):
        self.confluence_bars = confluence_bars
        self.timeframes = timeframes

    def transform_to_heikin_ashi(self, df: pd.DataFrame) -> pd.DataFrame:
        
        if len(df) < 2:
            return df.copy()

        ha_df = df.copy()

        # Calculate Heikin-Ashi values
        ha_df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
        ha_df['HA_Open'] = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
        ha_df.loc[0, 'HA_Open'] = df.loc[0, 'Open']  # First HA Open = first regular Open

        ha_df['HA_High'] = pd.concat([df['High'], ha_df[['HA_Open', 'HA_Close']].max(axis=1)], axis=1).max(axis=1)
        ha_df['HA_Low'] = pd.concat([df['Low'], ha_df[['HA_Open', 'HA_Close']].min(axis=1)], axis=1).min(axis=1)

        return ha_df[['HA_Open', 'HA_High', 'HA_Low', 'HA_Close', 'Volume']]

    def get_heikin_ashi_signal(self, df: pd.DataFrame) -> str:
        
        ha_df = self.transform_to_heikin_ashi(df)
        if len(ha_df) < 2:
            return "neutral"

        current_ha = ha_df.iloc[-1]
        prev_ha = ha_df.iloc[-2]

        # Bullish signal: HA Close > HA Open (filled candle)
        if current_ha['HA_Close'] > current_ha['HA_Open']:
            return "buy"
        # Bearish signal: HA Close < HA Open (filled candle)
        elif current_ha['HA_Close'] < current_ha['HA_Open']:
            return "sell"
        else:
            return "neutral"

    def check_confluence(self, df: pd.DataFrame, multi_tf_analyzer: 'MultiTimeframeAnalyzer') -> bool:
        
        confluence_count = 0

        for tf in self.timeframes:
            if tf == 'daily':
                tf_df = df
            else:
                # For demo purposes, use daily data as proxy
                # In real implementation, would need actual multi-timeframe data
                tf_df = df

            ha_signal = self.get_heikin_ashi_signal(tf_df)
            if ha_signal != "neutral":
                confluence_count += 1

        return confluence_count >= self.confluence_bars


class GARCHForecaster:
    

    def __init__(self, p: int = GARCH_P, q: int = GARCH_Q, forecast_horizon: int = FORECAST_HORIZON):
        self.p = p
        self.q = q
        self.forecast_horizon = forecast_horizon
        self.model = None

    def fit_garch_model(self, returns: pd.Series) -> bool:
        
        if len(returns) < 50:  # Minimum data requirement
            logger.warning("Insufficient data for GARCH fitting")
            return False

        try:
            from arch import arch_model
            # Fit GARCH(1,1) model
            self.model = arch_model(returns, vol='Garch', p=self.p, q=self.q, rescale=False)
            self.model_fit = self.model.fit(disp='off')
            return True
        except ImportError:
            logger.warning("arch library not available for GARCH modeling")
            return False
        except Exception as e:
            logger.error(f"Error fitting GARCH model: {e}")
            return False

    def forecast_volatility(self, returns: pd.Series) -> float:
        
        if self.model is None:
            if not self.fit_garch_model(returns):
                return self._fallback_atr_volatility(returns)

        try:
            # Forecast next period volatility
            forecasts = self.model_fit.forecast(horizon=self.forecast_horizon)
            # Return the forecasted volatility for the next period
            next_vol = forecasts.variance.iloc[-1, 0] ** 0.5  # Take square root for volatility
            return next_vol
        except Exception as e:
            logger.error(f"Error forecasting volatility: {e}")
            return self._fallback_atr_volatility(returns)

    def _fallback_atr_volatility(self, returns: pd.Series) -> float:
        
        # Simple volatility estimate using standard deviation of returns
        if len(returns) < 10:
            return 0.02  # Default volatility

        return returns.std() * (252 ** 0.5)  # Annualized volatility

    def should_shorten_periods(self, df: pd.DataFrame) -> bool:
        
        returns = df['Close'].pct_change().dropna()
        if len(returns) < 20:
            return False

        forecast_vol = self.forecast_volatility(returns)
        return forecast_vol > 0.02  # High volatility threshold


    

    def __init__(self, lookback: int = LOOKBACK, tolerance: float = TOLERANCE, min_confidence: float = MIN_CONFIDENCE):
        self.lookback = lookback
        self.tolerance = tolerance
        self.min_confidence = min_confidence
        self.patterns_detected = []

    def detect_peaks_troughs(self, df: pd.DataFrame) -> tuple:
        
        if not SCIPY_AVAILABLE:
            logger.warning("scipy not available for peak detection")
            return [], []

        if len(df) < self.lookback:
            return [], []

        # Use recent data for pattern detection
        recent_df = df.tail(self.lookback)
        prices = recent_df['Close'].values

        # Find peaks (local maxima)
        peaks, _ = find_peaks(prices, prominence=0.01 * prices.mean(), distance=5)
        # Find troughs (local minima) by inverting the signal
        troughs, _ = find_peaks(-prices, prominence=0.01 * prices.mean(), distance=5)

        return peaks, troughs

    def validate_gartley_pattern(self, points: list) -> tuple[str, float]:
        
        if len(points) < 5:
            return "neutral", 0.0

        # Extract price levels
        x, a, b, c, d = points

        # Calculate ratios
        xa = abs(a - x)
        ab = abs(b - a)
        bc = abs(c - b)
        cd = abs(d - c)

        if xa == 0:
            return "neutral", 0.0

        # Gartley bullish ratios
        ab_xa_ratio = ab / xa
        bc_ab_ratio = bc / ab if ab != 0 else 0
        cd_ab_ratio = cd / ab if ab != 0 else 0

        # Check bullish Gartley ratios
        ab_ratio_ok = abs(ab_xa_ratio - 0.618) <= self.tolerance
        bc_ratio_ok = 0.382 <= bc_ab_ratio <= 0.886
        cd_ratio_ok = abs(cd_ab_ratio - 1.272) <= self.tolerance * 2  # More tolerance for CD

        if ab_ratio_ok and bc_ratio_ok and cd_ratio_ok:
            confidence = (1 - abs(ab_xa_ratio - 0.618) / 0.618 +
                         1 - abs(bc_ab_ratio - 0.618) / 0.618 +
                         1 - abs(cd_ab_ratio - 1.272) / 1.272) / 3
            return "buy", min(confidence, 1.0)

        # Gartley bearish ratios (inverted)
        ab_xa_ratio_inv = ab / xa
        bc_ab_ratio_inv = bc / ab if ab != 0 else 0
        cd_ab_ratio_inv = cd / ab if ab != 0 else 0

        ab_ratio_ok_inv = abs(ab_xa_ratio_inv - 0.618) <= self.tolerance
        bc_ratio_ok_inv = 0.382 <= bc_ab_ratio_inv <= 0.886
        cd_ratio_ok_inv = abs(cd_ab_ratio_inv - 1.272) <= self.tolerance * 2

        if ab_ratio_ok_inv and bc_ratio_ok_inv and cd_ratio_ok_inv:
            confidence = (1 - abs(ab_xa_ratio_inv - 0.618) / 0.618 +
                         1 - abs(bc_ab_ratio_inv - 0.618) / 0.618 +
                         1 - abs(cd_ab_ratio_inv - 1.272) / 1.272) / 3
            return "sell", min(confidence, 1.0)

        return "neutral", 0.0

    def validate_butterfly_pattern(self, points: list) -> tuple[str, float]:
        
        if len(points) < 5:
            return "neutral", 0.0

        x, a, b, c, d = points

        xa = abs(a - x)
        ab = abs(b - a)
        bc = abs(c - b)
        cd = abs(d - c)

        if xa == 0:
            return "neutral", 0.0

        # Butterfly bullish ratios
        ab_xa_ratio = ab / xa
        bc_ab_ratio = bc / ab if ab != 0 else 0
        cd_ab_ratio = cd / ab if ab != 0 else 0

        ab_ratio_ok = abs(ab_xa_ratio - 0.786) <= self.tolerance
        bc_ratio_ok = 0.382 <= bc_ab_ratio <= 0.886
        cd_ratio_ok = abs(cd_ab_ratio - 1.618) <= self.tolerance * 2

        if ab_ratio_ok and bc_ratio_ok and cd_ratio_ok:
            confidence = (1 - abs(ab_xa_ratio - 0.786) / 0.786 +
                         1 - abs(bc_ab_ratio - 0.618) / 0.618 +
                         1 - abs(cd_ab_ratio - 1.618) / 1.618) / 3
            return "buy", min(confidence, 1.0)

        # Butterfly bearish ratios
        ab_ratio_ok_inv = abs(ab_xa_ratio - 0.786) <= self.tolerance
        bc_ratio_ok_inv = 0.382 <= bc_ab_ratio <= 0.886
        cd_ratio_ok_inv = abs(cd_ab_ratio - 1.618) <= self.tolerance * 2

class HMMRegimeDetector:
    

    def __init__(self, n_states: int = HMM_STATES, n_iter: int = HMM_ITER):
        self.n_states = n_states
        self.n_iter = n_iter
        self.model = None
        self.current_regime = None
        self.regime_history = []

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        if len(df) < 20:
            logger.info(f"Insufficient data for HMM features: {len(df)} rows")
            return pd.DataFrame()

        # Filter invalid data rows
        mask = (
            (df['Close'] >= 0) &  # Allow zero prices
            (df['Volume'] >= 0) &
            (df['High'] >= df['Low']) &
            (df['High'] >= df['Close']) &
            (df['Low'] <= df['Close'])
        )
        df = df[mask].copy()
        if len(df) < 20:
            logger.warning("Insufficient valid data after filtering")
            return pd.DataFrame()

        features = pd.DataFrame(index=df.index)
        logger.info(f"Starting feature preparation for DataFrame shape: {df.shape}")

        # Returns
        returns = df['Close'].pct_change().replace([np.inf, -np.inf], 0).fillna(0).clip(-0.5, 0.5)
        features['returns'] = returns
        logger.info(f"After returns calculation: shape={len(returns)}, inf={np.isinf(returns).any()}, nan={returns.isna().any()}, min={returns.min()}, max={returns.max()}")

        # Volatility
        atr = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14).fillna(0).replace([np.inf, -np.inf], 0)
        close_prev = df['Close'].shift(1).fillna(df['Close'].mean())
        vol = atr / close_prev.replace(0, 1e-8)  # Ensure denominator >0
        vol = vol.replace([np.inf, -np.inf], 0.01).fillna(0.01).clip(0.001, 1.0)
        features['vol'] = vol
        logger.info(f"After volatility calculation: shape={len(vol)}, inf={np.isinf(vol).any()}, nan={vol.isna().any()}, min={vol.min()}, max={vol.max()}")

        # Volume
        vol_log = np.log1p(df['Volume'].fillna(1).clip(lower=0)).replace([np.inf, -np.inf], 0).clip(0, 15)  # Lower clip for banking stock.
        features['vol_log'] = vol_log
        logger.info(f"After volume_log calculation: shape={len(vol_log)}, inf={np.isinf(vol_log).any()}, nan={vol_log.isna().any()}, min={vol_log.min()}, max={vol_log.max()}")

        # Features
        features = pd.concat([returns, vol, vol_log], axis=1).dropna().replace([np.inf, -np.inf], 0).fillna(0).astype('float64')
        features.columns = ['returns', 'vol', 'vol_log']

        # Final assert
        if not np.isfinite(features).all().all():
            logger.error(f"Non-finite values in features: min={features.min().min()}, max={features.max().max()}, nans={features.isna().sum().sum()}, infs={np.isinf(features).sum().sum()}")
            return pd.DataFrame()  # Trigger fallback

        logger.info(f"Feature preparation complete. Final shape: {features.shape}")
        return features

    def fit_hmm_model(self, features: pd.DataFrame, df: pd.DataFrame):
        
        if not HMMLEARN_AVAILABLE:
            logger.warning("hmmlearn not available for regime detection")
            regime = self._simple_regime(df)
            return regime, 0.5

        if features.shape[0] < 50 or np.any(~np.isfinite(features.values)):
            logger.info("Skipping HMM fit due to bad data")
            # Fallback regime using basic indicators
            if TALIB_AVAILABLE:
                try:
                    adx = talib.ADX(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
                    adx_val = adx[-1] if len(adx) > 0 and not np.isnan(adx[-1]) else 20
                    ema20 = talib.EMA(df['Close'].values, timeperiod=20)
                    ema50 = talib.EMA(df['Close'].values, timeperiod=50)
                    ema20_val = ema20[-1] if len(ema20) > 0 and not np.isnan(ema20[-1]) else df['Close'].iloc[-1]
                    ema50_val = ema50[-1] if len(ema50) > 0 and not np.isnan(ema50[-1]) else df['Close'].iloc[-1]
                except:
                    adx_val, ema20_val, ema50_val = 20, df['Close'].iloc[-1], df['Close'].iloc[-1]
            else:
                adx_val = 20
                ema20_val = df['Close'].ewm(span=20).mean().iloc[-1]
                ema50_val = df['Close'].ewm(span=50).mean().iloc[-1]

            if adx_val > 25:
                regime = 'bull' if ema20_val > ema50_val else 'bear'
            else:
                regime = 'sideways'
            return regime, 0.5

        try:
            from hmmlearn.hmm import GaussianHMM

            # Prepare data for HMM - final clean
            X = features.values.astype('float64')
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X = np.clip(X, -1, 1)

            if np.any(~np.isfinite(X)):
                inf_rows, inf_cols = np.where(np.isinf(X))
                logger.warning(f"Non-finite values in X before fit: inf at positions rows={inf_rows}, cols={inf_cols}")
                # Fallback
                if TALIB_AVAILABLE:
                    try:
                        adx = talib.ADX(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
                        adx_val = adx[-1] if len(adx) > 0 and not np.isnan(adx[-1]) else 20
                        ema20 = talib.EMA(df['Close'].values, timeperiod=20)
                        ema50 = talib.EMA(df['Close'].values, timeperiod=50)
                        ema20_val = ema20[-1] if len(ema20) > 0 and not np.isnan(ema20[-1]) else df['Close'].iloc[-1]
                        ema50_val = ema50[-1] if len(ema50) > 0 and not np.isnan(ema50[-1]) else df['Close'].iloc[-1]
                    except:
                        adx_val, ema20_val, ema50_val = 20, df['Close'].iloc[-1], df['Close'].iloc[-1]
                else:
                    adx_val = 20
                    ema20_val = df['Close'].ewm(span=20).mean().iloc[-1]
                    ema50_val = df['Close'].ewm(span=50).mean().iloc[-1]

                if adx_val > 25:
                    regime = 'bull' if ema20_val > ema50_val else 'bear'
                else:
                    regime = 'sideways'
                return regime, 0.5

            # Fit HMM model
            self.model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=self.n_iter,
                random_state=42
            )

            self.model.fit(X)

            # Decode the most likely sequence of states
            hidden_states = self.model.predict(X)

            # Store regime history
            self.regime_history = hidden_states.tolist()

            # Infer current regime
            current_regime = self.model.predict(features.iloc[-1:].values)[0]
            self.current_regime = current_regime
            regime = self.get_regime_signal(current_regime)

            logger.info(f"HMM model fitted with {self.n_states} states")
            return regime, 1.0

        except Exception as e:
            logger.error(f"Error fitting HMM model: {e}")
            logger.error(f"Feature stats: min={np.min(X) if len(X)>0 else 'N/A'}, max={np.max(X) if len(X)>0 else 'N/A'}, nan_count={np.isnan(X).sum() if len(X)>0 else 0}")
            adx = talib.ADX(df['High'], df['Low'], df['Close'], 14)
            ema20 = talib.EMA(df['Close'], 20)
            ema50 = talib.EMA(df['Close'], 50)
            if adx.iloc[-1] > 25:
                regime = 'bull' if ema20.iloc[-1] > ema50.iloc[-1] else 'bear'
            else:
                regime = 'sideways'
            return regime, 0.5

    def _simple_regime(self, df: pd.DataFrame) -> int:
        
        if len(df) < 50:
            return 1  # sideways

        try:
            if TALIB_AVAILABLE:
                adx = talib.ADX(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
                adx_val = adx[-1] if len(adx) > 0 and not np.isnan(adx[-1]) else 20
            else:
                # Simple trend strength proxy
                adx_val = 20

            ema20 = df['Close'].ewm(span=20).mean()
            if len(ema20) >= 2:
                ema_slope = (ema20.iloc[-1] - ema20.iloc[-2]) / ema20.iloc[-2]
            else:
                ema_slope = 0

            if adx_val > 25:
                return 2 if ema_slope > 0 else 0  # bull or bear
            else:
                return 1  # sideways
        except:
            return 1

    def infer_current_regime(self, features: pd.DataFrame) -> int:
        
        if self.model is None or len(features) == 0:
            return 0  # Default to bear regime

        try:
            # Use latest features to predict current regime
            latest_features = features.iloc[-1:].values
            current_regime = self.model.predict(latest_features)[0]

            self.current_regime = current_regime
            return current_regime

        except Exception as e:
            logger.error(f"Error inferring current regime: {e}")
            return 0

    def get_regime_signal(self, regime: int) -> str:
        
        # Map regimes to market conditions
        # This mapping may need adjustment based on actual regime characteristics
        regime_signals = {
            0: "bear_regime",    # Low returns, high volatility
            1: "sideways_regime", # Moderate returns, moderate volatility
            2: "bull_regime"     # High returns, low volatility
        }

        return regime_signals.get(regime, "neutral_regime")

    def get_transition_probabilities(self) -> dict:
        
        if self.model is None:
            return {}

        try:
            transmat = self.model.transmat_
            return {
                'transitions': transmat.tolist(),
                'regime_stability': transmat.diagonal().tolist()
            }
        except:
            return {}

    def detect_regime_change(self, new_regime: int) -> bool:
        
        if self.current_regime is None:
            return False

        return new_regime != self.current_regime

    def get_hmm_signal(self, df: pd.DataFrame) -> str:
        
        features = self.prepare_features(df)

        if len(features) < 20:
            return "neutral"

        # Fit model if not already fitted
        if self.model is None:
            self.fit_hmm_model(features, df)

        # Infer current regime
        current_regime = self.infer_current_regime(features)
        signal = self.get_regime_signal(current_regime)

        # Log regime changes
        if self.detect_regime_change(current_regime):
            logger.info(f"Regime change detected: {self.current_regime} -> {current_regime} ({signal})")

        return signal

    def adjust_weights_for_regime(self, base_weights: dict, regime_signal: str) -> dict:
        
        weights = base_weights.copy()

        if regime_signal == "bull_regime":
            # Increase trend-following weights in bull markets
            trend_indicators = ['SMA', 'EMA', 'MACD', 'TrendStrength']
            for indicator in trend_indicators:
                if indicator in weights:
                    weights[indicator] *= 1.2

        elif regime_signal == "bear_regime":
            # Increase trend-following weights in bear markets (for short signals)
            trend_indicators = ['SMA', 'EMA', 'MACD', 'TrendStrength']
            for indicator in trend_indicators:
                if indicator in weights:
                    weights[indicator] *= 1.2

        elif regime_signal == "sideways_regime":
            # Increase oscillator weights in sideways markets
            oscillator_indicators = ['RSI', 'Stochastic', 'WilliamsR', 'CCI']
            for indicator in oscillator_indicators:
                if indicator in weights:
                    weights[indicator] *= 1.5

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        return weights


class LSTMPredictor:
    

    def __init__(self, window: int = LSTM_WINDOW, epochs: int = LSTM_EPOCHS, batch_size: int = LSTM_BATCH):
        self.window = window
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = MinMaxScaler()
        self.fallback_model = None
        self.feature_columns = [
            'Close', 'High', 'Low', 'Open', 'Volume',
            'RSI', 'MACD', 'SMA', 'EMA', 'Bollinger',
            'Stochastic', 'WilliamsR', 'CCI', 'TrendStrength',
            'VPVR', 'HeikinAshi', 'Harmonic', 'HMM'
        ]
        self._initialize_model()

    def _initialize_model(self):
        
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available, using RandomForest fallback")
            self.fallback_model = RandomForestClassifier(n_estimators=100, random_state=42)
            return

        try:
            self.model = Sequential()
            self.model.add(Input(shape=(self.window, len(self.feature_columns))))
            self.model.add(LSTM(50, return_sequences=True))
            self.model.add(LSTM(50))
            self.model.add(Dense(1, activation='sigmoid'))
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            logger.info("LSTM model initialized")
        except Exception as e:
            logger.error(f"Error initializing LSTM model: {e}")
            self.fallback_model = RandomForestClassifier(n_estimators=100, random_state=42)

    def prepare_features(self, df: pd.DataFrame, signals: Dict[str, str]) -> pd.DataFrame:
        
        features = pd.DataFrame(index=df.index)

        # Add OHLCV data
        for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
            if col in df.columns:
                features[col] = df[col]
            else:
                features[col] = 0

        # Add technical signals as numeric
        for indicator in self.feature_columns[5:]:  # Skip OHLCV
            if indicator in signals:
                signal = signals[indicator]
                if signal == "buy":
                    features[indicator] = 1
                elif signal == "sell":
                    features[indicator] = -1
                else:
                    features[indicator] = 0
            else:
                features[indicator] = 0

        # Fill any missing values
        features = features.fillna(0)

        return features

    def create_sequences(self, features: pd.DataFrame) -> tuple:
        
        if len(features) < self.window + 1:
            return np.array([]), np.array([])

        # Scale features
        scaled_features = self.scaler.fit_transform(features.values)

        X, y = [], []
        for i in range(self.window, len(scaled_features)):
            X.append(scaled_features[i-self.window:i])
            # Target: 1 if next close > current close, 0 otherwise
            next_close = features.iloc[i]['Close']
            current_close = features.iloc[i-1]['Close']
            y.append(1 if next_close > current_close else 0)

        return np.array(X), np.array(y)

    def train_model(self, df: pd.DataFrame, signals: Dict[str, str], symbol: str = "") -> bool:
        
        effective_samples = len(df) - self.window
        logger.info(f"Effective samples for LSTM training ({symbol}): {effective_samples} (df len: {len(df)}, window: {self.window})")
        if effective_samples < 50:
            logger.warning(f"Insufficient effective samples for LSTM training: {effective_samples} < 50")
            return False

        try:
            features = self.prepare_features(df, signals)
            X, y = self.create_sequences(features)

            if len(X) == 0:
                logger.warning("No sequences created for training")
                return False

            # Use fallback if data is too small
            if len(X) < 50:
                logger.info("Using RandomForest fallback due to small dataset (<50 samples)")
                self.fallback_model = RandomForestClassifier(n_estimators=100, random_state=42)
                X_flat = X.reshape(X.shape[0], -1)
                self.fallback_model.fit(X_flat, y)
                return True

            # Train LSTM
            self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
            logger.info(f"LSTM model trained for {symbol} with {len(X)} samples")
            return True

        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            # Fallback to RandomForest
            try:
                features = self.prepare_features(df, signals)
                X, y = self.create_sequences(features)
                if len(X) > 0:
                    X_flat = X.reshape(X.shape[0], -1)
                    self.fallback_model = RandomForestClassifier(n_estimators=100, random_state=42)
                    self.fallback_model.fit(X_flat, y)
                    return True
            except Exception as e2:
                logger.error(f"Fallback training also failed: {e2}")
            return False

    def predict_signal(self, df: pd.DataFrame, signals: Dict[str, str]) -> tuple[str, float]:
        
        if len(df) < self.window:
            return "neutral", 0.5

        try:
            features = self.prepare_features(df, signals)
            if len(features) < self.window:
                return "neutral", 0.5

            # Get latest window of data
            latest_features = features.tail(self.window)
            scaled_features = self.scaler.transform(latest_features.values)
            X_pred = scaled_features.reshape(1, self.window, len(self.feature_columns))

            if self.model is not None:
                prediction = self.model.predict(X_pred, verbose=0)[0][0]
            elif self.fallback_model is not None:
                X_flat = X_pred.reshape(1, -1)
                prediction = self.fallback_model.predict_proba(X_flat)[0][1]
            else:
                return "neutral", 0.5

            confidence = prediction
            if prediction > 0.6:
                return "buy", confidence
            elif prediction < 0.4:
                return "sell", 1 - confidence
            else:
                return "neutral", 0.5

        except Exception as e:
            logger.error(f"Error predicting with LSTM model: {e}")
            return "neutral", 0.5

    def save_model(self, symbol: str) -> bool:
        
        try:
            if self.model is not None:
                self.model.save(f"models/lstm_{symbol}.h5")
            elif self.fallback_model is not None:
                joblib.dump(self.fallback_model, f"models/rf_{symbol}.pkl")
            return True
        except Exception as e:
            logger.error(f"Error saving model for {symbol}: {e}")
            return False

    def load_model(self, symbol: str) -> bool:
        
        try:
            if TENSORFLOW_AVAILABLE:
                from tensorflow.keras.models import load_model
                self.model = load_model(f"models/lstm_{symbol}.h5")
                return True
            elif os.path.exists(f"models/rf_{symbol}.pkl"):
                self.fallback_model = joblib.load(f"models/rf_{symbol}.pkl")
                return True
        except Exception as e:
            logger.error(f"Error loading model for {symbol}: {e}")
        return False


class EnhancedVaRCalculator(BacktestValidator):
    

    def __init__(self):
        super().__init__()
        self.mc_paths = MC_VA_PATHS
        self.confidence_levels = VA_CONFIDENCE
        self.stress_scenarios = STRESS_SCENARIOS

    def compute_monte_carlo_var(self, df: pd.DataFrame, portfolio_value: float = 100000) -> Dict[str, float]:
        
        if len(df) < 30:
            return {'var_95': 0.0, 'var_99': 0.0, 'es_95': 0.0, 'es_99': 0.0}

        try:
            returns = df['Close'].pct_change().dropna()
            if len(returns) < 10:
                return {'var_95': 0.0, 'var_99': 0.0, 'es_95': 0.0, 'es_99': 0.0}

            # Calculate correlation matrix (for multi-asset, but using single asset for now)
            corr_matrix = np.array([[1.0]])  # Single asset correlation

            # Fit distribution to returns
            mu, sigma = np.mean(returns), np.std(returns)

            # Generate correlated random returns
            simulated_returns = np.random.normal(mu, sigma, (self.mc_paths, 1))

            # Calculate portfolio losses
            portfolio_losses = portfolio_value * simulated_returns.flatten()

            # Calculate VaR and ES
            results = {}
            for conf in self.confidence_levels:
                var_percentile = (1 - conf) * 100
                var_value = np.percentile(portfolio_losses, var_percentile)
                results[f'var_{int(conf*100)}'] = -var_value  # Positive VaR value

                # Expected Shortfall (ES)
                tail_losses = portfolio_losses[portfolio_losses <= var_value]
                es_value = np.mean(tail_losses) if len(tail_losses) > 0 else var_value
                results[f'es_{int(conf*100)}'] = -es_value

            return results

        except Exception as e:
            logger.error(f"Error computing Monte Carlo VaR: {e}")
            return {'var_95': 0.0, 'var_99': 0.0, 'es_95': 0.0, 'es_99': 0.0}

    def apply_stress_scenarios(self, df: pd.DataFrame, base_var: Dict[str, float]) -> Dict[str, float]:
        
        stressed_results = {}

        for scenario in self.stress_scenarios:
            # Increase volatility by stress factor
            stressed_df = df.copy()
            stressed_df['Close'] = df['Close'] * (1 + (scenario - 1) * np.random.normal(0, 0.1, len(df)))

            # Recalculate VaR under stress
            stress_var = self.compute_monte_carlo_var(stressed_df)

            for key, value in stress_var.items():
                stressed_results[f'{key}_stress_{scenario}'] = value

        return stressed_results

    def calculate_risk_adjustment_factor(self, var_95: float, portfolio_value: float = 100000) -> float:
        
        if portfolio_value == 0:
            return 0.0

        var_ratio = var_95 / portfolio_value

        # High risk threshold: 5% of portfolio
        if var_ratio > 0.05:
            return -0.2  # Downgrade ensemble score
        elif var_ratio > 0.02:
            return -0.1  # Moderate downgrade
        else:
            return 0.0   # No adjustment

    def get_comprehensive_risk_metrics(self, df: pd.DataFrame, portfolio_value: float = 100000) -> Dict[str, Any]:
        
        base_var = self.compute_monte_carlo_var(df, portfolio_value)
        stress_var = self.apply_stress_scenarios(df, base_var)

        # Combine results
        risk_metrics = {**base_var, **stress_var}

        # Add risk adjustment factor
        risk_adjustment = self.calculate_risk_adjustment_factor(base_var.get('var_95', 0), portfolio_value)
        risk_metrics['risk_adjustment_factor'] = risk_adjustment

        # Determine risk level
        var_95_ratio = base_var.get('var_95', 0) / portfolio_value
        if var_95_ratio > 0.05:
            risk_metrics['risk_level'] = 'high'
        elif var_95_ratio > 0.02:
            risk_metrics['risk_level'] = 'medium'
        else:
            risk_metrics['risk_level'] = 'low'

        return risk_metrics

    def export_to_state(self, risk_metrics: Dict[str, Any], state: State) -> State:
        
        if 'risk_assessment' not in state:
            state['risk_assessment'] = {}

        state['risk_assessment']['var_metrics'] = risk_metrics
        logger.info(f"Exported VaR metrics to state: {risk_metrics.get('risk_level', 'unknown')} risk level")

        return state


class ParameterOptimizer:
    

    def __init__(self):
        self.grid_params = GRID_SEARCH_PARAMS
        self.best_params = {}
        self.optimization_results = {}

    def optimize_rsi_parameters(self, df: pd.DataFrame) -> Dict[str, Any]:
        
        if len(df) < 50:
            return {'period': 14, 'score': 0.5}

        best_score = 0
        best_period = 14

        for period in self.grid_params['rsi_period']:
            if len(df) < period * 2:
                continue

            try:
                # Calculate RSI with current period
                rsi = self._calculate_rsi(df, period)
                if rsi is None or rsi.empty:
                    continue

                # Evaluate performance
                score = self._evaluate_rsi_performance(df, rsi, period)
                if score > best_score:
                    best_score = score
                    best_period = period
            except Exception as e:
                logger.warning(f"Error optimizing RSI period {period}: {e}")
                continue

        return {'period': best_period, 'score': best_score}

    def optimize_macd_parameters(self, df: pd.DataFrame) -> Dict[str, Any]:
        
        if len(df) < 100:
            return {'fast': 12, 'slow': 26, 'signal': 9, 'score': 0.5}

        best_score = 0
        best_params = {'fast': 12, 'slow': 26, 'signal': 9}

        for fast in self.grid_params['macd_fast']:
            for slow in self.grid_params['macd_slow']:
                if fast >= slow or len(df) < slow * 2:
                    continue

                try:
                    # Calculate MACD with current parameters
                    macd, signal = self._calculate_macd(df, fast, slow, 9)
                    if macd is None or signal is None:
                        continue

                    # Evaluate performance
                    score = self._evaluate_macd_performance(df, macd, signal)
                    if score > best_score:
                        best_score = score
                        best_params = {'fast': fast, 'slow': slow, 'signal': 9}
                except Exception as e:
                    logger.warning(f"Error optimizing MACD params {fast},{slow}: {e}")
                    continue

        best_params['score'] = best_score
        return best_params

    def optimize_stochastic_parameters(self, df: pd.DataFrame) -> Dict[str, Any]:
        
        if len(df) < 50:
            return {'k_period': 14, 'd_period': 3, 'score': 0.5}

        best_score = 0
        best_params = {'k_period': 14, 'd_period': 3}

        for k_period in self.grid_params['stoch_k']:
            for d_period in self.grid_params['stoch_d']:
                if len(df) < k_period * 2:
                    continue

                try:
                    # Calculate Stochastic with current parameters
                    k, d = self._calculate_stochastic(df, k_period, d_period)
                    if k is None or d is None:
                        continue

                    # Evaluate performance
                    score = self._evaluate_stochastic_performance(df, k, d)
                    if score > best_score:
                        best_score = score
                        best_params = {'k_period': k_period, 'd_period': d_period}
                except Exception as e:
                    logger.warning(f"Error optimizing Stochastic params {k_period},{d_period}: {e}")
                    continue

        best_params['score'] = best_score
        return best_params

    def optimize_all_parameters(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        
        results = {}

        logger.info("Starting parameter optimization...")

        results['rsi'] = self.optimize_rsi_parameters(df)
        logger.info(f"RSI optimization complete: {results['rsi']}")

        results['macd'] = self.optimize_macd_parameters(df)
        logger.info(f"MACD optimization complete: {results['macd']}")

        results['stochastic'] = self.optimize_stochastic_parameters(df)
        logger.info(f"Stochastic optimization complete: {results['stochastic']}")

        self.optimization_results = results
        return results

    def _calculate_rsi(self, df: pd.DataFrame, period: int) -> pd.Series:
        
        if len(df) < period:
            return None

        price_changes = df['Close'].diff()
        gains = price_changes.where(price_changes > 0, 0)
        losses = -price_changes.where(price_changes < 0, 0)

        avg_gain = gains.rolling(period).mean()
        avg_loss = losses.rolling(period).mean()

        rs = avg_gain / avg_loss.replace(0, 1e-9)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(self, df: pd.DataFrame, fast: int, slow: int, signal: int) -> tuple:
        
        if len(df) < slow:
            return None, None

        ema_fast = df['Close'].ewm(span=fast).mean()
        ema_slow = df['Close'].ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()

        return macd, signal_line

    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int, d_period: int) -> tuple:
        
        if len(df) < k_period:
            return None, None

        high_max = df['High'].rolling(k_period).max()
        low_min = df['Low'].rolling(k_period).min()

        k_percent = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(d_period).mean()

        return k_percent, d_percent

    def _evaluate_rsi_performance(self, df: pd.DataFrame, rsi: pd.Series, period: int) -> float:
        
        if rsi.empty or len(rsi) < 10:
            return 0.5

        # Simple evaluation: how well RSI predicts reversals
        score = 0
        valid_signals = 0

        for i in range(period, len(rsi) - 1):
            if pd.isna(rsi.iloc[i]):
                continue

            current_rsi = rsi.iloc[i]
            future_return = (df['Close'].iloc[i+1] - df['Close'].iloc[i]) / df['Close'].iloc[i]

            # Check if oversold/overbought signals are followed by reversals
            if current_rsi <= RSI_OVERSOLD and future_return > 0:
                score += 1
            elif current_rsi >= RSI_OVERBOUGHT and future_return < 0:
                score += 1

            valid_signals += 1

        return score / valid_signals if valid_signals > 0 else 0.5

    def _evaluate_macd_performance(self, df: pd.DataFrame, macd: pd.Series, signal: pd.Series) -> float:
        
        if macd.empty or signal.empty or len(macd) < 10:
            return 0.5

        score = 0
        valid_signals = 0

        for i in range(1, min(len(macd), len(signal))):
            if pd.isna(macd.iloc[i]) or pd.isna(signal.iloc[i]):
                continue

            current_macd = macd.iloc[i]
            current_signal = signal.iloc[i]
            future_return = (df['Close'].iloc[i+1] - df['Close'].iloc[i]) / df['Close'].iloc[i] if i+1 < len(df) else 0

            # Check if MACD cross signals predict price movement
            if current_macd > current_signal and future_return > 0:
                score += 1
            elif current_macd < current_signal and future_return < 0:
                score += 1

            valid_signals += 1

        return score / valid_signals if valid_signals > 0 else 0.5

    def _evaluate_stochastic_performance(self, df: pd.DataFrame, k: pd.Series, d: pd.Series) -> float:
        
        if k.empty or d.empty or len(k) < 10:
            return 0.5

        score = 0
        valid_signals = 0

        for i in range(1, min(len(k), len(d))):
            if pd.isna(k.iloc[i]) or pd.isna(d.iloc[i]):
                continue

            current_k = k.iloc[i]
            current_d = d.iloc[i]
            future_return = (df['Close'].iloc[i+1] - df['Close'].iloc[i]) / df['Close'].iloc[i] if i+1 < len(df) else 0

            # Check if Stochastic signals predict price movement
            if current_k < 20 and current_d < 20 and future_return > 0:
                score += 1
            elif current_k > 80 and current_d > 80 and future_return < 0:
                score += 1

            valid_signals += 1

        return score / valid_signals if valid_signals > 0 else 0.5


class DataValidator:
    

    def __init__(self, max_period: int = 50):
        self.max_period = max_period
        self.logger = logging.getLogger(__name__)

    def validate_dataframe(self, df: pd.DataFrame, symbol: str = "") -> tuple[bool, pd.DataFrame, dict]:
        
        validation_info = {
            'original_length': len(df),
            'warnings': [],
            'errors': [],
            'data_quality_score': 1.0
        }

        if df is None or df.empty:
            validation_info['errors'].append("DataFrame is None or empty")
            validation_info['data_quality_score'] = 0.0
            return False, df, validation_info

        # Ensure consistent column names
        df = self._standardize_columns(df)

        # Check minimum length requirement
        if len(df) < self.max_period:
            validation_info['warnings'].append(
                f"Insufficient data: {len(df)} rows < {self.max_period} required"
            )
            validation_info['data_quality_score'] *= 0.7

        # Validate OHLCV consistency
        consistency_issues = self._validate_ohlcv_consistency(df)
        if consistency_issues:
            validation_info['errors'].extend(consistency_issues)
            validation_info['data_quality_score'] *= 0.5

        # Handle missing data
        df_cleaned, missing_info = self._handle_missing_data(df)
        validation_info.update(missing_info)

        # Detect outliers
        outlier_info = self._detect_outliers(df_cleaned)
        validation_info.update(outlier_info)

        # Calculate final validity
        is_valid = len(validation_info['errors']) == 0

        if symbol:
            self.logger.info(f"Data validation for {symbol}: valid={is_valid}, score={validation_info['data_quality_score']:.2f}")

        return is_valid, df_cleaned, validation_info

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        
        column_mapping = {
            'close': 'Close',
            'high': 'High',
            'low': 'Low',
            'open': 'Open',
            'volume': 'Volume'
        }

        df_copy = df.copy()
        for old_col, new_col in column_mapping.items():
            if old_col in df_copy.columns and new_col not in df_copy.columns:
                df_copy[new_col] = df_copy[old_col]
                df_copy.drop(columns=[old_col], inplace=True)

        return df_copy

    def _validate_ohlcv_consistency(self, df: pd.DataFrame) -> list[str]:
        
        issues = []

        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
            return issues

        # Check OHLC relationships
        invalid_high_low = (df['High'] < df['Low']).sum()
        if invalid_high_low > 0:
            issues.append(f"{invalid_high_low} rows have High < Low")

        invalid_close_range = ((df['Close'] > df['High']) | (df['Close'] < df['Low'])).sum()
        if invalid_close_range > 0:
            issues.append(f"{invalid_close_range} rows have Close outside High-Low range")

        # Check for negative values (only for prices that shouldn't be negative)
        negative_prices = (df[['High', 'Low', 'Close']] < 0).any(axis=1).sum()
        if negative_prices > 0:
            issues.append(f"{negative_prices} rows have negative prices")
        # Open price can be zero in some cases (e.g., suspended stocks)
        negative_open = (df['Open'] < 0).sum()
        if negative_open > 0:
            issues.append(f"{negative_open} rows have negative open prices")

        negative_volume = (df['Volume'] < 0).sum()
        if negative_volume > 0:
            issues.append(f"{negative_volume} rows have negative volume")

        return issues

    def _handle_missing_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        
        df_copy = df.copy()
        missing_info = {'missing_data_handled': False, 'interpolation_applied': False}

        # Check for missing values
        missing_mask = df_copy.isnull().any(axis=1)
        missing_count = missing_mask.sum()
        total_rows = len(df_copy)

        if missing_count > 0:
            missing_percentage = (missing_count / total_rows) * 100

            if missing_percentage < 5:
                # Interpolate missing values
                numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in numeric_cols:
                    if col in df_copy.columns:
                        df_copy[col] = df_copy[col].interpolate(method='linear')

                missing_info['missing_data_handled'] = True
                missing_info['interpolation_applied'] = True
                self.logger.info(f"Interpolated {missing_count} missing values ({missing_percentage:.1f}%)")
            else:
                missing_info['warnings'] = [f"High missing data: {missing_percentage:.1f}% gaps - consider data quality"]

        return df_copy, missing_info

    def _detect_outliers(self, df: pd.DataFrame) -> dict:
        
        outlier_info = {'outliers_detected': 0, 'outlier_columns': []}

        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

        for col in numeric_cols:
            if col in df.columns:
                # Calculate z-scores
                mean_val = df[col].mean()
                std_val = df[col].std()

                if std_val > 0:
                    z_scores = abs((df[col] - mean_val) / std_val)
                    outliers = (z_scores > 3).sum()

                    if outliers > 0:
                        outlier_info['outliers_detected'] += outliers
                        outlier_info['outlier_columns'].append(col)
                        self.logger.warning(f"Detected {outliers} outliers in {col} (z-score > 3)")

        return outlier_info


class PatternRecognition:
    
    
    def __init__(self, min_lookback_candles: int = 50, min_lookback_charts: int = 100):
        self.min_lookback_candles = min_lookback_candles
        self.min_lookback_charts = min_lookback_charts
        self.reliability_weights = {
            'engulfing': 0.8, 'hammer': 0.8, 'shooting_star': 0.8,
            'doji': 0.6, 'morning_star': 0.7, 'evening_star': 0.7,
            'three_soldiers': 0.75, 'three_crows': 0.75,
            'double_top': 0.6, 'double_bottom': 0.6,
            'head_shoulders': 0.5, 'inverse_head_shoulders': 0.5,
            'triangle': 0.5,
            'divergence': 0.7
        }
        self.recency_window = 10  # Boost if within last N bars
    
    def calculate_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        
        indicators = {}
        
        close = df['Close'].values if len(df) > 0 else np.array([])
        high = df['High'].values if 'High' in df.columns else close
        low = df['Low'].values if 'Low' in df.columns else close
        
        if len(close) < 20:
            return indicators
        
        try:
            if TALIB_AVAILABLE:
                # RSI
                indicators['RSI'] = pd.Series(talib.RSI(close, timeperiod=14), index=df.index)
                
                # MACD
                macd, _, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
                indicators['MACD'] = pd.Series(macd, index=df.index)
                
                # CCI
                indicators['CCI'] = pd.Series(talib.CCI(high, low, close, timeperiod=20), index=df.index)
            else:
                # Basic RSI
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                indicators['RSI'] = 100 - (100 / (1 + rs))
                
                # Basic MACD
                ema12 = df['Close'].ewm(span=12).mean()
                ema26 = df['Close'].ewm(span=26).mean()
                indicators['MACD'] = ema12 - ema26
                
                # Basic CCI
                tp = (df['High'] + df['Low'] + df['Close']) / 3
                sma_tp = tp.rolling(20).mean()
                mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
                indicators['CCI'] = (tp - sma_tp) / (0.015 * mad)
                
        except Exception as e:
            logger.warning(f"Error calculating indicators for patterns: {e}")
        
        return {k: v.dropna() for k, v in indicators.items() if len(v) > 0}
    
    def find_peaks_troughs(self, series: pd.Series, window: int = 5) -> tuple[List[int], List[int]]:
        
        if len(series) < 2 * window:
            return [], []
        
        peaks, troughs = [], []
        values = series.values
        
        if SCIPY_AVAILABLE:
            try:
                from scipy.signal import find_peaks
                peaks_idx, _ = find_peaks(values, distance=window)
                troughs_idx, _ = find_peaks(-values, distance=window)
            except:
                peaks_idx, troughs_idx = [], []
        else:
            # Simple rolling method
            for i in range(window, len(values) - window):
                if all(values[i] >= values[i - window:i + window + 1]):
                    peaks.append(i)
                if all(values[i] <= values[i - window:i + window + 1]):
                    troughs.append(i)
            peaks_idx, troughs_idx = peaks, troughs
        
        return peaks_idx.tolist(), troughs_idx.tolist()
    
    def detect_divergences(self, df: pd.DataFrame, indicators: Dict[str, pd.Series]) -> Dict[str, float]:
        
        divergences = {}
        price_peaks, price_troughs = self.find_peaks_troughs(df['Close'], window=5)
        recent_len = min(50, len(df))  # Look at recent 50 bars
        
        for osc in ['RSI', 'MACD', 'CCI']:
            if osc not in indicators or len(indicators[osc]) < 20:
                continue
            
            osc_series = indicators[osc].tail(recent_len)
            osc_peaks, osc_troughs = self.find_peaks_troughs(osc_series, window=5)
            
            # Align indices to original df
            osc_peaks = [len(df) - recent_len + idx for idx in osc_peaks]
            osc_troughs = [len(df) - recent_len + idx for idx in osc_troughs]
            
            price_recent_peaks = [p for p in price_peaks if len(df) - recent_len <= p < len(df)]
            price_recent_troughs = [t for t in price_troughs if len(df) - recent_len <= t < len(df)]
            
            # Find last two troughs/peaks for divergence check
            if len(price_recent_troughs) >= 2:
                t1, t2 = price_recent_troughs[-2], price_recent_troughs[-1]  # Recent troughs
                if t1 in osc_troughs and t2 in osc_troughs:
                    price_slope = (df['Close'].iloc[t2] - df['Close'].iloc[t1]) / (t2 - t1)
                    osc_slope = (osc_series.iloc[t2 - (len(df) - recent_len)] - osc_series.iloc[t1 - (len(df) - recent_len)]) / (t2 - t1)
                    
                    # Regular bullish: price lower low, osc higher low
                    if price_slope < 0 and osc_slope > 0:
                        magnitude = abs(price_slope - osc_slope)
                        strength = min(magnitude * 10, 1.0)  # Normalize
                        divergences[f'{osc}_regular_bullish'] = strength
                    # Hidden bullish: price higher low, osc lower low
                    elif price_slope > 0 and osc_slope < 0:
                        magnitude = abs(price_slope - osc_slope)
                        strength = min(magnitude * 10, 1.0)
                        divergences[f'{osc}_hidden_bullish'] = strength
            
            if len(price_recent_peaks) >= 2:
                p1, p2 = price_recent_peaks[-2], price_recent_peaks[-1]
                if p1 in osc_peaks and p2 in osc_peaks:
                    price_slope = (df['Close'].iloc[p2] - df['Close'].iloc[p1]) / (p2 - p1)
                    osc_slope = (osc_series.iloc[p2 - (len(df) - recent_len)] - osc_series.iloc[p1 - (len(df) - recent_len)]) / (p2 - p1)
                    
                    # Regular bearish: price higher high, osc lower high
                    if price_slope > 0 and osc_slope < 0:
                        magnitude = abs(price_slope - osc_slope)
                        strength = min(magnitude * 10, 1.0)
                        divergences[f'{osc}_regular_bearish'] = strength
                    # Hidden bearish: price lower high, osc higher high
                    elif price_slope < 0 and osc_slope > 0:
                        magnitude = abs(price_slope - osc_slope)
                        strength = min(magnitude * 10, 1.0)
                        divergences[f'{osc}_hidden_bearish'] = strength
        
        return divergences
    
    def detect_candlestick_patterns(self, df: pd.DataFrame) -> Dict[str, float]:
        
        patterns = {}
        if len(df) < 3:
            return patterns
        
        # Get last 3 candles
        ohlc = df[['Open', 'High', 'Low', 'Close']].tail(3)
        body_size = abs(ohlc['Close'] - ohlc['Open'])
        upper_shadow = ohlc['High'] - np.maximum(ohlc['Open'], ohlc['Close'])
        lower_shadow = np.minimum(ohlc['Open'], ohlc['Close']) - ohlc['Low']
        total_range = ohlc['High'] - ohlc['Low']
        
        current = ohlc.iloc[-1]
        prev = ohlc.iloc[-2]
        prev_prev = ohlc.iloc[-3] if len(ohlc) > 2 else None
        
        tolerance = 0.1  # 10% tolerance for sizes
        
        # Doji variants
        body_ratio = body_size.iloc[-1] / total_range.iloc[-1]
        if body_ratio < 0.1:  # Neutral Doji
            patterns['doji'] = 0.6
            if lower_shadow.iloc[-1] > 2 * body_size.iloc[-1] and upper_shadow.iloc[-1] < body_size.iloc[-1]:
                patterns['dragonfly_doji'] = 0.7  # Bullish
            elif upper_shadow.iloc[-1] > 2 * body_size.iloc[-1] and lower_shadow.iloc[-1] < body_size.iloc[-1]:
                patterns['gravestone_doji'] = 0.7  # Bearish
        
        # Hammer / Shooting Star
        body_current = body_size.iloc[-1]
        if lower_shadow.iloc[-1] > 2 * body_current and upper_shadow.iloc[-1] < 0.5 * body_current and body_current < 0.3 * total_range.iloc[-1]:
            patterns['hammer'] = 0.8 if current['Close'] > current['Open'] else 0.7  # Stronger if green
        if upper_shadow.iloc[-1] > 2 * body_current and lower_shadow.iloc[-1] < 0.5 * body_current and body_current < 0.3 * total_range.iloc[-1]:
            patterns['shooting_star'] = 0.8 if current['Close'] < current['Open'] else 0.7  # Stronger if red
        
        # Engulfing
        prev_body = body_size.iloc[-2]
        if (current['Close'] > current['Open'] and prev['Close'] < prev['Open'] and  # Bullish engulfing
            current['Open'] < prev['Close'] and current['Close'] > prev['Open'] and
            body_size.iloc[-1] > prev_body * 1.1):
            patterns['bullish_engulfing'] = 0.8
        elif (current['Close'] < current['Open'] and prev['Close'] > prev['Open'] and  # Bearish
              current['Open'] > prev['Close'] and current['Close'] < prev['Open'] and
              body_size.iloc[-1] > prev_body * 1.1):
            patterns['bearish_engulfing'] = 0.8
        
        # Morning / Evening Star
        if prev_prev is not None:
            if (prev_prev['Close'] < prev_prev['Open'] and  # Bearish first
                body_ratio.iloc[-2] < 0.3 and  # Small body second (star)
                current['Close'] > current['Open'] and current['Close'] > (prev_prev['Open'] + prev_prev['Close']) / 2):  # Bullish third
                patterns['morning_star'] = 0.7
            elif (prev_prev['Close'] > prev_prev['Open'] and
                  body_ratio.iloc[-2] < 0.3 and
                  current['Close'] < current['Open'] and current['Close'] < (prev_prev['Open'] + prev_prev['Close']) / 2):
                patterns['evening_star'] = 0.7
        
        # Three White Soldiers / Three Black Crows
        if len(ohlc) >= 3:
            all_bullish = all(ohlc['Close'] > ohlc['Open']) and ohlc['Close'].is_monotonic_increasing
            all_bearish = all(ohlc['Close'] < ohlc['Open']) and ohlc['Close'].is_monotonic_decreasing
            if all_bullish and body_size.mean() > total_range.mean() * 0.6:
                patterns['three_white_soldiers'] = 0.75
            if all_bearish and body_size.mean() > total_range.mean() * 0.6:
                patterns['three_black_crows'] = 0.75
        
        return {k: v for k, v in patterns.items() if v > 0}
    
    def detect_chart_patterns(self, df: pd.DataFrame) -> Dict[str, float]:
        
        patterns = {}
        if len(df) < self.min_lookback_charts:
            return patterns
        
        price_peaks, price_troughs = self.find_peaks_troughs(df['Close'], window=10)
        if len(price_peaks) < 3 or len(price_troughs) < 2:
            return patterns
        
        # Recent peaks/troughs
        recent_peaks = price_peaks[-5:] if len(price_peaks) >= 5 else price_peaks
        recent_troughs = price_troughs[-5:] if len(price_troughs) >= 5 else price_troughs
        
        # Double Top: two similar highs with trough in between
        if len(recent_peaks) >= 2:
            p1, p2 = recent_peaks[-2], recent_peaks[-1]
            peak1, peak2 = df['Close'].iloc[p1], df['Close'].iloc[p2]
            if abs(peak1 - peak2) / peak1 < 0.02:  # Within 2%
                trough_between = [t for t in recent_troughs if p1 < t < p2]
                if trough_between:
                    confidence = 1 - (abs(peak1 - peak2) / peak1)  # Closer = higher conf
                    patterns['double_top'] = confidence * 0.6
                    if df['Close'].iloc[-1] < min(peak1, peak2):  # Breakdown
                        patterns['double_top_breakdown'] = confidence
        
        # Double Bottom: similar
        if len(recent_troughs) >= 2:
            t1, t2 = recent_troughs[-2], recent_troughs[-1]
            trough1, trough2 = df['Close'].iloc[t1], df['Close'].iloc[t2]
            if abs(trough1 - trough2) / trough1 < 0.02:
                peak_between = [p for p in recent_peaks if t1 < p < t2]
                if peak_between:
                    confidence = 1 - (abs(trough1 - trough2) / trough1)
                    patterns['double_bottom'] = confidence * 0.6
                    if df['Close'].iloc[-1] > max(trough1, trough2):  # Breakout
                        patterns['double_bottom_breakout'] = confidence
        
        # Head & Shoulders: three peaks, middle highest, shoulders similar
        if len(recent_peaks) >= 3:
            s1, h, s2 = recent_peaks[-3], recent_peaks[-2], recent_peaks[-1]
            ps1, ph, ps2 = df['Close'].iloc[s1], df['Close'].iloc[h], df['Close'].iloc[s2]
            if ph > ps1 and ph > ps2 and abs(ps1 - ps2) / ps1 < 0.05:  # Shoulders similar
                troughs_between = [t for t in recent_troughs if s1 < t < s2]
                if len(troughs_between) >= 2:
                    confidence = 0.8 if abs(ps1 - ps2) / ps1 < 0.02 else 0.5
                    patterns['head_shoulders'] = confidence * 0.5
                    neckline = (df['Close'].iloc[troughs_between[0]] + df['Close'].iloc[troughs_between[-1]]) / 2
                    if df['Close'].iloc[-1] < neckline:
                        patterns['head_shoulders_breakdown'] = confidence
        
        # Basic Triangle: check if highs decreasing, lows increasing over last 20-50 bars
        recent_df = df.tail(50)
        highs = recent_df['High'].rolling(5).max()
        lows = recent_df['Low'].rolling(5).min()
        high_slope = (highs.iloc[-1] - highs.iloc[0]) / len(highs)
        low_slope = (lows.iloc[-1] - lows.iloc[0]) / len(lows)
        
        if high_slope < 0 and low_slope > 0 and abs(high_slope) > 0.001 and abs(low_slope) > 0.001:
            # Converging
            convergence = abs(highs.iloc[-1] - lows.iloc[-1]) / (highs.iloc[0] - lows.iloc[0])
            confidence = 0.6 if convergence < 0.5 else 0.4
            patterns['symmetrical_triangle'] = confidence
            if low_slope > abs(high_slope):
                patterns['ascending_triangle'] = confidence * 1.1  # Bullish bias
            elif high_slope < -abs(low_slope):
                patterns['descending_triangle'] = confidence * 1.1  # Bearish bias
        
        return {k: v for k, v in patterns.items() if v > 0}
    
    def _apply_recency_boost(self, pattern: str, completion_bar: int, current_bar: int) -> float:
        
        distance = current_bar - completion_bar
        if distance <= self.recency_window:
            return 1.2
        else:
            return max(0.5, 1.0 - (distance - self.recency_window) / 20)  # Decay over 20 bars
    
    def analyze_patterns(self, df: pd.DataFrame) -> tuple[str, float]:
        
        if len(df) < self.min_lookback_candles:
            return "neutral", 0.0
        
        indicators = self.calculate_indicators(df)
        divergences = self.detect_divergences(df, indicators)
        candles = self.detect_candlestick_patterns(df)
        charts = self.detect_chart_patterns(df)
        
        all_patterns = {**divergences, **candles, **charts}
        if not all_patterns:
            return "neutral", 0.0
        
        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0
        current_bar = len(df) - 1
        
        for pattern, conf in all_patterns.items():
            weight = self.reliability_weights.get(pattern.split('_')[0], 0.5)
            recency = self._apply_recency_boost(pattern, current_bar, current_bar)  # Assume recent
            adjusted_conf = conf * weight * recency
            total_weight += weight
            
            if 'bullish' in pattern or 'bottom' in pattern or 'hammer' in pattern or 'morning' in pattern or 'soldiers' in pattern or 'dragonfly' in pattern:
                buy_score += adjusted_conf
            elif 'bearish' in pattern or 'top' in pattern or 'shooting' in pattern or 'evening' in pattern or 'crows' in pattern or 'gravestone' in pattern or 'head_shoulders' in pattern:
                sell_score += adjusted_conf
        
        if total_weight == 0:
            return "neutral", 0.0
        
        net_score = (buy_score - sell_score) / total_weight
        overall_conf = (buy_score + sell_score) / total_weight  # Total pattern strength
        
        if net_score > 0.3:
            return "buy", min(overall_conf, 1.0)
        elif net_score < -0.3:
            return "sell", min(overall_conf, 1.0)
        else:
            return "neutral", overall_conf
class TechnicalAnalysisPipeline:
    """
    A modular pipeline for running a simplified, robust technical analysis workflow.
    """
    def __init__(self):
        self.data_validator = DataValidator()
        self.adaptive_calc = AdaptiveParameterCalculator()
        self.ichimoku = IchimokuCloud()
        self.vpvr = VPVRProfile()
        self.hmm = HMMRegimeDetector()
        self.trend_scorer = TrendStrengthScorer()
        self.ensemble_gen = EnsembleSignalGenerator()

    def run(self, state: State) -> State:
        """
        Executes the full technical analysis pipeline for each stock.
        """
        stock_data = state.get("stock_data", {})
        all_technical_signals = {}

        for symbol, df in stock_data.items():
            try:
                is_valid, df_cleaned, validation_info = self.data_validator.validate_dataframe(df, symbol)
                
                if not is_valid:
                    logger.warning(f"Skipping {symbol} due to data validation failures.")
                    all_technical_signals[symbol] = {
                        "error": "Data validation failed",
                        "validation_info": validation_info
                    }
                    continue
                
                # Core signal generation
                try:
                    # Convert DataFrame to HistoricalData format if needed
                    if isinstance(df_cleaned, pd.DataFrame):
                        historical_data = df_cleaned.reset_index().to_dict('records')
                        historical_data = [record for record in historical_data if isinstance(record, dict)]
                    else:
                        historical_data = df_cleaned
                    validate_data(historical_data)
                except (InsufficientDataError, ConstantPriceError) as e:
                    logger.warning(f"Data validation failed for {symbol}: {e}")
                    all_technical_signals[symbol] = {
                        "error": str(e),
                        "validation_info": validation_info
                    }
                    continue

                signals = self._calculate_core_signals(df_cleaned, symbol)
                
                # Ensembling and final decision
                ensemble_result = self.ensemble_gen.generate_ensemble_signal(signals, df_cleaned)
                
                final_signals = {
                    "ensemble_signal": ensemble_result['signal'],
                    "ensemble_score": ensemble_result['score'],
                    "contributing_signals": ensemble_result['contributing_signals'],
                    "raw_signals": signals,
                    "validation_info": validation_info
                }
                
                all_technical_signals[symbol] = final_signals
                logger.info(f"Successfully generated technical analysis for {symbol}")

            except Exception as e:
                logger.error(f"Unhandled error in technical analysis pipeline for {symbol}: {e}", exc_info=True)
                all_technical_signals[symbol] = {"error": str(e)}

        return {"technical_signals": all_technical_signals}

    def _calculate_core_signals(self, df: pd.DataFrame, symbol: str) -> Dict[str, str]:
        """
        Calculates a focused set of technical indicators.
        """
        signals = {}
        
        # 1. Basic Indicators (RSI, MACD, etc.)
        rsi_period = self.adaptive_calc.adaptive_rsi_period(df)
        base_signals = _calculate_technical_indicators_with_retry(
            df, rsi_period=rsi_period, symbol=symbol, use_talib=TALIB_AVAILABLE
        )
        signals.update(base_signals)
        
        # 2. Ichimoku Cloud
        signals['Ichimoku'] = self.ichimoku.get_ichimoku_signal(df)
        
        # 3. Volume Profile (VPVR)
        signals['VolumeProfile'] = self.vpvr.get_vpvr_signal(df)
        
        # 4. HMM Regime Detection
        features = self.hmm.prepare_features(df)
        if not features.empty:
            regime, conf = self.hmm.fit_hmm_model(features, df)
            signals['Regime'] = regime.replace('_regime', '') if conf > 0.5 else 'neutral'
        else:
            signals['Regime'] = 'neutral'
            
        # 5. Trend Strength
        signals['TrendStrength'] = "strong" if self.trend_scorer.score_trend_strength(df) > 0.6 else "weak"
        
        return signals

def technical_analysis_agent(state: State) -> State:
    """
    AnalysisActor: Perform technical analysis.
    """
    logger.info("AnalysisActor started")
    pipeline = TechnicalAnalysisPipeline()
    result_state = pipeline.run(state)
    symbols = list(state.get("stock_data", {}).keys())
    logger.info(f"AnalysisActor completed for {len(symbols)} symbols: {symbols}")
    return result_state


def _calculate_technical_indicators_with_retry(
    df: pd.DataFrame,
    rsi_period: int = 14,
    symbol: str = "",
    use_talib: bool = True
) -> Dict[str, str]:
    
    # Apply retry decorator dynamically
    if use_talib and TALIB_AVAILABLE:
        retry_func = retry_indicator_calculation(fallback_method=_calculate_technical_indicators_basic)(_calculate_technical_indicators_talib)
        return retry_func(df, rsi_period=rsi_period, symbol=symbol)
    else:
        retry_func = retry_indicator_calculation()(_calculate_technical_indicators_basic)
        return retry_func(df, symbol=symbol)


def _calculate_technical_indicators_talib(df: pd.DataFrame, rsi_period: int = 14, symbol: str = "") -> Dict[str, str]:
    """
    Calculates technical indicators using the TA-Lib library.
    Applies India-specific parameters for .NS symbols.
    """
    from config.config import INDIA_SPECIFIC_PARAMS, RSI_OVERBOUGHT, RSI_OVERSOLD
    
    signals = {}
    if not TALIB_AVAILABLE or df.empty or len(df) < 20:  # Minimum data requirement for most indicators
        return signals

    close = df['Close'].values.astype(float)
    high = df['High'].values.astype(float)
    low = df['Low'].values.astype(float)
    
    # Detect Indian stock for params
    is_indian = symbol.endswith('.NS')
    rsi_oversold = INDIA_SPECIFIC_PARAMS['RSI_OVERSOLD'] if is_indian else RSI_OVERSOLD
    rsi_overbought = INDIA_SPECIFIC_PARAMS['RSI_OVERBOUGHT'] if is_indian else RSI_OVERBOUGHT
    macd_fast = INDIA_SPECIFIC_PARAMS['MACD_FAST'] if is_indian else 12
    macd_slow = INDIA_SPECIFIC_PARAMS['MACD_SLOW'] if is_indian else 26
    macd_signal_period = INDIA_SPECIFIC_PARAMS['MACD_SIGNAL'] if is_indian else 9

    # RSI
    if len(close) < rsi_period:
        signals['RSI'] = "neutral"
    else:
        rsi = talib.RSI(close, timeperiod=rsi_period)
        if len(rsi) > 0 and not np.isnan(rsi[-1]):
            signals['RSI'] = "buy" if rsi[-1] < rsi_oversold else "sell" if rsi[-1] > rsi_overbought else "neutral"
        else:
            signals['RSI'] = "neutral"

    # MACD
    # Minimum data requirement for MACD (slow for slow EMA + signal for signal line)
    if len(close) < macd_slow + macd_signal_period:
        signals['MACD'] = "neutral"
    else:
        macd, macdsignal, _ = talib.MACD(close, fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal_period)
        if len(macd) > 0 and len(macdsignal) > 0 and not np.isnan(macd[-1]) and not np.isnan(macdsignal[-1]):
            signals['MACD'] = "buy" if macd[-1] > macdsignal[-1] else "sell" if macd[-1] < macdsignal[-1] else "neutral"
        else:
            signals['MACD'] = "neutral"

    # Bollinger Bands
    if len(close) < 20:  # Minimum data requirement for Bollinger Bands
        signals['Bollinger'] = "neutral"
    else:
        upper, _, lower = talib.BBANDS(close)
        if len(upper) > 0 and len(lower) > 0 and not np.isnan(upper[-1]) and not np.isnan(lower[-1]) and not np.isnan(close[-1]):
            signals['Bollinger'] = "buy" if close[-1] < lower[-1] else "sell" if close[-1] > upper[-1] else "neutral"
        else:
            signals['Bollinger'] = "neutral"

    # Stochastic Oscillator
    # Minimum data requirement for Stochastic (14 for %K + 3 for %D)
    if len(high) < 14 + 3 or len(low) < 14 + 3 or len(close) < 14 + 3:
        signals['Stochastic'] = "neutral"
    else:
        slowk, slowd = talib.STOCH(high, low, close)
        if len(slowk) > 0 and len(slowd) > 0 and not np.isnan(slowk[-1]) and not np.isnan(slowd[-1]):
            signals['Stochastic'] = "buy" if slowk[-1] < 20 and slowd[-1] < 20 else "sell" if slowk[-1] > 80 and slowd[-1] > 80 else "neutral"
        else:
            signals['Stochastic'] = "neutral"

    return signals


def _calculate_technical_indicators_basic(df: pd.DataFrame, rsi_period: int = 14, symbol: str = "") -> Dict[str, str]:
    """
    A basic, fallback implementation of technical indicators without TA-Lib.
    Applies India-specific parameters for NSE stocks.
    """
    signals = {}
    if df.empty or len(df) < 20:  # Minimum data requirement for most indicators
        signals = {'RSI': 'neutral', 'MACD': 'neutral', 'Bollinger': 'neutral', 'Stochastic': 'neutral'}
        return signals

    # FIXED: Use settings instead of direct config access
    is_indian_stock = symbol.endswith('.NS')
    rsi_oversold = settings.india_specific_params['RSI_OVERSOLD'] if is_indian_stock else settings.rsi_oversold
    rsi_overbought = settings.india_specific_params['RSI_OVERBOUGHT'] if is_indian_stock else settings.rsi_overbought
    macd_fast = settings.india_specific_params['MACD_FAST'] if is_indian_stock else 12
    macd_slow = settings.india_specific_params['MACD_SLOW'] if is_indian_stock else 26
    macd_signal = settings.india_specific_params['MACD_SIGNAL'] if is_indian_stock else 9

    # RSI - Fixed implementation using Wilder's smoothing
    if len(df) < rsi_period + 1:
        signals['RSI'] = "neutral"
    else:
        delta = df['Close'].diff(1)
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        # Initial simple moving averages
        avg_gain = gain.rolling(window=rsi_period, min_periods=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period, min_periods=rsi_period).mean()

        # FIXED: Apply Wilder's smoothing for subsequent values
        for i in range(rsi_period, len(df)):
            if pd.notna(gain.iloc[i]):
                avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (rsi_period - 1) + gain.iloc[i]) / rsi_period
            if pd.notna(loss.iloc[i]):
                avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (rsi_period - 1) + loss.iloc[i]) / rsi_period

        rs = avg_gain / avg_loss.replace(0, 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        # Handle constant price case (zero volatility)
        if avg_gain.iloc[-1] == 0 and avg_loss.iloc[-1] == 0:
            signals['RSI'] = "neutral"
        else:
            signals['RSI'] = "buy" if pd.notna(rsi.iloc[-1]) and rsi.iloc[-1] < rsi_oversold else "sell" if pd.notna(rsi.iloc[-1]) and rsi.iloc[-1] > rsi_overbought else "neutral"

    # MACD - Using TA-Lib exact EMA initialization
    def calculate_ema_talib(series: pd.Series, period: int) -> pd.Series:
        """Calculate EMA using TA-Lib initialization method."""
        n = len(series)
        if n == 0:
            return pd.Series(np.full(n, np.nan), index=series.index)
        
        alpha = 2.0 / (period + 1.0)
        ema = np.full(n, np.nan)
        
        # Find first non-NaN index
        first_valid_idx = series.first_valid_index()
        if first_valid_idx is None or pd.isna(first_valid_idx):
            return pd.Series(ema, index=series.index)
        
        first_valid_pos = series.index.get_loc(first_valid_idx)
        if n - first_valid_pos < period:
            # Not enough data after first valid
            for i in range(first_valid_pos, n):
                if not pd.isna(series.iloc[i]):
                    ema[i] = series.iloc[i]  # Just copy if insufficient
            return pd.Series(ema, index=series.index)
        
        # Find the position for initialization: first_valid_pos + period - 1
        init_pos = first_valid_pos + period - 1
        if init_pos >= n:
            init_pos = n - 1
        
        # FIXED: SMA over the first 'period' valid values starting from first_valid
        valid_start = first_valid_pos
        valid_data = series.iloc[valid_start:init_pos + 1].dropna()
        if len(valid_data) >= period:
            sma_init = valid_data.iloc[0:period].mean()
        else:
            # If less than period valid, use all available
            sma_init = valid_data.mean()
        
        ema[init_pos] = sma_init
        
        # Forward fill the recursive calculation
        for i in range(init_pos + 1, n):
            if pd.isna(series.iloc[i]):
                continue  # Skip NaNs in input
            ema[i] = alpha * series.iloc[i] + (1.0 - alpha) * ema[i - 1]
        
        # Backward fill if needed, but usually not for EMA
        # For positions between first_valid and init_pos, we can approximate by simple average or leave NaN
        # But to match TA-Lib, leave as NaN until init_pos
        
        return pd.Series(ema, index=series.index)

    if len(df) < macd_slow + macd_signal:  # Minimum data requirement for MACD
        signals['MACD'] = "neutral"
    else:
        close = df['Close']
        ema_fast = calculate_ema_talib(close, macd_fast)
        ema_slow = calculate_ema_talib(close, macd_slow)
        macd_line = ema_fast - ema_slow
        signal_line = calculate_ema_talib(macd_line, macd_signal)
        
        # Ensure no NaN in final values
        if pd.isna(macd_line.iloc[-1]) or pd.isna(signal_line.iloc[-1]):
            signals['MACD'] = "neutral"
        else:
            signals['MACD'] = "buy" if macd_line.iloc[-1] > signal_line.iloc[-1] else "sell" if macd_line.iloc[-1] < signal_line.iloc[-1] else "neutral"

    # Bollinger Bands
    if len(df) < 20:  # Minimum data requirement for Bollinger Bands
        signals['Bollinger'] = "neutral"
    else:
        sma20 = df['Close'].rolling(window=20).mean()
        std20 = df['Close'].rolling(window=20).std()
        upper_band = sma20 + (std20 * 2)
        lower_band = sma20 - (std20 * 2)
        signals['Bollinger'] = "buy" if df['Close'].iloc[-1] < lower_band.iloc[-1] else "sell" if df['Close'].iloc[-1] > upper_band.iloc[-1] else "neutral"

    # Stochastic - Fixed implementation to match TA-Lib STOCH (slowk, slowd)
    if len(df) < 14 + 3 + 3:  # Minimum for fastk (14), slowk (3), slowd (3)
        signals['Stochastic'] = "neutral"
    else:
        low14 = df['Low'].rolling(window=14).min()
        high14 = df['High'].rolling(window=14).max()
        
        # Fast %K
        fastk = 100.0 * (df['Close'] - low14) / (high14 - low14).replace(0, 1e-9)
        
        # Slow %K (3-period SMA of fast %K)
        slowk = fastk.rolling(window=3).mean()
        
        # Slow %D (3-period SMA of slow %K)
        slowd = slowk.rolling(window=3).mean()
        
        # Generate signal using slowk and slowd to match TA-Lib
        signals['Stochastic'] = "buy" if pd.notna(slowk.iloc[-1]) and pd.notna(slowd.iloc[-1]) and slowk.iloc[-1] < 20 and slowd.iloc[-1] < 20 else "sell" if pd.notna(slowk.iloc[-1]) and pd.notna(slowd.iloc[-1]) and slowk.iloc[-1] > 80 and slowd.iloc[-1] > 80 else "neutral"
    
    return signals


    


