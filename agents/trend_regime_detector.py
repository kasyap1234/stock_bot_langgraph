"""
Trend Regime Detector

This module implements trend regime detection and classification using multiple
trend analysis techniques including ADX, moving averages, linear regression,
and momentum indicators to identify trending vs ranging market conditions.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import warnings

logger = logging.getLogger(__name__)

# Try to import statistical libraries with fallbacks
try:
    from scipy import stats
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    logger.warning("scipy not available, using basic trend detection")
    SCIPY_AVAILABLE = False

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available, using basic regression")
    SKLEARN_AVAILABLE = False


class TrendRegime(Enum):
    """Trend regime types"""
    STRONG_UPTREND = "strong_uptrend"       # Strong bullish trend
    WEAK_UPTREND = "weak_uptrend"           # Weak bullish trend
    RANGING = "ranging"                     # Sideways/consolidating market
    WEAK_DOWNTREND = "weak_downtrend"       # Weak bearish trend
    STRONG_DOWNTREND = "strong_downtrend"   # Strong bearish trend


class TrendDirection(Enum):
    """Trend direction types"""
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"


@dataclass
class TrendMetrics:
    """Comprehensive trend analysis metrics"""
    direction: TrendDirection
    strength: float  # 0.0 to 1.0
    slope: float
    r_squared: float
    adx_value: float
    momentum: float
    trend_consistency: float
    support_resistance_strength: float
    breakout_probability: float
    timestamp: datetime


@dataclass
class TrendRegimeResult:
    """Result of trend regime classification"""
    regime: TrendRegime
    metrics: TrendMetrics
    regime_probabilities: Dict[TrendRegime, float]
    confidence: float
    trend_duration: int  # Days in current trend
    last_trend_change: datetime
    key_levels: Dict[str, float]  # Support/resistance levels


class TrendStrengthAnalyzer:
    """Analyzes trend strength using multiple indicators"""
    
    def __init__(self, adx_period: int = 14, momentum_period: int = 10):
        """
        Initialize trend strength analyzer
        
        Args:
            adx_period: Period for ADX calculation
            momentum_period: Period for momentum calculation
        """
        self.adx_period = adx_period
        self.momentum_period = momentum_period
        
    def calculate_adx(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average Directional Index (ADX)"""
        try:
            high = df['High']
            low = df['Low']
            close = df['Close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = np.abs(high - close.shift(1))
            tr3 = np.abs(low - close.shift(1))
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # Calculate Directional Movement
            high_diff = high - high.shift(1)
            low_diff = low.shift(1) - low
            
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            
            # Smooth the values using Wilder's smoothing
            tr_smooth = self._wilders_smoothing(pd.Series(tr), self.adx_period)
            plus_dm_smooth = self._wilders_smoothing(pd.Series(plus_dm), self.adx_period)
            minus_dm_smooth = self._wilders_smoothing(pd.Series(minus_dm), self.adx_period)
            
            # Calculate Directional Indicators
            plus_di = 100 * (plus_dm_smooth / (tr_smooth + 1e-10))
            minus_di = 100 * (minus_dm_smooth / (tr_smooth + 1e-10))
            
            # Calculate DX
            di_sum = plus_di + minus_di
            dx = 100 * np.abs(plus_di - minus_di) / (di_sum + 1e-10)
            
            # Calculate ADX using Wilder's smoothing
            adx = self._wilders_smoothing(dx, self.adx_period)
            
            # Ensure ADX values are reasonable
            adx = adx.fillna(0)
            
            # If ADX is still all zeros, use a simple trend strength measure
            if adx.sum() == 0:
                # Fallback: use price momentum as trend strength
                close = df['Close']
                price_changes = close.pct_change().abs()
                trend_strength = price_changes.rolling(window=self.adx_period).mean() * 100
                return trend_strength.fillna(0)
            
            return adx
            
        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")
            return pd.Series(index=df.index, data=0.0)
    
    def _wilders_smoothing(self, series: pd.Series, period: int) -> pd.Series:
        """Apply Wilder's smoothing (exponential moving average)"""
        alpha = 1.0 / period
        return series.ewm(alpha=alpha, adjust=False).mean()
    
    def calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate various momentum indicators"""
        close = df['Close']
        
        indicators = {}
        
        # Rate of Change (ROC)
        indicators['roc'] = close.pct_change(periods=self.momentum_period) * 100
        
        # Momentum
        indicators['momentum'] = close / close.shift(self.momentum_period) - 1
        
        # Price oscillator
        ema_fast = close.ewm(span=12).mean()
        ema_slow = close.ewm(span=26).mean()
        indicators['price_oscillator'] = (ema_fast - ema_slow) / ema_slow * 100
        
        # Relative Strength Index (RSI)
        indicators['rsi'] = self._calculate_rsi(close, period=14)
        
        return indicators
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def calculate_trend_consistency(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate trend consistency score"""
        try:
            close = df['Close']
            
            # Calculate rolling correlation with time (trend consistency)
            consistency_scores = []
            
            for i in range(len(close)):
                start_idx = max(0, i - window + 1)
                window_prices = close.iloc[start_idx:i+1]
                
                if len(window_prices) < 5:
                    consistency_scores.append(0.0)
                    continue
                
                # Create time series (x-axis)
                x = np.arange(len(window_prices))
                y = window_prices.values
                
                # Calculate correlation coefficient
                if len(x) > 1 and np.std(y) > 0:
                    correlation = np.corrcoef(x, y)[0, 1]
                    consistency_scores.append(abs(correlation))
                else:
                    consistency_scores.append(0.0)
            
            return pd.Series(consistency_scores, index=close.index)
            
        except Exception as e:
            logger.error(f"Error calculating trend consistency: {e}")
            return pd.Series(index=df.index, data=0.0)


class TrendDirectionAnalyzer:
    """Analyzes trend direction using multiple methods"""
    
    def __init__(self, ma_periods: List[int] = None, regression_window: int = 20):
        """
        Initialize trend direction analyzer
        
        Args:
            ma_periods: Moving average periods for trend analysis
            regression_window: Window for linear regression analysis
        """
        self.ma_periods = ma_periods or [10, 20, 50, 100]
        self.regression_window = regression_window
        
    def calculate_moving_average_trends(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate trend signals from multiple moving averages"""
        close = df['Close']
        ma_trends = {}
        
        for period in self.ma_periods:
            if len(close) >= period:
                ma = close.rolling(window=period).mean()
                
                # Trend direction: 1 for up, -1 for down, 0 for flat
                ma_slope = ma.diff()
                trend_signal = np.where(ma_slope > 0, 1, np.where(ma_slope < 0, -1, 0))
                
                ma_trends[f'ma_{period}_trend'] = pd.Series(trend_signal, index=close.index)
                ma_trends[f'ma_{period}_slope'] = ma_slope / close * 100  # Percentage slope
        
        return ma_trends
    
    def calculate_linear_regression_trend(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate trend using rolling linear regression"""
        close = df['Close']
        
        slopes = []
        r_squared_values = []
        
        for i in range(len(close)):
            start_idx = max(0, i - self.regression_window + 1)
            window_prices = close.iloc[start_idx:i+1]
            
            if len(window_prices) < 5:
                slopes.append(0.0)
                r_squared_values.append(0.0)
                continue
            
            # Prepare data for regression
            x = np.arange(len(window_prices)).reshape(-1, 1)
            y = window_prices.values
            
            try:
                if SKLEARN_AVAILABLE:
                    # Use sklearn for regression
                    model = LinearRegression()
                    model.fit(x, y)
                    slope = model.coef_[0]
                    r_squared = model.score(x, y)
                else:
                    # Use numpy for simple linear regression
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x.flatten(), y)
                    r_squared = r_value ** 2
                
                # Normalize slope by price level
                normalized_slope = slope / window_prices.iloc[-1] * 100
                
                slopes.append(normalized_slope)
                r_squared_values.append(max(0, r_squared))
                
            except Exception as e:
                logger.warning(f"Error in regression calculation: {e}")
                slopes.append(0.0)
                r_squared_values.append(0.0)
        
        return {
            'regression_slope': pd.Series(slopes, index=close.index),
            'regression_r_squared': pd.Series(r_squared_values, index=close.index)
        }
    
    def calculate_breakout_signals(self, df: pd.DataFrame, lookback: int = 20) -> Dict[str, pd.Series]:
        """Calculate breakout and breakdown signals"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        volume = df['Volume']
        
        # Rolling highs and lows
        rolling_high = high.rolling(window=lookback).max()
        rolling_low = low.rolling(window=lookback).min()
        
        # Breakout signals
        breakout_up = close > rolling_high.shift(1)
        breakout_down = close < rolling_low.shift(1)
        
        # Volume confirmation
        avg_volume = volume.rolling(window=lookback).mean()
        volume_surge = volume > avg_volume * 1.5
        
        # Confirmed breakouts
        confirmed_breakout_up = breakout_up & volume_surge
        confirmed_breakout_down = breakout_down & volume_surge
        
        return {
            'breakout_up': breakout_up.astype(int),
            'breakout_down': breakout_down.astype(int),
            'confirmed_breakout_up': confirmed_breakout_up.astype(int),
            'confirmed_breakout_down': confirmed_breakout_down.astype(int),
            'rolling_high': rolling_high,
            'rolling_low': rolling_low
        }


class SupportResistanceAnalyzer:
    """Analyzes support and resistance levels"""
    
    def __init__(self, min_touches: int = 2, tolerance: float = 0.02):
        """
        Initialize support/resistance analyzer
        
        Args:
            min_touches: Minimum number of touches to confirm level
            tolerance: Price tolerance for level identification (as percentage)
        """
        self.min_touches = min_touches
        self.tolerance = tolerance
        
    def find_support_resistance_levels(self, df: pd.DataFrame, 
                                     lookback: int = 50) -> Dict[str, List[float]]:
        """Find significant support and resistance levels"""
        try:
            high = df['High'].tail(lookback)
            low = df['Low'].tail(lookback)
            close = df['Close'].tail(lookback)
            
            # Find local peaks and troughs
            if SCIPY_AVAILABLE:
                # Use scipy for peak detection
                high_peaks, _ = find_peaks(high.values, distance=5)
                low_peaks, _ = find_peaks(-low.values, distance=5)
            else:
                # Simple peak detection
                high_peaks = self._simple_peak_detection(high.values, True)
                low_peaks = self._simple_peak_detection(low.values, False)
            
            # Extract resistance levels (peaks)
            resistance_levels = []
            if len(high_peaks) > 0:
                resistance_prices = high.iloc[high_peaks].values
                resistance_levels = self._cluster_levels(resistance_prices)
            
            # Extract support levels (troughs)
            support_levels = []
            if len(low_peaks) > 0:
                support_prices = low.iloc[low_peaks].values
                support_levels = self._cluster_levels(support_prices)
            
            # Current price for context
            current_price = close.iloc[-1]
            
            # Filter levels by significance
            significant_resistance = [
                level for level in resistance_levels 
                if level > current_price and self._count_touches(df, level, 'resistance') >= self.min_touches
            ]
            
            significant_support = [
                level for level in support_levels 
                if level < current_price and self._count_touches(df, level, 'support') >= self.min_touches
            ]
            
            return {
                'resistance_levels': significant_resistance,
                'support_levels': significant_support,
                'all_resistance': resistance_levels,
                'all_support': support_levels
            }
            
        except Exception as e:
            logger.error(f"Error finding support/resistance levels: {e}")
            return {
                'resistance_levels': [],
                'support_levels': [],
                'all_resistance': [],
                'all_support': []
            }
    
    def _simple_peak_detection(self, data: np.ndarray, find_maxima: bool) -> List[int]:
        """Simple peak detection without scipy"""
        peaks = []
        
        if find_maxima:
            for i in range(1, len(data) - 1):
                if data[i] > data[i-1] and data[i] > data[i+1]:
                    peaks.append(i)
        else:
            for i in range(1, len(data) - 1):
                if data[i] < data[i-1] and data[i] < data[i+1]:
                    peaks.append(i)
        
        return peaks
    
    def _cluster_levels(self, prices: np.ndarray) -> List[float]:
        """Cluster similar price levels together"""
        if len(prices) == 0:
            return []
        
        # Sort prices
        sorted_prices = np.sort(prices)
        clusters = []
        current_cluster = [sorted_prices[0]]
        
        for price in sorted_prices[1:]:
            # Check if price is within tolerance of current cluster
            cluster_center = np.mean(current_cluster)
            if abs(price - cluster_center) / cluster_center <= self.tolerance:
                current_cluster.append(price)
            else:
                # Start new cluster
                if len(current_cluster) >= 1:
                    clusters.append(np.mean(current_cluster))
                current_cluster = [price]
        
        # Add final cluster
        if len(current_cluster) >= 1:
            clusters.append(np.mean(current_cluster))
        
        return clusters
    
    def _count_touches(self, df: pd.DataFrame, level: float, level_type: str) -> int:
        """Count how many times price has touched a level"""
        if level_type == 'resistance':
            # Count times high price came close to resistance level
            touches = ((df['High'] >= level * (1 - self.tolerance)) & 
                      (df['High'] <= level * (1 + self.tolerance))).sum()
        else:
            # Count times low price came close to support level
            touches = ((df['Low'] >= level * (1 - self.tolerance)) & 
                      (df['Low'] <= level * (1 + self.tolerance))).sum()
        
        return touches
    
    def calculate_level_strength(self, df: pd.DataFrame, level: float, 
                               level_type: str) -> float:
        """Calculate strength of a support/resistance level"""
        touches = self._count_touches(df, level, level_type)
        
        # Calculate volume at touches
        if level_type == 'resistance':
            near_level = ((df['High'] >= level * (1 - self.tolerance)) & 
                         (df['High'] <= level * (1 + self.tolerance)))
        else:
            near_level = ((df['Low'] >= level * (1 - self.tolerance)) & 
                         (df['Low'] <= level * (1 + self.tolerance)))
        
        if near_level.sum() > 0:
            avg_volume_at_level = df.loc[near_level, 'Volume'].mean()
            avg_volume_overall = df['Volume'].mean()
            volume_ratio = avg_volume_at_level / (avg_volume_overall + 1e-10)
        else:
            volume_ratio = 1.0
        
        # Strength score combines touches and volume
        strength = min(touches / 5.0, 1.0) * min(volume_ratio, 2.0) / 2.0
        
        return strength


class TrendRegimeDetector:
    """Main trend regime detection system"""
    
    def __init__(self, 
                 adx_period: int = 14,
                 momentum_period: int = 10,
                 regression_window: int = 20,
                 ma_periods: List[int] = None):
        """
        Initialize trend regime detector
        
        Args:
            adx_period: Period for ADX calculation
            momentum_period: Period for momentum indicators
            regression_window: Window for regression analysis
            ma_periods: Moving average periods
        """
        self.strength_analyzer = TrendStrengthAnalyzer(adx_period, momentum_period)
        self.direction_analyzer = TrendDirectionAnalyzer(ma_periods, regression_window)
        self.sr_analyzer = SupportResistanceAnalyzer()
        
        self.current_regime = TrendRegime.RANGING
        self.regime_history = []
        self.last_update = None
        
        # Regime classification thresholds
        self.thresholds = {
            'strong_trend_adx': 25,      # ADX > 25 for strong trend
            'weak_trend_adx': 15,        # ADX > 15 for weak trend
            'strong_slope_threshold': 2.0,  # Slope > 2% for strong trend
            'weak_slope_threshold': 0.5,    # Slope > 0.5% for weak trend
            'r_squared_threshold': 0.3,     # R² > 0.3 for trend validity
            'consistency_threshold': 0.6    # Consistency > 0.6 for trend
        }
    
    def detect_trend_regime(self, df: pd.DataFrame) -> TrendRegimeResult:
        """Detect current trend regime"""
        try:
            # Calculate all trend metrics
            metrics = self._calculate_comprehensive_metrics(df)
            
            # Classify regime
            regime = self._classify_trend_regime(metrics)
            
            # Calculate regime probabilities
            regime_probabilities = self._calculate_regime_probabilities(metrics)
            
            # Calculate confidence
            confidence = self._calculate_confidence(metrics, regime)
            
            # Estimate trend duration
            trend_duration = self._estimate_trend_duration(df, regime)
            
            # Find key support/resistance levels
            key_levels = self._find_key_levels(df)
            
            # Update internal state
            self.current_regime = regime
            self.last_update = datetime.now()
            
            # Add to history
            self.regime_history.append({
                'timestamp': self.last_update,
                'regime': regime,
                'confidence': confidence,
                'adx': metrics.adx_value,
                'slope': metrics.slope
            })
            
            # Keep history limited
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]
            
            return TrendRegimeResult(
                regime=regime,
                metrics=metrics,
                regime_probabilities=regime_probabilities,
                confidence=confidence,
                trend_duration=trend_duration,
                last_trend_change=self.last_update - timedelta(days=trend_duration),
                key_levels=key_levels
            )
            
        except Exception as e:
            logger.error(f"Error detecting trend regime: {e}")
            return self._create_fallback_result(df)
    
    def _calculate_comprehensive_metrics(self, df: pd.DataFrame) -> TrendMetrics:
        """Calculate comprehensive trend analysis metrics"""
        try:
            # ADX for trend strength
            adx = self.strength_analyzer.calculate_adx(df)
            current_adx = adx.iloc[-1] if len(adx) > 0 else 0.0
            
            # Linear regression for trend direction and strength
            regression_data = self.direction_analyzer.calculate_linear_regression_trend(df)
            slope = regression_data['regression_slope'].iloc[-1] if len(regression_data['regression_slope']) > 0 else 0.0
            r_squared = regression_data['regression_r_squared'].iloc[-1] if len(regression_data['regression_r_squared']) > 0 else 0.0
            
            # Momentum indicators
            momentum_indicators = self.strength_analyzer.calculate_momentum_indicators(df)
            momentum = momentum_indicators['momentum'].iloc[-1] if len(momentum_indicators['momentum']) > 0 else 0.0
            
            # Trend consistency
            consistency = self.strength_analyzer.calculate_trend_consistency(df)
            trend_consistency = consistency.iloc[-1] if len(consistency) > 0 else 0.0
            
            # Support/resistance analysis
            sr_levels = self.sr_analyzer.find_support_resistance_levels(df)
            sr_strength = len(sr_levels['resistance_levels']) + len(sr_levels['support_levels'])
            sr_strength = min(sr_strength / 5.0, 1.0)  # Normalize to 0-1
            
            # Breakout analysis
            breakout_data = self.direction_analyzer.calculate_breakout_signals(df)
            recent_breakouts = (breakout_data['confirmed_breakout_up'].tail(5).sum() + 
                              breakout_data['confirmed_breakout_down'].tail(5).sum())
            breakout_probability = min(recent_breakouts / 2.0, 1.0)
            
            # Determine overall direction
            if slope > self.thresholds['weak_slope_threshold']:
                direction = TrendDirection.UP
            elif slope < -self.thresholds['weak_slope_threshold']:
                direction = TrendDirection.DOWN
            else:
                direction = TrendDirection.SIDEWAYS
            
            # Calculate overall trend strength (0-1)
            adx_strength = min(current_adx / 50.0, 1.0)  # Normalize ADX
            slope_strength = min(abs(slope) / 5.0, 1.0)  # Normalize slope
            overall_strength = (adx_strength + slope_strength + trend_consistency) / 3.0
            
            return TrendMetrics(
                direction=direction,
                strength=overall_strength,
                slope=slope,
                r_squared=r_squared,
                adx_value=current_adx,
                momentum=momentum,
                trend_consistency=trend_consistency,
                support_resistance_strength=sr_strength,
                breakout_probability=breakout_probability,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error calculating trend metrics: {e}")
            return TrendMetrics(
                direction=TrendDirection.SIDEWAYS,
                strength=0.0,
                slope=0.0,
                r_squared=0.0,
                adx_value=0.0,
                momentum=0.0,
                trend_consistency=0.0,
                support_resistance_strength=0.0,
                breakout_probability=0.0,
                timestamp=datetime.now()
            )
    
    def _classify_trend_regime(self, metrics: TrendMetrics) -> TrendRegime:
        """Classify trend regime based on metrics"""
        adx = metrics.adx_value
        slope = metrics.slope
        r_squared = metrics.r_squared
        consistency = metrics.trend_consistency
        
        # Strong trend criteria
        strong_trend = (adx > self.thresholds['strong_trend_adx'] and 
                       abs(slope) > self.thresholds['strong_slope_threshold'] and
                       r_squared > self.thresholds['r_squared_threshold'])
        
        # Weak trend criteria
        weak_trend = (adx > self.thresholds['weak_trend_adx'] and 
                     abs(slope) > self.thresholds['weak_slope_threshold'])
        
        # Direction classification
        if strong_trend:
            if slope > 0:
                return TrendRegime.STRONG_UPTREND
            else:
                return TrendRegime.STRONG_DOWNTREND
        elif weak_trend:
            if slope > 0:
                return TrendRegime.WEAK_UPTREND
            else:
                return TrendRegime.WEAK_DOWNTREND
        else:
            return TrendRegime.RANGING
    
    def _calculate_regime_probabilities(self, metrics: TrendMetrics) -> Dict[TrendRegime, float]:
        """Calculate probabilities for each trend regime"""
        adx = metrics.adx_value
        slope = metrics.slope
        r_squared = metrics.r_squared
        
        # Initialize probabilities
        probs = {regime: 0.0 for regime in TrendRegime}
        
        # ADX-based trend strength probability
        trend_prob = min(adx / 30.0, 1.0)  # Normalize ADX to probability
        ranging_prob = 1.0 - trend_prob
        
        # Slope-based direction probability
        if abs(slope) > self.thresholds['strong_slope_threshold']:
            strong_prob = min(abs(slope) / 5.0, 1.0)
            weak_prob = 1.0 - strong_prob
        else:
            strong_prob = 0.0
            weak_prob = min(abs(slope) / self.thresholds['strong_slope_threshold'], 1.0)
        
        # Distribute probabilities based on direction
        if slope > 0:  # Upward trend
            probs[TrendRegime.STRONG_UPTREND] = trend_prob * strong_prob
            probs[TrendRegime.WEAK_UPTREND] = trend_prob * weak_prob
        else:  # Downward trend
            probs[TrendRegime.STRONG_DOWNTREND] = trend_prob * strong_prob
            probs[TrendRegime.WEAK_DOWNTREND] = trend_prob * weak_prob
        
        # Ranging probability
        probs[TrendRegime.RANGING] = ranging_prob
        
        # Normalize probabilities
        total_prob = sum(probs.values())
        if total_prob > 0:
            probs = {k: v / total_prob for k, v in probs.items()}
        else:
            probs = {regime: 0.2 for regime in TrendRegime}
        
        return probs
    
    def _calculate_confidence(self, metrics: TrendMetrics, regime: TrendRegime) -> float:
        """Calculate confidence in regime classification"""
        # Base confidence on multiple factors
        adx_confidence = min(metrics.adx_value / 30.0, 1.0)
        r_squared_confidence = metrics.r_squared
        consistency_confidence = metrics.trend_consistency
        
        # Average confidence with weights
        confidence = (0.4 * adx_confidence + 
                     0.3 * r_squared_confidence + 
                     0.3 * consistency_confidence)
        
        return min(max(confidence, 0.1), 1.0)
    
    def _estimate_trend_duration(self, df: pd.DataFrame, regime: TrendRegime) -> int:
        """Estimate how long the current trend has been active"""
        if len(df) < 10:
            return 1
        
        # Simple approach: look for trend changes in recent data
        recent_data = df.tail(30)  # Last 30 days
        
        # Calculate rolling regime classification
        duration = 1
        current_direction = self._get_regime_direction(regime)
        
        for i in range(len(recent_data) - 2, 0, -1):
            window_data = recent_data.iloc[:i+1]
            if len(window_data) < 5:
                break
                
            window_metrics = self._calculate_comprehensive_metrics(window_data)
            window_regime = self._classify_trend_regime(window_metrics)
            window_direction = self._get_regime_direction(window_regime)
            
            if window_direction == current_direction:
                duration += 1
            else:
                break
        
        return min(duration, 30)  # Cap at 30 days
    
    def _get_regime_direction(self, regime: TrendRegime) -> str:
        """Get simplified direction from regime"""
        if regime in [TrendRegime.STRONG_UPTREND, TrendRegime.WEAK_UPTREND]:
            return 'up'
        elif regime in [TrendRegime.STRONG_DOWNTREND, TrendRegime.WEAK_DOWNTREND]:
            return 'down'
        else:
            return 'sideways'
    
    def _find_key_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        """Find key support and resistance levels"""
        try:
            sr_levels = self.sr_analyzer.find_support_resistance_levels(df)
            current_price = df['Close'].iloc[-1]
            
            # Find nearest levels
            resistance_levels = sr_levels['resistance_levels']
            support_levels = sr_levels['support_levels']
            
            nearest_resistance = None
            nearest_support = None
            
            if resistance_levels:
                nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price))
            
            if support_levels:
                nearest_support = min(support_levels, key=lambda x: abs(x - current_price))
            
            return {
                'current_price': current_price,
                'nearest_resistance': nearest_resistance,
                'nearest_support': nearest_support,
                'all_resistance': resistance_levels,
                'all_support': support_levels
            }
            
        except Exception as e:
            logger.error(f"Error finding key levels: {e}")
            return {'current_price': df['Close'].iloc[-1] if len(df) > 0 else 0.0}
    
    def _create_fallback_result(self, df: pd.DataFrame) -> TrendRegimeResult:
        """Create fallback result when detection fails"""
        current_price = df['Close'].iloc[-1] if len(df) > 0 else 0.0
        
        metrics = TrendMetrics(
            direction=TrendDirection.SIDEWAYS,
            strength=0.0,
            slope=0.0,
            r_squared=0.0,
            adx_value=0.0,
            momentum=0.0,
            trend_consistency=0.0,
            support_resistance_strength=0.0,
            breakout_probability=0.0,
            timestamp=datetime.now()
        )
        
        return TrendRegimeResult(
            regime=TrendRegime.RANGING,
            metrics=metrics,
            regime_probabilities={regime: 0.2 for regime in TrendRegime},
            confidence=0.5,
            trend_duration=1,
            last_trend_change=datetime.now(),
            key_levels={'current_price': current_price}
        )
    
    def get_regime_specific_parameters(self, regime: TrendRegime) -> Dict[str, float]:
        """Get trading parameters adjusted for trend regime"""
        regime_params = {
            TrendRegime.STRONG_UPTREND: {
                'trend_following_weight': 1.5,
                'mean_reversion_weight': 0.3,
                'breakout_sensitivity': 1.2,
                'position_hold_multiplier': 1.3,
                'stop_loss_distance': 0.8,
                'take_profit_multiplier': 1.4
            },
            TrendRegime.WEAK_UPTREND: {
                'trend_following_weight': 1.2,
                'mean_reversion_weight': 0.6,
                'breakout_sensitivity': 1.0,
                'position_hold_multiplier': 1.1,
                'stop_loss_distance': 0.9,
                'take_profit_multiplier': 1.2
            },
            TrendRegime.RANGING: {
                'trend_following_weight': 0.5,
                'mean_reversion_weight': 1.5,
                'breakout_sensitivity': 0.8,
                'position_hold_multiplier': 0.8,
                'stop_loss_distance': 1.0,
                'take_profit_multiplier': 0.9
            },
            TrendRegime.WEAK_DOWNTREND: {
                'trend_following_weight': 1.2,
                'mean_reversion_weight': 0.6,
                'breakout_sensitivity': 1.0,
                'position_hold_multiplier': 0.9,
                'stop_loss_distance': 0.9,
                'take_profit_multiplier': 1.1
            },
            TrendRegime.STRONG_DOWNTREND: {
                'trend_following_weight': 1.5,
                'mean_reversion_weight': 0.3,
                'breakout_sensitivity': 1.2,
                'position_hold_multiplier': 0.7,
                'stop_loss_distance': 0.8,
                'take_profit_multiplier': 1.3
            }
        }
        
        return regime_params.get(regime, regime_params[TrendRegime.RANGING])
    
    def get_trend_statistics(self) -> Dict[str, Any]:
        """Get trend regime statistics"""
        if not self.regime_history:
            return {}
        
        # Calculate regime distribution
        regime_counts = {}
        confidence_sum = {}
        adx_sum = {}
        
        for entry in self.regime_history:
            regime = entry['regime'].value
            confidence = entry['confidence']
            adx = entry['adx']
            
            if regime not in regime_counts:
                regime_counts[regime] = 0
                confidence_sum[regime] = 0.0
                adx_sum[regime] = 0.0
                
            regime_counts[regime] += 1
            confidence_sum[regime] += confidence
            adx_sum[regime] += adx
        
        # Calculate statistics
        total_entries = len(self.regime_history)
        regime_distribution = {
            regime: count / total_entries 
            for regime, count in regime_counts.items()
        }
        
        avg_confidence = {
            regime: confidence_sum[regime] / regime_counts[regime]
            for regime in regime_counts.keys()
        }
        
        avg_adx = {
            regime: adx_sum[regime] / regime_counts[regime]
            for regime in regime_counts.keys()
        }
        
        return {
            'total_detections': total_entries,
            'regime_distribution': regime_distribution,
            'average_confidence': avg_confidence,
            'average_adx': avg_adx,
            'current_regime': self.current_regime.value,
            'last_update': self.last_update.isoformat() if self.last_update else None
        }
    
    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """Update regime classification thresholds"""
        self.thresholds.update(new_thresholds)
        logger.info(f"Updated trend regime thresholds: {self.thresholds}")