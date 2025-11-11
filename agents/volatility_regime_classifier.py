"""
Volatility Regime Classifier using GARCH Models

This module implements volatility regime detection and classification using GARCH models
for volatility forecasting and regime-based classification of market conditions.
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

# Try to import GARCH libraries with fallbacks
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    logger.warning("arch library not available, using simplified volatility estimation")
    ARCH_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available, using basic clustering")
    SKLEARN_AVAILABLE = False


class VolatilityRegime(Enum):
    """Volatility regime types"""
    LOW = "low"           # Low volatility period
    NORMAL = "normal"     # Normal volatility period
    HIGH = "high"         # High volatility period
    EXTREME = "extreme"   # Extreme volatility period


@dataclass
class VolatilityMetrics:
    """Volatility metrics and statistics"""
    current_volatility: float
    forecasted_volatility: float
    volatility_percentile: float
    regime: VolatilityRegime
    confidence: float
    garch_params: Dict[str, float]
    timestamp: datetime


@dataclass
class VolatilityRegimeResult:
    """Result of volatility regime classification"""
    regime: VolatilityRegime
    metrics: VolatilityMetrics
    regime_probabilities: Dict[VolatilityRegime, float]
    regime_thresholds: Dict[str, float]
    forecast_horizon: int


class GARCHVolatilityEstimator:
    """GARCH-based volatility estimation and forecasting"""
    
    def __init__(self, p: int = 1, q: int = 1, mean_model: str = 'Constant'):
        """
        Initialize GARCH volatility estimator
        
        Args:
            p: GARCH lag order
            q: ARCH lag order  
            mean_model: Mean model specification ('Constant', 'Zero', 'AR')
        """
        self.p = p
        self.q = q
        self.mean_model = mean_model
        self.model = None
        self.fitted_model = None
        self.is_fitted = False
        
    def fit(self, returns: pd.Series, update_freq: int = 0) -> bool:
        """
        Fit GARCH model to return series
        
        Args:
            returns: Time series of returns
            update_freq: Update frequency for model re-estimation
            
        Returns:
            bool: Success status
        """
        try:
            if not ARCH_AVAILABLE:
                logger.warning("ARCH library not available, using simplified volatility")
                return False
                
            # Clean returns data
            returns_clean = returns.dropna()
            if len(returns_clean) < 50:
                logger.warning(f"Insufficient data for GARCH fitting: {len(returns_clean)} observations")
                return False
                
            # Convert to percentage returns for better numerical stability
            returns_pct = returns_clean * 100
            
            # Create GARCH model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                self.model = arch_model(
                    returns_pct,
                    vol='GARCH',
                    p=self.p,
                    q=self.q,
                    mean=self.mean_model,
                    dist='normal'
                )
                
                # Fit model
                self.fitted_model = self.model.fit(
                    update_freq=update_freq,
                    disp='off',
                    show_warning=False
                )
                
            self.is_fitted = True
            logger.info(f"GARCH({self.p},{self.q}) model fitted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error fitting GARCH model: {e}")
            self.is_fitted = False
            return False
    
    def forecast_volatility(self, horizon: int = 1) -> Tuple[float, float]:
        """
        Forecast volatility using fitted GARCH model
        
        Args:
            horizon: Forecast horizon in periods
            
        Returns:
            Tuple of (forecasted_volatility, confidence_interval_width)
        """
        try:
            if not self.is_fitted or self.fitted_model is None:
                logger.warning("GARCH model not fitted, cannot forecast")
                return 0.0, 0.0
                
            # Generate forecast
            forecast = self.fitted_model.forecast(horizon=horizon, method='simulation')
            
            # Extract volatility forecast (convert back from percentage)
            vol_forecast = np.sqrt(forecast.variance.iloc[-1, 0]) / 100
            
            # Calculate confidence interval width (simplified)
            ci_width = vol_forecast * 0.2  # Approximate 20% confidence interval
            
            return float(vol_forecast), float(ci_width)
            
        except Exception as e:
            logger.error(f"Error forecasting volatility: {e}")
            return 0.0, 0.0
    
    def get_model_parameters(self) -> Dict[str, float]:
        """Get fitted GARCH model parameters"""
        if not self.is_fitted or self.fitted_model is None:
            return {}
            
        try:
            params = self.fitted_model.params
            return {
                'omega': float(params.get('omega', 0)),
                'alpha[1]': float(params.get('alpha[1]', 0)),
                'beta[1]': float(params.get('beta[1]', 0)),
                'loglikelihood': float(self.fitted_model.loglikelihood),
                'aic': float(self.fitted_model.aic),
                'bic': float(self.fitted_model.bic)
            }
        except Exception as e:
            logger.error(f"Error extracting GARCH parameters: {e}")
            return {}
    
    def calculate_conditional_volatility(self, returns: pd.Series) -> pd.Series:
        """Calculate conditional volatility series"""
        if not self.is_fitted or self.fitted_model is None:
            # Fallback to rolling volatility
            return returns.rolling(window=20, min_periods=1).std()
            
        try:
            # Get conditional volatility from fitted model
            cond_vol = self.fitted_model.conditional_volatility / 100  # Convert from percentage
            return pd.Series(cond_vol, index=returns.index[-len(cond_vol):])
            
        except Exception as e:
            logger.error(f"Error calculating conditional volatility: {e}")
            return returns.rolling(window=20, min_periods=1).std()


class VolatilityRegimeClassifier:
    """Classifier for volatility regimes based on GARCH estimates"""
    
    def __init__(self, lookback_window: int = 252, regime_thresholds: Dict[str, float] = None):
        """
        Initialize volatility regime classifier
        
        Args:
            lookback_window: Window for calculating volatility percentiles
            regime_thresholds: Custom thresholds for regime classification
        """
        self.lookback_window = lookback_window
        self.garch_estimator = GARCHVolatilityEstimator()
        
        # Default regime thresholds (percentiles)
        self.regime_thresholds = regime_thresholds or {
            'low_threshold': 0.25,      # 25th percentile
            'normal_threshold': 0.75,   # 75th percentile  
            'high_threshold': 0.95      # 95th percentile
        }
        
        self.volatility_history = []
        self.regime_history = []
        
    def calculate_realized_volatility(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """Calculate realized volatility using various estimators"""
        try:
            # Standard deviation (most common)
            vol_std = returns.rolling(window=window, min_periods=1).std()
            
            # Parkinson estimator (if high/low data available)
            # For now, use standard deviation
            realized_vol = vol_std
            
            # Annualize volatility (assuming daily data)
            realized_vol_annual = realized_vol * np.sqrt(252)
            
            return realized_vol_annual.ffill().fillna(0)
            
        except Exception as e:
            logger.error(f"Error calculating realized volatility: {e}")
            return pd.Series(index=returns.index, data=0.0)
    
    def fit_garch_model(self, returns: pd.Series) -> bool:
        """Fit GARCH model to return series"""
        return self.garch_estimator.fit(returns)
    
    def classify_volatility_regime(self, returns: pd.Series, 
                                 forecast_horizon: int = 1) -> VolatilityRegimeResult:
        """
        Classify current volatility regime
        
        Args:
            returns: Time series of returns
            forecast_horizon: Forecast horizon for volatility prediction
            
        Returns:
            VolatilityRegimeResult with regime classification
        """
        try:
            # Calculate current realized volatility
            realized_vol = self.calculate_realized_volatility(returns)
            current_vol = realized_vol.iloc[-1] if len(realized_vol) > 0 else 0.0
            
            # Fit GARCH model if not already fitted or if we have new data
            if not self.garch_estimator.is_fitted:
                garch_success = self.fit_garch_model(returns)
            else:
                garch_success = True
            
            # Forecast volatility
            if garch_success:
                forecasted_vol, forecast_ci = self.garch_estimator.forecast_volatility(forecast_horizon)
                garch_params = self.garch_estimator.get_model_parameters()
            else:
                # Fallback forecast using simple methods
                forecasted_vol = self._simple_volatility_forecast(returns)
                forecast_ci = forecasted_vol * 0.2
                garch_params = {}
            
            # Calculate volatility percentile
            vol_percentile = self._calculate_volatility_percentile(current_vol, realized_vol)
            
            # Classify regime
            regime = self._classify_regime(vol_percentile)
            
            # Calculate regime probabilities
            regime_probabilities = self._calculate_regime_probabilities(vol_percentile)
            
            # Calculate confidence
            confidence = self._calculate_classification_confidence(vol_percentile, regime)
            
            # Create metrics object
            metrics = VolatilityMetrics(
                current_volatility=current_vol,
                forecasted_volatility=forecasted_vol,
                volatility_percentile=vol_percentile,
                regime=regime,
                confidence=confidence,
                garch_params=garch_params,
                timestamp=datetime.now()
            )
            
            # Update history
            self.volatility_history.append(current_vol)
            self.regime_history.append(regime)
            
            # Keep history limited
            if len(self.volatility_history) > self.lookback_window:
                self.volatility_history = self.volatility_history[-self.lookback_window:]
                self.regime_history = self.regime_history[-self.lookback_window:]
            
            return VolatilityRegimeResult(
                regime=regime,
                metrics=metrics,
                regime_probabilities=regime_probabilities,
                regime_thresholds=self.regime_thresholds,
                forecast_horizon=forecast_horizon
            )
            
        except Exception as e:
            logger.error(f"Error classifying volatility regime: {e}")
            return self._create_fallback_result()
    
    def _simple_volatility_forecast(self, returns: pd.Series) -> float:
        """Simple volatility forecast when GARCH is not available"""
        try:
            # Use exponentially weighted moving average
            recent_vol = returns.rolling(window=20, min_periods=1).std().iloc[-1]
            long_term_vol = returns.rolling(window=60, min_periods=1).std().iloc[-1]
            
            # Weighted average with more weight on recent volatility
            forecast = 0.7 * recent_vol + 0.3 * long_term_vol
            
            # Annualize
            return forecast * np.sqrt(252)
            
        except Exception as e:
            logger.error(f"Error in simple volatility forecast: {e}")
            return 0.2  # Default 20% annual volatility
    
    def _calculate_volatility_percentile(self, current_vol: float, 
                                       vol_series: pd.Series) -> float:
        """Calculate percentile of current volatility in historical distribution"""
        try:
            if len(vol_series) < 10:
                return 0.5  # Default to median
                
            # Use recent history for percentile calculation
            recent_vol = vol_series.tail(self.lookback_window)
            percentile = (recent_vol < current_vol).mean()
            
            return float(percentile)
            
        except Exception as e:
            logger.error(f"Error calculating volatility percentile: {e}")
            return 0.5
    
    def _classify_regime(self, vol_percentile: float) -> VolatilityRegime:
        """Classify volatility regime based on percentile"""
        if vol_percentile <= self.regime_thresholds['low_threshold']:
            return VolatilityRegime.LOW
        elif vol_percentile <= self.regime_thresholds['normal_threshold']:
            return VolatilityRegime.NORMAL
        elif vol_percentile <= self.regime_thresholds['high_threshold']:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME
    
    def _calculate_regime_probabilities(self, vol_percentile: float) -> Dict[VolatilityRegime, float]:
        """Calculate probabilities for each volatility regime"""
        # Use smooth transitions between regimes
        low_prob = max(0, 1 - 4 * vol_percentile) if vol_percentile <= 0.25 else 0
        normal_prob = max(0, 1 - 4 * abs(vol_percentile - 0.5)) if 0.25 < vol_percentile <= 0.75 else 0
        high_prob = max(0, 4 * (vol_percentile - 0.75)) if 0.75 < vol_percentile <= 0.95 else 0
        extreme_prob = max(0, 20 * (vol_percentile - 0.95)) if vol_percentile > 0.95 else 0
        
        # Normalize probabilities
        total = low_prob + normal_prob + high_prob + extreme_prob
        if total > 0:
            return {
                VolatilityRegime.LOW: low_prob / total,
                VolatilityRegime.NORMAL: normal_prob / total,
                VolatilityRegime.HIGH: high_prob / total,
                VolatilityRegime.EXTREME: extreme_prob / total
            }
        else:
            # Default uniform distribution
            return {regime: 0.25 for regime in VolatilityRegime}
    
    def _calculate_classification_confidence(self, vol_percentile: float, 
                                          regime: VolatilityRegime) -> float:
        """Calculate confidence in regime classification"""
        # Higher confidence when percentile is far from thresholds
        thresholds = [0, self.regime_thresholds['low_threshold'], 
                     self.regime_thresholds['normal_threshold'],
                     self.regime_thresholds['high_threshold'], 1.0]
        
        # Find which interval the percentile falls into
        for i in range(len(thresholds) - 1):
            if thresholds[i] <= vol_percentile <= thresholds[i + 1]:
                interval_center = (thresholds[i] + thresholds[i + 1]) / 2
                interval_width = thresholds[i + 1] - thresholds[i]
                
                # Distance from center (normalized)
                distance_from_center = abs(vol_percentile - interval_center) / (interval_width / 2)
                
                # Confidence is higher when closer to center of interval
                confidence = 0.5 + 0.5 * (1 - distance_from_center)
                return min(max(confidence, 0.1), 1.0)
        
        return 0.5  # Default confidence
    
    def _create_fallback_result(self) -> VolatilityRegimeResult:
        """Create fallback result when classification fails"""
        metrics = VolatilityMetrics(
            current_volatility=0.2,
            forecasted_volatility=0.2,
            volatility_percentile=0.5,
            regime=VolatilityRegime.NORMAL,
            confidence=0.5,
            garch_params={},
            timestamp=datetime.now()
        )
        
        return VolatilityRegimeResult(
            regime=VolatilityRegime.NORMAL,
            metrics=metrics,
            regime_probabilities={regime: 0.25 for regime in VolatilityRegime},
            regime_thresholds=self.regime_thresholds,
            forecast_horizon=1
        )
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get statistics about volatility regime classification"""
        if not self.regime_history:
            return {}
        
        # Calculate regime distribution
        regime_counts = {}
        for regime in self.regime_history:
            regime_counts[regime.value] = regime_counts.get(regime.value, 0) + 1
        
        total_count = len(self.regime_history)
        regime_distribution = {k: v / total_count for k, v in regime_counts.items()}
        
        # Calculate volatility statistics
        vol_stats = {}
        if self.volatility_history:
            vol_array = np.array(self.volatility_history)
            vol_stats = {
                'mean': float(np.mean(vol_array)),
                'std': float(np.std(vol_array)),
                'min': float(np.min(vol_array)),
                'max': float(np.max(vol_array)),
                'median': float(np.median(vol_array))
            }
        
        return {
            'total_classifications': total_count,
            'regime_distribution': regime_distribution,
            'volatility_statistics': vol_stats,
            'current_regime': self.regime_history[-1].value if self.regime_history else None,
            'garch_fitted': self.garch_estimator.is_fitted
        }
    
    def update_regime_thresholds(self, new_thresholds: Dict[str, float]):
        """Update regime classification thresholds"""
        self.regime_thresholds.update(new_thresholds)
        logger.info(f"Updated volatility regime thresholds: {self.regime_thresholds}")
    
    def get_volatility_forecast_accuracy(self, actual_returns: pd.Series, 
                                       forecast_horizon: int = 1) -> Dict[str, float]:
        """Evaluate volatility forecast accuracy"""
        if not self.garch_estimator.is_fitted:
            return {}
        
        try:
            # Calculate forecast errors for recent period
            forecast_errors = []
            actual_vols = []
            
            # Use rolling window to evaluate forecasts
            for i in range(len(actual_returns) - forecast_horizon - 20, len(actual_returns) - forecast_horizon):
                if i < 20:
                    continue
                    
                # Fit model on data up to point i
                train_data = actual_returns.iloc[:i]
                temp_estimator = GARCHVolatilityEstimator()
                
                if temp_estimator.fit(train_data):
                    forecast_vol, _ = temp_estimator.forecast_volatility(forecast_horizon)
                    
                    # Calculate actual volatility for forecast period
                    future_returns = actual_returns.iloc[i:i+forecast_horizon]
                    actual_vol = future_returns.std() * np.sqrt(252)
                    
                    forecast_errors.append(abs(forecast_vol - actual_vol))
                    actual_vols.append(actual_vol)
            
            if forecast_errors:
                mae = np.mean(forecast_errors)  # Mean Absolute Error
                rmse = np.sqrt(np.mean(np.array(forecast_errors) ** 2))  # Root Mean Square Error
                mape = np.mean(np.array(forecast_errors) / (np.array(actual_vols) + 1e-10)) * 100  # Mean Absolute Percentage Error
                
                return {
                    'mean_absolute_error': float(mae),
                    'root_mean_square_error': float(rmse),
                    'mean_absolute_percentage_error': float(mape),
                    'forecast_samples': len(forecast_errors)
                }
            
        except Exception as e:
            logger.error(f"Error evaluating forecast accuracy: {e}")
        
        return {}


class VolatilityRegimeDetector:
    """Main volatility regime detection system"""
    
    def __init__(self, lookback_window: int = 252):
        """Initialize volatility regime detector"""
        self.classifier = VolatilityRegimeClassifier(lookback_window)
        self.current_regime = VolatilityRegime.NORMAL
        self.last_update = None
        
    def detect_volatility_regime(self, returns: pd.Series, 
                               forecast_horizon: int = 1) -> VolatilityRegimeResult:
        """Detect current volatility regime"""
        try:
            result = self.classifier.classify_volatility_regime(returns, forecast_horizon)
            
            # Update internal state
            self.current_regime = result.regime
            self.last_update = datetime.now()
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting volatility regime: {e}")
            return self.classifier._create_fallback_result()
    
    def get_regime_adjusted_parameters(self, base_params: Dict[str, float]) -> Dict[str, float]:
        """Get parameters adjusted for current volatility regime"""
        regime_adjustments = {
            VolatilityRegime.LOW: {
                'position_size_multiplier': 1.2,
                'stop_loss_multiplier': 1.1,
                'volatility_target_multiplier': 0.8,
                'rebalance_frequency_multiplier': 0.8
            },
            VolatilityRegime.NORMAL: {
                'position_size_multiplier': 1.0,
                'stop_loss_multiplier': 1.0,
                'volatility_target_multiplier': 1.0,
                'rebalance_frequency_multiplier': 1.0
            },
            VolatilityRegime.HIGH: {
                'position_size_multiplier': 0.7,
                'stop_loss_multiplier': 0.8,
                'volatility_target_multiplier': 1.3,
                'rebalance_frequency_multiplier': 1.2
            },
            VolatilityRegime.EXTREME: {
                'position_size_multiplier': 0.4,
                'stop_loss_multiplier': 0.6,
                'volatility_target_multiplier': 1.8,
                'rebalance_frequency_multiplier': 1.5
            }
        }
        
        adjustments = regime_adjustments.get(self.current_regime, regime_adjustments[VolatilityRegime.NORMAL])
        
        # Apply adjustments to base parameters
        adjusted_params = base_params.copy()
        for param, multiplier in adjustments.items():
            if param in adjusted_params:
                adjusted_params[param] *= multiplier
            else:
                adjusted_params[param] = multiplier
                
        return adjusted_params
    
    def get_volatility_statistics(self) -> Dict[str, Any]:
        """Get volatility regime statistics"""
        stats = self.classifier.get_regime_statistics()
        stats.update({
            'current_regime': self.current_regime.value,
            'last_update': self.last_update.isoformat() if self.last_update else None
        })
        return stats