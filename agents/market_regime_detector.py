"""
Market Regime Detection System using Hidden Markov Models

This module implements a comprehensive market regime detection system that identifies
different market states (bull, bear, volatile, stable) using Hidden Markov Models
and various market indicators.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pickle
import os
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import required libraries with fallbacks
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    logger.warning("hmmlearn not available, using simplified regime detection")
    HMM_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available, using basic preprocessing")
    SKLEARN_AVAILABLE = False


class MarketRegime(Enum):
    """Market regime types"""
    BULL = "bull"           # Strong upward trend, low volatility
    BEAR = "bear"           # Strong downward trend, low volatility  
    VOLATILE = "volatile"   # High volatility, unclear direction
    STABLE = "stable"       # Low volatility, sideways movement


@dataclass
class RegimeFeatures:
    """Features used for regime detection"""
    returns: float
    volatility: float
    trend_strength: float
    volume_ratio: float
    momentum: float
    timestamp: datetime


@dataclass
class RegimeDetectionResult:
    """Result of regime detection"""
    current_regime: MarketRegime
    regime_probabilities: Dict[MarketRegime, float]
    confidence: float
    features: RegimeFeatures
    regime_duration: int  # Days in current regime
    last_regime_change: datetime


class HiddenMarkovRegimeModel:
    """Hidden Markov Model for market regime detection"""
    
    def __init__(self, n_states: int = 4, random_state: int = 42):
        self.n_states = n_states
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_names = ['returns', 'volatility', 'trend_strength', 'volume_ratio', 'momentum']
        self.regime_mapping = {
            0: MarketRegime.BULL,
            1: MarketRegime.BEAR, 
            2: MarketRegime.VOLATILE,
            3: MarketRegime.STABLE
        }
        self.is_trained = False
        
        if HMM_AVAILABLE:
            # Initialize Gaussian HMM with full covariance
            self.model = hmm.GaussianHMM(
                n_components=n_states,
                covariance_type="full",
                random_state=random_state,
                n_iter=100
            )
        
        if SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()
            
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for HMM training/prediction"""
        features = []
        
        # Calculate returns
        returns = df['Close'].pct_change().fillna(0)
        
        # Calculate volatility (rolling standard deviation of returns)
        volatility = returns.rolling(window=20, min_periods=1).std().fillna(0)
        
        # Calculate trend strength using ADX-like measure
        trend_strength = self._calculate_trend_strength(df)
        
        # Calculate volume ratio (current volume / average volume)
        avg_volume = df['Volume'].rolling(window=20, min_periods=1).mean()
        volume_ratio = (df['Volume'] / avg_volume).fillna(1.0)
        
        # Calculate momentum (rate of change)
        momentum = df['Close'].pct_change(periods=10).fillna(0)
        
        # Combine features
        feature_matrix = np.column_stack([
            returns.values,
            volatility.values,
            trend_strength,  # Already numpy array
            volume_ratio.values,
            momentum.values
        ])
        
        # Handle any remaining NaN values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return feature_matrix
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate trend strength similar to ADX"""
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
            
            # Smooth the values
            period = 14
            tr_smooth = pd.Series(tr).rolling(window=period, min_periods=1).mean()
            plus_dm_smooth = pd.Series(plus_dm).rolling(window=period, min_periods=1).mean()
            minus_dm_smooth = pd.Series(minus_dm).rolling(window=period, min_periods=1).mean()
            
            # Calculate Directional Indicators (avoid division by zero)
            plus_di = 100 * (plus_dm_smooth / (tr_smooth + 1e-10))
            minus_di = 100 * (minus_dm_smooth / (tr_smooth + 1e-10))
            
            # Calculate DX
            di_sum = plus_di + minus_di
            dx = 100 * np.abs(plus_di - minus_di) / (di_sum + 1e-10)
            
            # Calculate ADX (trend strength)
            adx = pd.Series(dx).rolling(window=period, min_periods=1).mean()
            
            # Ensure same length as input data and handle NaN values
            adx = adx.reindex(df.index, fill_value=0).fillna(0)
            
            # Normalize to 0-1 range
            adx_values = adx.values / 100.0
            
            # Ensure no negative values and clip to reasonable range
            adx_values = np.clip(adx_values, 0, 1)
            
            return adx_values
            
        except Exception as e:
            logger.warning(f"Error calculating trend strength: {e}")
            # Return simple trend strength based on price momentum
            try:
                returns = df['Close'].pct_change().fillna(0)
                momentum = returns.rolling(window=14, min_periods=1).std().fillna(0)
                return np.clip(momentum.values * 10, 0, 1)  # Scale and clip
            except:
                return np.zeros(len(df))
    
    def train(self, market_data: Dict[str, pd.DataFrame], 
              min_samples: int = 100) -> bool:
        """Train the HMM on historical market data"""
        if not HMM_AVAILABLE:
            logger.warning("HMM not available, using fallback regime detection")
            return False
            
        try:
            # Combine data from all symbols
            all_features = []
            
            for symbol, df in market_data.items():
                if len(df) < min_samples:
                    logger.warning(f"Insufficient data for {symbol}: {len(df)} samples")
                    continue
                    
                features = self.prepare_features(df)
                all_features.append(features)
                
            if not all_features:
                logger.error("No sufficient data for training")
                return False
                
            # Concatenate all features
            X = np.vstack(all_features)
            
            # Scale features
            if SKLEARN_AVAILABLE and self.scaler is not None:
                X = self.scaler.fit_transform(X)
            
            # Train HMM
            self.model.fit(X)
            self.is_trained = True
            
            logger.info(f"HMM trained on {len(X)} samples from {len(all_features)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Error training HMM: {e}")
            return False
    
    def predict_regime(self, df: pd.DataFrame) -> RegimeDetectionResult:
        """Predict current market regime"""
        try:
            # Prepare features
            features = self.prepare_features(df)
            
            if not self.is_trained or not HMM_AVAILABLE:
                # Fallback to rule-based regime detection
                return self._fallback_regime_detection(df, features)
            
            # Use only the most recent features for prediction
            recent_features = features[-1:].reshape(1, -1)
            
            # Scale features
            if SKLEARN_AVAILABLE and self.scaler is not None:
                recent_features = self.scaler.transform(recent_features)
            
            # Predict regime
            regime_probs = self.model.predict_proba(recent_features)[0]
            predicted_state = np.argmax(regime_probs)
            current_regime = self.regime_mapping[predicted_state]
            
            # Calculate confidence (max probability)
            confidence = float(np.max(regime_probs))
            
            # Create regime probabilities dict
            regime_probabilities = {
                regime: float(regime_probs[state]) 
                for state, regime in self.regime_mapping.items()
            }
            
            # Calculate regime duration (simplified)
            regime_duration = self._estimate_regime_duration(features, predicted_state)
            
            # Create features object
            latest_features = RegimeFeatures(
                returns=float(features[-1, 0]),
                volatility=float(features[-1, 1]),
                trend_strength=float(features[-1, 2]),
                volume_ratio=float(features[-1, 3]),
                momentum=float(features[-1, 4]),
                timestamp=datetime.now()
            )
            
            return RegimeDetectionResult(
                current_regime=current_regime,
                regime_probabilities=regime_probabilities,
                confidence=confidence,
                features=latest_features,
                regime_duration=regime_duration,
                last_regime_change=datetime.now() - timedelta(days=regime_duration)
            )
            
        except Exception as e:
            logger.error(f"Error predicting regime: {e}")
            # Return fallback result
            return self._fallback_regime_detection(df, features if 'features' in locals() else None)
    
    def _fallback_regime_detection(self, df: pd.DataFrame, 
                                 features: Optional[np.ndarray] = None) -> RegimeDetectionResult:
        """Fallback rule-based regime detection when HMM is not available"""
        try:
            if features is None:
                features = self.prepare_features(df)
            
            if len(features) == 0:
                # Return default stable regime
                return RegimeDetectionResult(
                    current_regime=MarketRegime.STABLE,
                    regime_probabilities={regime: 0.25 for regime in MarketRegime},
                    confidence=0.5,
                    features=RegimeFeatures(0, 0, 0, 1, 0, datetime.now()),
                    regime_duration=1,
                    last_regime_change=datetime.now()
                )
            
            # Get latest features
            latest = features[-1]
            returns, volatility, trend_strength, volume_ratio, momentum = latest
            
            # Rule-based classification
            regime_scores = {
                MarketRegime.BULL: 0.0,
                MarketRegime.BEAR: 0.0,
                MarketRegime.VOLATILE: 0.0,
                MarketRegime.STABLE: 0.0
            }
            
            # Bull market: positive returns, low volatility, strong trend
            if returns > 0.01 and volatility < 0.02 and trend_strength > 0.5:
                regime_scores[MarketRegime.BULL] += 0.8
            elif returns > 0 and trend_strength > 0.3:
                regime_scores[MarketRegime.BULL] += 0.4
                
            # Bear market: negative returns, low volatility, strong trend
            if returns < -0.01 and volatility < 0.02 and trend_strength > 0.5:
                regime_scores[MarketRegime.BEAR] += 0.8
            elif returns < 0 and trend_strength > 0.3:
                regime_scores[MarketRegime.BEAR] += 0.4
                
            # Volatile market: high volatility regardless of direction
            if volatility > 0.03:
                regime_scores[MarketRegime.VOLATILE] += 0.6
            if abs(momentum) > 0.05:
                regime_scores[MarketRegime.VOLATILE] += 0.3
                
            # Stable market: low volatility, weak trend
            if volatility < 0.015 and trend_strength < 0.3:
                regime_scores[MarketRegime.STABLE] += 0.7
            if abs(returns) < 0.005:
                regime_scores[MarketRegime.STABLE] += 0.2
            
            # Normalize scores to probabilities
            total_score = sum(regime_scores.values())
            if total_score > 0:
                regime_probabilities = {k: v/total_score for k, v in regime_scores.items()}
            else:
                regime_probabilities = {regime: 0.25 for regime in MarketRegime}
            
            # Select regime with highest probability
            current_regime = max(regime_probabilities, key=regime_probabilities.get)
            confidence = regime_probabilities[current_regime]
            
            # Create features object
            latest_features = RegimeFeatures(
                returns=float(returns),
                volatility=float(volatility),
                trend_strength=float(trend_strength),
                volume_ratio=float(volume_ratio),
                momentum=float(momentum),
                timestamp=datetime.now()
            )
            
            return RegimeDetectionResult(
                current_regime=current_regime,
                regime_probabilities=regime_probabilities,
                confidence=confidence,
                features=latest_features,
                regime_duration=1,  # Simplified
                last_regime_change=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in fallback regime detection: {e}")
            # Return safe default
            return RegimeDetectionResult(
                current_regime=MarketRegime.STABLE,
                regime_probabilities={regime: 0.25 for regime in MarketRegime},
                confidence=0.5,
                features=RegimeFeatures(0, 0, 0, 1, 0, datetime.now()),
                regime_duration=1,
                last_regime_change=datetime.now()
            )
    
    def _estimate_regime_duration(self, features: np.ndarray, current_state: int) -> int:
        """Estimate how long the current regime has been active"""
        if len(features) < 10:
            return 1
            
        # Simple approach: look back and see when features significantly changed
        recent_window = min(30, len(features))
        recent_features = features[-recent_window:]
        
        # Calculate feature stability (inverse of variance)
        feature_vars = np.var(recent_features, axis=0)
        stability = 1.0 / (1.0 + np.mean(feature_vars))
        
        # Estimate duration based on stability (more stable = longer duration)
        estimated_duration = int(stability * 20) + 1
        return min(estimated_duration, recent_window)
    
    def save_model(self, filepath: str) -> bool:
        """Save trained model to file"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'regime_mapping': self.regime_mapping,
                'is_trained': self.is_trained,
                'feature_names': self.feature_names
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
                
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model from file"""
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Model file not found: {filepath}")
                return False
                
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.regime_mapping = model_data['regime_mapping']
            self.is_trained = model_data['is_trained']
            self.feature_names = model_data['feature_names']
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False


class MarketRegimeDetector:
    """Main market regime detection system"""
    
    def __init__(self, model_path: str = "data/models/regime_model.pkl"):
        self.hmm_model = HiddenMarkovRegimeModel()
        self.model_path = model_path
        self.current_regime = MarketRegime.STABLE
        self.regime_history = []
        self.last_update = None
        
        # Try to load existing model
        self.hmm_model.load_model(model_path)
        
    def detect_current_regime(self, market_data: pd.DataFrame) -> RegimeDetectionResult:
        """Detect current market regime"""
        try:
            result = self.hmm_model.predict_regime(market_data)
            
            # Update internal state
            self.current_regime = result.current_regime
            self.last_update = datetime.now()
            
            # Add to history
            self.regime_history.append({
                'timestamp': self.last_update,
                'regime': result.current_regime,
                'confidence': result.confidence
            })
            
            # Keep only recent history (last 100 entries)
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]
                
            return result
            
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            # Return safe default
            return RegimeDetectionResult(
                current_regime=MarketRegime.STABLE,
                regime_probabilities={regime: 0.25 for regime in MarketRegime},
                confidence=0.5,
                features=RegimeFeatures(0, 0, 0, 1, 0, datetime.now()),
                regime_duration=1,
                last_regime_change=datetime.now()
            )
    
    def get_regime_specific_parameters(self, regime: MarketRegime) -> Dict[str, Any]:
        """Get parameter adjustments for different market regimes"""
        regime_params = {
            MarketRegime.BULL: {
                'trend_following_weight': 1.2,
                'mean_reversion_weight': 0.8,
                'volatility_adjustment': 0.9,
                'position_size_multiplier': 1.1,
                'stop_loss_multiplier': 1.2,
                'rsi_overbought': 75,
                'rsi_oversold': 20,
                'macd_sensitivity': 0.8
            },
            MarketRegime.BEAR: {
                'trend_following_weight': 1.2,
                'mean_reversion_weight': 0.8,
                'volatility_adjustment': 0.9,
                'position_size_multiplier': 0.8,
                'stop_loss_multiplier': 0.8,
                'rsi_overbought': 80,
                'rsi_oversold': 25,
                'macd_sensitivity': 0.8
            },
            MarketRegime.VOLATILE: {
                'trend_following_weight': 0.7,
                'mean_reversion_weight': 1.3,
                'volatility_adjustment': 1.5,
                'position_size_multiplier': 0.6,
                'stop_loss_multiplier': 0.7,
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'macd_sensitivity': 1.2
            },
            MarketRegime.STABLE: {
                'trend_following_weight': 0.9,
                'mean_reversion_weight': 1.1,
                'volatility_adjustment': 1.0,
                'position_size_multiplier': 1.0,
                'stop_loss_multiplier': 1.0,
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'macd_sensitivity': 1.0
            }
        }
        
        return regime_params.get(regime, regime_params[MarketRegime.STABLE])
    
    def train_on_historical_data(self, market_data: Dict[str, pd.DataFrame]) -> bool:
        """Train the regime detection model on historical data"""
        try:
            success = self.hmm_model.train(market_data)
            
            if success:
                # Save the trained model
                self.hmm_model.save_model(self.model_path)
                logger.info("Regime detection model trained and saved successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Error training regime detection model: {e}")
            return False
    
    def get_regime_transition_probability(self, from_regime: MarketRegime, 
                                       to_regime: MarketRegime) -> float:
        """Get probability of transitioning from one regime to another"""
        if not self.hmm_model.is_trained or not HMM_AVAILABLE:
            # Return default transition probabilities
            if from_regime == to_regime:
                return 0.7  # High probability of staying in same regime
            else:
                return 0.1  # Low probability of changing regime
        
        try:
            # Get transition matrix from trained HMM
            transition_matrix = self.hmm_model.model.transmat_
            
            # Map regimes to states
            from_state = None
            to_state = None
            
            for state, regime in self.hmm_model.regime_mapping.items():
                if regime == from_regime:
                    from_state = state
                if regime == to_regime:
                    to_state = state
            
            if from_state is not None and to_state is not None:
                return float(transition_matrix[from_state, to_state])
            else:
                return 0.25  # Default uniform probability
                
        except Exception as e:
            logger.error(f"Error getting transition probability: {e}")
            return 0.25
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get statistics about regime detection performance"""
        if not self.regime_history:
            return {}
        
        # Calculate regime distribution
        regime_counts = {}
        confidence_sum = {}
        
        for entry in self.regime_history:
            regime = entry['regime']
            confidence = entry['confidence']
            
            if regime not in regime_counts:
                regime_counts[regime] = 0
                confidence_sum[regime] = 0.0
                
            regime_counts[regime] += 1
            confidence_sum[regime] += confidence
        
        # Calculate statistics
        total_entries = len(self.regime_history)
        regime_distribution = {
            regime.value: count / total_entries 
            for regime, count in regime_counts.items()
        }
        
        avg_confidence = {
            regime.value: confidence_sum[regime] / regime_counts[regime]
            for regime in regime_counts.keys()
        }
        
        return {
            'total_detections': total_entries,
            'regime_distribution': regime_distribution,
            'average_confidence': avg_confidence,
            'current_regime': self.current_regime.value,
            'last_update': self.last_update.isoformat() if self.last_update else None
        }