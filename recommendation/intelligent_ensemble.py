"""
Intelligent Ensemble Decision Engine for Stock Bot Accuracy Improvements

This module implements machine learning-based signal combination, conflict resolution,
and Bayesian confidence estimation for trading recommendations.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of trading signals"""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    RISK = "risk"
    MACRO = "macro"
    MONTE_CARLO = "monte_carlo"
    BACKTEST = "backtest"


@dataclass
class Signal:
    """Individual trading signal with metadata"""
    signal_type: SignalType
    strength: float  # -1 to 1, where 1 is strong buy, -1 is strong sell
    confidence: float  # 0 to 1, confidence in the signal
    timestamp: pd.Timestamp
    source: str  # Source identifier (e.g., "RSI", "MACD", "P/E_ratio")
    metadata: Dict[str, Any]  # Additional signal-specific data


@dataclass
class MarketContext:
    """Current market context for signal interpretation"""
    volatility_regime: str  # "low", "medium", "high"
    trend_regime: str  # "trending", "ranging", "transitional"
    market_sentiment: str  # "bearish", "neutral", "bullish"
    correlation_regime: str  # "low", "medium", "high"
    volume_regime: str  # "low", "normal", "high"


@dataclass
class Recommendation:
    """Final trading recommendation with confidence and reasoning"""
    action: str  # "BUY", "SELL", "HOLD"
    strength: float  # -1 to 1, strength of recommendation
    confidence: float  # 0 to 1, confidence in recommendation
    probability_estimates: Dict[str, float]  # P(BUY), P(SELL), P(HOLD)
    reasoning: str  # Human-readable explanation
    signal_contributions: Dict[str, float]  # Individual signal contributions
    risk_assessment: Dict[str, float]  # Risk metrics


class AdaptiveSignalCombiner:
    """Machine learning-based signal weighting and combination"""
    
    def __init__(self):
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance = {}
        self.performance_history = []
        
    def extract_features(self, signals: List[Signal], market_context: MarketContext) -> np.ndarray:
        """Extract features from signals and market context for ML model"""
        features = []
        
        # Signal strength features by type
        signal_strengths = {st: 0.0 for st in SignalType}
        signal_confidences = {st: 0.0 for st in SignalType}
        signal_counts = {st: 0 for st in SignalType}
        
        for signal in signals:
            signal_strengths[signal.signal_type] += signal.strength
            signal_confidences[signal.signal_type] += signal.confidence
            signal_counts[signal.signal_type] += 1
        
        # Average strengths and confidences
        for st in SignalType:
            if signal_counts[st] > 0:
                features.extend([
                    signal_strengths[st] / signal_counts[st],  # Average strength
                    signal_confidences[st] / signal_counts[st],  # Average confidence
                    signal_counts[st]  # Count of signals
                ])
            else:
                features.extend([0.0, 0.0, 0])
        
        # Market context features (encoded)
        volatility_encoding = {"low": 0, "medium": 1, "high": 2}
        trend_encoding = {"ranging": 0, "transitional": 1, "trending": 2}
        sentiment_encoding = {"bearish": -1, "neutral": 0, "bullish": 1}
        correlation_encoding = {"low": 0, "medium": 1, "high": 2}
        volume_encoding = {"low": 0, "normal": 1, "high": 2}
        
        features.extend([
            volatility_encoding.get(market_context.volatility_regime, 1),
            trend_encoding.get(market_context.trend_regime, 1),
            sentiment_encoding.get(market_context.market_sentiment, 0),
            correlation_encoding.get(market_context.correlation_regime, 1),
            volume_encoding.get(market_context.volume_regime, 1)
        ])
        
        # Signal consensus features
        all_strengths = [s.strength for s in signals]
        if all_strengths:
            features.extend([
                np.mean(all_strengths),
                np.std(all_strengths),
                np.max(all_strengths),
                np.min(all_strengths),
                len([s for s in all_strengths if s > 0.5]),  # Strong buy signals
                len([s for s in all_strengths if s < -0.5])  # Strong sell signals
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0, 0])
        
        # Signal freshness (time-based decay)
        current_time = pd.Timestamp.now()
        if signals:
            avg_age_hours = np.mean([(current_time - s.timestamp).total_seconds() / 3600 
                                   for s in signals if hasattr(s, 'timestamp')])
            features.append(min(avg_age_hours, 24))  # Cap at 24 hours
        else:
            features.append(0.0)
        
        return np.array(features).reshape(1, -1)
    
    def calculate_dynamic_weights(self, signals: List[Signal], 
                                market_context: MarketContext) -> Dict[str, float]:
        """Calculate dynamic weights for signals based on ML model and market conditions"""
        
        if not self.is_trained:
            # Use rule-based weights if model not trained
            return self._calculate_rule_based_weights(signals, market_context)
        
        try:
            # Extract features for prediction
            features = self.extract_features(signals, market_context)
            features_scaled = self.scaler.transform(features)
            
            # Get feature importance from trained model
            importance_scores = self.rf_model.feature_importances_
            
            # Calculate weights based on signal type and current importance
            weights = {}
            signal_type_importance = {}
            
            # Map feature importance back to signal types
            feature_idx = 0
            for st in SignalType:
                # Each signal type has 3 features (strength, confidence, count)
                type_importance = np.mean(importance_scores[feature_idx:feature_idx+3])
                signal_type_importance[st] = type_importance
                feature_idx += 3
            
            # Calculate individual signal weights
            for signal in signals:
                base_weight = signal_type_importance.get(signal.signal_type, 0.1)
                
                # Adjust weight based on signal confidence and market context
                confidence_multiplier = 1.0 + (signal.confidence - 0.5)  # 0.5 to 1.5
                
                # Market context adjustments
                context_multiplier = self._get_context_multiplier(signal, market_context)
                
                final_weight = base_weight * confidence_multiplier * context_multiplier
                weights[signal.source] = final_weight
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            
            logger.debug(f"ML-based dynamic weights calculated: {weights}")
            return weights
            
        except Exception as e:
            logger.warning(f"Error in ML-based weight calculation: {e}. Falling back to rule-based.")
            return self._calculate_rule_based_weights(signals, market_context)
    
    def _calculate_rule_based_weights(self, signals: List[Signal], 
                                    market_context: MarketContext) -> Dict[str, float]:
        """Fallback rule-based weight calculation"""
        
        base_weights = {
            SignalType.TECHNICAL: 0.30,
            SignalType.FUNDAMENTAL: 0.15,
            SignalType.SENTIMENT: 0.20,
            SignalType.RISK: 0.15,
            SignalType.MACRO: 0.10,
            SignalType.MONTE_CARLO: 0.05,
            SignalType.BACKTEST: 0.05
        }
        
        weights = {}
        
        for signal in signals:
            base_weight = base_weights.get(signal.signal_type, 0.1)
            
            # Adjust based on market context
            context_multiplier = self._get_context_multiplier(signal, market_context)
            
            # Adjust based on signal confidence
            confidence_multiplier = 0.5 + signal.confidence
            
            final_weight = base_weight * context_multiplier * confidence_multiplier
            weights[signal.source] = final_weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _get_context_multiplier(self, signal: Signal, market_context: MarketContext) -> float:
        """Calculate context-based multiplier for signal weight"""
        
        multiplier = 1.0
        
        # Volatility regime adjustments
        if market_context.volatility_regime == "high":
            if signal.signal_type in [SignalType.RISK, SignalType.MONTE_CARLO]:
                multiplier *= 1.3  # Increase risk signal importance
            elif signal.signal_type == SignalType.TECHNICAL:
                multiplier *= 0.8  # Reduce technical signal importance
        elif market_context.volatility_regime == "low":
            if signal.signal_type == SignalType.TECHNICAL:
                multiplier *= 1.2  # Increase technical signal importance
        
        # Trend regime adjustments
        if market_context.trend_regime == "trending":
            if signal.signal_type == SignalType.TECHNICAL:
                multiplier *= 1.1  # Favor technical in trending markets
        elif market_context.trend_regime == "ranging":
            if signal.signal_type in [SignalType.FUNDAMENTAL, SignalType.SENTIMENT]:
                multiplier *= 1.1  # Favor fundamental/sentiment in ranging markets
        
        # Sentiment regime adjustments
        if market_context.market_sentiment in ["bullish", "bearish"]:
            if signal.signal_type in [SignalType.SENTIMENT, SignalType.MACRO]:
                multiplier *= 1.1  # Favor sentiment signals in extreme sentiment
        
        return multiplier
    
    def train_model(self, historical_signals: List[List[Signal]], 
                   historical_contexts: List[MarketContext],
                   historical_outcomes: List[float]):
        """Train the ML model on historical data"""
        
        if len(historical_signals) < 50:
            logger.warning("Insufficient training data for ML model")
            return
        
        try:
            # Prepare training data
            X = []
            y = historical_outcomes
            
            for signals, context in zip(historical_signals, historical_contexts):
                features = self.extract_features(signals, context)
                X.append(features.flatten())
            
            X = np.array(X)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.rf_model.fit(X_scaled, y)
            self.is_trained = True
            
            # Store feature importance
            feature_names = self._get_feature_names()
            self.feature_importance = dict(zip(feature_names, self.rf_model.feature_importances_))
            
            logger.info(f"ML model trained on {len(X)} samples")
            logger.debug(f"Top feature importances: {sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]}")
            
        except Exception as e:
            logger.error(f"Error training ML model: {e}")
            self.is_trained = False
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names for interpretability"""
        names = []
        
        # Signal type features
        for st in SignalType:
            names.extend([f"{st.value}_strength", f"{st.value}_confidence", f"{st.value}_count"])
        
        # Market context features
        names.extend([
            "volatility_regime", "trend_regime", "market_sentiment", 
            "correlation_regime", "volume_regime"
        ])
        
        # Consensus features
        names.extend([
            "mean_strength", "std_strength", "max_strength", "min_strength",
            "strong_buy_count", "strong_sell_count"
        ])
        
        # Freshness feature
        names.append("avg_age_hours")
        
        return names


class SignalConflictResolver:
    """Sophisticated conflict resolution for contradicting signals"""
    
    def __init__(self):
        self.resolution_strategies = {
            'confidence_weighted': self._resolve_by_confidence,
            'recency_weighted': self._resolve_by_recency,
            'source_reliability': self._resolve_by_source_reliability,
            'consensus_based': self._resolve_by_consensus
        }
        
        # Source reliability scores (can be updated based on historical performance)
        self.source_reliability = {
            'MACD': 0.85, 'RSI': 0.80, 'EMA': 0.75, 'SMA': 0.70,
            'Bollinger': 0.78, 'Stochastic': 0.82, 'LSTM': 0.90,
            'HMM': 0.88, 'P/E_ratio': 0.85, 'ROE': 0.80,
            'sentiment_composite': 0.75, 'news_sentiment': 0.70
        }
    
    def resolve_conflicts(self, conflicting_signals: List[Signal], 
                         strategy: str = 'confidence_weighted') -> Signal:
        """Resolve conflicts between contradicting signals"""
        
        if not conflicting_signals:
            raise ValueError("No signals provided for conflict resolution")
        
        if len(conflicting_signals) == 1:
            return conflicting_signals[0]
        
        # Check if signals are actually conflicting
        strengths = [s.strength for s in conflicting_signals]
        if not self._are_conflicting(strengths):
            # No real conflict, return strongest signal
            return max(conflicting_signals, key=lambda s: abs(s.strength))
        
        resolver = self.resolution_strategies.get(strategy, self._resolve_by_confidence)
        resolved_signal = resolver(conflicting_signals)
        
        logger.debug(f"Resolved conflict using {strategy}: {len(conflicting_signals)} signals -> strength={resolved_signal.strength:.3f}")
        
        return resolved_signal
    
    def _are_conflicting(self, strengths: List[float], threshold: float = 0.3) -> bool:
        """Check if signals are actually conflicting"""
        positive_signals = [s for s in strengths if s > threshold]
        negative_signals = [s for s in strengths if s < -threshold]
        
        return len(positive_signals) > 0 and len(negative_signals) > 0
    
    def _resolve_by_confidence(self, signals: List[Signal]) -> Signal:
        """Resolve conflict by weighting signals by confidence"""
        
        total_weighted_strength = 0.0
        total_confidence = 0.0
        
        for signal in signals:
            weight = signal.confidence ** 2  # Square confidence for emphasis
            total_weighted_strength += signal.strength * weight
            total_confidence += weight
        
        if total_confidence == 0:
            return signals[0]  # Fallback
        
        resolved_strength = total_weighted_strength / total_confidence
        avg_confidence = np.mean([s.confidence for s in signals])
        
        # Create resolved signal
        return Signal(
            signal_type=signals[0].signal_type,
            strength=resolved_strength,
            confidence=avg_confidence * 0.8,  # Reduce confidence due to conflict
            timestamp=max(s.timestamp for s in signals if hasattr(s, 'timestamp')),
            source="conflict_resolved",
            metadata={'resolution_method': 'confidence_weighted', 'original_count': len(signals)}
        )
    
    def _resolve_by_recency(self, signals: List[Signal]) -> Signal:
        """Resolve conflict by favoring more recent signals"""
        
        current_time = pd.Timestamp.now()
        
        # Calculate recency weights (exponential decay)
        weights = []
        for signal in signals:
            if hasattr(signal, 'timestamp'):
                age_hours = (current_time - signal.timestamp).total_seconds() / 3600
                weight = np.exp(-age_hours / 6)  # 6-hour half-life
            else:
                weight = 1.0  # Default weight if no timestamp
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0] * len(signals)
            total_weight = len(signals)
        
        weights = [w / total_weight for w in weights]
        
        # Calculate weighted average
        resolved_strength = sum(s.strength * w for s, w in zip(signals, weights))
        avg_confidence = sum(s.confidence * w for s, w in zip(signals, weights))
        
        return Signal(
            signal_type=signals[0].signal_type,
            strength=resolved_strength,
            confidence=avg_confidence * 0.9,  # Slight confidence reduction
            timestamp=max(s.timestamp for s in signals if hasattr(s, 'timestamp')),
            source="recency_resolved",
            metadata={'resolution_method': 'recency_weighted', 'original_count': len(signals)}
        )
    
    def _resolve_by_source_reliability(self, signals: List[Signal]) -> Signal:
        """Resolve conflict by weighting signals by source reliability"""
        
        total_weighted_strength = 0.0
        total_reliability = 0.0
        
        for signal in signals:
            reliability = self.source_reliability.get(signal.source, 0.5)
            total_weighted_strength += signal.strength * reliability
            total_reliability += reliability
        
        if total_reliability == 0:
            return signals[0]  # Fallback
        
        resolved_strength = total_weighted_strength / total_reliability
        avg_confidence = np.mean([s.confidence for s in signals])
        
        return Signal(
            signal_type=signals[0].signal_type,
            strength=resolved_strength,
            confidence=avg_confidence * 0.85,  # Confidence reduction due to conflict
            timestamp=max(s.timestamp for s in signals if hasattr(s, 'timestamp')),
            source="reliability_resolved",
            metadata={'resolution_method': 'source_reliability', 'original_count': len(signals)}
        )
    
    def _resolve_by_consensus(self, signals: List[Signal]) -> Signal:
        """Resolve conflict by finding consensus among majority"""
        
        # Group signals by direction
        buy_signals = [s for s in signals if s.strength > 0.1]
        sell_signals = [s for s in signals if s.strength < -0.1]
        neutral_signals = [s for s in signals if -0.1 <= s.strength <= 0.1]
        
        # Find majority direction
        if len(buy_signals) > len(sell_signals) and len(buy_signals) > len(neutral_signals):
            majority_signals = buy_signals
        elif len(sell_signals) > len(buy_signals) and len(sell_signals) > len(neutral_signals):
            majority_signals = sell_signals
        else:
            majority_signals = neutral_signals
        
        if not majority_signals:
            majority_signals = signals  # Fallback to all signals
        
        # Calculate consensus strength and confidence
        avg_strength = np.mean([s.strength for s in majority_signals])
        avg_confidence = np.mean([s.confidence for s in majority_signals])
        
        # Adjust confidence based on consensus strength
        consensus_ratio = len(majority_signals) / len(signals)
        adjusted_confidence = avg_confidence * consensus_ratio
        
        return Signal(
            signal_type=signals[0].signal_type,
            strength=avg_strength,
            confidence=adjusted_confidence,
            timestamp=max(s.timestamp for s in signals if hasattr(s, 'timestamp')),
            source="consensus_resolved",
            metadata={'resolution_method': 'consensus_based', 'original_count': len(signals), 'consensus_ratio': consensus_ratio}
        )
    
    def update_source_reliability(self, source: str, performance_score: float):
        """Update source reliability based on historical performance"""
        
        if source in self.source_reliability:
            # Exponential moving average update
            alpha = 0.1  # Learning rate
            self.source_reliability[source] = (
                alpha * performance_score + 
                (1 - alpha) * self.source_reliability[source]
            )
        else:
            self.source_reliability[source] = performance_score
        
        logger.debug(f"Updated reliability for {source}: {self.source_reliability[source]:.3f}")


class BayesianConfidenceEstimator:
    """Bayesian confidence estimation for trading recommendations"""
    
    def __init__(self):
        self.bayesian_model = BayesianRidge()
        self.is_trained = False
        self.prior_probabilities = {'BUY': 0.33, 'SELL': 0.33, 'HOLD': 0.34}
        self.historical_accuracy = {}
        
    def estimate_confidence(self, signals: List[Signal], 
                          market_context: MarketContext,
                          composite_strength: float) -> Tuple[float, Dict[str, float]]:
        """Estimate confidence and probability distribution for recommendation"""
        
        # Calculate base confidence from signal consensus
        base_confidence = self._calculate_signal_consensus_confidence(signals)
        
        # Adjust confidence based on market context
        context_adjustment = self._get_context_confidence_adjustment(market_context)
        
        # Calculate probability estimates
        probabilities = self._calculate_action_probabilities(composite_strength, base_confidence)
        
        # Final confidence is the maximum probability (certainty in recommendation)
        final_confidence = max(probabilities.values()) * context_adjustment
        
        # Apply Bayesian updates if model is trained
        if self.is_trained:
            try:
                bayesian_confidence = self._get_bayesian_confidence(signals, market_context)
                final_confidence = 0.7 * final_confidence + 0.3 * bayesian_confidence
            except Exception as e:
                logger.warning(f"Bayesian confidence estimation failed: {e}")
        
        logger.debug(f"Confidence estimation: base={base_confidence:.3f}, context_adj={context_adjustment:.3f}, final={final_confidence:.3f}")
        
        return final_confidence, probabilities
    
    def _calculate_signal_consensus_confidence(self, signals: List[Signal]) -> float:
        """Calculate confidence based on signal consensus"""
        
        if not signals:
            return 0.5
        
        strengths = [s.strength for s in signals]
        confidences = [s.confidence for s in signals]
        
        # Consensus metrics
        mean_strength = np.mean(np.abs(strengths))
        std_strength = np.std(strengths)
        mean_confidence = np.mean(confidences)
        
        # Agreement score (lower std = higher agreement)
        agreement_score = 1.0 / (1.0 + std_strength)
        
        # Signal strength score
        strength_score = min(mean_strength * 2, 1.0)  # Scale to [0, 1]
        
        # Combined confidence
        consensus_confidence = (
            0.4 * agreement_score + 
            0.3 * strength_score + 
            0.3 * mean_confidence
        )
        
        return np.clip(consensus_confidence, 0.0, 1.0)
    
    def _get_context_confidence_adjustment(self, market_context: MarketContext) -> float:
        """Adjust confidence based on market context"""
        
        adjustment = 1.0
        
        # Volatility adjustments
        if market_context.volatility_regime == "high":
            adjustment *= 0.8  # Reduce confidence in high volatility
        elif market_context.volatility_regime == "low":
            adjustment *= 1.1  # Increase confidence in low volatility
        
        # Trend adjustments
        if market_context.trend_regime == "trending":
            adjustment *= 1.05  # Slight boost in trending markets
        elif market_context.trend_regime == "transitional":
            adjustment *= 0.9  # Reduce confidence in transitional periods
        
        # Correlation adjustments
        if market_context.correlation_regime == "high":
            adjustment *= 0.95  # Slight reduction in high correlation (systematic risk)
        
        return np.clip(adjustment, 0.5, 1.5)
    
    def _calculate_action_probabilities(self, composite_strength: float, 
                                     base_confidence: float) -> Dict[str, float]:
        """Calculate probability distribution over actions"""
        
        # Convert composite strength to probabilities using softmax-like function
        # Adjust temperature based on confidence (higher confidence = more decisive)
        temperature = 2.0 / (1.0 + base_confidence)  # Range: [1.0, 2.0]
        
        # Define action scores based on composite strength
        buy_score = max(0, composite_strength) / temperature
        sell_score = max(0, -composite_strength) / temperature
        hold_score = (1.0 - abs(composite_strength)) / temperature
        
        # Apply softmax
        scores = np.array([buy_score, sell_score, hold_score])
        exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
        probabilities = exp_scores / np.sum(exp_scores)
        
        return {
            'BUY': probabilities[0],
            'SELL': probabilities[1],
            'HOLD': probabilities[2]
        }
    
    def _get_bayesian_confidence(self, signals: List[Signal], 
                               market_context: MarketContext) -> float:
        """Get confidence estimate from trained Bayesian model"""
        
        if not self.is_trained:
            return 0.5
        
        # Prepare features (simplified version)
        features = []
        
        # Signal statistics
        strengths = [s.strength for s in signals]
        confidences = [s.confidence for s in signals]
        
        features.extend([
            np.mean(strengths), np.std(strengths),
            np.mean(confidences), np.std(confidences),
            len(signals)
        ])
        
        # Market context (encoded)
        context_encoding = {
            'volatility_regime': {"low": 0, "medium": 1, "high": 2}.get(market_context.volatility_regime, 1),
            'trend_regime': {"ranging": 0, "transitional": 1, "trending": 2}.get(market_context.trend_regime, 1),
            'market_sentiment': {"bearish": -1, "neutral": 0, "bullish": 1}.get(market_context.market_sentiment, 0)
        }
        
        features.extend(list(context_encoding.values()))
        
        try:
            # Predict confidence
            X = np.array(features).reshape(1, -1)
            confidence_pred = self.bayesian_model.predict(X)[0]
            
            # Get prediction uncertainty
            _, pred_std = self.bayesian_model.predict(X, return_std=True)
            uncertainty = pred_std[0]
            
            # Adjust confidence based on uncertainty
            adjusted_confidence = confidence_pred * (1.0 - uncertainty)
            
            return np.clip(adjusted_confidence, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Bayesian confidence prediction failed: {e}")
            return 0.5
    
    def train_bayesian_model(self, historical_features: List[List[float]], 
                           historical_confidences: List[float]):
        """Train Bayesian model on historical confidence data"""
        
        if len(historical_features) < 30:
            logger.warning("Insufficient data for Bayesian model training")
            return
        
        try:
            X = np.array(historical_features)
            y = np.array(historical_confidences)
            
            self.bayesian_model.fit(X, y)
            self.is_trained = True
            
            logger.info(f"Bayesian confidence model trained on {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Error training Bayesian model: {e}")
            self.is_trained = False
    
    def update_prior_probabilities(self, action_counts: Dict[str, int]):
        """Update prior probabilities based on historical action distribution"""
        
        total_actions = sum(action_counts.values())
        if total_actions > 0:
            for action in self.prior_probabilities:
                count = action_counts.get(action, 0)
                self.prior_probabilities[action] = count / total_actions
        
        logger.debug(f"Updated prior probabilities: {self.prior_probabilities}")


class StackingEnsemble:
    """Stacking ensemble that uses a meta-learner to combine base signal predictions"""
    
    def __init__(self):
        # In a full implementation, this would be a trained model
        # For now, we'll use a simple weighted average with learned weights
        self.meta_weights = {
            SignalType.TECHNICAL: 0.25,
            SignalType.FUNDAMENTAL: 0.15,
            SignalType.SENTIMENT: 0.20,
            SignalType.RISK: 0.15,
            SignalType.MACRO: 0.10,
            SignalType.MONTE_CARLO: 0.05,
            SignalType.BACKTEST: 0.10
        }
    
    def predict(self, signals: List[Signal], market_context: MarketContext) -> float:
        """Generate prediction using stacking ensemble"""
        # Group signals by type
        signals_by_type = {}
        for signal in signals:
            if signal.signal_type not in signals_by_type:
                signals_by_type[signal.signal_type] = []
            signals_by_type[signal.signal_type].append(signal)
        
        # Calculate average strength for each type
        type_strengths = {}
        for signal_type, type_signals in signals_by_type.items():
            avg_strength = np.mean([s.strength for s in type_signals])
            type_strengths[signal_type] = avg_strength
        
        # Apply meta-learner weights
        stacked_strength = 0.0
        for signal_type, strength in type_strengths.items():
            weight = self.meta_weights.get(signal_type, 0.0)
            stacked_strength += strength * weight
        
        # Adjust based on market context
        stacked_strength = self._apply_context_adjustment(stacked_strength, market_context)
        
        return stacked_strength
    
    def _apply_context_adjustment(self, strength: float, market_context: MarketContext) -> float:
        """Apply market context-based adjustments to stacked prediction"""
        # Adjust based on volatility regime
        if market_context.volatility_regime == "high":
            # Reduce signal strength in high volatility
            strength *= 0.8
        elif market_context.volatility_regime == "low":
            # Slightly increase signal strength in low volatility
            strength *= 1.1
        
        # Adjust based on trend regime
        if market_context.trend_regime == "trending":
            # Increase signal strength in trending markets
            strength *= 1.1
        elif market_context.trend_regime == "ranging":
            # Reduce signal strength in ranging markets
            strength *= 0.9
        
        return np.clip(strength, -1.0, 1.0)


class BoostingEnsemble:
    """Boosting ensemble that sequentially trains weak learners to correct errors"""
    
    def __init__(self):
        # For simplicity, we'll simulate boosting with weighted voting
        # where weights are based on signal confidence and context
        self.n_boosting_rounds = 3
        self.learning_rate = 0.1
    
    def predict(self, signals: List[Signal], market_context: MarketContext) -> float:
        """Generate prediction using boosting ensemble"""
        # Initialize prediction
        prediction = 0.0
        
        # Simulate boosting rounds
        for i in range(self.n_boosting_rounds):
            # Calculate weighted prediction for this round
            round_prediction = self._calculate_boosting_round(signals, market_context, i)
            
            # Apply learning rate
            prediction += self.learning_rate * round_prediction
        
        return np.clip(prediction, -1.0, 1.0)
    
    def _calculate_boosting_round(self, signals: List[Signal], market_context: MarketContext, round_num: int) -> float:
        """Calculate prediction for a single boosting round"""
        # In a real boosting implementation, this would train a weak learner
        # For now, we'll use a simple weighted average with context adjustments
        total_weighted_strength = 0.0
        total_weight = 0.0
        
        for signal in signals:
            # Calculate weight based on confidence and context
            base_weight = signal.confidence
            context_weight = self._get_context_weight(signal, market_context, round_num)
            final_weight = base_weight * context_weight
            
            total_weighted_strength += signal.strength * final_weight
            total_weight += final_weight
        
        if total_weight > 0:
            return total_weighted_strength / total_weight
        else:
            return 0.0
    
    def _get_context_weight(self, signal: Signal, market_context: MarketContext, round_num: int) -> float:
        """Calculate context-based weight for a signal in a boosting round"""
        weight = 1.0
        
        # Adjust based on signal type and market context
        if market_context.volatility_regime == "high":
            if signal.signal_type in [SignalType.RISK, SignalType.MONTE_CARLO]:
                weight *= 1.3  # Increase risk signal importance
            elif signal.signal_type == SignalType.TECHNICAL:
                weight *= 0.8  # Reduce technical signal importance
        elif market_context.volatility_regime == "low":
            if signal.signal_type == SignalType.TECHNICAL:
                weight *= 1.2  # Increase technical signal importance
        
        # Adjust based on trend regime
        if market_context.trend_regime == "trending":
            if signal.signal_type == SignalType.TECHNICAL:
                weight *= 1.1  # Favor technical in trending markets
        elif market_context.trend_regime == "ranging":
            if signal.signal_type in [SignalType.FUNDAMENTAL, SignalType.SENTIMENT]:
                weight *= 1.1  # Favor fundamental/sentiment in ranging markets
        
        # Adjust based on round number (e.g., focus on different aspects)
        if round_num == 0:
            # First round: focus on strong signals
            if abs(signal.strength) > 0.5:
                weight *= 1.2
        elif round_num == 1:
            # Second round: focus on medium signals
            if 0.2 <= abs(signal.strength) <= 0.5:
                weight *= 1.2
        elif round_num == 2:
            # Third round: focus on weak signals to correct errors
            if abs(signal.strength) < 0.2:
                weight *= 1.2
        
        return weight


class DynamicWeighter:
    """Dynamically adjusts signal weights based on market conditions and performance"""
    
    def __init__(self):
        # Historical performance tracking for each signal source
        self.performance_history = {}
        self.decay_factor = 0.95  # How much to decay old performance scores
    
    def adjust_weights(self, original_weights: Dict[str, float], market_context: MarketContext) -> Dict[str, float]:
        """Adjust signal weights based on market context and historical performance"""
        adjusted_weights = {}
        
        for source, weight in original_weights.items():
            # Get performance adjustment for this source
            perf_adjustment = self._get_performance_adjustment(source)
            
            # Get context adjustment
            context_adjustment = self._get_context_adjustment(source, market_context)
            
            # Apply adjustments
            adjusted_weight = weight * perf_adjustment * context_adjustment
            adjusted_weights[source] = adjusted_weight
        
        # Normalize weights to sum to 1
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}
        
        return adjusted_weights
    
    def _get_performance_adjustment(self, source: str) -> float:
        """Get weight adjustment based on historical performance of the signal source"""
        # Get historical accuracy for this source
        historical_accuracy = self.performance_history.get(source, 0.5)  # Default to 50%
        
        # Map accuracy to adjustment factor
        # If accuracy > 50%, increase weight; if < 50%, decrease weight
        adjustment = 0.5 + historical_accuracy  # Range: 0.5 to 1.5
        
        return adjustment
    
    def _get_context_adjustment(self, source: str, market_context: MarketContext) -> float:
        """Get weight adjustment based on market context"""
        adjustment = 1.0
        
        # Example: adjust based on signal type (extracted from source name)
        # In a real implementation, this would be more sophisticated
        if 'tech' in source.lower() or any(t in source.lower() for t in ['rsi', 'macd', 'sma', 'ema']):
            # Technical signals
            if market_context.volatility_regime == "high":
                adjustment *= 0.8  # Reduce weight in high volatility
            elif market_context.volatility_regime == "low":
                adjustment *= 1.1  # Increase weight in low volatility
            
            if market_context.trend_regime == "trending":
                adjustment *= 1.1  # Increase weight in trending markets
        elif 'sentiment' in source.lower() or 'news' in source.lower():
            # Sentiment signals
            if market_context.market_sentiment in ["bullish", "bearish"]:
                adjustment *= 1.1  # Increase weight in extreme sentiment
            elif market_context.correlation_regime == "high":
                adjustment *= 0.9  # Reduce weight in high correlation (herding)
        
        return adjustment
    
    def update_performance(self, source: str, accuracy: float):
        """Update historical performance for a signal source"""
        if source in self.performance_history:
            # Apply decay to old performance and add new performance
            old_perf = self.performance_history[source]
            self.performance_history[source] = self.decay_factor * old_perf + (1 - self.decay_factor) * accuracy
        else:
            self.performance_history[source] = accuracy


class IntelligentEnsembleEngine:
    """Main ensemble engine that combines all components"""
    
    def __init__(self):
        self.signal_combiner = AdaptiveSignalCombiner()
        self.conflict_resolver = SignalConflictResolver()
        self.confidence_estimator = BayesianConfidenceEstimator()
        # Add new components for advanced ensembling
        self.stacking_ensemble = StackingEnsemble()
        self.boosting_ensemble = BoostingEnsemble()
        self.dynamic_weighter = DynamicWeighter()
        
    def generate_ensemble_recommendation(self, signals: List[Signal],
                                       market_context: MarketContext) -> Recommendation:
        """Generate final recommendation using intelligent ensemble methods"""
        
        if not signals:
            return self._generate_default_recommendation("No signals available")
        
        try:
            # Step 1: Resolve conflicts within signal groups
            resolved_signals = self._resolve_signal_conflicts(signals)
            
            # Step 2: Calculate dynamic weights
            weights = self.signal_combiner.calculate_dynamic_weights(resolved_signals, market_context)
            
            # NEW: Apply advanced ensemble methods
            composite_strength = self._apply_advanced_ensemble_methods(resolved_signals, weights, market_context)
            
            # Step 4: Estimate confidence and probabilities
            confidence, probabilities = self.confidence_estimator.estimate_confidence(
                resolved_signals, market_context, composite_strength
            )
            
            # Step 5: Determine final action
            action = self._determine_action(composite_strength, probabilities, confidence)
            
            # Step 6: Generate reasoning
            reasoning = self._generate_reasoning(action, composite_strength, resolved_signals, weights)
            
            # Step 7: Calculate risk assessment
            risk_assessment = self._calculate_risk_assessment(resolved_signals, market_context)
            
            # Step 8: Calculate signal contributions
            signal_contributions = self._calculate_signal_contributions(resolved_signals, weights)
            
            recommendation = Recommendation(
                action=action,
                strength=composite_strength,
                confidence=confidence,
                probability_estimates=probabilities,
                reasoning=reasoning,
                signal_contributions=signal_contributions,
                risk_assessment=risk_assessment
            )
            
            logger.info(f"Generated ensemble recommendation: {action} (strength={composite_strength:.3f}, confidence={confidence:.3f})")
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error generating ensemble recommendation: {e}")
            return self._generate_default_recommendation(f"Error in ensemble processing: {str(e)}")
    
    def _apply_advanced_ensemble_methods(self, signals: List[Signal], weights: Dict[str, float],
                                       market_context: MarketContext) -> float:
        """Apply advanced ensemble methods like stacking, boosting, and dynamic weighting"""
        
        # Calculate base weighted composite
        base_composite = self._calculate_weighted_composite(signals, weights)
        
        # Apply stacking ensemble
        stacking_contribution = self.stacking_ensemble.predict(signals, market_context)
        
        # Apply boosting ensemble
        boosting_contribution = self.boosting_ensemble.predict(signals, market_context)
        
        # Apply dynamic weighting based on market context
        dynamic_composite = self.dynamic_weighter.adjust_weights(weights, market_context)
        dynamic_contribution = self._calculate_weighted_composite(signals, dynamic_composite)
        
        # Combine contributions using learned weights
        # For now, use equal weights; in a full implementation, these would be learned
        final_composite = (0.4 * base_composite +
                          0.2 * stacking_contribution +
                          0.2 * boosting_contribution +
                          0.2 * dynamic_contribution)
        
        return final_composite
    
    def _resolve_signal_conflicts(self, signals: List[Signal]) -> List[Signal]:
        """Resolve conflicts within signal groups"""
        
        # Group signals by type and source
        signal_groups = {}
        for signal in signals:
            key = (signal.signal_type, signal.source)
            if key not in signal_groups:
                signal_groups[key] = []
            signal_groups[key].append(signal)
        
        resolved_signals = []
        
        for (signal_type, source), group_signals in signal_groups.items():
            if len(group_signals) == 1:
                resolved_signals.append(group_signals[0])
            else:
                # Resolve conflicts within the group
                resolved_signal = self.conflict_resolver.resolve_conflicts(group_signals)
                resolved_signals.append(resolved_signal)
        
        return resolved_signals
    
    def _calculate_weighted_composite(self, signals: List[Signal], 
                                    weights: Dict[str, float]) -> float:
        """Calculate weighted composite strength"""
        
        total_weighted_strength = 0.0
        total_weight = 0.0
        
        for signal in signals:
            weight = weights.get(signal.source, 0.0)
            total_weighted_strength += signal.strength * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return total_weighted_strength / total_weight
    
    def _determine_action(self, composite_strength: float, 
                         probabilities: Dict[str, float], 
                         confidence: float) -> str:
        """Determine final action based on composite strength and probabilities"""
        
        # Use probability-based decision with confidence threshold
        max_prob_action = max(probabilities, key=probabilities.get)
        max_prob = probabilities[max_prob_action]
        
        # Require minimum confidence for BUY/SELL decisions
        min_confidence_threshold = 0.6
        
        if max_prob > 0.5 and confidence > min_confidence_threshold:
            return max_prob_action
        elif abs(composite_strength) > 0.3 and confidence > 0.5:
            # Fallback to strength-based decision
            if composite_strength > 0.1:
                return "BUY"
            elif composite_strength < -0.1:
                return "SELL"
        
        return "HOLD"
    
    def _generate_reasoning(self, action: str, composite_strength: float,
                          signals: List[Signal], weights: Dict[str, float]) -> str:
        """Generate human-readable reasoning for the recommendation"""
        
        # Find top contributing signals
        signal_impacts = []
        for signal in signals:
            weight = weights.get(signal.source, 0.0)
            impact = abs(signal.strength * weight)
            signal_impacts.append((signal, impact))
        
        # Sort by impact
        signal_impacts.sort(key=lambda x: x[1], reverse=True)
        top_signals = signal_impacts[:3]  # Top 3 contributors
        
        reasoning_parts = [f"Recommendation: {action} (strength: {composite_strength:.3f})"]
        
        if top_signals:
            reasoning_parts.append("Key factors:")
            for signal, impact in top_signals:
                direction = "bullish" if signal.strength > 0 else "bearish"
                reasoning_parts.append(
                    f"- {signal.source} ({signal.signal_type.value}): {direction} "
                    f"signal (strength: {signal.strength:.3f}, impact: {impact:.3f})"
                )
        
        return " ".join(reasoning_parts)
    
    def _calculate_risk_assessment(self, signals: List[Signal], 
                                 market_context: MarketContext) -> Dict[str, float]:
        """Calculate risk metrics for the recommendation"""
        
        # Extract risk-related signals
        risk_signals = [s for s in signals if s.signal_type == SignalType.RISK]
        
        # Base risk metrics
        volatility_risk = {"low": 0.2, "medium": 0.5, "high": 0.8}.get(
            market_context.volatility_regime, 0.5
        )
        
        correlation_risk = {"low": 0.2, "medium": 0.5, "high": 0.8}.get(
            market_context.correlation_regime, 0.5
        )
        
        # Signal consensus risk (higher disagreement = higher risk)
        if len(signals) > 1:
            strengths = [s.strength for s in signals]
            consensus_risk = np.std(strengths)  # Standard deviation as risk measure
        else:
            consensus_risk = 0.5
        
        # Monte Carlo risk (if available)
        monte_carlo_signals = [s for s in signals if s.signal_type == SignalType.MONTE_CARLO]
        monte_carlo_risk = 0.5
        if monte_carlo_signals:
            # Assume Monte Carlo signal strength represents risk (negative = higher risk)
            mc_strength = np.mean([s.strength for s in monte_carlo_signals])
            monte_carlo_risk = max(0.0, 1.0 - mc_strength)  # Convert to risk measure
        
        return {
            'volatility_risk': volatility_risk,
            'correlation_risk': correlation_risk,
            'consensus_risk': min(consensus_risk, 1.0),
            'monte_carlo_risk': monte_carlo_risk,
            'overall_risk': np.mean([volatility_risk, correlation_risk, consensus_risk, monte_carlo_risk])
        }
    
    def _calculate_signal_contributions(self, signals: List[Signal], 
                                      weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate individual signal contributions to final recommendation"""
        
        contributions = {}
        
        for signal in signals:
            weight = weights.get(signal.source, 0.0)
            contribution = signal.strength * weight
            contributions[signal.source] = contribution
        
        return contributions
    
    def _generate_default_recommendation(self, reason: str) -> Recommendation:
        """Generate default HOLD recommendation when processing fails"""
        
        return Recommendation(
            action="HOLD",
            strength=0.0,
            confidence=0.1,
            probability_estimates={'BUY': 0.33, 'SELL': 0.33, 'HOLD': 0.34},
            reasoning=f"Default recommendation due to: {reason}",
            signal_contributions={},
            risk_assessment={'overall_risk': 0.5}
        )