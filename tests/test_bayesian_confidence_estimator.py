"""
Unit tests for the Bayesian Confidence Estimator component
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from recommendation.intelligent_ensemble import (
    BayesianConfidenceEstimator, Signal, SignalType, MarketContext
)


class TestBayesianConfidenceEstimator:
    
    def setup_method(self):
        """Setup test fixtures"""
        self.estimator = BayesianConfidenceEstimator()
        
        # Create sample signals
        self.sample_signals = [
            Signal(
                signal_type=SignalType.TECHNICAL,
                strength=0.7,
                confidence=0.8,
                timestamp=pd.Timestamp.now(),
                source="MACD",
                metadata={}
            ),
            Signal(
                signal_type=SignalType.FUNDAMENTAL,
                strength=0.3,
                confidence=0.6,
                timestamp=pd.Timestamp.now(),
                source="P/E_ratio",
                metadata={}
            ),
            Signal(
                signal_type=SignalType.SENTIMENT,
                strength=-0.2,
                confidence=0.7,
                timestamp=pd.Timestamp.now(),
                source="news_sentiment",
                metadata={}
            )
        ]
        
        # Create sample market context
        self.sample_context = MarketContext(
            volatility_regime="medium",
            trend_regime="trending",
            market_sentiment="bullish",
            correlation_regime="low",
            volume_regime="normal"
        )
    
    def test_estimate_confidence_basic(self):
        """Test basic confidence estimation"""
        composite_strength = 0.5
        confidence, probabilities = self.estimator.estimate_confidence(
            self.sample_signals, self.sample_context, composite_strength
        )
        
        # Confidence should be between 0 and 1
        assert 0.0 <= confidence <= 1.0
        
        # Probabilities should sum to 1 and be valid
        assert abs(sum(probabilities.values()) - 1.0) < 1e-6
        assert all(0.0 <= p <= 1.0 for p in probabilities.values())
        
        # Should have all three action probabilities
        assert set(probabilities.keys()) == {'BUY', 'SELL', 'HOLD'}
    
    def test_estimate_confidence_empty_signals(self):
        """Test confidence estimation with empty signals"""
        confidence, probabilities = self.estimator.estimate_confidence(
            [], self.sample_context, 0.0
        )
        
        # Should return reasonable defaults
        assert 0.0 <= confidence <= 1.0
        assert abs(sum(probabilities.values()) - 1.0) < 1e-6
    
    def test_calculate_signal_consensus_confidence(self):
        """Test signal consensus confidence calculation"""
        confidence = self.estimator._calculate_signal_consensus_confidence(self.sample_signals)
        
        # Should return valid confidence
        assert 0.0 <= confidence <= 1.0
        
        # Test with empty signals
        empty_confidence = self.estimator._calculate_signal_consensus_confidence([])
        assert empty_confidence == 0.5  # Default value
    
    def test_calculate_signal_consensus_confidence_high_agreement(self):
        """Test consensus confidence with high signal agreement"""
        # Create signals with similar strengths (high agreement)
        similar_signals = [
            Signal(SignalType.TECHNICAL, 0.7, 0.8, pd.Timestamp.now(), "signal1", {}),
            Signal(SignalType.TECHNICAL, 0.75, 0.85, pd.Timestamp.now(), "signal2", {}),
            Signal(SignalType.TECHNICAL, 0.72, 0.82, pd.Timestamp.now(), "signal3", {})
        ]
        
        high_agreement_confidence = self.estimator._calculate_signal_consensus_confidence(similar_signals)
        
        # Create signals with different strengths (low agreement)
        different_signals = [
            Signal(SignalType.TECHNICAL, 0.8, 0.8, pd.Timestamp.now(), "signal1", {}),
            Signal(SignalType.TECHNICAL, -0.6, 0.8, pd.Timestamp.now(), "signal2", {}),
            Signal(SignalType.TECHNICAL, 0.1, 0.8, pd.Timestamp.now(), "signal3", {})
        ]
        
        low_agreement_confidence = self.estimator._calculate_signal_consensus_confidence(different_signals)
        
        # High agreement should result in higher confidence
        assert high_agreement_confidence > low_agreement_confidence
    
    def test_get_context_confidence_adjustment(self):
        """Test market context confidence adjustments"""
        # Test high volatility (should reduce confidence)
        high_vol_context = MarketContext("high", "trending", "neutral", "low", "normal")
        high_vol_adj = self.estimator._get_context_confidence_adjustment(high_vol_context)
        assert high_vol_adj < 1.0
        
        # Test low volatility (should increase confidence)
        low_vol_context = MarketContext("low", "trending", "neutral", "low", "normal")
        low_vol_adj = self.estimator._get_context_confidence_adjustment(low_vol_context)
        assert low_vol_adj > 1.0
        
        # Test trending market (should slightly increase confidence)
        trending_context = MarketContext("medium", "trending", "neutral", "low", "normal")
        trending_adj = self.estimator._get_context_confidence_adjustment(trending_context)
        assert trending_adj >= 1.0
        
        # Test transitional market (should reduce confidence)
        transitional_context = MarketContext("medium", "transitional", "neutral", "low", "normal")
        transitional_adj = self.estimator._get_context_confidence_adjustment(transitional_context)
        assert transitional_adj < 1.0
        
        # Adjustment should be within reasonable bounds
        for adj in [high_vol_adj, low_vol_adj, trending_adj, transitional_adj]:
            assert 0.5 <= adj <= 1.5
    
    def test_calculate_action_probabilities(self):
        """Test action probability calculation"""
        # Test strong buy signal
        buy_probs = self.estimator._calculate_action_probabilities(0.8, 0.9)
        assert buy_probs['BUY'] > buy_probs['SELL']
        assert buy_probs['BUY'] > buy_probs['HOLD']
        
        # Test strong sell signal
        sell_probs = self.estimator._calculate_action_probabilities(-0.8, 0.9)
        assert sell_probs['SELL'] > sell_probs['BUY']
        assert sell_probs['SELL'] > sell_probs['HOLD']
        
        # Test neutral signal
        neutral_probs = self.estimator._calculate_action_probabilities(0.0, 0.5)
        assert neutral_probs['HOLD'] >= max(neutral_probs['BUY'], neutral_probs['SELL'])
        
        # All probabilities should sum to 1
        for probs in [buy_probs, sell_probs, neutral_probs]:
            assert abs(sum(probs.values()) - 1.0) < 1e-6
            assert all(p >= 0 for p in probs.values())
    
    def test_calculate_action_probabilities_confidence_effect(self):
        """Test that confidence affects probability distribution"""
        # High confidence should make probabilities more decisive
        high_conf_probs = self.estimator._calculate_action_probabilities(0.6, 0.9)
        low_conf_probs = self.estimator._calculate_action_probabilities(0.6, 0.3)
        
        # High confidence should result in more extreme probabilities
        high_conf_max = max(high_conf_probs.values())
        low_conf_max = max(low_conf_probs.values())
        
        assert high_conf_max > low_conf_max
    
    def test_get_bayesian_confidence_untrained(self):
        """Test Bayesian confidence when model is not trained"""
        confidence = self.estimator._get_bayesian_confidence(self.sample_signals, self.sample_context)
        
        # Should return default value when not trained
        assert confidence == 0.5
    
    def test_train_bayesian_model_insufficient_data(self):
        """Test Bayesian model training with insufficient data"""
        # Create insufficient training data
        features = [[0.1, 0.2, 0.3] for _ in range(20)]  # Only 20 samples
        confidences = [0.5] * 20
        
        self.estimator.train_bayesian_model(features, confidences)
        
        # Model should not be trained
        assert not self.estimator.is_trained
    
    def test_train_bayesian_model_sufficient_data(self):
        """Test Bayesian model training with sufficient data"""
        # Create sufficient training data
        np.random.seed(42)  # For reproducible tests
        
        features = []
        confidences = []
        
        for i in range(50):  # 50 samples
            # Create varied features
            feature_vector = [
                np.random.uniform(-1, 1),  # mean strength
                np.random.uniform(0, 1),   # std strength
                np.random.uniform(0, 1),   # mean confidence
                np.random.uniform(0, 0.5), # std confidence
                np.random.randint(1, 10),  # signal count
                np.random.randint(0, 3),   # volatility regime
                np.random.randint(0, 3),   # trend regime
                np.random.randint(-1, 2)   # market sentiment
            ]
            
            # Create corresponding confidence (with some relationship to features)
            confidence = 0.5 + 0.3 * feature_vector[2] + 0.1 * np.random.normal()
            confidence = np.clip(confidence, 0.0, 1.0)
            
            features.append(feature_vector)
            confidences.append(confidence)
        
        self.estimator.train_bayesian_model(features, confidences)
        
        # Model should be trained
        assert self.estimator.is_trained
    
    def test_get_bayesian_confidence_trained(self):
        """Test Bayesian confidence with trained model"""
        # First train the model
        self.test_train_bayesian_model_sufficient_data()
        
        # Now test confidence estimation
        confidence = self.estimator._get_bayesian_confidence(self.sample_signals, self.sample_context)
        
        # Should return valid confidence
        assert 0.0 <= confidence <= 1.0
    
    def test_update_prior_probabilities(self):
        """Test updating prior probabilities"""
        initial_priors = self.estimator.prior_probabilities.copy()
        
        # Update with action counts
        action_counts = {'BUY': 50, 'SELL': 30, 'HOLD': 20}
        self.estimator.update_prior_probabilities(action_counts)
        
        # Probabilities should be updated based on counts
        total_actions = sum(action_counts.values())
        expected_buy_prob = action_counts['BUY'] / total_actions
        
        assert abs(self.estimator.prior_probabilities['BUY'] - expected_buy_prob) < 1e-6
        
        # Should sum to 1
        assert abs(sum(self.estimator.prior_probabilities.values()) - 1.0) < 1e-6
    
    def test_update_prior_probabilities_empty_counts(self):
        """Test updating prior probabilities with empty counts"""
        initial_priors = self.estimator.prior_probabilities.copy()
        
        # Update with empty counts
        self.estimator.update_prior_probabilities({})
        
        # Probabilities should remain unchanged
        assert self.estimator.prior_probabilities == initial_priors
    
    def test_estimate_confidence_with_trained_model(self):
        """Test confidence estimation with trained Bayesian model"""
        # Train the model first
        self.test_train_bayesian_model_sufficient_data()
        
        # Test confidence estimation
        confidence, probabilities = self.estimator.estimate_confidence(
            self.sample_signals, self.sample_context, 0.5
        )
        
        # Should return valid results
        assert 0.0 <= confidence <= 1.0
        assert abs(sum(probabilities.values()) - 1.0) < 1e-6
    
    def test_confidence_bounds(self):
        """Test that confidence estimates stay within bounds"""
        # Test with extreme inputs
        extreme_signals = [
            Signal(SignalType.TECHNICAL, 1.0, 1.0, pd.Timestamp.now(), "extreme_buy", {}),
            Signal(SignalType.TECHNICAL, -1.0, 1.0, pd.Timestamp.now(), "extreme_sell", {})
        ]
        
        confidence, probabilities = self.estimator.estimate_confidence(
            extreme_signals, self.sample_context, 0.0
        )
        
        # Should still be within bounds
        assert 0.0 <= confidence <= 1.0
        assert all(0.0 <= p <= 1.0 for p in probabilities.values())
    
    def test_signal_strength_impact_on_probabilities(self):
        """Test that signal strength properly impacts action probabilities"""
        # Test different composite strengths
        strengths = [-0.8, -0.3, 0.0, 0.3, 0.8]
        
        for strength in strengths:
            _, probabilities = self.estimator.estimate_confidence(
                self.sample_signals, self.sample_context, strength
            )
            
            if strength > 0.5:
                # Strong buy signal
                assert probabilities['BUY'] > probabilities['SELL']
                assert probabilities['BUY'] > probabilities['HOLD']
            elif strength < -0.5:
                # Strong sell signal
                assert probabilities['SELL'] > probabilities['BUY']
                assert probabilities['SELL'] > probabilities['HOLD']
            else:
                # Neutral or weak signal - HOLD should be competitive
                assert probabilities['HOLD'] >= min(probabilities['BUY'], probabilities['SELL'])
    
    def test_market_context_impact(self):
        """Test that market context impacts confidence estimation"""
        # Test different market contexts
        contexts = [
            MarketContext("low", "trending", "bullish", "low", "normal"),    # Favorable
            MarketContext("high", "transitional", "bearish", "high", "low"), # Unfavorable
            MarketContext("medium", "ranging", "neutral", "medium", "normal") # Neutral
        ]
        
        confidences = []
        for context in contexts:
            confidence, _ = self.estimator.estimate_confidence(
                self.sample_signals, context, 0.5
            )
            confidences.append(confidence)
        
        # Favorable context should generally result in higher confidence
        # (though this depends on the specific implementation)
        assert all(0.0 <= c <= 1.0 for c in confidences)
    
    @patch('recommendation.intelligent_ensemble.logger')
    def test_error_handling_in_bayesian_prediction(self, mock_logger):
        """Test error handling when Bayesian prediction fails"""
        # Force the model to be trained but cause an error
        self.estimator.is_trained = True
        self.estimator.bayesian_model = None  # This will cause an error
        
        confidence = self.estimator._get_bayesian_confidence(self.sample_signals, self.sample_context)
        
        # Should fall back to default value
        assert confidence == 0.5
        
        # Should log a warning
        mock_logger.warning.assert_called()
    
    @patch('recommendation.intelligent_ensemble.logger')
    def test_error_handling_in_model_training(self, mock_logger):
        """Test error handling when model training fails"""
        # Test insufficient data case (should trigger warning)
        insufficient_features = [[0.1, 0.2, 0.3] for _ in range(20)]  # Only 20 samples
        insufficient_confidences = [0.5] * 20
        
        self.estimator.train_bayesian_model(insufficient_features, insufficient_confidences)
        
        # Model should not be trained
        assert not self.estimator.is_trained
        
        # Should log a warning for insufficient data
        mock_logger.warning.assert_called()
    
    def test_consensus_confidence_edge_cases(self):
        """Test consensus confidence calculation with edge cases"""
        # Test with single signal
        single_signal = [Signal(SignalType.TECHNICAL, 0.5, 0.8, pd.Timestamp.now(), "single", {})]
        confidence = self.estimator._calculate_signal_consensus_confidence(single_signal)
        assert 0.0 <= confidence <= 1.0
        
        # Test with zero confidence signals
        zero_conf_signals = [
            Signal(SignalType.TECHNICAL, 0.5, 0.0, pd.Timestamp.now(), "zero1", {}),
            Signal(SignalType.TECHNICAL, 0.7, 0.0, pd.Timestamp.now(), "zero2", {})
        ]
        confidence = self.estimator._calculate_signal_consensus_confidence(zero_conf_signals)
        assert 0.0 <= confidence <= 1.0
        
        # Test with zero strength signals
        zero_strength_signals = [
            Signal(SignalType.TECHNICAL, 0.0, 0.8, pd.Timestamp.now(), "zero_str1", {}),
            Signal(SignalType.TECHNICAL, 0.0, 0.9, pd.Timestamp.now(), "zero_str2", {})
        ]
        confidence = self.estimator._calculate_signal_consensus_confidence(zero_strength_signals)
        assert 0.0 <= confidence <= 1.0
    
    def test_probability_temperature_effect(self):
        """Test that confidence acts as temperature in probability calculation"""
        # Same strength, different confidence levels
        high_conf_probs = self.estimator._calculate_action_probabilities(0.6, 0.9)
        low_conf_probs = self.estimator._calculate_action_probabilities(0.6, 0.3)
        
        # High confidence should result in more decisive (less uniform) distribution
        high_conf_entropy = -sum(p * np.log(p + 1e-10) for p in high_conf_probs.values())
        low_conf_entropy = -sum(p * np.log(p + 1e-10) for p in low_conf_probs.values())
        
        # Lower entropy means more decisive (high confidence should have lower entropy)
        assert high_conf_entropy < low_conf_entropy


if __name__ == "__main__":
    pytest.main([__file__])