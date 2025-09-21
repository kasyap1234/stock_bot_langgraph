"""
Unit tests for the Adaptive Signal Combiner component
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from recommendation.intelligent_ensemble import (
    AdaptiveSignalCombiner, Signal, SignalType, MarketContext
)


class TestAdaptiveSignalCombiner:
    
    def setup_method(self):
        """Setup test fixtures"""
        self.combiner = AdaptiveSignalCombiner()
        
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
    
    def test_extract_features_basic(self):
        """Test basic feature extraction"""
        features = self.combiner.extract_features(self.sample_signals, self.sample_context)
        
        # Should return 2D array with single row
        assert features.shape[0] == 1
        assert features.shape[1] > 0
        
        # Check that features are numeric
        assert np.all(np.isfinite(features))
    
    def test_extract_features_empty_signals(self):
        """Test feature extraction with empty signals"""
        features = self.combiner.extract_features([], self.sample_context)
        
        assert features.shape[0] == 1
        assert features.shape[1] > 0
        
        # Most features should be zero for empty signals
        non_zero_features = np.count_nonzero(features)
        assert non_zero_features <= 5  # Only market context features should be non-zero
    
    def test_calculate_dynamic_weights_untrained(self):
        """Test weight calculation when model is not trained"""
        weights = self.combiner.calculate_dynamic_weights(self.sample_signals, self.sample_context)
        
        # Should return weights for all signals
        assert len(weights) == len(self.sample_signals)
        
        # All weights should be positive and sum to 1
        assert all(w >= 0 for w in weights.values())
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        
        # Check that signal sources are used as keys
        expected_sources = {s.source for s in self.sample_signals}
        assert set(weights.keys()) == expected_sources
    
    def test_calculate_rule_based_weights(self):
        """Test rule-based weight calculation"""
        weights = self.combiner._calculate_rule_based_weights(self.sample_signals, self.sample_context)
        
        # Should return normalized weights
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        
        # Technical signals should have higher base weight
        technical_weight = weights.get("MACD", 0)
        fundamental_weight = weights.get("P/E_ratio", 0)
        assert technical_weight > 0
        assert fundamental_weight > 0
    
    def test_get_context_multiplier_high_volatility(self):
        """Test context multiplier in high volatility regime"""
        high_vol_context = MarketContext(
            volatility_regime="high",
            trend_regime="trending",
            market_sentiment="neutral",
            correlation_regime="medium",
            volume_regime="normal"
        )
        
        # Technical signal should get reduced multiplier
        technical_signal = Signal(
            signal_type=SignalType.TECHNICAL,
            strength=0.5,
            confidence=0.8,
            timestamp=pd.Timestamp.now(),
            source="RSI",
            metadata={}
        )
        
        multiplier = self.combiner._get_context_multiplier(technical_signal, high_vol_context)
        assert multiplier < 1.0  # Should be reduced
        
        # Risk signal should get increased multiplier
        risk_signal = Signal(
            signal_type=SignalType.RISK,
            strength=0.5,
            confidence=0.8,
            timestamp=pd.Timestamp.now(),
            source="VaR",
            metadata={}
        )
        
        risk_multiplier = self.combiner._get_context_multiplier(risk_signal, high_vol_context)
        assert risk_multiplier > 1.0  # Should be increased
    
    def test_get_context_multiplier_trending_market(self):
        """Test context multiplier in trending market"""
        trending_context = MarketContext(
            volatility_regime="low",
            trend_regime="trending",
            market_sentiment="neutral",
            correlation_regime="low",
            volume_regime="normal"
        )
        
        technical_signal = Signal(
            signal_type=SignalType.TECHNICAL,
            strength=0.5,
            confidence=0.8,
            timestamp=pd.Timestamp.now(),
            source="MACD",
            metadata={}
        )
        
        multiplier = self.combiner._get_context_multiplier(technical_signal, trending_context)
        
        # Should get boost from both low volatility and trending market
        assert multiplier > 1.0
    
    def test_train_model_insufficient_data(self):
        """Test model training with insufficient data"""
        # Try to train with very little data
        historical_signals = [self.sample_signals] * 10  # Only 10 samples
        historical_contexts = [self.sample_context] * 10
        historical_outcomes = [0.5] * 10
        
        self.combiner.train_model(historical_signals, historical_contexts, historical_outcomes)
        
        # Model should not be trained due to insufficient data
        assert not self.combiner.is_trained
    
    def test_train_model_sufficient_data(self):
        """Test model training with sufficient data"""
        # Create sufficient training data
        np.random.seed(42)  # For reproducible tests
        
        historical_signals = []
        historical_contexts = []
        historical_outcomes = []
        
        for i in range(100):  # 100 samples
            # Create varied signals
            signals = [
                Signal(
                    signal_type=SignalType.TECHNICAL,
                    strength=np.random.uniform(-1, 1),
                    confidence=np.random.uniform(0.5, 1.0),
                    timestamp=pd.Timestamp.now(),
                    source="MACD",
                    metadata={}
                ),
                Signal(
                    signal_type=SignalType.FUNDAMENTAL,
                    strength=np.random.uniform(-1, 1),
                    confidence=np.random.uniform(0.5, 1.0),
                    timestamp=pd.Timestamp.now(),
                    source="P/E_ratio",
                    metadata={}
                )
            ]
            
            # Create varied contexts
            context = MarketContext(
                volatility_regime=np.random.choice(["low", "medium", "high"]),
                trend_regime=np.random.choice(["ranging", "trending", "transitional"]),
                market_sentiment=np.random.choice(["bearish", "neutral", "bullish"]),
                correlation_regime=np.random.choice(["low", "medium", "high"]),
                volume_regime=np.random.choice(["low", "normal", "high"])
            )
            
            outcome = np.random.uniform(-1, 1)
            
            historical_signals.append(signals)
            historical_contexts.append(context)
            historical_outcomes.append(outcome)
        
        self.combiner.train_model(historical_signals, historical_contexts, historical_outcomes)
        
        # Model should be trained
        assert self.combiner.is_trained
        assert len(self.combiner.feature_importance) > 0
    
    def test_get_feature_names(self):
        """Test feature name generation"""
        feature_names = self.combiner._get_feature_names()
        
        # Should have names for all signal types
        assert len(feature_names) > 0
        
        # Should include signal type features
        assert any("technical" in name.lower() for name in feature_names)
        assert any("fundamental" in name.lower() for name in feature_names)
        
        # Should include market context features
        assert any("volatility" in name.lower() for name in feature_names)
        assert any("trend" in name.lower() for name in feature_names)
    
    def test_calculate_dynamic_weights_with_trained_model(self):
        """Test weight calculation with trained model"""
        # First train the model
        self.test_train_model_sufficient_data()
        
        # Now test weight calculation
        weights = self.combiner.calculate_dynamic_weights(self.sample_signals, self.sample_context)
        
        # Should return valid weights
        assert len(weights) == len(self.sample_signals)
        assert all(w >= 0 for w in weights.values())
        assert abs(sum(weights.values()) - 1.0) < 1e-6
    
    def test_signal_type_coverage(self):
        """Test that all signal types are handled in feature extraction"""
        # Create signals of all types
        all_type_signals = []
        for signal_type in SignalType:
            signal = Signal(
                signal_type=signal_type,
                strength=0.5,
                confidence=0.8,
                timestamp=pd.Timestamp.now(),
                source=f"test_{signal_type.value}",
                metadata={}
            )
            all_type_signals.append(signal)
        
        # Extract features
        features = self.combiner.extract_features(all_type_signals, self.sample_context)
        
        # Should handle all signal types without error
        assert features.shape[0] == 1
        assert np.all(np.isfinite(features))
    
    def test_weight_normalization(self):
        """Test that weights are properly normalized"""
        # Create signals with extreme confidence values
        extreme_signals = [
            Signal(
                signal_type=SignalType.TECHNICAL,
                strength=0.9,
                confidence=0.1,  # Very low confidence
                timestamp=pd.Timestamp.now(),
                source="low_conf_signal",
                metadata={}
            ),
            Signal(
                signal_type=SignalType.FUNDAMENTAL,
                strength=0.2,
                confidence=0.99,  # Very high confidence
                timestamp=pd.Timestamp.now(),
                source="high_conf_signal",
                metadata={}
            )
        ]
        
        weights = self.combiner.calculate_dynamic_weights(extreme_signals, self.sample_context)
        
        # Weights should still be normalized
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        
        # High confidence signal should have higher weight
        assert weights["high_conf_signal"] > weights["low_conf_signal"]
    
    def test_market_context_encoding(self):
        """Test that market context is properly encoded in features"""
        # Test different market contexts
        contexts = [
            MarketContext("low", "ranging", "bearish", "low", "low"),
            MarketContext("high", "trending", "bullish", "high", "high"),
            MarketContext("medium", "transitional", "neutral", "medium", "normal")
        ]
        
        features_list = []
        for context in contexts:
            features = self.combiner.extract_features(self.sample_signals, context)
            features_list.append(features)
        
        # Features should be different for different contexts
        assert not np.array_equal(features_list[0], features_list[1])
        assert not np.array_equal(features_list[1], features_list[2])
    
    @patch('recommendation.intelligent_ensemble.logger')
    def test_error_handling_in_ml_weights(self, mock_logger):
        """Test error handling when ML model fails"""
        # Force the model to be marked as trained but cause an error
        self.combiner.is_trained = True
        self.combiner.rf_model = None  # This will cause an error
        
        # Should fall back to rule-based weights
        weights = self.combiner.calculate_dynamic_weights(self.sample_signals, self.sample_context)
        
        # Should still return valid weights
        assert len(weights) == len(self.sample_signals)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        
        # Should log a warning
        mock_logger.warning.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])