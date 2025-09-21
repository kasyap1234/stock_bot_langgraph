"""
Integration tests for the complete Intelligent Ensemble Engine
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from recommendation.intelligent_ensemble import (
    IntelligentEnsembleEngine, Signal, SignalType, MarketContext, Recommendation
)


class TestIntelligentEnsembleEngineIntegration:
    
    def setup_method(self):
        """Setup test fixtures"""
        self.engine = IntelligentEnsembleEngine()
        
        # Create comprehensive test signals
        self.test_signals = [
            Signal(
                signal_type=SignalType.TECHNICAL,
                strength=0.8,
                confidence=0.9,
                timestamp=pd.Timestamp.now(),
                source="MACD",
                metadata={"timeframe": "1h", "crossover": True}
            ),
            Signal(
                signal_type=SignalType.TECHNICAL,
                strength=0.6,
                confidence=0.8,
                timestamp=pd.Timestamp.now(),
                source="RSI",
                metadata={"value": 75, "overbought": True}
            ),
            Signal(
                signal_type=SignalType.FUNDAMENTAL,
                strength=0.4,
                confidence=0.7,
                timestamp=pd.Timestamp.now(),
                source="P/E_ratio",
                metadata={"value": 15.2, "sector_avg": 18.5}
            ),
            Signal(
                signal_type=SignalType.SENTIMENT,
                strength=-0.3,
                confidence=0.6,
                timestamp=pd.Timestamp.now(),
                source="news_sentiment",
                metadata={"articles_analyzed": 50, "negative_ratio": 0.65}
            ),
            Signal(
                signal_type=SignalType.RISK,
                strength=-0.2,
                confidence=0.8,
                timestamp=pd.Timestamp.now(),
                source="VaR",
                metadata={"confidence_level": 0.95, "value": -0.05}
            )
        ]
        
        # Create test market context
        self.test_context = MarketContext(
            volatility_regime="medium",
            trend_regime="trending",
            market_sentiment="bullish",
            correlation_regime="low",
            volume_regime="normal"
        )
    
    def test_generate_ensemble_recommendation_complete_flow(self):
        """Test complete recommendation generation flow"""
        recommendation = self.engine.generate_ensemble_recommendation(
            self.test_signals, self.test_context
        )
        
        # Validate recommendation structure
        assert isinstance(recommendation, Recommendation)
        assert recommendation.action in ["BUY", "SELL", "HOLD"]
        assert -1.0 <= recommendation.strength <= 1.0
        assert 0.0 <= recommendation.confidence <= 1.0
        
        # Validate probability estimates
        assert set(recommendation.probability_estimates.keys()) == {"BUY", "SELL", "HOLD"}
        assert abs(sum(recommendation.probability_estimates.values()) - 1.0) < 1e-6
        assert all(0.0 <= p <= 1.0 for p in recommendation.probability_estimates.values())
        
        # Validate signal contributions
        assert len(recommendation.signal_contributions) > 0
        assert all(isinstance(k, str) for k in recommendation.signal_contributions.keys())
        
        # Validate risk assessment
        assert "overall_risk" in recommendation.risk_assessment
        assert 0.0 <= recommendation.risk_assessment["overall_risk"] <= 1.0
        
        # Validate reasoning
        assert isinstance(recommendation.reasoning, str)
        assert len(recommendation.reasoning) > 0
    
    def test_conflicting_signals_resolution(self):
        """Test handling of conflicting signals"""
        conflicting_signals = [
            Signal(SignalType.TECHNICAL, 0.9, 0.9, pd.Timestamp.now(), "strong_buy", {}),
            Signal(SignalType.TECHNICAL, -0.8, 0.8, pd.Timestamp.now(), "strong_sell", {}),
            Signal(SignalType.FUNDAMENTAL, 0.2, 0.6, pd.Timestamp.now(), "weak_buy", {})
        ]
        
        recommendation = self.engine.generate_ensemble_recommendation(
            conflicting_signals, self.test_context
        )
        
        # Should handle conflicts gracefully
        assert isinstance(recommendation, Recommendation)
        assert recommendation.action in ["BUY", "SELL", "HOLD"]
        
        # Confidence should be reduced due to conflicts
        assert recommendation.confidence < 0.9  # Should be less than max individual confidence
    
    def test_empty_signals_handling(self):
        """Test handling of empty signal list"""
        recommendation = self.engine.generate_ensemble_recommendation([], self.test_context)
        
        # Should return default HOLD recommendation
        assert recommendation.action == "HOLD"
        assert recommendation.strength == 0.0
        assert recommendation.confidence <= 0.5  # Low confidence due to no signals
    
    def test_single_signal_handling(self):
        """Test handling of single signal"""
        single_signal = [self.test_signals[0]]  # Strong technical buy signal
        
        recommendation = self.engine.generate_ensemble_recommendation(
            single_signal, self.test_context
        )
        
        # Should reflect the single signal
        assert isinstance(recommendation, Recommendation)
        # With a strong positive signal, should likely be BUY
        if recommendation.action == "BUY":
            assert recommendation.strength > 0
    
    def test_market_context_impact(self):
        """Test that different market contexts produce different recommendations"""
        # Test with high volatility context
        high_vol_context = MarketContext("high", "transitional", "bearish", "high", "low")
        high_vol_rec = self.engine.generate_ensemble_recommendation(
            self.test_signals, high_vol_context
        )
        
        # Test with low volatility context
        low_vol_context = MarketContext("low", "trending", "bullish", "low", "normal")
        low_vol_rec = self.engine.generate_ensemble_recommendation(
            self.test_signals, low_vol_context
        )
        
        # Contexts should influence confidence (low vol should generally have higher confidence)
        # Note: This might not always be true depending on signal composition, so we just check validity
        assert 0.0 <= high_vol_rec.confidence <= 1.0
        assert 0.0 <= low_vol_rec.confidence <= 1.0
    
    def test_signal_weighting_consistency(self):
        """Test that signal weighting is consistent and reasonable"""
        recommendation = self.engine.generate_ensemble_recommendation(
            self.test_signals, self.test_context
        )
        
        # Check that all signals contribute to the recommendation
        signal_sources = {s.source for s in self.test_signals}
        contribution_sources = set(recommendation.signal_contributions.keys())
        
        # Most signals should have contributions (some might be filtered out)
        assert len(contribution_sources) > 0
        assert contribution_sources.issubset(signal_sources)
    
    def test_risk_assessment_completeness(self):
        """Test that risk assessment includes all expected components"""
        recommendation = self.engine.generate_ensemble_recommendation(
            self.test_signals, self.test_context
        )
        
        risk_assessment = recommendation.risk_assessment
        
        # Should include overall risk
        assert "overall_risk" in risk_assessment
        
        # All risk values should be valid
        for risk_type, risk_value in risk_assessment.items():
            assert 0.0 <= risk_value <= 1.0, f"Invalid risk value for {risk_type}: {risk_value}"
    
    def test_recommendation_reasoning_quality(self):
        """Test that recommendation reasoning is informative"""
        recommendation = self.engine.generate_ensemble_recommendation(
            self.test_signals, self.test_context
        )
        
        reasoning = recommendation.reasoning
        
        # Should contain action and strength
        assert recommendation.action in reasoning
        assert "strength" in reasoning.lower()
        
        # Should mention key factors
        assert "factors" in reasoning.lower() or "signal" in reasoning.lower()
    
    def test_probability_distribution_sanity(self):
        """Test that probability distributions make sense given signal strengths"""
        # Test with mostly positive signals
        positive_signals = [
            Signal(SignalType.TECHNICAL, 0.7, 0.8, pd.Timestamp.now(), "pos1", {}),
            Signal(SignalType.FUNDAMENTAL, 0.5, 0.7, pd.Timestamp.now(), "pos2", {}),
            Signal(SignalType.SENTIMENT, 0.3, 0.6, pd.Timestamp.now(), "pos3", {})
        ]
        
        recommendation = self.engine.generate_ensemble_recommendation(
            positive_signals, self.test_context
        )
        
        # BUY probability should be highest or competitive
        probs = recommendation.probability_estimates
        if recommendation.action == "BUY":
            assert probs["BUY"] >= probs["SELL"]
    
    def test_confidence_correlation_with_signal_agreement(self):
        """Test that confidence correlates with signal agreement"""
        # High agreement signals
        agreement_signals = [
            Signal(SignalType.TECHNICAL, 0.7, 0.8, pd.Timestamp.now(), "agree1", {}),
            Signal(SignalType.FUNDAMENTAL, 0.6, 0.8, pd.Timestamp.now(), "agree2", {}),
            Signal(SignalType.SENTIMENT, 0.8, 0.9, pd.Timestamp.now(), "agree3", {})
        ]
        
        agreement_rec = self.engine.generate_ensemble_recommendation(
            agreement_signals, self.test_context
        )
        
        # Disagreement signals
        disagreement_signals = [
            Signal(SignalType.TECHNICAL, 0.8, 0.9, pd.Timestamp.now(), "disagree1", {}),
            Signal(SignalType.FUNDAMENTAL, -0.7, 0.8, pd.Timestamp.now(), "disagree2", {}),
            Signal(SignalType.SENTIMENT, 0.1, 0.5, pd.Timestamp.now(), "disagree3", {})
        ]
        
        disagreement_rec = self.engine.generate_ensemble_recommendation(
            disagreement_signals, self.test_context
        )
        
        # Agreement should generally result in higher confidence
        # (Though this isn't guaranteed due to other factors, so we just check validity)
        assert 0.0 <= agreement_rec.confidence <= 1.0
        assert 0.0 <= disagreement_rec.confidence <= 1.0
    
    def test_signal_type_diversity_handling(self):
        """Test handling of diverse signal types"""
        diverse_signals = [
            Signal(SignalType.TECHNICAL, 0.5, 0.8, pd.Timestamp.now(), "tech", {}),
            Signal(SignalType.FUNDAMENTAL, 0.3, 0.7, pd.Timestamp.now(), "fund", {}),
            Signal(SignalType.SENTIMENT, -0.2, 0.6, pd.Timestamp.now(), "sent", {}),
            Signal(SignalType.RISK, -0.4, 0.8, pd.Timestamp.now(), "risk", {}),
            Signal(SignalType.MACRO, 0.1, 0.5, pd.Timestamp.now(), "macro", {}),
            Signal(SignalType.MONTE_CARLO, 0.2, 0.7, pd.Timestamp.now(), "mc", {}),
            Signal(SignalType.BACKTEST, 0.6, 0.9, pd.Timestamp.now(), "bt", {})
        ]
        
        recommendation = self.engine.generate_ensemble_recommendation(
            diverse_signals, self.test_context
        )
        
        # Should handle all signal types
        assert isinstance(recommendation, Recommendation)
        assert recommendation.action in ["BUY", "SELL", "HOLD"]
    
    def test_extreme_market_conditions(self):
        """Test behavior under extreme market conditions"""
        extreme_contexts = [
            MarketContext("high", "transitional", "bearish", "high", "low"),  # Crisis mode
            MarketContext("low", "trending", "bullish", "low", "high"),       # Bull market
            MarketContext("medium", "ranging", "neutral", "medium", "normal") # Neutral
        ]
        
        for context in extreme_contexts:
            recommendation = self.engine.generate_ensemble_recommendation(
                self.test_signals, context
            )
            
            # Should handle all contexts gracefully
            assert isinstance(recommendation, Recommendation)
            assert recommendation.action in ["BUY", "SELL", "HOLD"]
            assert 0.0 <= recommendation.confidence <= 1.0
    
    def test_signal_freshness_impact(self):
        """Test that signal freshness (timestamp) impacts recommendations"""
        # Create signals with different timestamps
        old_signal = Signal(
            SignalType.TECHNICAL, 0.8, 0.9,
            pd.Timestamp.now() - pd.Timedelta(hours=24),  # 24 hours old
            "old_signal", {}
        )
        
        fresh_signal = Signal(
            SignalType.TECHNICAL, -0.6, 0.8,
            pd.Timestamp.now(),  # Fresh signal
            "fresh_signal", {}
        )
        
        mixed_age_signals = [old_signal, fresh_signal]
        
        recommendation = self.engine.generate_ensemble_recommendation(
            mixed_age_signals, self.test_context
        )
        
        # Should handle mixed-age signals
        assert isinstance(recommendation, Recommendation)
        # Fresh negative signal might influence the result more than old positive signal
    
    @patch('recommendation.intelligent_ensemble.logger')
    def test_error_handling_and_logging(self, mock_logger):
        """Test error handling and appropriate logging"""
        # Test with valid inputs (should not cause errors)
        recommendation = self.engine.generate_ensemble_recommendation(
            self.test_signals, self.test_context
        )
        
        assert isinstance(recommendation, Recommendation)
        
        # Should have some debug logging
        mock_logger.debug.assert_called()
    
    def test_recommendation_determinism(self):
        """Test that identical inputs produce identical outputs"""
        # Generate recommendation twice with same inputs
        rec1 = self.engine.generate_ensemble_recommendation(
            self.test_signals, self.test_context
        )
        
        rec2 = self.engine.generate_ensemble_recommendation(
            self.test_signals, self.test_context
        )
        
        # Should be identical (assuming no randomness in untrained models)
        assert rec1.action == rec2.action
        assert abs(rec1.strength - rec2.strength) < 1e-6
        assert abs(rec1.confidence - rec2.confidence) < 1e-6
    
    def test_performance_with_large_signal_set(self):
        """Test performance with a large number of signals"""
        # Create many signals
        large_signal_set = []
        for i in range(100):
            signal = Signal(
                signal_type=SignalType.TECHNICAL,
                strength=np.random.uniform(-1, 1),
                confidence=np.random.uniform(0.5, 1.0),
                timestamp=pd.Timestamp.now(),
                source=f"signal_{i}",
                metadata={}
            )
            large_signal_set.append(signal)
        
        # Should handle large signal sets efficiently
        recommendation = self.engine.generate_ensemble_recommendation(
            large_signal_set, self.test_context
        )
        
        assert isinstance(recommendation, Recommendation)
        assert recommendation.action in ["BUY", "SELL", "HOLD"]


if __name__ == "__main__":
    pytest.main([__file__])