"""
Unit tests for the Signal Conflict Resolver component
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from recommendation.intelligent_ensemble import (
    SignalConflictResolver, Signal, SignalType, MarketContext
)


class TestSignalConflictResolver:
    
    def setup_method(self):
        """Setup test fixtures"""
        self.resolver = SignalConflictResolver()
        
        # Create conflicting signals for testing
        self.conflicting_signals = [
            Signal(
                signal_type=SignalType.TECHNICAL,
                strength=0.8,  # Strong buy
                confidence=0.9,
                timestamp=pd.Timestamp.now(),
                source="MACD",
                metadata={}
            ),
            Signal(
                signal_type=SignalType.TECHNICAL,
                strength=-0.7,  # Strong sell
                confidence=0.8,
                timestamp=pd.Timestamp.now(),
                source="RSI",
                metadata={}
            ),
            Signal(
                signal_type=SignalType.FUNDAMENTAL,
                strength=0.3,  # Weak buy
                confidence=0.6,
                timestamp=pd.Timestamp.now(),
                source="P/E_ratio",
                metadata={}
            )
        ]
        
        # Create non-conflicting signals
        self.non_conflicting_signals = [
            Signal(
                signal_type=SignalType.TECHNICAL,
                strength=0.7,
                confidence=0.8,
                timestamp=pd.Timestamp.now(),
                source="MACD",
                metadata={}
            ),
            Signal(
                signal_type=SignalType.TECHNICAL,
                strength=0.5,
                confidence=0.7,
                timestamp=pd.Timestamp.now(),
                source="RSI",
                metadata={}
            )
        ]
    
    def test_resolve_conflicts_empty_list(self):
        """Test conflict resolution with empty signal list"""
        with pytest.raises(ValueError, match="No signals provided"):
            self.resolver.resolve_conflicts([])
    
    def test_resolve_conflicts_single_signal(self):
        """Test conflict resolution with single signal"""
        single_signal = [self.conflicting_signals[0]]
        resolved = self.resolver.resolve_conflicts(single_signal)
        
        # Should return the same signal
        assert resolved.strength == single_signal[0].strength
        assert resolved.confidence == single_signal[0].confidence
        assert resolved.source == single_signal[0].source
    
    def test_resolve_conflicts_no_actual_conflict(self):
        """Test conflict resolution when signals don't actually conflict"""
        resolved = self.resolver.resolve_conflicts(self.non_conflicting_signals)
        
        # Should return the strongest signal
        expected_strongest = max(self.non_conflicting_signals, key=lambda s: abs(s.strength))
        assert resolved.strength == expected_strongest.strength
    
    def test_are_conflicting_true(self):
        """Test conflict detection with actually conflicting signals"""
        strengths = [0.8, -0.7, 0.3]  # Mix of strong buy, strong sell, weak buy
        assert self.resolver._are_conflicting(strengths)
    
    def test_are_conflicting_false(self):
        """Test conflict detection with non-conflicting signals"""
        strengths = [0.7, 0.5, 0.3]  # All positive
        assert not self.resolver._are_conflicting(strengths)
        
        strengths = [-0.7, -0.5, -0.3]  # All negative
        assert not self.resolver._are_conflicting(strengths)
        
        strengths = [0.1, -0.1, 0.05]  # All weak signals
        assert not self.resolver._are_conflicting(strengths)
    
    def test_resolve_by_confidence(self):
        """Test confidence-weighted conflict resolution"""
        resolved = self.resolver._resolve_by_confidence(self.conflicting_signals)
        
        # Should be a valid signal
        assert isinstance(resolved, Signal)
        assert -1.0 <= resolved.strength <= 1.0
        assert 0.0 <= resolved.confidence <= 1.0
        assert resolved.source == "conflict_resolved"
        
        # Confidence should be reduced due to conflict
        max_original_confidence = max(s.confidence for s in self.conflicting_signals)
        assert resolved.confidence < max_original_confidence
        
        # Should have metadata about resolution
        assert resolved.metadata['resolution_method'] == 'confidence_weighted'
        assert resolved.metadata['original_count'] == len(self.conflicting_signals)
    
    def test_resolve_by_recency(self):
        """Test recency-weighted conflict resolution"""
        # Create signals with different timestamps
        old_signal = Signal(
            signal_type=SignalType.TECHNICAL,
            strength=0.8,
            confidence=0.9,
            timestamp=pd.Timestamp.now() - pd.Timedelta(hours=12),
            source="old_signal",
            metadata={}
        )
        
        recent_signal = Signal(
            signal_type=SignalType.TECHNICAL,
            strength=-0.6,
            confidence=0.7,
            timestamp=pd.Timestamp.now(),
            source="recent_signal",
            metadata={}
        )
        
        signals_with_timestamps = [old_signal, recent_signal]
        resolved = self.resolver._resolve_by_recency(signals_with_timestamps)
        
        # Should favor more recent signal
        assert resolved.source == "recency_resolved"
        assert resolved.metadata['resolution_method'] == 'recency_weighted'
        
        # Strength should be closer to recent signal due to higher weight
        # (Recent signal has negative strength, so resolved should be negative or close to 0)
        assert resolved.strength <= 0.2  # Should be influenced by recent negative signal
    
    def test_resolve_by_source_reliability(self):
        """Test source reliability-weighted conflict resolution"""
        # Create signals with different reliability scores
        high_reliability_signal = Signal(
            signal_type=SignalType.TECHNICAL,
            strength=0.7,
            confidence=0.8,
            timestamp=pd.Timestamp.now(),
            source="MACD",  # High reliability in default settings
            metadata={}
        )
        
        low_reliability_signal = Signal(
            signal_type=SignalType.SENTIMENT,
            strength=-0.6,
            confidence=0.7,
            timestamp=pd.Timestamp.now(),
            source="unknown_source",  # Low reliability (not in default dict)
            metadata={}
        )
        
        signals = [high_reliability_signal, low_reliability_signal]
        resolved = self.resolver._resolve_by_source_reliability(signals)
        
        # Should favor high reliability source
        assert resolved.source == "reliability_resolved"
        assert resolved.metadata['resolution_method'] == 'source_reliability'
        
        # Should be closer to high reliability signal (positive)
        assert resolved.strength > 0
    
    def test_resolve_by_consensus(self):
        """Test consensus-based conflict resolution"""
        # Create signals where majority agrees
        consensus_signals = [
            Signal(SignalType.TECHNICAL, 0.8, 0.9, pd.Timestamp.now(), "signal1", {}),
            Signal(SignalType.TECHNICAL, 0.6, 0.8, pd.Timestamp.now(), "signal2", {}),
            Signal(SignalType.TECHNICAL, 0.7, 0.85, pd.Timestamp.now(), "signal3", {}),
            Signal(SignalType.TECHNICAL, -0.4, 0.7, pd.Timestamp.now(), "signal4", {})  # Minority
        ]
        
        resolved = self.resolver._resolve_by_consensus(consensus_signals)
        
        # Should favor majority (positive signals)
        assert resolved.strength > 0
        assert resolved.source == "consensus_resolved"
        assert resolved.metadata['resolution_method'] == 'consensus_based'
        
        # Consensus ratio should be in metadata
        assert 'consensus_ratio' in resolved.metadata
        assert resolved.metadata['consensus_ratio'] == 0.75  # 3 out of 4 signals
    
    def test_resolve_conflicts_different_strategies(self):
        """Test conflict resolution with different strategies"""
        strategies = ['confidence_weighted', 'recency_weighted', 'source_reliability', 'consensus_based']
        
        for strategy in strategies:
            resolved = self.resolver.resolve_conflicts(self.conflicting_signals, strategy=strategy)
            
            # All strategies should return valid signals
            assert isinstance(resolved, Signal)
            assert -1.0 <= resolved.strength <= 1.0
            assert 0.0 <= resolved.confidence <= 1.0
            assert resolved.metadata['resolution_method'] == strategy
    
    def test_resolve_conflicts_invalid_strategy(self):
        """Test conflict resolution with invalid strategy"""
        # Should fall back to confidence_weighted
        resolved = self.resolver.resolve_conflicts(self.conflicting_signals, strategy='invalid_strategy')
        
        assert isinstance(resolved, Signal)
        # Should use default confidence_weighted method
        assert resolved.source == "conflict_resolved"
    
    def test_update_source_reliability(self):
        """Test updating source reliability scores"""
        initial_reliability = self.resolver.source_reliability.get("MACD", 0.5)
        
        # Update with high performance
        self.resolver.update_source_reliability("MACD", 0.95)
        
        # Reliability should increase (but not jump to 0.95 due to exponential moving average)
        new_reliability = self.resolver.source_reliability["MACD"]
        assert new_reliability > initial_reliability
        assert new_reliability < 0.95  # Should be smoothed
        
        # Update with low performance
        self.resolver.update_source_reliability("MACD", 0.3)
        
        # Reliability should decrease
        updated_reliability = self.resolver.source_reliability["MACD"]
        assert updated_reliability < new_reliability
    
    def test_update_source_reliability_new_source(self):
        """Test updating reliability for new source"""
        new_source = "new_indicator"
        assert new_source not in self.resolver.source_reliability
        
        self.resolver.update_source_reliability(new_source, 0.8)
        
        # Should add new source with given reliability
        assert new_source in self.resolver.source_reliability
        assert self.resolver.source_reliability[new_source] == 0.8
    
    def test_confidence_reduction_in_conflicts(self):
        """Test that confidence is appropriately reduced when resolving conflicts"""
        # Test all resolution methods reduce confidence
        strategies = ['confidence_weighted', 'recency_weighted', 'source_reliability', 'consensus_based']
        
        max_original_confidence = max(s.confidence for s in self.conflicting_signals)
        
        for strategy in strategies:
            resolved = self.resolver.resolve_conflicts(self.conflicting_signals, strategy=strategy)
            
            # Confidence should be reduced due to conflict (except consensus which might not always reduce)
            if strategy != 'consensus_based':
                assert resolved.confidence < max_original_confidence
            
            # But should still be reasonable
            assert resolved.confidence > 0.1
    
    def test_timestamp_handling(self):
        """Test proper timestamp handling in resolved signals"""
        resolved = self.resolver.resolve_conflicts(self.conflicting_signals)
        
        # Should have the most recent timestamp
        expected_timestamp = max(s.timestamp for s in self.conflicting_signals if hasattr(s, 'timestamp'))
        assert resolved.timestamp == expected_timestamp
    
    def test_zero_confidence_signals(self):
        """Test handling of signals with zero confidence"""
        zero_conf_signals = [
            Signal(SignalType.TECHNICAL, 0.8, 0.0, pd.Timestamp.now(), "zero_conf1", {}),
            Signal(SignalType.TECHNICAL, -0.6, 0.0, pd.Timestamp.now(), "zero_conf2", {}),
        ]
        
        resolved = self.resolver.resolve_conflicts(zero_conf_signals)
        
        # Should handle gracefully without division by zero
        assert isinstance(resolved, Signal)
        assert np.isfinite(resolved.strength)
        assert np.isfinite(resolved.confidence)
    
    def test_extreme_strength_values(self):
        """Test handling of extreme strength values"""
        extreme_signals = [
            Signal(SignalType.TECHNICAL, 1.0, 0.9, pd.Timestamp.now(), "max_buy", {}),
            Signal(SignalType.TECHNICAL, -1.0, 0.9, pd.Timestamp.now(), "max_sell", {}),
        ]
        
        resolved = self.resolver.resolve_conflicts(extreme_signals)
        
        # Should handle extreme values and stay within bounds
        assert -1.0 <= resolved.strength <= 1.0
        assert 0.0 <= resolved.confidence <= 1.0
    
    @patch('recommendation.intelligent_ensemble.logger')
    def test_logging(self, mock_logger):
        """Test that appropriate logging occurs"""
        self.resolver.resolve_conflicts(self.conflicting_signals)
        
        # Should log debug information about resolution
        mock_logger.debug.assert_called()
        
        # Test reliability update logging
        self.resolver.update_source_reliability("test_source", 0.8)
        mock_logger.debug.assert_called()
    
    def test_metadata_preservation(self):
        """Test that resolution metadata is properly set"""
        resolved = self.resolver.resolve_conflicts(self.conflicting_signals, strategy='confidence_weighted')
        
        # Should have resolution metadata
        assert 'resolution_method' in resolved.metadata
        assert 'original_count' in resolved.metadata
        assert resolved.metadata['resolution_method'] == 'confidence_weighted'
        assert resolved.metadata['original_count'] == len(self.conflicting_signals)
    
    def test_signal_type_consistency(self):
        """Test that resolved signal maintains signal type from input"""
        # All signals have same type
        same_type_signals = [
            Signal(SignalType.TECHNICAL, 0.8, 0.9, pd.Timestamp.now(), "tech1", {}),
            Signal(SignalType.TECHNICAL, -0.6, 0.8, pd.Timestamp.now(), "tech2", {}),
        ]
        
        resolved = self.resolver.resolve_conflicts(same_type_signals)
        assert resolved.signal_type == SignalType.TECHNICAL
        
        # Mixed types - should use first signal's type
        mixed_type_signals = [
            Signal(SignalType.TECHNICAL, 0.8, 0.9, pd.Timestamp.now(), "tech", {}),
            Signal(SignalType.FUNDAMENTAL, -0.6, 0.8, pd.Timestamp.now(), "fund", {}),
        ]
        
        resolved = self.resolver.resolve_conflicts(mixed_type_signals)
        assert resolved.signal_type == SignalType.TECHNICAL


if __name__ == "__main__":
    pytest.main([__file__])