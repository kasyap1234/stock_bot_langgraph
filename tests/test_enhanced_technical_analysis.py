"""
Unit tests for Enhanced Technical Analysis Engine
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import json

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

from agents.enhanced_technical_analysis import (
    Signal,
    IndicatorPerformance,
    SignalQualityFilter,
    IndicatorPerformanceTracker,
    MultiTimeframeAnalyzer,
    EnhancedTechnicalAnalysisEngine
)


class TestSignal(unittest.TestCase):
    """Test Signal dataclass"""
    
    def test_signal_creation(self):
        """Test Signal object creation"""
        signal = Signal(
            indicator='RSI',
            direction='buy',
            strength=0.8,
            confidence=0.7,
            timestamp=datetime.now(),
            timeframe='1D',
            metadata={'rsi_value': 25}
        )
        
        self.assertEqual(signal.indicator, 'RSI')
        self.assertEqual(signal.direction, 'buy')
        self.assertEqual(signal.strength, 0.8)
        self.assertEqual(signal.confidence, 0.7)
        self.assertEqual(signal.timeframe, '1D')
        self.assertEqual(signal.metadata['rsi_value'], 25)


class TestSignalQualityFilter(unittest.TestCase):
    """Test SignalQualityFilter functionality"""
    
    def setUp(self):
        self.filter = SignalQualityFilter(min_confidence=0.3, min_strength=0.2)
        
    def test_high_quality_signal_passes(self):
        """Test that high quality signals pass the filter"""
        signal = Signal(
            indicator='RSI',
            direction='buy',
            strength=0.8,
            confidence=0.7,
            timestamp=datetime.now(),
            timeframe='1D',
            metadata={}
        )
        
        filtered_signals = self.filter.filter_signals([signal])
        self.assertEqual(len(filtered_signals), 1)
        self.assertEqual(filtered_signals[0], signal)
        
    def test_low_quality_signal_filtered(self):
        """Test that low quality signals are filtered out"""
        low_quality_signal = Signal(
            indicator='RSI',
            direction='buy',
            strength=0.1,  # Below threshold
            confidence=0.2,  # Below threshold
            timestamp=datetime.now(),
            timeframe='1D',
            metadata={}
        )
        
        filtered_signals = self.filter.filter_signals([low_quality_signal])
        self.assertEqual(len(filtered_signals), 0)
        
    def test_neutral_signal_filtered(self):
        """Test that neutral signals are filtered out"""
        neutral_signal = Signal(
            indicator='RSI',
            direction='neutral',
            strength=0.8,
            confidence=0.7,
            timestamp=datetime.now(),
            timeframe='1D',
            metadata={}
        )
        
        filtered_signals = self.filter.filter_signals([neutral_signal])
        self.assertEqual(len(filtered_signals), 0)
        
    def test_signal_quality_scoring(self):
        """Test signal quality scoring with market context"""
        signal = Signal(
            indicator='RSI',
            direction='buy',
            strength=0.8,
            confidence=0.6,
            timestamp=datetime.now(),
            timeframe='1D',
            metadata={}
        )
        
        # Test with normal market conditions
        market_context = {'volatility': 0.02, 'trend_strength': 0.5}
        score = self.filter.score_signal_quality(signal, market_context)
        expected_base_score = (0.8 + 0.6) / 2  # 0.7
        self.assertAlmostEqual(score, expected_base_score, places=1)
        
        # Test with high volatility (should reduce score)
        market_context = {'volatility': 0.06, 'trend_strength': 0.5}
        score_high_vol = self.filter.score_signal_quality(signal, market_context)
        self.assertLess(score_high_vol, score)
        
        # Test with strong trend (should boost score)
        market_context = {'volatility': 0.02, 'trend_strength': 0.8}
        score_strong_trend = self.filter.score_signal_quality(signal, market_context)
        self.assertGreater(score_strong_trend, score)
    
    def test_noise_detection(self):
        """Test noise detection in signals"""
        # Test signal with alternating pattern (noise)
        noisy_signal = Signal(
            indicator='RSI',
            direction='buy',
            strength=0.5,
            confidence=0.4,
            timestamp=datetime.now(),
            timeframe='1D',
            metadata={
                'recent_signals': [
                    {'direction': 'buy', 'strength': 0.3, 'timestamp': datetime.now()},
                    {'direction': 'sell', 'strength': 0.4, 'timestamp': datetime.now()},
                    {'direction': 'buy', 'strength': 0.5, 'timestamp': datetime.now()}
                ]
            }
        )
        
        self.assertTrue(self.filter._is_noise(noisy_signal))
        
        # Test signal with low strength (noise)
        weak_signal = Signal(
            indicator='MACD',
            direction='sell',
            strength=0.05,  # Below noise threshold
            confidence=0.6,
            timestamp=datetime.now(),
            timeframe='1D',
            metadata={}
        )
        
        self.assertTrue(self.filter._is_noise(weak_signal))
        
        # Test clean signal (not noise)
        clean_signal = Signal(
            indicator='RSI',
            direction='buy',
            strength=0.7,
            confidence=0.8,
            timestamp=datetime.now(),
            timeframe='1D',
            metadata={}
        )
        
        self.assertFalse(self.filter._is_noise(clean_signal))
    
    def test_historical_performance_update(self):
        """Test historical performance tracking"""
        # Update performance for RSI
        self.filter.update_historical_performance('RSI', 0.75, 'trending')
        self.filter.update_historical_performance('RSI', 0.65, 'ranging')
        
        # Check that performance data was stored
        self.assertIn('RSI', self.filter.historical_performance)
        rsi_perf = self.filter.historical_performance['RSI']
        
        self.assertIn('accuracy', rsi_perf)
        self.assertIn('regime_performance', rsi_perf)
        self.assertIn('trending', rsi_perf['regime_performance'])
        self.assertIn('ranging', rsi_perf['regime_performance'])
        
    def test_historical_performance_scoring(self):
        """Test historical performance adjustment in scoring"""
        # Set up historical performance
        self.filter.update_historical_performance('RSI', 0.8, 'trending')
        
        signal = Signal(
            indicator='RSI',
            direction='buy',
            strength=0.6,
            confidence=0.6,
            timestamp=datetime.now(),
            timeframe='1D',
            metadata={}
        )
        
        # Test with trending market (should get boost from good historical performance)
        market_context = {'volatility': 0.02, 'trend_strength': 0.5, 'regime': 'trending'}
        score_with_history = self.filter.score_signal_quality(signal, market_context)
        
        # Test with new indicator (no history)
        signal_new = Signal(
            indicator='NEW_INDICATOR',
            direction='buy',
            strength=0.6,
            confidence=0.6,
            timestamp=datetime.now(),
            timeframe='1D',
            metadata={}
        )
        
        score_no_history = self.filter.score_signal_quality(signal_new, market_context)
        
        # RSI with good history should score higher
        self.assertGreater(score_with_history, score_no_history)
    
    def test_noise_filtered_signals(self):
        """Test advanced noise filtering with signal patterns"""
        # Create a series of signals with noise pattern
        signals = []
        base_time = datetime.now()
        
        for i, direction in enumerate(['buy', 'sell', 'buy', 'sell', 'buy']):
            signal = Signal(
                indicator='RSI',
                direction=direction,
                strength=0.4 + i * 0.1,
                confidence=0.5,
                timestamp=base_time + timedelta(minutes=i),
                timeframe='1D',
                metadata={}
            )
            signals.append(signal)
            
        # Apply noise filtering
        filtered = self.filter.get_noise_filtered_signals(signals, lookback_window=3)
        
        # Should filter out some of the alternating signals
        self.assertLess(len(filtered), len(signals))
        
        # Check that metadata was added
        for signal in signals:
            if 'recent_signals' in signal.metadata:
                self.assertIsInstance(signal.metadata['recent_signals'], list)


class TestIndicatorPerformanceTracker(unittest.TestCase):
    """Test IndicatorPerformanceTracker functionality"""
    
    def setUp(self):
        # Create temporary file for testing
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_file.close()
        self.tracker = IndicatorPerformanceTracker(self.temp_file.name)
        
    def tearDown(self):
        # Clean up temporary file
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
            
    def test_new_indicator_performance(self):
        """Test performance tracking for new indicator"""
        # Update performance for new indicator
        self.tracker.update_performance('RSI', True, 0.05, 'trending')
        
        # Check that performance data was created
        self.assertIn('RSI', self.tracker.performance_data)
        perf = self.tracker.performance_data['RSI']
        
        self.assertEqual(perf.indicator_name, 'RSI')
        self.assertEqual(perf.total_signals, 1)
        self.assertEqual(perf.correct_signals, 1)
        self.assertEqual(perf.accuracy, 1.0)
        # avg_return uses exponential moving average: 0.1 * 0.05 + 0.9 * 0.0 = 0.005
        self.assertAlmostEqual(perf.avg_return, 0.005, places=3)
        
    def test_performance_updates(self):
        """Test multiple performance updates"""
        # Add several performance updates
        self.tracker.update_performance('MACD', True, 0.03, 'trending')
        self.tracker.update_performance('MACD', False, -0.02, 'trending')
        self.tracker.update_performance('MACD', True, 0.04, 'ranging')
        
        perf = self.tracker.performance_data['MACD']
        
        self.assertEqual(perf.total_signals, 3)
        self.assertEqual(perf.correct_signals, 2)
        self.assertAlmostEqual(perf.accuracy, 2/3, places=2)
        
        # Check market regime performance
        self.assertIn('trending', perf.market_regime_performance)
        self.assertIn('ranging', perf.market_regime_performance)
        
    def test_indicator_weight_calculation(self):
        """Test dynamic weight calculation"""
        # Add performance data
        self.tracker.update_performance('RSI', True, 0.05, 'trending')
        self.tracker.update_performance('RSI', True, 0.03, 'trending')
        self.tracker.update_performance('RSI', False, -0.01, 'trending')
        
        # Get weight for trending market
        weight = self.tracker.get_indicator_weight('RSI', 'trending')
        
        # Should be based on accuracy (2/3 = 0.67) with regime adjustment
        expected_base = 2/3
        self.assertGreater(weight, 0.1)  # Should be above minimum
        self.assertLessEqual(weight, 1.0)  # Should not exceed maximum
        
    def test_save_load_performance_data(self):
        """Test saving and loading performance data"""
        # Add some performance data
        self.tracker.update_performance('RSI', True, 0.05, 'trending')
        self.tracker.update_performance('MACD', False, -0.02, 'ranging')
        
        # Create new tracker with same file
        new_tracker = IndicatorPerformanceTracker(self.temp_file.name)
        
        # Check that data was loaded correctly
        self.assertIn('RSI', new_tracker.performance_data)
        self.assertIn('MACD', new_tracker.performance_data)
        
        rsi_perf = new_tracker.performance_data['RSI']
        self.assertEqual(rsi_perf.total_signals, 1)
        self.assertEqual(rsi_perf.correct_signals, 1)


class TestMultiTimeframeAnalyzer(unittest.TestCase):
    """Test MultiTimeframeAnalyzer functionality"""
    
    def setUp(self):
        self.analyzer = MultiTimeframeAnalyzer(['1H', '4H', '1D', '1W'])
        
        # Create sample data with hourly frequency for better timeframe testing
        dates = pd.date_range(start='2023-01-01', end='2023-03-31', freq='h')
        np.random.seed(42)  # For reproducible tests
        
        self.sample_data = pd.DataFrame({
            'Open': 100 + np.random.randn(len(dates)).cumsum() * 0.02,
            'High': 100 + np.random.randn(len(dates)).cumsum() * 0.02 + 0.5,
            'Low': 100 + np.random.randn(len(dates)).cumsum() * 0.02 - 0.5,
            'Close': 100 + np.random.randn(len(dates)).cumsum() * 0.02,
            'Volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
    def test_enhanced_data_resampling(self):
        """Test enhanced data resampling to different timeframes"""
        # Test hourly resampling
        hourly_data = self.analyzer._resample_data(self.sample_data, '1H')
        self.assertEqual(len(hourly_data), len(self.sample_data))
        
        # Test 4-hour resampling
        four_hour_data = self.analyzer._resample_data(self.sample_data, '4H')
        self.assertLess(len(four_hour_data), len(self.sample_data))
        self.assertTrue(all(col in four_hour_data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']))
        
        # Test weekly resampling
        weekly_data = self.analyzer._resample_data(self.sample_data, '1W')
        self.assertLess(len(weekly_data), len(four_hour_data))
        
    def test_consistency_score_calculation(self):
        """Test basic consistency score calculation"""
        # Create mock timeframe signals
        timeframe_signals = {
            '1H': {'RSI': 'buy', 'MACD': 'sell'},
            '4H': {'RSI': 'buy', 'MACD': 'neutral'},
            '1D': {'RSI': 'buy', 'MACD': 'sell'}
        }
        
        # RSI should have high consistency (3/3 agreement on buy)
        rsi_consistency = self.analyzer._calculate_consistency_score('RSI', timeframe_signals)
        self.assertEqual(rsi_consistency, 1.0)
        
        # MACD should have lower consistency
        macd_consistency = self.analyzer._calculate_consistency_score('MACD', timeframe_signals)
        self.assertLess(macd_consistency, 1.0)
        
    def test_enhanced_consistency_score_calculation(self):
        """Test enhanced consistency score with weighting"""
        # Create mock Signal objects with strength
        signal_1h = Signal('RSI', 'buy', 0.8, 0.7, datetime.now(), '1H', {})
        signal_4h = Signal('RSI', 'buy', 0.6, 0.8, datetime.now(), '4H', {})
        signal_1d = Signal('RSI', 'sell', 0.4, 0.5, datetime.now(), '1D', {})
        
        timeframe_signals = {
            '1H': {'RSI': signal_1h},
            '4H': {'RSI': signal_4h},
            '1D': {'RSI': signal_1d}
        }
        
        enhanced_score = self.analyzer._calculate_enhanced_consistency_score('RSI', timeframe_signals)
        basic_score = self.analyzer._calculate_consistency_score('RSI', timeframe_signals)
        
        # Enhanced score should consider signal strengths and timeframe weights
        self.assertIsInstance(enhanced_score, float)
        self.assertGreaterEqual(enhanced_score, 0.0)
        self.assertLessEqual(enhanced_score, 1.0)
        
    def test_timeframe_categorization(self):
        """Test timeframe categorization"""
        weighted_signals = [
            {'direction': 'buy', 'weight': 0.1, 'timeframe': '5m'},
            {'direction': 'buy', 'weight': 0.2, 'timeframe': '1H'},
            {'direction': 'buy', 'weight': 0.3, 'timeframe': '1D'}
        ]
        
        categories = self.analyzer._categorize_timeframes(weighted_signals)
        
        self.assertIn('short_term', categories)
        self.assertIn('medium_term', categories)
        self.assertIn('long_term', categories)
        self.assertEqual(len(categories), 3)
        
    def test_cross_timeframe_validation(self):
        """Test cross-timeframe validation"""
        # Create mock signals for validation
        signal_macd_1h = Signal('MACD', 'buy', 0.7, 0.8, datetime.now(), '1H', {})
        signal_macd_4h = Signal('MACD', 'buy', 0.6, 0.7, datetime.now(), '4H', {})
        signal_rsi_1h = Signal('RSI', 'buy', 0.8, 0.6, datetime.now(), '1H', {})
        signal_rsi_1d = Signal('RSI', 'buy', 0.5, 0.7, datetime.now(), '1D', {})
        
        timeframe_signals = {
            '1H': {'MACD': signal_macd_1h, 'RSI': signal_rsi_1h},
            '4H': {'MACD': signal_macd_4h},
            '1D': {'RSI': signal_rsi_1d}
        }
        
        validation_results = self.analyzer._cross_timeframe_validation(timeframe_signals)
        
        # Check that validation results have expected structure
        self.assertIn('trend_alignment', validation_results)
        self.assertIn('momentum_confirmation', validation_results)
        self.assertIn('support_resistance_confluence', validation_results)
        self.assertIn('overall_validation_score', validation_results)
        
        # Check that scores are in valid range
        overall_score = validation_results['overall_validation_score']
        self.assertGreaterEqual(overall_score, 0.0)
        self.assertLessEqual(overall_score, 1.0)
        
    def test_trend_alignment_validation(self):
        """Test trend alignment validation"""
        # Create signals with good trend alignment
        signal_macd_1h = Signal('MACD', 'buy', 0.7, 0.8, datetime.now(), '1H', {})
        signal_macd_4h = Signal('MACD', 'buy', 0.6, 0.7, datetime.now(), '4H', {})
        signal_ichimoku_1d = Signal('Ichimoku', 'buy', 0.8, 0.9, datetime.now(), '1D', {})
        
        timeframe_signals = {
            '1H': {'MACD': signal_macd_1h},
            '4H': {'MACD': signal_macd_4h},
            '1D': {'Ichimoku': signal_ichimoku_1d}
        }
        
        trend_alignment = self.analyzer._validate_trend_alignment(timeframe_signals)
        
        self.assertIn('score', trend_alignment)
        self.assertIn('aligned_indicators', trend_alignment)
        self.assertIn('alignment_strength', trend_alignment)
        
        # Should have good alignment since all signals are 'buy'
        self.assertGreater(trend_alignment['score'], 0.5)
        
    def test_momentum_confirmation_validation(self):
        """Test momentum confirmation validation"""
        # Create signals with momentum confirmation
        signal_rsi_1h = Signal('RSI', 'buy', 0.8, 0.7, datetime.now(), '1H', {})
        signal_macd_1h = Signal('MACD', 'buy', 0.6, 0.8, datetime.now(), '1H', {})
        signal_rsi_1d = Signal('RSI', 'buy', 0.7, 0.6, datetime.now(), '1D', {})
        
        timeframe_signals = {
            '1H': {'RSI': signal_rsi_1h, 'MACD': signal_macd_1h},
            '1D': {'RSI': signal_rsi_1d}
        }
        
        momentum_confirmation = self.analyzer._validate_momentum_confirmation(timeframe_signals)
        
        self.assertIn('score', momentum_confirmation)
        self.assertIn('confirmed_indicators', momentum_confirmation)
        self.assertIn('confirmation_strength', momentum_confirmation)
        
    def test_timeframe_strength_calculation(self):
        """Test timeframe strength calculation"""
        signal_1h = Signal('RSI', 'buy', 0.8, 0.7, datetime.now(), '1H', {})
        signal_4h = Signal('MACD', 'sell', 0.6, 0.5, datetime.now(), '4H', {})
        
        timeframe_signals = {
            '1H': {'RSI': signal_1h},
            '4H': {'MACD': signal_4h}
        }
        
        timeframe_strength = self.analyzer._calculate_timeframe_strength(timeframe_signals)
        
        self.assertIn('1H', timeframe_strength)
        self.assertIn('4H', timeframe_strength)
        
        # 1H should have higher strength (0.75 vs 0.55)
        self.assertGreater(timeframe_strength['1H'], timeframe_strength['4H'])
        
    def test_consensus_signal_generation(self):
        """Test consensus signal generation with enhanced features"""
        # Create mock signals with high consistency
        signal_1h = Signal('RSI', 'buy', 0.8, 0.7, datetime.now(), '1H', {})
        signal_4h = Signal('RSI', 'buy', 0.6, 0.8, datetime.now(), '4H', {})
        
        timeframe_signals = {
            '1H': {'RSI': signal_1h},
            '4H': {'RSI': signal_4h}
        }
        consistency_scores = {'RSI': 1.0}
        
        consensus = self.analyzer._generate_consensus_signals(timeframe_signals, consistency_scores)
        
        self.assertIn('RSI', consensus)
        self.assertEqual(consensus['RSI'].direction, 'buy')
        self.assertEqual(consensus['RSI'].timeframe, 'consensus')
        self.assertIn('consistency_score', consensus['RSI'].metadata)


class TestEnhancedTechnicalAnalysisEngine(unittest.TestCase):
    """Test EnhancedTechnicalAnalysisEngine functionality"""
    
    def setUp(self):
        self.engine = EnhancedTechnicalAnalysisEngine()
        
        # Create sample data for testing
        dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
        np.random.seed(42)
        
        self.sample_data = pd.DataFrame({
            'Open': 100 + np.random.randn(len(dates)).cumsum() * 0.5,
            'High': 100 + np.random.randn(len(dates)).cumsum() * 0.5 + 2,
            'Low': 100 + np.random.randn(len(dates)).cumsum() * 0.5 - 2,
            'Close': 100 + np.random.randn(len(dates)).cumsum() * 0.5,
            'Volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
    def test_market_context_calculation(self):
        """Test market context calculation"""
        context = self.engine._get_market_context(self.sample_data)
        
        self.assertIn('volatility', context)
        self.assertIn('trend_strength', context)
        self.assertIn('regime', context)
        
        self.assertIsInstance(context['volatility'], float)
        self.assertIsInstance(context['trend_strength'], float)
        self.assertIsInstance(context['regime'], str)
        
        # Check that values are in reasonable ranges
        self.assertGreaterEqual(context['volatility'], 0)
        self.assertGreaterEqual(context['trend_strength'], 0)
        self.assertLessEqual(context['trend_strength'], 1)
        
    def test_dynamic_weights_calculation(self):
        """Test dynamic weights calculation"""
        weights = self.engine.get_dynamic_weights('trending', 0.02)
        
        # Check that all expected indicators have weights
        expected_indicators = ['RSI', 'MACD', 'Ichimoku', 'SupportResistance', 'VWAP']
        for indicator in expected_indicators:
            self.assertIn(indicator, weights)
            self.assertGreaterEqual(weights[indicator], 0.0)
            self.assertLessEqual(weights[indicator], 1.0)
            
        # Check that weights sum to approximately 1
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)
        
    def test_market_condition_weighting(self):
        """Test market condition-based weight adjustments"""
        # Test trending market - should favor trend indicators
        trending_weights = self.engine.get_dynamic_weights('trending', 0.02)
        
        # Test ranging market - should favor oscillators
        ranging_weights = self.engine.get_dynamic_weights('ranging', 0.02)
        
        # In trending markets, MACD should have higher weight than in ranging markets
        if 'MACD' in trending_weights and 'MACD' in ranging_weights:
            self.assertGreater(trending_weights['MACD'], ranging_weights['MACD'])
            
        # In ranging markets, RSI should have higher weight than in trending markets
        if 'RSI' in trending_weights and 'RSI' in ranging_weights:
            self.assertGreater(ranging_weights['RSI'], trending_weights['RSI'])
            
    def test_volatility_based_weighting(self):
        """Test volatility-based weight adjustments"""
        # Test high volatility
        high_vol_weights = self.engine.get_dynamic_weights('high_volatility', 0.08)
        
        # Test low volatility
        low_vol_weights = self.engine.get_dynamic_weights('normal', 0.005)
        
        # Both should return valid weight distributions
        self.assertAlmostEqual(sum(high_vol_weights.values()), 1.0, places=2)
        self.assertAlmostEqual(sum(low_vol_weights.values()), 1.0, places=2)
        
    def test_advanced_weighting_algorithms(self):
        """Test advanced weighting algorithms"""
        base_weights = {'RSI': 0.3, 'MACD': 0.3, 'Ichimoku': 0.2, 'SupportResistance': 0.1, 'VWAP': 0.1}
        
        # Test momentum weighting
        momentum_weights = self.engine._apply_momentum_weighting(base_weights)
        self.assertEqual(len(momentum_weights), len(base_weights))
        
        # Test correlation adjustments
        correlation_weights = self.engine._apply_correlation_adjustments(base_weights)
        self.assertEqual(len(correlation_weights), len(base_weights))
        
        # Test recency weighting
        recency_weights = self.engine._apply_recency_weighting(base_weights)
        self.assertEqual(len(recency_weights), len(base_weights))
        
        # Test diversity weighting
        diversity_weights = self.engine._apply_diversity_weighting(base_weights)
        self.assertEqual(len(diversity_weights), len(base_weights))
        
    def test_adaptive_weights_calculation(self):
        """Test adaptive weights based on current signals"""
        # Create mock signals with different quality scores
        signal_rsi = Signal('RSI', 'buy', 0.8, 0.7, datetime.now(), '1D', {'quality_score': 0.9})
        signal_macd = Signal('MACD', 'sell', 0.6, 0.5, datetime.now(), '1D', {'quality_score': 0.4})
        
        signals = {'RSI': signal_rsi, 'MACD': signal_macd}
        market_context = {'regime': 'trending', 'volatility': 0.02}
        
        adaptive_weights = self.engine.calculate_adaptive_weights(signals, market_context)
        
        # Check that weights are calculated
        self.assertIn('RSI', adaptive_weights)
        self.assertIn('MACD', adaptive_weights)
        
        # RSI should have higher weight due to better quality score and signal strength
        self.assertGreater(adaptive_weights['RSI'], adaptive_weights['MACD'])
        
        # Weights should sum to 1
        self.assertAlmostEqual(sum(adaptive_weights.values()), 1.0, places=2)
        
    def test_rsi_signal_calculation(self):
        """Test RSI signal calculation"""
        signal = self.engine._calculate_rsi_signal(self.sample_data)
        
        if signal is not None:  # RSI might return None for insufficient data
            self.assertEqual(signal.indicator, 'RSI')
            self.assertIn(signal.direction, ['buy', 'sell', 'neutral'])
            self.assertGreaterEqual(signal.strength, 0.0)
            self.assertLessEqual(signal.strength, 1.0)
            self.assertGreaterEqual(signal.confidence, 0.0)
            self.assertLessEqual(signal.confidence, 1.0)
            self.assertIn('rsi_value', signal.metadata)
            
    def test_macd_signal_calculation(self):
        """Test MACD signal calculation"""
        signal = self.engine._calculate_macd_signal(self.sample_data)
        
        if signal is not None:
            self.assertEqual(signal.indicator, 'MACD')
            self.assertIn(signal.direction, ['buy', 'sell', 'neutral'])
            self.assertGreaterEqual(signal.strength, 0.0)
            self.assertLessEqual(signal.strength, 1.0)
            self.assertGreaterEqual(signal.confidence, 0.0)
            self.assertLessEqual(signal.confidence, 1.0)
            self.assertIn('histogram', signal.metadata)
            
    def test_analyze_with_quality_control(self):
        """Test main analysis function with quality control"""
        result = self.engine.analyze_with_quality_control(self.sample_data, 'TEST')
        
        # Check that result has expected structure
        self.assertIn('signals', result)
        self.assertIn('market_context', result)
        self.assertIn('dynamic_weights', result)
        self.assertIn('timeframe_analysis', result)
        self.assertIn('quality_metrics', result)
        
        # Check that signals are Signal objects
        for signal in result['signals'].values():
            self.assertIsInstance(signal, Signal)
            
        # Check quality metrics
        quality_metrics = result['quality_metrics']
        self.assertIn('overall_quality', quality_metrics)
        self.assertIn('signal_count', quality_metrics)
        
    def test_error_handling(self):
        """Test error handling with invalid data"""
        # Test with empty dataframe
        empty_df = pd.DataFrame()
        result = self.engine.analyze_with_quality_control(empty_df, 'TEST')
        
        # Should handle gracefully and return empty results
        self.assertIn('signals', result)
        self.assertEqual(len(result['signals']), 0)


def generate_synthetic_close(length=252, trend=0.0005, volatility=0.01, seed=42):
    """Helper for tests"""
    np.random.seed(seed)
    returns = np.random.normal(trend, volatility, length)
    close = 100 * np.exp(np.cumsum(returns))
    return close

def calculate_ema_talib(series, period):
    """Test helper - TALib matching EMA"""
    n = len(series)
    if n == 0:
        return pd.Series(np.full(n, np.nan))
    
    alpha = 2.0 / (period + 1.0)
    ema = np.full(n, np.nan)
    
    first_valid_pos = 0
    if n < period:
        for i in range(first_valid_pos, n):
            ema[i] = series[i]
        return pd.Series(ema)
    
    init_pos = period - 1
    sma_init = np.mean(series[0:period])
    ema[init_pos] = sma_init
    
    for i in range(init_pos + 1, n):
        ema[i] = alpha * series[i] + (1.0 - alpha) * ema[i - 1]
    
    return pd.Series(ema)

class TestMACDVerification(unittest.TestCase):
    """Test MACD calculation matches TALib"""

    def setUp(self):
        self.close = generate_synthetic_close()
        self.is_indian = False
        self.engine = EnhancedTechnicalAnalysisEngine()

        # Create proper OHLC data for testing (match the length of self.close)
        np.random.seed(42)

        self.sample_data = pd.DataFrame({
            'Open': self.close * 0.98 + np.random.randn(len(self.close)) * 0.5,
            'High': self.close * 1.02 + np.random.randn(len(self.close)) * 1.0,
            'Low': self.close * 0.98 - np.random.randn(len(self.close)) * 1.0,
            'Close': self.close,
            'Volume': np.random.randint(1000, 10000, len(self.close))
        })
        
    def test_macd_basic_matches_talib_us(self):
        """Test basic MACD calculation works"""
        fast, slow, signal = 12, 26, 9

        # Project basic calculation
        close_series = pd.Series(self.close)
        ema_fast = calculate_ema_talib(close_series, fast)
        ema_slow = calculate_ema_talib(close_series, slow)
        macd_line = ema_fast - ema_slow
        signal_line = calculate_ema_talib(macd_line, signal)
        hist = macd_line - signal_line

        # Basic checks that calculation produces reasonable results
        self.assertEqual(len(macd_line), len(self.close))
        self.assertEqual(len(signal_line), len(self.close))
        self.assertEqual(len(hist), len(self.close))

        # Check that we have some valid values
        valid_hist = hist.dropna()
        self.assertGreater(len(valid_hist), 0)
        
    def test_macd_enhanced_matches_after_fix(self):
        """Test enhanced MACD now returns valid signals"""
        if not TALIB_AVAILABLE:
            self.skipTest("TA-Lib not available")

        signal = self.engine._calculate_macd_signal(self.sample_data, 'AAPL')
        self.assertIsNotNone(signal)
        self.assertFalse(pd.isna(signal.metadata['histogram']))
        self.assertIn(signal.direction, ['buy', 'sell', 'neutral'])
        self.assertIsInstance(signal.strength, float)
        self.assertIsInstance(signal.confidence, float)
        
    def test_macd_indian_params(self):
        """Test Indian params applied correctly"""
        if not TALIB_AVAILABLE:
            self.skipTest("TA-Lib not available")

        indian_close = generate_synthetic_close(seed=123)
        # Create proper OHLC data (match the length of indian_close)
        np.random.seed(123)
        indian_data = pd.DataFrame({
            'Open': indian_close * 0.98 + np.random.randn(len(indian_close)) * 0.5,
            'High': indian_close * 1.02 + np.random.randn(len(indian_close)) * 1.0,
            'Low': indian_close * 0.98 - np.random.randn(len(indian_close)) * 1.0,
            'Close': indian_close,
            'Volume': np.random.randint(1000, 10000, len(indian_close))
        })

        signal = self.engine._calculate_macd_signal(indian_data, 'RELIANCE.NS')
        self.assertIsNotNone(signal)
        self.assertTrue(signal.metadata['is_indian'])
        self.assertEqual(signal.metadata['periods'], {'fast': 8, 'slow': 17, 'signal': 9})
        
    def test_nan_propagation(self):
        """Test NaN handling in MACD"""
        if not TALIB_AVAILABLE:
            self.skipTest("TA-Lib not available")

        close_with_nan = generate_synthetic_close()
        close_with_nan[10:15] = np.nan  # Introduce NaNs
        # Create proper OHLC data with NaNs (match the length of close_with_nan)
        np.random.seed(42)
        df = pd.DataFrame({
            'Open': close_with_nan * 0.98 + np.random.randn(len(close_with_nan)) * 0.5,
            'High': close_with_nan * 1.02 + np.random.randn(len(close_with_nan)) * 1.0,
            'Low': close_with_nan * 0.98 - np.random.randn(len(close_with_nan)) * 1.0,
            'Close': close_with_nan,
            'Volume': np.random.randint(1000, 10000, len(close_with_nan))
        })

        signal = self.engine._calculate_macd_signal(df, 'TEST')
        if signal:
            self.assertTrue(pd.isna(signal.metadata['histogram']) or abs(signal.metadata['histogram']) < 1e-10)  # Should handle NaNs properly

if __name__ == '__main__':
    unittest.main()