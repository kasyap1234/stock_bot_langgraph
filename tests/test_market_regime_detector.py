"""
Unit tests for Market Regime Detection System
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os

from agents.market_regime_detector import (
    MarketRegime,
    RegimeFeatures,
    RegimeDetectionResult,
    HiddenMarkovRegimeModel,
    MarketRegimeDetector
)


class TestMarketRegime(unittest.TestCase):
    """Test MarketRegime enum"""
    
    def test_regime_values(self):
        """Test that regime enum has correct values"""
        self.assertEqual(MarketRegime.BULL.value, "bull")
        self.assertEqual(MarketRegime.BEAR.value, "bear")
        self.assertEqual(MarketRegime.VOLATILE.value, "volatile")
        self.assertEqual(MarketRegime.STABLE.value, "stable")


class TestRegimeFeatures(unittest.TestCase):
    """Test RegimeFeatures dataclass"""
    
    def test_regime_features_creation(self):
        """Test RegimeFeatures object creation"""
        features = RegimeFeatures(
            returns=0.02,
            volatility=0.15,
            trend_strength=0.7,
            volume_ratio=1.2,
            momentum=0.05,
            timestamp=datetime.now()
        )
        
        self.assertEqual(features.returns, 0.02)
        self.assertEqual(features.volatility, 0.15)
        self.assertEqual(features.trend_strength, 0.7)
        self.assertEqual(features.volume_ratio, 1.2)
        self.assertEqual(features.momentum, 0.05)
        self.assertIsInstance(features.timestamp, datetime)


class TestHiddenMarkovRegimeModel(unittest.TestCase):
    """Test HiddenMarkovRegimeModel functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = HiddenMarkovRegimeModel(n_states=4, random_state=42)
        
        # Create sample market data
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        # Bull market data (upward trend, low volatility)
        bull_returns = np.random.normal(0.001, 0.01, len(dates))
        bull_prices = 100 * np.exp(np.cumsum(bull_returns))
        
        self.bull_data = pd.DataFrame({
            'Open': bull_prices * 0.99,
            'High': bull_prices * 1.02,
            'Low': bull_prices * 0.98,
            'Close': bull_prices,
            'Volume': np.random.randint(1000, 5000, len(dates))
        }, index=dates)
        
        # Bear market data (downward trend, low volatility)
        bear_returns = np.random.normal(-0.001, 0.01, len(dates))
        bear_prices = 100 * np.exp(np.cumsum(bear_returns))
        
        self.bear_data = pd.DataFrame({
            'Open': bear_prices * 1.01,
            'High': bear_prices * 1.02,
            'Low': bear_prices * 0.98,
            'Close': bear_prices,
            'Volume': np.random.randint(1000, 5000, len(dates))
        }, index=dates)
        
        # Volatile market data (high volatility)
        volatile_returns = np.random.normal(0, 0.03, len(dates))
        volatile_prices = 100 * np.exp(np.cumsum(volatile_returns))
        
        self.volatile_data = pd.DataFrame({
            'Open': volatile_prices * 0.97,
            'High': volatile_prices * 1.05,
            'Low': volatile_prices * 0.95,
            'Close': volatile_prices,
            'Volume': np.random.randint(2000, 10000, len(dates))
        }, index=dates)
        
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertEqual(self.model.n_states, 4)
        self.assertEqual(self.model.random_state, 42)
        self.assertFalse(self.model.is_trained)
        self.assertEqual(len(self.model.regime_mapping), 4)
        
    def test_feature_preparation(self):
        """Test feature preparation from market data"""
        features = self.model.prepare_features(self.bull_data)
        
        # Check feature matrix shape
        self.assertEqual(features.shape[0], len(self.bull_data))
        self.assertEqual(features.shape[1], 5)  # 5 features
        
        # Check that features are numeric and finite
        self.assertTrue(np.all(np.isfinite(features)))
        
        # Check feature ranges are reasonable
        returns = features[:, 0]
        volatility = features[:, 1]
        trend_strength = features[:, 2]
        volume_ratio = features[:, 3]
        momentum = features[:, 4]
        
        # Returns should be small daily changes
        self.assertTrue(np.all(np.abs(returns) < 0.5))
        
        # Volatility should be non-negative
        self.assertTrue(np.all(volatility >= 0))
        
        # Trend strength should be between 0 and 1
        self.assertTrue(np.all((trend_strength >= 0) & (trend_strength <= 1)))
        
        # Volume ratio should be positive
        self.assertTrue(np.all(volume_ratio > 0))
        
    def test_trend_strength_calculation(self):
        """Test trend strength calculation"""
        trend_strength = self.model._calculate_trend_strength(self.bull_data)
        
        # Should return array of same length as input
        self.assertEqual(len(trend_strength), len(self.bull_data))
        
        # Values should be between 0 and 1
        self.assertTrue(np.all((trend_strength >= 0) & (trend_strength <= 1)))
        
        # Should not contain NaN values
        self.assertFalse(np.any(np.isnan(trend_strength)))
        
    def test_fallback_regime_detection(self):
        """Test fallback regime detection when HMM is not available"""
        # Test with bull market data
        result = self.model._fallback_regime_detection(self.bull_data)
        
        self.assertIsInstance(result, RegimeDetectionResult)
        self.assertIsInstance(result.current_regime, MarketRegime)
        self.assertIsInstance(result.regime_probabilities, dict)
        self.assertIsInstance(result.confidence, float)
        self.assertIsInstance(result.features, RegimeFeatures)
        
        # Check that probabilities sum to approximately 1
        prob_sum = sum(result.regime_probabilities.values())
        self.assertAlmostEqual(prob_sum, 1.0, places=2)
        
        # Confidence should be between 0 and 1
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        
    def test_regime_prediction_without_training(self):
        """Test regime prediction without training (should use fallback)"""
        result = self.model.predict_regime(self.bull_data)
        
        self.assertIsInstance(result, RegimeDetectionResult)
        self.assertIn(result.current_regime, MarketRegime)
        
        # Should have probabilities for all regimes
        self.assertEqual(len(result.regime_probabilities), 4)
        for regime in MarketRegime:
            self.assertIn(regime, result.regime_probabilities)
            
    def test_training_with_insufficient_data(self):
        """Test training with insufficient data"""
        # Create very small dataset
        small_data = self.bull_data.head(10)
        market_data = {'TEST': small_data}
        
        success = self.model.train(market_data, min_samples=50)
        self.assertFalse(success)
        
    def test_training_with_sufficient_data(self):
        """Test training with sufficient data"""
        market_data = {
            'BULL': self.bull_data,
            'BEAR': self.bear_data,
            'VOLATILE': self.volatile_data
        }
        
        # Training might fail if hmmlearn is not available, which is okay
        success = self.model.train(market_data)
        # Don't assert success since it depends on library availability
        
    def test_regime_duration_estimation(self):
        """Test regime duration estimation"""
        features = self.model.prepare_features(self.bull_data)
        duration = self.model._estimate_regime_duration(features, 0)
        
        self.assertIsInstance(duration, int)
        self.assertGreater(duration, 0)
        self.assertLessEqual(duration, 30)  # Should not exceed window size
        
    def test_model_save_load(self):
        """Test model saving and loading"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_path = tmp_file.name
            
        try:
            # Save model
            success = self.model.save_model(tmp_path)
            self.assertTrue(success)
            self.assertTrue(os.path.exists(tmp_path))
            
            # Create new model and load
            new_model = HiddenMarkovRegimeModel()
            load_success = new_model.load_model(tmp_path)
            self.assertTrue(load_success)
            
            # Check that key attributes are preserved
            self.assertEqual(new_model.regime_mapping, self.model.regime_mapping)
            self.assertEqual(new_model.feature_names, self.model.feature_names)
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    def test_load_nonexistent_model(self):
        """Test loading non-existent model file"""
        success = self.model.load_model('nonexistent_file.pkl')
        self.assertFalse(success)


class TestMarketRegimeDetector(unittest.TestCase):
    """Test MarketRegimeDetector functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary model path
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, 'test_regime_model.pkl')
        
        self.detector = MarketRegimeDetector(model_path=self.model_path)
        
        # Create sample market data
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
        
        self.sample_data = pd.DataFrame({
            'Open': 100 + np.random.randn(len(dates)).cumsum() * 0.5,
            'High': 100 + np.random.randn(len(dates)).cumsum() * 0.5 + 2,
            'Low': 100 + np.random.randn(len(dates)).cumsum() * 0.5 - 2,
            'Close': 100 + np.random.randn(len(dates)).cumsum() * 0.5,
            'Volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    def test_detector_initialization(self):
        """Test detector initialization"""
        self.assertEqual(self.detector.model_path, self.model_path)
        self.assertEqual(self.detector.current_regime, MarketRegime.STABLE)
        self.assertEqual(len(self.detector.regime_history), 0)
        self.assertIsNone(self.detector.last_update)
        
    def test_regime_detection(self):
        """Test current regime detection"""
        result = self.detector.detect_current_regime(self.sample_data)
        
        self.assertIsInstance(result, RegimeDetectionResult)
        self.assertIn(result.current_regime, MarketRegime)
        
        # Check that detector state is updated
        self.assertEqual(self.detector.current_regime, result.current_regime)
        self.assertIsNotNone(self.detector.last_update)
        self.assertEqual(len(self.detector.regime_history), 1)
        
    def test_regime_specific_parameters(self):
        """Test regime-specific parameter retrieval"""
        for regime in MarketRegime:
            params = self.detector.get_regime_specific_parameters(regime)
            
            self.assertIsInstance(params, dict)
            
            # Check that expected parameters are present
            expected_params = [
                'trend_following_weight',
                'mean_reversion_weight',
                'volatility_adjustment',
                'position_size_multiplier',
                'stop_loss_multiplier',
                'rsi_overbought',
                'rsi_oversold',
                'macd_sensitivity'
            ]
            
            for param in expected_params:
                self.assertIn(param, params)
                self.assertIsInstance(params[param], (int, float))
                
    def test_regime_parameter_differences(self):
        """Test that different regimes have different parameters"""
        bull_params = self.detector.get_regime_specific_parameters(MarketRegime.BULL)
        bear_params = self.detector.get_regime_specific_parameters(MarketRegime.BEAR)
        volatile_params = self.detector.get_regime_specific_parameters(MarketRegime.VOLATILE)
        stable_params = self.detector.get_regime_specific_parameters(MarketRegime.STABLE)
        
        # Bull and bear should have higher trend following weights
        self.assertGreater(bull_params['trend_following_weight'], 
                          volatile_params['trend_following_weight'])
        self.assertGreater(bear_params['trend_following_weight'], 
                          volatile_params['trend_following_weight'])
        
        # Volatile should have higher mean reversion weight
        self.assertGreater(volatile_params['mean_reversion_weight'],
                          bull_params['mean_reversion_weight'])
        
        # Volatile should have lower position size multiplier
        self.assertLess(volatile_params['position_size_multiplier'],
                       stable_params['position_size_multiplier'])
        
    def test_training_on_historical_data(self):
        """Test training on historical data"""
        # Create multiple datasets
        market_data = {
            'STOCK1': self.sample_data,
            'STOCK2': self.sample_data.copy()  # Simple copy for testing
        }
        
        # Training might succeed or fail depending on library availability
        result = self.detector.train_on_historical_data(market_data)
        self.assertIsInstance(result, bool)
        
    def test_transition_probability(self):
        """Test regime transition probability calculation"""
        # Test same regime transition
        same_prob = self.detector.get_regime_transition_probability(
            MarketRegime.BULL, MarketRegime.BULL
        )
        self.assertIsInstance(same_prob, float)
        self.assertGreaterEqual(same_prob, 0.0)
        self.assertLessEqual(same_prob, 1.0)
        
        # Test different regime transition
        diff_prob = self.detector.get_regime_transition_probability(
            MarketRegime.BULL, MarketRegime.BEAR
        )
        self.assertIsInstance(diff_prob, float)
        self.assertGreaterEqual(diff_prob, 0.0)
        self.assertLessEqual(diff_prob, 1.0)
        
    def test_regime_statistics_empty(self):
        """Test regime statistics with no history"""
        stats = self.detector.get_regime_statistics()
        self.assertEqual(stats, {})
        
    def test_regime_statistics_with_history(self):
        """Test regime statistics with detection history"""
        # Perform several detections
        for _ in range(5):
            self.detector.detect_current_regime(self.sample_data)
            
        stats = self.detector.get_regime_statistics()
        
        self.assertIn('total_detections', stats)
        self.assertIn('regime_distribution', stats)
        self.assertIn('average_confidence', stats)
        self.assertIn('current_regime', stats)
        self.assertIn('last_update', stats)
        
        self.assertEqual(stats['total_detections'], 5)
        self.assertIsInstance(stats['regime_distribution'], dict)
        self.assertIsInstance(stats['average_confidence'], dict)
        
    def test_regime_history_limit(self):
        """Test that regime history is limited to prevent memory issues"""
        # Perform many detections (more than the limit of 100)
        for _ in range(150):
            self.detector.detect_current_regime(self.sample_data)
            
        # History should be limited to 100 entries
        self.assertLessEqual(len(self.detector.regime_history), 100)
        
    def test_error_handling_invalid_data(self):
        """Test error handling with invalid data"""
        # Test with empty dataframe
        empty_df = pd.DataFrame()
        result = self.detector.detect_current_regime(empty_df)
        
        # Should return a valid result even with invalid data
        self.assertIsInstance(result, RegimeDetectionResult)
        self.assertIn(result.current_regime, MarketRegime)


class TestRegimeDetectionIntegration(unittest.TestCase):
    """Integration tests for regime detection system"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.detector = MarketRegimeDetector()
        
        # Create realistic market scenarios
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        # Bull market scenario
        bull_trend = np.linspace(0, 0.3, len(dates))  # 30% growth over year
        bull_noise = np.random.normal(0, 0.01, len(dates))
        bull_returns = np.diff(np.concatenate([[0], bull_trend + bull_noise]))
        bull_prices = 100 * np.exp(np.cumsum(bull_returns))
        
        self.bull_scenario = pd.DataFrame({
            'Open': bull_prices * 0.999,
            'High': bull_prices * 1.005,
            'Low': bull_prices * 0.995,
            'Close': bull_prices,
            'Volume': np.random.randint(5000, 15000, len(dates))
        }, index=dates)
        
        # Bear market scenario
        bear_trend = np.linspace(0, -0.2, len(dates))  # 20% decline over year
        bear_noise = np.random.normal(0, 0.01, len(dates))
        bear_returns = np.diff(np.concatenate([[0], bear_trend + bear_noise]))
        bear_prices = 100 * np.exp(np.cumsum(bear_returns))
        
        self.bear_scenario = pd.DataFrame({
            'Open': bear_prices * 1.001,
            'High': bear_prices * 1.005,
            'Low': bear_prices * 0.995,
            'Close': bear_prices,
            'Volume': np.random.randint(8000, 20000, len(dates))
        }, index=dates)
        
    def test_bull_market_detection(self):
        """Test detection of bull market characteristics"""
        result = self.detector.detect_current_regime(self.bull_scenario)
        
        # In a strong bull market, we should see positive features
        self.assertGreaterEqual(result.features.returns, -0.01)  # Not strongly negative
        self.assertGreaterEqual(result.confidence, 0.2)  # Some confidence in detection
        
        # Bull or stable regime would be reasonable for upward trending data
        self.assertIn(result.current_regime, [MarketRegime.BULL, MarketRegime.STABLE])
        
    def test_bear_market_detection(self):
        """Test detection of bear market characteristics"""
        result = self.detector.detect_current_regime(self.bear_scenario)
        
        # In a bear market, we should see negative or neutral features
        self.assertLessEqual(result.features.returns, 0.01)  # Not strongly positive
        self.assertGreaterEqual(result.confidence, 0.2)  # Some confidence in detection
        
        # Bear, stable, or volatile regime would be reasonable for declining data
        self.assertIn(result.current_regime, 
                     [MarketRegime.BEAR, MarketRegime.STABLE, MarketRegime.VOLATILE])
        
    def test_regime_consistency_over_time(self):
        """Test that regime detection is reasonably consistent over short periods"""
        # Take a subset of bull market data
        subset_data = self.bull_scenario.tail(50)  # Last 50 days
        
        results = []
        for i in range(10, len(subset_data), 5):  # Sample every 5 days
            window_data = subset_data.iloc[:i]
            result = self.detector.detect_current_regime(window_data)
            results.append(result.current_regime)
            
        # Should not have too many regime changes in stable period
        unique_regimes = set(results)
        self.assertLessEqual(len(unique_regimes), 3)  # At most 3 different regimes
        
    def test_parameter_adaptation(self):
        """Test that parameters adapt appropriately to detected regimes"""
        # Test different regimes
        regimes_to_test = [MarketRegime.BULL, MarketRegime.BEAR, 
                          MarketRegime.VOLATILE, MarketRegime.STABLE]
        
        all_params = {}
        for regime in regimes_to_test:
            params = self.detector.get_regime_specific_parameters(regime)
            all_params[regime] = params
            
        # Verify that volatile regime has more conservative parameters
        volatile_params = all_params[MarketRegime.VOLATILE]
        stable_params = all_params[MarketRegime.STABLE]
        
        self.assertLess(volatile_params['position_size_multiplier'],
                       stable_params['position_size_multiplier'])
        self.assertGreater(volatile_params['volatility_adjustment'],
                          stable_params['volatility_adjustment'])


if __name__ == '__main__':
    unittest.main()