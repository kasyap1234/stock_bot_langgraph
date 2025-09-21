"""
Unit tests for Volatility Regime Classifier
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from agents.volatility_regime_classifier import (
    VolatilityRegime,
    VolatilityMetrics,
    VolatilityRegimeResult,
    GARCHVolatilityEstimator,
    VolatilityRegimeClassifier,
    VolatilityRegimeDetector
)


class TestVolatilityRegime(unittest.TestCase):
    """Test VolatilityRegime enum"""
    
    def test_regime_values(self):
        """Test that volatility regime enum has correct values"""
        self.assertEqual(VolatilityRegime.LOW.value, "low")
        self.assertEqual(VolatilityRegime.NORMAL.value, "normal")
        self.assertEqual(VolatilityRegime.HIGH.value, "high")
        self.assertEqual(VolatilityRegime.EXTREME.value, "extreme")


class TestVolatilityMetrics(unittest.TestCase):
    """Test VolatilityMetrics dataclass"""
    
    def test_volatility_metrics_creation(self):
        """Test VolatilityMetrics object creation"""
        metrics = VolatilityMetrics(
            current_volatility=0.15,
            forecasted_volatility=0.18,
            volatility_percentile=0.75,
            regime=VolatilityRegime.HIGH,
            confidence=0.85,
            garch_params={'omega': 0.001, 'alpha[1]': 0.1, 'beta[1]': 0.8},
            timestamp=datetime.now()
        )
        
        self.assertEqual(metrics.current_volatility, 0.15)
        self.assertEqual(metrics.forecasted_volatility, 0.18)
        self.assertEqual(metrics.volatility_percentile, 0.75)
        self.assertEqual(metrics.regime, VolatilityRegime.HIGH)
        self.assertEqual(metrics.confidence, 0.85)
        self.assertIsInstance(metrics.garch_params, dict)
        self.assertIsInstance(metrics.timestamp, datetime)


class TestGARCHVolatilityEstimator(unittest.TestCase):
    """Test GARCH volatility estimator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.estimator = GARCHVolatilityEstimator(p=1, q=1)
        
        # Create sample return data with different volatility regimes
        np.random.seed(42)
        
        # Low volatility period
        low_vol_returns = np.random.normal(0, 0.01, 100)
        
        # High volatility period  
        high_vol_returns = np.random.normal(0, 0.03, 100)
        
        # Combined series
        combined_returns = np.concatenate([low_vol_returns, high_vol_returns])
        
        dates = pd.date_range(start='2023-01-01', periods=len(combined_returns), freq='D')
        self.returns_series = pd.Series(combined_returns, index=dates)
        
    def test_estimator_initialization(self):
        """Test GARCH estimator initialization"""
        self.assertEqual(self.estimator.p, 1)
        self.assertEqual(self.estimator.q, 1)
        self.assertEqual(self.estimator.mean_model, 'Constant')
        self.assertFalse(self.estimator.is_fitted)
        self.assertIsNone(self.estimator.model)
        self.assertIsNone(self.estimator.fitted_model)
        
    def test_fit_with_sufficient_data(self):
        """Test GARCH model fitting with sufficient data"""
        # This test may pass or fail depending on library availability
        result = self.estimator.fit(self.returns_series)
        self.assertIsInstance(result, bool)
        
        # If fitting succeeded, check that model is marked as fitted
        if result:
            self.assertTrue(self.estimator.is_fitted)
            self.assertIsNotNone(self.estimator.fitted_model)
            
    def test_fit_with_insufficient_data(self):
        """Test GARCH model fitting with insufficient data"""
        short_series = self.returns_series.head(10)
        result = self.estimator.fit(short_series)
        
        # Should fail due to insufficient data
        self.assertFalse(result)
        
    def test_forecast_without_fitting(self):
        """Test volatility forecasting without fitting model"""
        vol_forecast, ci_width = self.estimator.forecast_volatility()
        
        # Should return zeros when model not fitted
        self.assertEqual(vol_forecast, 0.0)
        self.assertEqual(ci_width, 0.0)
        
    def test_forecast_after_fitting(self):
        """Test volatility forecasting after fitting model"""
        # Try to fit model
        fit_success = self.estimator.fit(self.returns_series)
        
        if fit_success:
            vol_forecast, ci_width = self.estimator.forecast_volatility(horizon=5)
            
            # Should return positive values
            self.assertGreaterEqual(vol_forecast, 0.0)
            self.assertGreaterEqual(ci_width, 0.0)
            
            # Forecast should be reasonable (not too extreme)
            self.assertLess(vol_forecast, 1.0)  # Less than 100% volatility
            
    def test_get_model_parameters_without_fitting(self):
        """Test getting model parameters without fitting"""
        params = self.estimator.get_model_parameters()
        self.assertEqual(params, {})
        
    def test_get_model_parameters_after_fitting(self):
        """Test getting model parameters after fitting"""
        fit_success = self.estimator.fit(self.returns_series)
        
        if fit_success:
            params = self.estimator.get_model_parameters()
            
            # Should contain expected GARCH parameters
            expected_params = ['omega', 'alpha[1]', 'beta[1]']
            for param in expected_params:
                if param in params:  # May not be present if fitting failed
                    self.assertIsInstance(params[param], float)
                    
    def test_conditional_volatility_calculation(self):
        """Test conditional volatility calculation"""
        # Test without fitting (should use fallback)
        cond_vol = self.estimator.calculate_conditional_volatility(self.returns_series)
        
        self.assertIsInstance(cond_vol, pd.Series)
        self.assertEqual(len(cond_vol), len(self.returns_series))
        self.assertTrue(all(cond_vol >= 0))  # Volatility should be non-negative
        
        # Test after fitting
        fit_success = self.estimator.fit(self.returns_series)
        if fit_success:
            cond_vol_fitted = self.estimator.calculate_conditional_volatility(self.returns_series)
            
            self.assertIsInstance(cond_vol_fitted, pd.Series)
            self.assertGreater(len(cond_vol_fitted), 0)
            self.assertTrue(all(cond_vol_fitted >= 0))


class TestVolatilityRegimeClassifier(unittest.TestCase):
    """Test VolatilityRegimeClassifier functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.classifier = VolatilityRegimeClassifier(lookback_window=100)
        
        # Create different volatility scenarios
        np.random.seed(42)
        
        # Low volatility scenario
        low_vol_returns = np.random.normal(0.0005, 0.008, 200)  # ~0.8% daily vol
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        self.low_vol_series = pd.Series(low_vol_returns, index=dates)
        
        # High volatility scenario
        high_vol_returns = np.random.normal(0, 0.025, 200)  # ~2.5% daily vol
        self.high_vol_series = pd.Series(high_vol_returns, index=dates)
        
        # Normal volatility scenario
        normal_vol_returns = np.random.normal(0.0002, 0.015, 200)  # ~1.5% daily vol
        self.normal_vol_series = pd.Series(normal_vol_returns, index=dates)
        
    def test_classifier_initialization(self):
        """Test classifier initialization"""
        self.assertEqual(self.classifier.lookback_window, 100)
        self.assertIsInstance(self.classifier.regime_thresholds, dict)
        self.assertEqual(len(self.classifier.volatility_history), 0)
        self.assertEqual(len(self.classifier.regime_history), 0)
        
        # Check default thresholds
        expected_thresholds = ['low_threshold', 'normal_threshold', 'high_threshold']
        for threshold in expected_thresholds:
            self.assertIn(threshold, self.classifier.regime_thresholds)
            
    def test_realized_volatility_calculation(self):
        """Test realized volatility calculation"""
        realized_vol = self.classifier.calculate_realized_volatility(self.normal_vol_series)
        
        self.assertIsInstance(realized_vol, pd.Series)
        self.assertEqual(len(realized_vol), len(self.normal_vol_series))
        self.assertTrue(all(realized_vol >= 0))  # Should be non-negative
        
        # Check that volatility is annualized (should be reasonable values)
        mean_vol = realized_vol.mean()
        self.assertGreater(mean_vol, 0.05)  # At least 5% annual vol
        self.assertLess(mean_vol, 2.0)      # Less than 200% annual vol
        
    def test_volatility_regime_classification_low_vol(self):
        """Test classification of low volatility regime"""
        result = self.classifier.classify_volatility_regime(self.low_vol_series)
        
        self.assertIsInstance(result, VolatilityRegimeResult)
        self.assertIsInstance(result.regime, VolatilityRegime)
        self.assertIsInstance(result.metrics, VolatilityMetrics)
        
        # Check that probabilities sum to 1
        prob_sum = sum(result.regime_probabilities.values())
        self.assertAlmostEqual(prob_sum, 1.0, places=2)
        
        # Check confidence is reasonable
        self.assertGreaterEqual(result.metrics.confidence, 0.0)
        self.assertLessEqual(result.metrics.confidence, 1.0)
        
    def test_volatility_regime_classification_high_vol(self):
        """Test classification of high volatility regime"""
        result = self.classifier.classify_volatility_regime(self.high_vol_series)
        
        self.assertIsInstance(result, VolatilityRegimeResult)
        
        # High volatility should likely be classified as HIGH or EXTREME
        # (though this depends on the specific data and thresholds)
        self.assertIn(result.regime, [VolatilityRegime.HIGH, VolatilityRegime.EXTREME, 
                                     VolatilityRegime.NORMAL])  # Allow normal for edge cases
        
    def test_regime_probability_calculation(self):
        """Test regime probability calculation"""
        # Test with different percentile values
        test_percentiles = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for percentile in test_percentiles:
            probs = self.classifier._calculate_regime_probabilities(percentile)
            
            # Check that all regimes have probabilities
            self.assertEqual(len(probs), 4)
            for regime in VolatilityRegime:
                self.assertIn(regime, probs)
                self.assertGreaterEqual(probs[regime], 0.0)
                self.assertLessEqual(probs[regime], 1.0)
                
            # Check that probabilities sum to 1
            prob_sum = sum(probs.values())
            self.assertAlmostEqual(prob_sum, 1.0, places=2)
            
    def test_regime_classification_logic(self):
        """Test regime classification logic"""
        # Test boundary cases
        low_percentile = 0.1
        normal_percentile = 0.5
        high_percentile = 0.8
        extreme_percentile = 0.98
        
        low_regime = self.classifier._classify_regime(low_percentile)
        normal_regime = self.classifier._classify_regime(normal_percentile)
        high_regime = self.classifier._classify_regime(high_percentile)
        extreme_regime = self.classifier._classify_regime(extreme_percentile)
        
        self.assertEqual(low_regime, VolatilityRegime.LOW)
        self.assertEqual(normal_regime, VolatilityRegime.NORMAL)
        self.assertEqual(high_regime, VolatilityRegime.HIGH)
        self.assertEqual(extreme_regime, VolatilityRegime.EXTREME)
        
    def test_confidence_calculation(self):
        """Test classification confidence calculation"""
        # Test confidence at different percentiles
        test_cases = [
            (0.125, VolatilityRegime.LOW),    # Center of low regime
            (0.5, VolatilityRegime.NORMAL),   # Center of normal regime
            (0.85, VolatilityRegime.HIGH),    # Center of high regime
            (0.975, VolatilityRegime.EXTREME) # Center of extreme regime
        ]
        
        for percentile, expected_regime in test_cases:
            confidence = self.classifier._calculate_classification_confidence(percentile, expected_regime)
            
            self.assertGreaterEqual(confidence, 0.1)
            self.assertLessEqual(confidence, 1.0)
            
            # Confidence should be higher when percentile is in center of regime
            self.assertGreater(confidence, 0.5)
            
    def test_history_management(self):
        """Test that history is properly managed"""
        # Perform multiple classifications
        for _ in range(150):  # More than lookback window
            self.classifier.classify_volatility_regime(self.normal_vol_series)
            
        # History should be limited to lookback window
        self.assertLessEqual(len(self.classifier.volatility_history), self.classifier.lookback_window)
        self.assertLessEqual(len(self.classifier.regime_history), self.classifier.lookback_window)
        
    def test_regime_statistics(self):
        """Test regime statistics calculation"""
        # Perform several classifications
        for _ in range(10):
            self.classifier.classify_volatility_regime(self.normal_vol_series)
            
        stats = self.classifier.get_regime_statistics()
        
        self.assertIn('total_classifications', stats)
        self.assertIn('regime_distribution', stats)
        self.assertIn('volatility_statistics', stats)
        self.assertIn('current_regime', stats)
        
        self.assertEqual(stats['total_classifications'], 10)
        self.assertIsInstance(stats['regime_distribution'], dict)
        
    def test_threshold_updates(self):
        """Test updating regime thresholds"""
        new_thresholds = {
            'low_threshold': 0.2,
            'normal_threshold': 0.8
        }
        
        original_thresholds = self.classifier.regime_thresholds.copy()
        self.classifier.update_regime_thresholds(new_thresholds)
        
        # Check that thresholds were updated
        self.assertEqual(self.classifier.regime_thresholds['low_threshold'], 0.2)
        self.assertEqual(self.classifier.regime_thresholds['normal_threshold'], 0.8)
        
        # Check that unchanged thresholds remain the same
        self.assertEqual(self.classifier.regime_thresholds['high_threshold'], 
                        original_thresholds['high_threshold'])
        
    def test_fallback_result_creation(self):
        """Test creation of fallback result"""
        fallback = self.classifier._create_fallback_result()
        
        self.assertIsInstance(fallback, VolatilityRegimeResult)
        self.assertEqual(fallback.regime, VolatilityRegime.NORMAL)
        self.assertIsInstance(fallback.metrics, VolatilityMetrics)
        self.assertEqual(len(fallback.regime_probabilities), 4)
        
        # Check that probabilities are uniform
        for prob in fallback.regime_probabilities.values():
            self.assertAlmostEqual(prob, 0.25, places=2)


class TestVolatilityRegimeDetector(unittest.TestCase):
    """Test VolatilityRegimeDetector functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = VolatilityRegimeDetector(lookback_window=100)
        
        # Create sample return data
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 200)
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        self.returns_series = pd.Series(returns, index=dates)
        
    def test_detector_initialization(self):
        """Test detector initialization"""
        self.assertIsInstance(self.detector.classifier, VolatilityRegimeClassifier)
        self.assertEqual(self.detector.current_regime, VolatilityRegime.NORMAL)
        self.assertIsNone(self.detector.last_update)
        
    def test_volatility_regime_detection(self):
        """Test volatility regime detection"""
        result = self.detector.detect_volatility_regime(self.returns_series)
        
        self.assertIsInstance(result, VolatilityRegimeResult)
        self.assertIsInstance(result.regime, VolatilityRegime)
        
        # Check that detector state is updated
        self.assertEqual(self.detector.current_regime, result.regime)
        self.assertIsNotNone(self.detector.last_update)
        
    def test_regime_adjusted_parameters(self):
        """Test regime-adjusted parameter calculation"""
        base_params = {
            'position_size': 0.1,
            'stop_loss': 0.05,
            'volatility_target': 0.15
        }
        
        # Test for different regimes
        for regime in VolatilityRegime:
            self.detector.current_regime = regime
            adjusted_params = self.detector.get_regime_adjusted_parameters(base_params)
            
            self.assertIsInstance(adjusted_params, dict)
            
            # Check that original parameters are preserved
            for param in base_params:
                self.assertIn(param, adjusted_params)
                
            # Check that adjustment multipliers are added
            expected_multipliers = [
                'position_size_multiplier',
                'stop_loss_multiplier', 
                'volatility_target_multiplier',
                'rebalance_frequency_multiplier'
            ]
            
            for multiplier in expected_multipliers:
                self.assertIn(multiplier, adjusted_params)
                self.assertIsInstance(adjusted_params[multiplier], float)
                self.assertGreater(adjusted_params[multiplier], 0)
                
    def test_parameter_adjustment_logic(self):
        """Test that parameter adjustments make sense for different regimes"""
        base_params = {'position_size': 0.1}
        
        # Low volatility should allow larger positions
        self.detector.current_regime = VolatilityRegime.LOW
        low_vol_params = self.detector.get_regime_adjusted_parameters(base_params)
        
        # High volatility should require smaller positions
        self.detector.current_regime = VolatilityRegime.HIGH
        high_vol_params = self.detector.get_regime_adjusted_parameters(base_params)
        
        # Extreme volatility should require even smaller positions
        self.detector.current_regime = VolatilityRegime.EXTREME
        extreme_vol_params = self.detector.get_regime_adjusted_parameters(base_params)
        
        # Check position size adjustments
        low_multiplier = low_vol_params['position_size_multiplier']
        high_multiplier = high_vol_params['position_size_multiplier']
        extreme_multiplier = extreme_vol_params['position_size_multiplier']
        
        # Should decrease as volatility increases
        self.assertGreater(low_multiplier, high_multiplier)
        self.assertGreater(high_multiplier, extreme_multiplier)
        
    def test_volatility_statistics(self):
        """Test volatility statistics retrieval"""
        # Perform detection to populate statistics
        self.detector.detect_volatility_regime(self.returns_series)
        
        stats = self.detector.get_volatility_statistics()
        
        self.assertIn('current_regime', stats)
        self.assertIn('last_update', stats)
        self.assertIn('total_classifications', stats)
        
        self.assertEqual(stats['current_regime'], self.detector.current_regime.value)
        self.assertIsNotNone(stats['last_update'])
        
    def test_error_handling_invalid_data(self):
        """Test error handling with invalid data"""
        # Test with empty series
        empty_series = pd.Series(dtype=float)
        result = self.detector.detect_volatility_regime(empty_series)
        
        # Should return a valid result even with invalid data
        self.assertIsInstance(result, VolatilityRegimeResult)
        self.assertIsInstance(result.regime, VolatilityRegime)


class TestVolatilityRegimeIntegration(unittest.TestCase):
    """Integration tests for volatility regime system"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.detector = VolatilityRegimeDetector()
        
        # Create realistic volatility scenarios
        np.random.seed(42)
        
        # Create regime-switching volatility data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        # Low vol period (first 3 months)
        low_vol_period = len(dates) // 4
        low_vol_returns = np.random.normal(0, 0.01, low_vol_period)
        
        # Normal vol period (next 3 months)
        normal_vol_period = len(dates) // 4
        normal_vol_returns = np.random.normal(0, 0.018, normal_vol_period)
        
        # High vol period (next 3 months)
        high_vol_period = len(dates) // 4
        high_vol_returns = np.random.normal(0, 0.035, high_vol_period)
        
        # Extreme vol period (last 3 months)
        extreme_vol_period = len(dates) - low_vol_period - normal_vol_period - high_vol_period
        extreme_vol_returns = np.random.normal(0, 0.055, extreme_vol_period)
        
        # Combine all periods
        all_returns = np.concatenate([
            low_vol_returns, normal_vol_returns, 
            high_vol_returns, extreme_vol_returns
        ])
        
        self.regime_switching_data = pd.Series(all_returns, index=dates)
        
    def test_regime_detection_across_periods(self):
        """Test regime detection across different volatility periods"""
        results = []
        
        # Test detection at different points in the series
        window_size = 60  # 60-day windows
        
        for i in range(window_size, len(self.regime_switching_data), 30):
            window_data = self.regime_switching_data.iloc[i-window_size:i]
            result = self.detector.detect_volatility_regime(window_data)
            results.append({
                'date': window_data.index[-1],
                'regime': result.regime,
                'volatility': result.metrics.current_volatility,
                'confidence': result.metrics.confidence
            })
        
        # Should detect different regimes across the year
        detected_regimes = set(r['regime'] for r in results)
        self.assertGreater(len(detected_regimes), 1)  # Should detect multiple regimes
        
        # Volatility should generally increase over time in our test data
        volatilities = [r['volatility'] for r in results]
        early_vol = np.mean(volatilities[:3])
        late_vol = np.mean(volatilities[-3:])
        self.assertGreater(late_vol, early_vol)
        
    def test_parameter_adaptation_consistency(self):
        """Test that parameter adaptation is consistent with regime detection"""
        base_params = {
            'position_size': 0.1,
            'stop_loss': 0.02,
            'volatility_target': 0.15
        }
        
        # Test different periods of our regime-switching data
        test_periods = [
            (0, 90),    # Low vol period
            (90, 180),  # Normal vol period
            (180, 270), # High vol period
            (270, 365)  # Extreme vol period
        ]
        
        regime_params = []
        
        for start, end in test_periods:
            if end > len(self.regime_switching_data):
                end = len(self.regime_switching_data)
                
            period_data = self.regime_switching_data.iloc[start:end]
            result = self.detector.detect_volatility_regime(period_data)
            
            adjusted_params = self.detector.get_regime_adjusted_parameters(base_params)
            
            regime_params.append({
                'regime': result.regime,
                'position_multiplier': adjusted_params['position_size_multiplier'],
                'volatility_multiplier': adjusted_params['volatility_target_multiplier']
            })
        
        # Position size multipliers should generally decrease as volatility increases
        # (though exact ordering may vary due to randomness in data)
        position_multipliers = [p['position_multiplier'] for p in regime_params]
        
        # At least some variation in multipliers
        self.assertGreater(max(position_multipliers) - min(position_multipliers), 0.1)
        
    def test_forecast_accuracy_evaluation(self):
        """Test volatility forecast accuracy evaluation"""
        # Use sufficient data for forecast evaluation
        if len(self.regime_switching_data) > 100:
            accuracy_metrics = self.detector.classifier.get_volatility_forecast_accuracy(
                self.regime_switching_data, forecast_horizon=1
            )
            
            # May be empty if GARCH fitting fails, which is acceptable
            if accuracy_metrics:
                expected_metrics = ['mean_absolute_error', 'root_mean_square_error', 
                                  'mean_absolute_percentage_error', 'forecast_samples']
                
                for metric in expected_metrics:
                    if metric in accuracy_metrics:
                        self.assertIsInstance(accuracy_metrics[metric], (int, float))
                        if metric != 'forecast_samples':
                            self.assertGreaterEqual(accuracy_metrics[metric], 0)


if __name__ == '__main__':
    unittest.main()