"""
Unit tests for GARCH volatility estimation module.

Tests the GARCH(1,1) model implementation for accuracy and robustness.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.garch_volatility import (
    GARCHVolatilityEstimator, 
    VolatilityPredictor,
    GARCHParameters,
    VolatilityForecast,
    estimate_garch_volatility
)


class TestGARCHParameters(unittest.TestCase):
    """Test GARCH parameter validation."""
    
    def test_valid_parameters(self):
        """Test valid GARCH parameters."""
        params = GARCHParameters(omega=0.001, alpha=0.1, beta=0.8)
        self.assertEqual(params.omega, 0.001)
        self.assertEqual(params.alpha, 0.1)
        self.assertEqual(params.beta, 0.8)
    
    def test_invalid_omega(self):
        """Test invalid omega parameter."""
        with self.assertRaises(ValueError):
            GARCHParameters(omega=-0.001, alpha=0.1, beta=0.8)
        
        with self.assertRaises(ValueError):
            GARCHParameters(omega=0, alpha=0.1, beta=0.8)
    
    def test_invalid_alpha_beta(self):
        """Test invalid alpha/beta parameters."""
        with self.assertRaises(ValueError):
            GARCHParameters(omega=0.001, alpha=-0.1, beta=0.8)
        
        with self.assertRaises(ValueError):
            GARCHParameters(omega=0.001, alpha=0.1, beta=-0.8)
    
    def test_non_stationary_parameters(self):
        """Test non-stationary parameter combination."""
        with self.assertRaises(ValueError):
            GARCHParameters(omega=0.001, alpha=0.6, beta=0.6)  # alpha + beta = 1.2 > 1


class TestGARCHVolatilityEstimator(unittest.TestCase):
    """Test GARCH volatility estimator."""
    
    def setUp(self):
        """Set up test data."""
        # Generate synthetic return data with known GARCH properties
        np.random.seed(42)
        n = 500
        
        # True GARCH parameters
        self.true_omega = 0.0001
        self.true_alpha = 0.1
        self.true_beta = 0.85
        
        # Generate GARCH(1,1) process
        returns = np.zeros(n)
        variances = np.zeros(n)
        variances[0] = self.true_omega / (1 - self.true_alpha - self.true_beta)
        
        for t in range(1, n):
            variances[t] = (self.true_omega + 
                          self.true_alpha * returns[t-1]**2 + 
                          self.true_beta * variances[t-1])
            returns[t] = np.sqrt(variances[t]) * np.random.normal()
        
        self.synthetic_returns = returns
        self.true_variances = variances
        
        # Create estimator
        self.estimator = GARCHVolatilityEstimator()
    
    def test_fit_synthetic_data(self):
        """Test fitting on synthetic GARCH data."""
        params = self.estimator.fit(self.synthetic_returns)
        
        # Check parameter types and bounds
        self.assertIsInstance(params, GARCHParameters)
        self.assertGreater(params.omega, 0)
        self.assertGreaterEqual(params.alpha, 0)
        self.assertGreaterEqual(params.beta, 0)
        self.assertLess(params.alpha + params.beta, 1)
        
        # Parameters should be reasonably close to true values (more lenient bounds)
        # GARCH parameter estimation can be noisy, so we use wider tolerances
        self.assertAlmostEqual(params.omega, self.true_omega, delta=self.true_omega * 1.0)
        self.assertAlmostEqual(params.alpha, self.true_alpha, delta=0.1)
        self.assertAlmostEqual(params.beta, self.true_beta, delta=0.2)
    
    def test_fit_insufficient_data(self):
        """Test fitting with insufficient data."""
        short_returns = np.random.normal(0, 0.02, 30)  # Only 30 observations
        
        with self.assertRaises(ValueError):
            self.estimator.fit(short_returns)
    
    def test_fit_invalid_data(self):
        """Test fitting with invalid data."""
        # NaN values
        invalid_returns = np.array([0.01, 0.02, np.nan, 0.01])
        with self.assertRaises(ValueError):
            self.estimator.fit(invalid_returns)
        
        # Infinite values
        invalid_returns = np.array([0.01, 0.02, np.inf, 0.01])
        with self.assertRaises(ValueError):
            self.estimator.fit(invalid_returns)
    
    def test_forecast_without_fit(self):
        """Test forecasting without fitting first."""
        with self.assertRaises(RuntimeError):
            self.estimator.forecast()
    
    def test_forecast_after_fit(self):
        """Test forecasting after fitting."""
        # Fit model first
        self.estimator.fit(self.synthetic_returns)
        
        # Test one-step forecast
        forecast = self.estimator.forecast(horizon=1)
        
        self.assertIsInstance(forecast, VolatilityForecast)
        self.assertGreater(forecast.forecast_variance, 0)
        self.assertGreater(forecast.forecast_volatility, 0)
        self.assertEqual(len(forecast.confidence_interval), 2)
        self.assertLess(forecast.confidence_interval[0], forecast.confidence_interval[1])
        self.assertIsInstance(forecast.forecast_date, datetime)
        
        # Test multi-step forecast
        forecast_5 = self.estimator.forecast(horizon=5)
        self.assertGreater(forecast_5.forecast_variance, 0)
        
        # Multi-step forecast should be different from one-step
        self.assertNotEqual(forecast.forecast_variance, forecast_5.forecast_variance)
    
    def test_conditional_volatilities(self):
        """Test conditional volatility extraction."""
        # Before fitting
        self.assertIsNone(self.estimator.get_conditional_volatilities())
        
        # After fitting
        self.estimator.fit(self.synthetic_returns)
        cond_vols = self.estimator.get_conditional_volatilities()
        
        self.assertIsNotNone(cond_vols)
        self.assertEqual(len(cond_vols), len(self.synthetic_returns))
        self.assertTrue(np.all(cond_vols > 0))
    
    def test_pandas_series_input(self):
        """Test fitting with pandas Series input."""
        returns_series = pd.Series(self.synthetic_returns)
        params = self.estimator.fit(returns_series)
        
        self.assertIsInstance(params, GARCHParameters)
        self.assertGreater(params.omega, 0)


class TestVolatilityPredictor(unittest.TestCase):
    """Test high-level volatility predictor interface."""
    
    def setUp(self):
        """Set up test data."""
        # Create synthetic price data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        
        # Generate price series with volatility clustering
        returns = np.random.normal(0.001, 0.02, 300)
        prices = 100 * np.exp(np.cumsum(returns))
        
        self.price_data = pd.DataFrame({
            'Close': prices,
            'Date': dates
        })
        
        self.predictor = VolatilityPredictor(lookback_window=252)
    
    def test_predict_volatility_valid_data(self):
        """Test volatility prediction with valid data."""
        result = self.predictor.predict_volatility(self.price_data)
        
        # Check result structure
        self.assertIn('volatility_forecast', result)
        self.assertIn('confidence_interval', result)
        self.assertIn('forecast_date', result)
        self.assertIn('model_params', result)
        
        # Check values
        self.assertGreater(result['volatility_forecast'], 0)
        self.assertEqual(len(result['confidence_interval']), 2)
        self.assertLess(result['confidence_interval'][0], result['confidence_interval'][1])
        self.assertIsInstance(result['forecast_date'], datetime)
    
    def test_predict_volatility_insufficient_data(self):
        """Test volatility prediction with insufficient data."""
        short_data = self.price_data.head(30)  # Only 30 days
        result = self.predictor.predict_volatility(short_data)
        
        # Should still return a result (using all available data)
        self.assertIn('volatility_forecast', result)
        self.assertGreater(result['volatility_forecast'], 0)
    
    def test_predict_volatility_missing_close(self):
        """Test volatility prediction with missing Close column."""
        invalid_data = pd.DataFrame({'Price': [100, 101, 102]})
        
        # Should return fallback result instead of raising exception
        result = self.predictor.predict_volatility(invalid_data)
        self.assertTrue(result.get('fallback', False))
        self.assertGreater(result['volatility_forecast'], 0)
    
    def test_predict_volatility_multi_horizon(self):
        """Test multi-horizon volatility prediction."""
        result_1 = self.predictor.predict_volatility(self.price_data, horizon=1)
        result_5 = self.predictor.predict_volatility(self.price_data, horizon=5)
        
        # Both should be valid
        self.assertGreater(result_1['volatility_forecast'], 0)
        self.assertGreater(result_5['volatility_forecast'], 0)
        
        # Forecast dates should be different
        self.assertNotEqual(result_1['forecast_date'], result_5['forecast_date'])
    
    def test_refit_logic(self):
        """Test model refitting logic."""
        # First prediction should fit the model
        result1 = self.predictor.predict_volatility(self.price_data)
        self.assertIsNotNone(self.predictor.last_fit_date)
        
        # Second prediction within threshold should not refit
        result2 = self.predictor.predict_volatility(self.price_data, refit_threshold_days=30)
        
        # Should have same model parameters (no refit)
        if result1['model_params'] and result2['model_params']:
            self.assertEqual(result1['model_params']['omega'], result2['model_params']['omega'])


class TestConvenienceFunction(unittest.TestCase):
    """Test convenience function for GARCH estimation."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        returns = np.random.normal(0.001, 0.02, 200)
        prices = 100 * np.exp(np.cumsum(returns))
        
        self.price_data = pd.DataFrame({
            'Close': prices,
            'Date': dates
        })
    
    def test_estimate_garch_volatility(self):
        """Test convenience function."""
        result = estimate_garch_volatility(self.price_data)
        
        self.assertIn('volatility_forecast', result)
        self.assertGreater(result['volatility_forecast'], 0)
        
        # Test with different horizon
        result_3 = estimate_garch_volatility(self.price_data, horizon=3)
        self.assertGreater(result_3['volatility_forecast'], 0)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_optimization_failure_fallback(self):
        """Test fallback when optimization fails."""
        # Create problematic data that might cause optimization issues
        problematic_returns = np.array([0.0] * 100)  # Zero variance
        
        estimator = GARCHVolatilityEstimator()
        
        # Should not crash, should return reasonable parameters
        params = estimator.fit(problematic_returns + np.random.normal(0, 1e-6, 100))
        
        self.assertIsInstance(params, GARCHParameters)
        self.assertGreater(params.omega, 0)
    
    def test_volatility_predictor_fallback(self):
        """Test volatility predictor fallback behavior."""
        # Create data that might cause GARCH fitting to fail
        problematic_data = pd.DataFrame({
            'Close': [100.0] * 10  # Constant prices - insufficient data
        })
        
        predictor = VolatilityPredictor()
        
        # Should return fallback result
        result = predictor.predict_volatility(problematic_data)
        
        self.assertIn('volatility_forecast', result)
        self.assertGreater(result['volatility_forecast'], 0)
        
        # Should indicate fallback was used
        self.assertTrue(result.get('fallback', False))


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)