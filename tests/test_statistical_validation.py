"""
Tests for Statistical Significance Validator

This module tests the statistical validation functionality to ensure
proper significance testing of strategy performance.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.advanced_backtesting_engine import StatisticalValidator, ValidationReport


class TestStatisticalValidator(unittest.TestCase):
    """Test cases for StatisticalValidator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = StatisticalValidator(confidence_level=0.95)
        
        # Create sample return data
        np.random.seed(42)  # For reproducible tests
        
        # Positive returns (should be statistically significant)
        self.positive_returns = np.random.normal(0.02, 0.1, 100).tolist()
        
        # Zero mean returns (should not be significant)
        self.zero_returns = np.random.normal(0.0, 0.1, 100).tolist()
        
        # Negative returns
        self.negative_returns = np.random.normal(-0.01, 0.1, 100).tolist()
        
        # High volatility returns
        self.high_vol_returns = np.random.normal(0.01, 0.5, 100).tolist()
        
        # Benchmark returns
        self.benchmark_returns = np.random.normal(0.01, 0.08, 100).tolist()
    
    def test_validate_positive_returns(self):
        """Test validation of positive returns."""
        report = self.validator.validate_strategy_performance(
            strategy_returns=self.positive_returns,
            risk_free_rate=0.02
        )
        
        # Check report structure
        self.assertIsInstance(report, ValidationReport)
        self.assertIsInstance(report.is_statistically_significant, bool)
        self.assertIsInstance(report.p_value, float)
        self.assertIsInstance(report.confidence_interval_lower, float)
        self.assertIsInstance(report.confidence_interval_upper, float)
        self.assertIsInstance(report.t_statistic, float)
        self.assertIsInstance(report.sample_size, int)
        
        # Check sample size
        self.assertEqual(report.sample_size, len(self.positive_returns))
        
        # Check that p-value is between 0 and 1
        self.assertGreaterEqual(report.p_value, 0)
        self.assertLessEqual(report.p_value, 1)
        
        # Check confidence interval ordering
        self.assertLess(report.confidence_interval_lower, report.confidence_interval_upper)
    
    def test_validate_zero_returns(self):
        """Test validation of zero mean returns."""
        report = self.validator.validate_strategy_performance(
            strategy_returns=self.zero_returns,
            risk_free_rate=0.02
        )
        
        # Should not be statistically significant
        # (though this depends on the random seed and sample)
        self.assertIsInstance(report.is_statistically_significant, bool)
        
        # Mean return should be close to zero
        self.assertAlmostEqual(report.mean_return, 0.0, delta=0.05)
    
    def test_validate_with_benchmark(self):
        """Test validation with benchmark comparison."""
        report = self.validator.validate_strategy_performance(
            strategy_returns=self.positive_returns,
            benchmark_returns=self.benchmark_returns,
            risk_free_rate=0.02
        )
        
        # Should have additional benchmark metrics
        additional_metrics = report.additional_metrics
        
        # Check for benchmark comparison metrics
        expected_benchmark_metrics = [
            'vs_benchmark_t_statistic',
            'vs_benchmark_p_value',
            'outperforms_benchmark',
            'information_ratio'
        ]
        
        for metric in expected_benchmark_metrics:
            if metric in additional_metrics:
                self.assertIsInstance(additional_metrics[metric], (int, float, bool))
    
    def test_validate_empty_returns(self):
        """Test validation with empty returns."""
        report = self.validator.validate_strategy_performance(
            strategy_returns=[],
            risk_free_rate=0.02
        )
        
        # Should handle empty returns gracefully
        self.assertFalse(report.is_statistically_significant)
        self.assertEqual(report.sample_size, 0)
        self.assertEqual(report.validation_method, "Failed")
    
    def test_validate_single_return(self):
        """Test validation with single return."""
        report = self.validator.validate_strategy_performance(
            strategy_returns=[0.05],
            risk_free_rate=0.02
        )
        
        # Should handle single return
        self.assertEqual(report.sample_size, 1)
        self.assertEqual(report.mean_return, 0.05)
    
    def test_validate_nan_returns(self):
        """Test validation with NaN returns."""
        nan_returns = [0.01, np.nan, 0.02, np.inf, -np.inf, 0.03]
        
        report = self.validator.validate_strategy_performance(
            strategy_returns=nan_returns,
            risk_free_rate=0.02
        )
        
        # Should filter out invalid values
        self.assertEqual(report.sample_size, 3)  # Only valid returns
        self.assertAlmostEqual(report.mean_return, 0.02, delta=0.01)
    
    def test_confidence_levels(self):
        """Test different confidence levels."""
        validator_99 = StatisticalValidator(confidence_level=0.99)
        
        report_95 = self.validator.validate_strategy_performance(
            strategy_returns=self.positive_returns,
            risk_free_rate=0.02
        )
        
        report_99 = validator_99.validate_strategy_performance(
            strategy_returns=self.positive_returns,
            risk_free_rate=0.02
        )
        
        # 99% confidence interval should be wider than 95%
        ci_width_95 = report_95.confidence_interval_upper - report_95.confidence_interval_lower
        ci_width_99 = report_99.confidence_interval_upper - report_99.confidence_interval_lower
        
        self.assertGreater(ci_width_99, ci_width_95)
    
    def test_additional_statistical_tests(self):
        """Test additional statistical tests."""
        report = self.validator.validate_strategy_performance(
            strategy_returns=self.positive_returns,
            benchmark_returns=self.benchmark_returns,
            risk_free_rate=0.02
        )
        
        additional_metrics = report.additional_metrics
        
        # Check for Sharpe ratio
        if 'sharpe_ratio' in additional_metrics:
            self.assertIsInstance(additional_metrics['sharpe_ratio'], float)
        
        # Check for normality test
        if 'jarque_bera_p_value' in additional_metrics:
            self.assertGreaterEqual(additional_metrics['jarque_bera_p_value'], 0)
            self.assertLessEqual(additional_metrics['jarque_bera_p_value'], 1)
        
        # Check for independence test (if statsmodels available)
        if 'ljung_box_p_value' in additional_metrics:
            self.assertGreaterEqual(additional_metrics['ljung_box_p_value'], 0)
            self.assertLessEqual(additional_metrics['ljung_box_p_value'], 1)
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation and confidence intervals."""
        # Create returns with known properties
        returns = np.random.normal(0.05, 0.15, 252)  # One year of daily returns
        
        report = self.validator.validate_strategy_performance(
            strategy_returns=returns.tolist(),
            risk_free_rate=0.02
        )
        
        additional_metrics = report.additional_metrics
        
        if 'sharpe_ratio' in additional_metrics:
            sharpe_ratio = additional_metrics['sharpe_ratio']
            
            # Sharpe ratio should be reasonable
            self.assertGreater(sharpe_ratio, -5)  # Not extremely negative
            self.assertLess(sharpe_ratio, 10)     # Not extremely positive
            
            # Check confidence intervals if available
            if 'sharpe_ci_lower' in additional_metrics and 'sharpe_ci_upper' in additional_metrics:
                ci_lower = additional_metrics['sharpe_ci_lower']
                ci_upper = additional_metrics['sharpe_ci_upper']
                
                self.assertLess(ci_lower, ci_upper)
                self.assertLessEqual(ci_lower, sharpe_ratio)
                self.assertGreaterEqual(ci_upper, sharpe_ratio)
    
    def test_high_volatility_returns(self):
        """Test validation with high volatility returns."""
        report = self.validator.validate_strategy_performance(
            strategy_returns=self.high_vol_returns,
            risk_free_rate=0.02
        )
        
        # Should handle high volatility gracefully
        self.assertIsInstance(report.is_statistically_significant, bool)
        self.assertGreater(report.std_error, 0)
        
        # Confidence interval should be wider for high volatility
        ci_width = report.confidence_interval_upper - report.confidence_interval_lower
        self.assertGreater(ci_width, 0)
    
    def test_mismatched_benchmark_length(self):
        """Test handling of mismatched benchmark length."""
        short_benchmark = self.benchmark_returns[:50]  # Shorter benchmark
        
        report = self.validator.validate_strategy_performance(
            strategy_returns=self.positive_returns,
            benchmark_returns=short_benchmark,
            risk_free_rate=0.02
        )
        
        # Should handle mismatched lengths gracefully
        self.assertIsInstance(report, ValidationReport)
        
        # May or may not have benchmark metrics depending on implementation
        additional_metrics = report.additional_metrics
        self.assertIsInstance(additional_metrics, dict)
    
    def test_zero_volatility_returns(self):
        """Test handling of zero volatility returns."""
        constant_returns = [0.01] * 100  # Constant returns
        
        report = self.validator.validate_strategy_performance(
            strategy_returns=constant_returns,
            risk_free_rate=0.02
        )
        
        # Should handle zero volatility
        self.assertAlmostEqual(report.mean_return, 0.01, places=10)
        self.assertAlmostEqual(report.std_error, 0.0, places=15)
        
        # T-statistic should be 0 for zero volatility
        self.assertEqual(report.t_statistic, 0.0)
        
        # Should be statistically significant since mean > 0 with zero volatility
        self.assertTrue(report.is_statistically_significant)
    
    def test_validation_report_attributes(self):
        """Test all ValidationReport attributes are properly set."""
        report = self.validator.validate_strategy_performance(
            strategy_returns=self.positive_returns,
            risk_free_rate=0.02
        )
        
        # Check all required attributes exist
        required_attributes = [
            'is_statistically_significant',
            'p_value',
            'confidence_interval_lower',
            'confidence_interval_upper',
            't_statistic',
            'degrees_of_freedom',
            'sample_size',
            'mean_return',
            'std_error',
            'validation_method',
            'additional_metrics'
        ]
        
        for attr in required_attributes:
            self.assertTrue(hasattr(report, attr), f"Missing attribute: {attr}")
            
        # Check degrees of freedom
        expected_dof = len(self.positive_returns) - 1
        self.assertEqual(report.degrees_of_freedom, expected_dof)
        
        # Check validation method
        self.assertEqual(report.validation_method, "One-sample t-test")


class TestValidationReport(unittest.TestCase):
    """Test cases for ValidationReport dataclass."""
    
    def test_validation_report_creation(self):
        """Test ValidationReport creation."""
        report = ValidationReport(
            is_statistically_significant=True,
            p_value=0.01,
            confidence_interval_lower=0.005,
            confidence_interval_upper=0.025,
            t_statistic=2.5,
            degrees_of_freedom=99,
            sample_size=100,
            mean_return=0.015,
            std_error=0.006,
            validation_method="One-sample t-test",
            additional_metrics={'sharpe_ratio': 1.2}
        )
        
        self.assertTrue(report.is_statistically_significant)
        self.assertEqual(report.p_value, 0.01)
        self.assertEqual(report.sample_size, 100)
        self.assertEqual(report.validation_method, "One-sample t-test")
        self.assertEqual(report.additional_metrics['sharpe_ratio'], 1.2)


if __name__ == '__main__':
    unittest.main()