"""
Unit tests for enhanced Kelly Criterion calculator.

Tests the Kelly Criterion implementation with volatility adjustments and
portfolio correlation considerations.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.enhanced_kelly_criterion import (
    EnhancedKellyCriterion,
    KellyParameters,
    KellyResult,
    PortfolioKellyResult,
    calculate_enhanced_kelly
)


class TestKellyParameters(unittest.TestCase):
    """Test Kelly parameters validation."""
    
    def test_valid_parameters(self):
        """Test valid Kelly parameters."""
        params = KellyParameters(
            expected_return=0.12,
            volatility=0.20,
            risk_free_rate=0.02
        )
        
        self.assertEqual(params.expected_return, 0.12)
        self.assertEqual(params.volatility, 0.20)
        self.assertEqual(params.risk_free_rate, 0.02)
    
    def test_invalid_volatility(self):
        """Test invalid volatility parameter."""
        with self.assertRaises(ValueError):
            KellyParameters(
                expected_return=0.12,
                volatility=0.0,
                risk_free_rate=0.02
            )
        
        with self.assertRaises(ValueError):
            KellyParameters(
                expected_return=0.12,
                volatility=-0.1,
                risk_free_rate=0.02
            )
    
    def test_low_expected_return_warning(self):
        """Test warning for low expected return."""
        with self.assertLogs(level='WARNING'):
            KellyParameters(
                expected_return=0.01,  # Lower than risk-free rate
                volatility=0.20,
                risk_free_rate=0.02
            )


class TestEnhancedKellyCriterion(unittest.TestCase):
    """Test enhanced Kelly Criterion calculator."""
    
    def setUp(self):
        """Set up test data."""
        self.calculator = EnhancedKellyCriterion(
            max_kelly_fraction=0.25,
            min_kelly_fraction=0.01
        )
        
        # Standard test parameters
        self.test_params = KellyParameters(
            expected_return=0.15,
            volatility=0.20,
            risk_free_rate=0.02,
            win_probability=0.6,
            avg_win=0.02,
            avg_loss=0.015
        )
        
        # Generate synthetic return data
        np.random.seed(42)
        n_periods = 252
        
        # Asset with positive expected return
        returns_1 = np.random.normal(0.15/252, 0.20/np.sqrt(252), n_periods)
        returns_2 = np.random.normal(0.10/252, 0.25/np.sqrt(252), n_periods)
        returns_3 = np.random.normal(0.08/252, 0.15/np.sqrt(252), n_periods)
        
        dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')
        
        self.returns_data = pd.DataFrame({
            'STOCK_A': returns_1,
            'STOCK_B': returns_2,
            'STOCK_C': returns_3
        }, index=dates)
    
    def test_initialization(self):
        """Test calculator initialization."""
        calculator = EnhancedKellyCriterion(
            max_kelly_fraction=0.3,
            min_kelly_fraction=0.005,
            volatility_adjustment_factor=1.5
        )
        
        self.assertEqual(calculator.max_kelly_fraction, 0.3)
        self.assertEqual(calculator.min_kelly_fraction, 0.005)
        self.assertEqual(calculator.volatility_adjustment_factor, 1.5)
    
    def test_continuous_kelly_calculation(self):
        """Test continuous Kelly formula."""
        result = self.calculator.calculate_kelly_fraction(
            self.test_params, 
            method='continuous'
        )
        
        # Check result structure
        self.assertIsInstance(result, KellyResult)
        self.assertEqual(result.calculation_method, 'continuous')
        
        # Check Kelly fraction is positive and reasonable
        self.assertGreater(result.kelly_fraction, 0)
        self.assertLess(result.kelly_fraction, 1)
        
        # Check adjustments
        self.assertGreater(result.adjusted_kelly, 0)
        self.assertLessEqual(result.adjusted_kelly, result.kelly_fraction)
        
        # Check confidence interval
        self.assertEqual(len(result.confidence_interval), 2)
        self.assertLess(result.confidence_interval[0], result.confidence_interval[1])
    
    def test_discrete_kelly_calculation(self):
        """Test discrete Kelly formula."""
        result = self.calculator.calculate_kelly_fraction(
            self.test_params, 
            method='discrete'
        )
        
        self.assertIsInstance(result, KellyResult)
        self.assertEqual(result.calculation_method, 'discrete')
        self.assertGreater(result.kelly_fraction, 0)
    
    def test_fractional_kelly_calculation(self):
        """Test fractional Kelly formula."""
        result = self.calculator.calculate_kelly_fraction(
            self.test_params, 
            method='fractional'
        )
        
        self.assertIsInstance(result, KellyResult)
        self.assertEqual(result.calculation_method, 'fractional')
        self.assertGreater(result.kelly_fraction, 0)
        
        # Fractional Kelly should be more conservative
        continuous_result = self.calculator.calculate_kelly_fraction(
            self.test_params, 
            method='continuous'
        )
        self.assertLessEqual(result.kelly_fraction, continuous_result.kelly_fraction)
    
    def test_invalid_method(self):
        """Test invalid calculation method."""
        with self.assertRaises(ValueError):
            self.calculator.calculate_kelly_fraction(
                self.test_params, 
                method='invalid_method'
            )
    
    def test_kelly_fraction_bounds(self):
        """Test Kelly fraction bounds enforcement."""
        # High expected return should be capped
        high_return_params = KellyParameters(
            expected_return=1.0,  # 100% expected return
            volatility=0.20,
            risk_free_rate=0.02
        )
        
        result = self.calculator.calculate_kelly_fraction(high_return_params)
        
        # Should be capped at max_kelly_fraction
        self.assertEqual(result.kelly_fraction, self.calculator.max_kelly_fraction)
        self.assertIn("capped", " ".join(result.warnings))
    
    def test_zero_expected_return(self):
        """Test handling of zero or negative expected return."""
        zero_return_params = KellyParameters(
            expected_return=0.02,  # Same as risk-free rate
            volatility=0.20,
            risk_free_rate=0.02
        )
        
        result = self.calculator.calculate_kelly_fraction(zero_return_params)
        
        # Should return minimum Kelly fraction
        self.assertEqual(result.kelly_fraction, self.calculator.min_kelly_fraction)
    
    def test_volatility_adjustment(self):
        """Test volatility-based adjustments."""
        # Low volatility asset
        low_vol_params = KellyParameters(
            expected_return=0.15,
            volatility=0.10,  # Low volatility
            risk_free_rate=0.02
        )
        
        # High volatility asset
        high_vol_params = KellyParameters(
            expected_return=0.15,
            volatility=0.50,  # High volatility
            risk_free_rate=0.02
        )
        
        low_vol_result = self.calculator.calculate_kelly_fraction(low_vol_params)
        high_vol_result = self.calculator.calculate_kelly_fraction(high_vol_params)
        
        # High volatility should result in lower adjusted Kelly
        self.assertGreater(low_vol_result.risk_adjusted_kelly, 
                          high_vol_result.risk_adjusted_kelly)
    
    def test_parameter_estimation_from_returns(self):
        """Test parameter estimation from historical returns."""
        returns_series = self.returns_data['STOCK_A']
        
        params = self.calculator.estimate_parameters_from_returns(
            returns_series, 
            risk_free_rate=0.02
        )
        
        # Check parameter structure
        self.assertIsInstance(params, KellyParameters)
        self.assertGreater(params.expected_return, 0)
        self.assertGreater(params.volatility, 0)
        self.assertEqual(params.risk_free_rate, 0.02)
        
        # Check win/loss statistics
        self.assertIsNotNone(params.win_probability)
        self.assertIsNotNone(params.avg_win)
        self.assertIsNotNone(params.avg_loss)
        
        self.assertGreaterEqual(params.win_probability, 0)
        self.assertLessEqual(params.win_probability, 1)
    
    def test_parameter_estimation_insufficient_data(self):
        """Test parameter estimation with insufficient data."""
        short_returns = pd.Series([0.01, 0.02, -0.01])  # Only 3 observations
        
        with self.assertRaises(ValueError):
            self.calculator.estimate_parameters_from_returns(short_returns)
    
    def test_parameter_estimation_methods(self):
        """Test different parameter estimation methods."""
        returns_series = self.returns_data['STOCK_A']
        
        # Test different methods
        for method in ['historical', 'robust', 'bayesian']:
            params = self.calculator.estimate_parameters_from_returns(
                returns_series, 
                estimation_method=method
            )
            
            self.assertIsInstance(params, KellyParameters)
            self.assertGreater(params.volatility, 0)
    
    def test_portfolio_kelly_calculation(self):
        """Test portfolio-level Kelly calculation."""
        # Create asset parameters
        asset_parameters = {}
        for column in self.returns_data.columns:
            returns_series = self.returns_data[column]
            asset_parameters[column] = self.calculator.estimate_parameters_from_returns(
                returns_series
            )
        
        # Create correlation matrix
        correlation_matrix = self.returns_data.corr().values
        asset_names = list(self.returns_data.columns)
        
        # Calculate portfolio Kelly
        portfolio_result = self.calculator.calculate_portfolio_kelly(
            asset_parameters, 
            correlation_matrix, 
            asset_names
        )
        
        # Check result structure
        self.assertIsInstance(portfolio_result, PortfolioKellyResult)
        
        # Check individual Kelly fractions
        self.assertEqual(len(portfolio_result.individual_kellys), len(asset_names))
        for asset_name in asset_names:
            self.assertIn(asset_name, portfolio_result.individual_kellys)
            self.assertGreaterEqual(portfolio_result.individual_kellys[asset_name], 0)
        
        # Check optimal weights
        self.assertEqual(len(portfolio_result.optimal_weights), len(asset_names))
        
        # Weights should be non-negative and sum to reasonable total
        total_weight = sum(portfolio_result.optimal_weights.values())
        self.assertGreater(total_weight, 0)
        self.assertLessEqual(total_weight, 1.0)  # Should not exceed 100% due to correlations
        
        # Check other metrics
        self.assertGreaterEqual(portfolio_result.correlation_adjustment, 0)
        self.assertLessEqual(portfolio_result.correlation_adjustment, 1)
        # Diversification benefit can be negative in some cases (concentrated portfolios)
        self.assertIsInstance(portfolio_result.diversification_benefit, (int, float))
    
    def test_portfolio_optimization_methods(self):
        """Test different portfolio optimization methods."""
        asset_parameters = {}
        for column in self.returns_data.columns:
            returns_series = self.returns_data[column]
            asset_parameters[column] = self.calculator.estimate_parameters_from_returns(
                returns_series
            )
        
        correlation_matrix = self.returns_data.corr().values
        asset_names = list(self.returns_data.columns)
        
        # Test different optimization methods
        for method in ['mean_variance', 'risk_parity', 'equal_risk']:
            portfolio_result = self.calculator.calculate_portfolio_kelly(
                asset_parameters, 
                correlation_matrix, 
                asset_names,
                method=method
            )
            
            self.assertIsInstance(portfolio_result, PortfolioKellyResult)
            self.assertEqual(len(portfolio_result.optimal_weights), len(asset_names))
    
    def test_correlation_adjustment(self):
        """Test correlation-based adjustments."""
        # Perfect correlation matrix
        perfect_corr = np.ones((3, 3))
        weights = [0.33, 0.33, 0.34]
        
        adjustment_perfect = self.calculator._calculate_correlation_adjustment(
            perfect_corr, weights
        )
        
        # Zero correlation matrix
        zero_corr = np.eye(3)
        adjustment_zero = self.calculator._calculate_correlation_adjustment(
            zero_corr, weights
        )
        
        # Perfect correlation should have lower adjustment than zero correlation
        self.assertLess(adjustment_perfect, adjustment_zero)
        
        # Both should be between 0 and 1
        self.assertGreaterEqual(adjustment_perfect, 0)
        self.assertLessEqual(adjustment_perfect, 1)
        self.assertGreaterEqual(adjustment_zero, 0)
        self.assertLessEqual(adjustment_zero, 1)
    
    def test_diversification_benefit(self):
        """Test diversification benefit calculation."""
        # Perfect correlation (no diversification)
        perfect_corr = np.ones((3, 3))
        weights = [0.33, 0.33, 0.34]
        
        benefit_perfect = self.calculator._calculate_diversification_benefit(
            perfect_corr, weights
        )
        
        # Zero correlation (perfect diversification)
        zero_corr = np.eye(3)
        benefit_zero = self.calculator._calculate_diversification_benefit(
            zero_corr, weights
        )
        
        # Zero correlation should have higher diversification benefit
        self.assertGreater(benefit_zero, benefit_perfect)
    
    def test_risk_budget_allocation(self):
        """Test risk budget allocation calculation."""
        asset_parameters = {
            'A': KellyParameters(0.15, 0.20, 0.02),
            'B': KellyParameters(0.12, 0.25, 0.02),
            'C': KellyParameters(0.10, 0.15, 0.02)
        }
        
        weights = {'A': 0.4, 'B': 0.3, 'C': 0.3}
        asset_names = ['A', 'B', 'C']
        
        risk_budget = self.calculator._calculate_risk_budget_allocation(
            asset_parameters, weights, asset_names
        )
        
        # Check structure
        self.assertEqual(len(risk_budget), 3)
        for name in asset_names:
            self.assertIn(name, risk_budget)
            self.assertGreaterEqual(risk_budget[name], 0)
            self.assertLessEqual(risk_budget[name], 1)
        
        # Risk budget should sum to 1
        self.assertAlmostEqual(sum(risk_budget.values()), 1.0, places=6)


class TestConvenienceFunction(unittest.TestCase):
    """Test convenience function for Kelly calculation."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n_periods = 100
        
        returns_1 = np.random.normal(0.001, 0.02, n_periods)
        returns_2 = np.random.normal(0.0008, 0.025, n_periods)
        
        dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')
        
        self.returns_data = pd.DataFrame({
            'A': returns_1,
            'B': returns_2
        }, index=dates)
    
    def test_calculate_enhanced_kelly_individual_only(self):
        """Test enhanced Kelly calculation without portfolio optimization."""
        result = calculate_enhanced_kelly(self.returns_data)
        
        # Check structure
        self.assertIn('individual_results', result)
        self.assertIn('portfolio_result', result)
        self.assertIn('total_kelly', result)
        
        # Check individual results
        individual_results = result['individual_results']
        self.assertEqual(len(individual_results), 2)
        
        for asset_name in ['A', 'B']:
            self.assertIn(asset_name, individual_results)
            asset_result = individual_results[asset_name]
            
            self.assertIn('kelly_fraction', asset_result)
            self.assertIn('adjusted_kelly', asset_result)
            self.assertIn('recommended_position', asset_result)
            
            self.assertGreaterEqual(asset_result['kelly_fraction'], 0)
            self.assertGreaterEqual(asset_result['adjusted_kelly'], 0)
            self.assertGreaterEqual(asset_result['recommended_position'], 0)
        
        # Portfolio result should be None (no correlation matrix provided)
        self.assertIsNone(result['portfolio_result'])
        
        # Total Kelly should be sum of individual positions
        expected_total = sum(
            individual_results[asset]['recommended_position'] 
            for asset in individual_results
        )
        self.assertAlmostEqual(result['total_kelly'], expected_total, places=6)
    
    def test_calculate_enhanced_kelly_with_portfolio(self):
        """Test enhanced Kelly calculation with portfolio optimization."""
        correlation_matrix = self.returns_data.corr().values
        
        result = calculate_enhanced_kelly(
            self.returns_data, 
            correlation_matrix=correlation_matrix
        )
        
        # Check structure
        self.assertIn('individual_results', result)
        self.assertIn('portfolio_result', result)
        
        # Portfolio result should not be None
        self.assertIsNotNone(result['portfolio_result'])
        
        portfolio_result = result['portfolio_result']
        self.assertIn('individual_kellys', portfolio_result)
        self.assertIn('optimal_weights', portfolio_result)
        self.assertIn('correlation_adjustment', portfolio_result)
    
    def test_calculate_enhanced_kelly_insufficient_data(self):
        """Test enhanced Kelly with insufficient data."""
        short_data = self.returns_data.head(10)  # Only 10 observations
        
        result = calculate_enhanced_kelly(short_data)
        
        # Should handle gracefully
        self.assertIn('individual_results', result)
        # Individual results should be empty due to insufficient data
        self.assertEqual(len(result['individual_results']), 0)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_invalid_correlation_matrix_size(self):
        """Test handling of mismatched correlation matrix size."""
        calculator = EnhancedKellyCriterion()
        
        # Create mismatched data
        asset_parameters = {
            'A': KellyParameters(0.15, 0.20, 0.02),
            'B': KellyParameters(0.12, 0.25, 0.02)
        }
        
        # 3x3 correlation matrix for 2 assets
        correlation_matrix = np.eye(3)
        asset_names = ['A', 'B']
        
        # Should handle gracefully (may use fallback method)
        result = calculator.calculate_portfolio_kelly(
            asset_parameters, correlation_matrix, asset_names
        )
        
        self.assertIsInstance(result, PortfolioKellyResult)
    
    def test_optimization_failure_fallback(self):
        """Test fallback when portfolio optimization fails."""
        calculator = EnhancedKellyCriterion()
        
        # Create problematic parameters that might cause optimization to fail
        asset_parameters = {
            'A': KellyParameters(0.0, 0.20, 0.02),  # Zero expected return
            'B': KellyParameters(0.0, 0.25, 0.02)   # Zero expected return
        }
        
        correlation_matrix = np.eye(2)
        asset_names = ['A', 'B']
        
        # Should not crash, should return fallback result
        result = calculator.calculate_portfolio_kelly(
            asset_parameters, correlation_matrix, asset_names
        )
        
        self.assertIsInstance(result, PortfolioKellyResult)
        self.assertEqual(len(result.optimal_weights), 2)
    
    def test_nan_returns_handling(self):
        """Test handling of NaN values in returns."""
        calculator = EnhancedKellyCriterion()
        
        # Create returns with NaN values
        returns_with_nan = pd.Series([0.01, np.nan, 0.02, -0.01, np.nan] * 20)
        
        # Should handle NaN values gracefully
        params = calculator.estimate_parameters_from_returns(returns_with_nan)
        
        self.assertIsInstance(params, KellyParameters)
        self.assertGreater(params.volatility, 0)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)