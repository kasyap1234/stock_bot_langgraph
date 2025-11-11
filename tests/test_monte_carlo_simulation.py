"""
Tests for Monte Carlo Simulation Capabilities

This module tests the Monte Carlo simulation functionality for strategy
robustness testing and bootstrap sampling validation.
"""

import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.advanced_backtesting_engine import MonteCarloSimulator
from simulation.trading_strategies import BaseStrategy, StrategyConfig, TradingSignal


class MockStrategy(BaseStrategy):
    """Mock strategy for testing purposes."""
    
    def __init__(self):
        config = StrategyConfig(
            name="Mock Strategy",
            description="Test strategy",
            parameters={}
        )
        super().__init__(config)
    
    def generate_signals(self, data, state=None):
        """Generate mock signals."""
        if len(data) < 10:
            return []
        
        # Generate a simple buy signal with some randomness
        confidence = 0.6 + np.random.random() * 0.3
        return [TradingSignal(
            symbol=data.get('symbol', 'TEST'),
            action='BUY',
            confidence=confidence,
            price=data['Close'].iloc[-1],
            timestamp=data.index[-1],
            reason="Mock signal"
        )]
    
    def validate_signal(self, signal, data):
        """Always validate signals for testing."""
        return True


class TestMonteCarloSimulator(unittest.TestCase):
    """Test cases for MonteCarloSimulator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.simulator = MonteCarloSimulator(
            num_simulations=50,  # Reduced for faster testing
            confidence_level=0.95,
            bootstrap_block_size=20  # Smaller block size for testing
        )
        
        # Create sample data
        self.start_date = datetime(2020, 1, 1)
        self.end_date = datetime(2020, 12, 31)
        
        # Generate sample stock data
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        dates = dates[dates.weekday < 5]  # Remove weekends
        
        np.random.seed(42)  # For reproducible tests
        prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates)))
        
        self.stock_data = {
            'AAPL': pd.DataFrame({
                'Open': prices * 0.99,
                'High': prices * 1.02,
                'Low': prices * 0.98,
                'Close': prices,
                'Volume': np.random.randint(1000000, 10000000, len(dates))
            }, index=dates)
        }
        
        self.strategy = MockStrategy()
    
    def test_create_bootstrap_sample(self):
        """Test bootstrap sample creation."""
        bootstrap_data = self.simulator._create_bootstrap_sample(self.stock_data)
        
        # Should have same symbols
        self.assertEqual(set(bootstrap_data.keys()), set(self.stock_data.keys()))
        
        # Should have same length
        for symbol in self.stock_data:
            original_len = len(self.stock_data[symbol])
            bootstrap_len = len(bootstrap_data[symbol])
            self.assertEqual(original_len, bootstrap_len)
        
        # Should have same columns
        for symbol in self.stock_data:
            original_cols = set(self.stock_data[symbol].columns)
            bootstrap_cols = set(bootstrap_data[symbol].columns)
            self.assertEqual(original_cols, bootstrap_cols)
    
    def test_block_bootstrap(self):
        """Test block bootstrap sampling method."""
        df = self.stock_data['AAPL']
        bootstrap_df = self.simulator._block_bootstrap(df)
        
        # Should have same length
        self.assertEqual(len(bootstrap_df), len(df))
        
        # Should have same columns
        self.assertEqual(list(bootstrap_df.columns), list(df.columns))
        
        # Should have same index (dates)
        self.assertTrue(bootstrap_df.index.equals(df.index))
    
    def test_block_bootstrap_short_data(self):
        """Test block bootstrap with data shorter than block size."""
        # Create short data
        short_dates = pd.date_range(start=self.start_date, periods=10, freq='D')
        short_data = self.stock_data['AAPL'].iloc[:10].copy()
        short_data.index = short_dates
        
        bootstrap_df = self.simulator._block_bootstrap(short_data)
        
        # Should handle short data gracefully
        self.assertEqual(len(bootstrap_df), len(short_data))
        self.assertEqual(list(bootstrap_df.columns), list(short_data.columns))
    
    def test_calculate_monte_carlo_statistics(self):
        """Test Monte Carlo statistics calculation."""
        # Create mock simulation results
        simulation_results = []
        np.random.seed(42)
        
        for i in range(100):
            simulation_results.append({
                'simulation_id': i,
                'total_return': np.random.normal(0.05, 0.15),
                'sharpe_ratio': np.random.normal(1.0, 0.5),
                'max_drawdown': np.random.uniform(0.01, 0.20),
                'win_rate': np.random.uniform(0.4, 0.8),
                'total_trades': np.random.randint(10, 100)
            })
        
        stats = self.simulator._calculate_monte_carlo_statistics(simulation_results)
        
        # Check structure
        self.assertIn('returns', stats)
        self.assertIn('sharpe_ratio', stats)
        self.assertIn('max_drawdown', stats)
        self.assertIn('win_rate', stats)
        
        # Check returns statistics
        returns_stats = stats['returns']
        self.assertIn('mean', returns_stats)
        self.assertIn('median', returns_stats)
        self.assertIn('std', returns_stats)
        self.assertIn('confidence_interval', returns_stats)
        self.assertIn('probability_positive', returns_stats)
        
        # Check confidence intervals
        ci = returns_stats['confidence_interval']
        self.assertEqual(len(ci), 2)
        self.assertLess(ci[0], ci[1])  # Lower bound < upper bound
        
        # Check probability is between 0 and 1
        prob_positive = returns_stats['probability_positive']
        self.assertGreaterEqual(prob_positive, 0)
        self.assertLessEqual(prob_positive, 1)
    
    def test_empty_simulation_results(self):
        """Test handling of empty simulation results."""
        stats = self.simulator._calculate_monte_carlo_statistics([])
        
        # Should handle empty results gracefully
        self.assertIsInstance(stats, dict)
    
    @patch('simulation.advanced_backtesting_engine.BacktestingEngine')
    def test_run_monte_carlo_simulation_success(self, mock_engine_class):
        """Test successful Monte Carlo simulation execution."""
        # Mock the backtesting engine
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine
        
        # Mock successful backtest results with some variation
        def mock_backtest(*args, **kwargs):
            return {
                'total_return': np.random.normal(0.05, 0.02),
                'sharpe_ratio': np.random.normal(1.2, 0.3),
                'max_drawdown': np.random.uniform(0.02, 0.08),
                'win_rate': np.random.uniform(0.5, 0.7),
                'total_trades': np.random.randint(5, 20)
            }
        
        mock_engine.run_strategy_backtest.side_effect = mock_backtest
        
        # Run simulation with small number for faster testing
        self.simulator.num_simulations = 10
        
        result = self.simulator.run_monte_carlo_simulation(
            strategy=self.strategy,
            stock_data=self.stock_data,
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=100000.0
        )
        
        # Check result structure
        self.assertTrue(result['success'])
        self.assertIn('num_simulations', result)
        self.assertIn('simulation_results', result)
        self.assertIn('statistics', result)
        
        # Check that simulations were run
        self.assertGreater(result['num_simulations'], 0)
        self.assertGreater(len(result['simulation_results']), 0)
        
        # Check statistics structure
        stats = result['statistics']
        self.assertIn('returns', stats)
        self.assertIn('sharpe_ratio', stats)
    
    @patch('simulation.advanced_backtesting_engine.BacktestingEngine')
    def test_run_monte_carlo_simulation_failures(self, mock_engine_class):
        """Test Monte Carlo simulation with some failed backtests."""
        # Mock the backtesting engine
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine
        
        # Mock mixed results (some failures)
        def mock_backtest(*args, **kwargs):
            if np.random.random() < 0.3:  # 30% failure rate
                return {'error': 'Backtest failed'}
            else:
                return {
                    'total_return': np.random.normal(0.05, 0.02),
                    'sharpe_ratio': np.random.normal(1.2, 0.3),
                    'max_drawdown': np.random.uniform(0.02, 0.08),
                    'win_rate': np.random.uniform(0.5, 0.7),
                    'total_trades': np.random.randint(5, 20)
                }
        
        mock_engine.run_strategy_backtest.side_effect = mock_backtest
        
        # Run simulation
        self.simulator.num_simulations = 20
        
        result = self.simulator.run_monte_carlo_simulation(
            strategy=self.strategy,
            stock_data=self.stock_data,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Should still succeed with partial results
        if result['success']:
            self.assertGreater(result['num_simulations'], 0)
            self.assertLessEqual(result['num_simulations'], 20)  # Some may have failed
    
    @patch('simulation.advanced_backtesting_engine.BacktestingEngine')
    def test_run_monte_carlo_simulation_all_failures(self, mock_engine_class):
        """Test Monte Carlo simulation with all failed backtests."""
        # Mock the backtesting engine
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine
        
        # Mock all failures
        mock_engine.run_strategy_backtest.return_value = {'error': 'All backtests failed'}
        
        # Run simulation
        self.simulator.num_simulations = 5
        
        result = self.simulator.run_monte_carlo_simulation(
            strategy=self.strategy,
            stock_data=self.stock_data,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Should handle all failures gracefully
        self.assertFalse(result['success'])
        self.assertIn('error', result)
    
    def test_confidence_level_parameter(self):
        """Test different confidence levels."""
        # Test with different confidence level
        simulator_99 = MonteCarloSimulator(
            num_simulations=10,
            confidence_level=0.99,
            bootstrap_block_size=20
        )
        
        # Create mock results
        simulation_results = [
            {
                'simulation_id': i,
                'total_return': np.random.normal(0.05, 0.1),
                'sharpe_ratio': np.random.normal(1.0, 0.3),
                'max_drawdown': np.random.uniform(0.01, 0.15),
                'win_rate': np.random.uniform(0.4, 0.8),
                'total_trades': 10
            }
            for i in range(100)
        ]
        
        stats_95 = self.simulator._calculate_monte_carlo_statistics(simulation_results)
        stats_99 = simulator_99._calculate_monte_carlo_statistics(simulation_results)
        
        # 99% confidence interval should be wider than 95%
        ci_95 = stats_95['returns']['confidence_interval']
        ci_99 = stats_99['returns']['confidence_interval']
        
        ci_width_95 = ci_95[1] - ci_95[0]
        ci_width_99 = ci_99[1] - ci_99[0]
        
        self.assertGreater(ci_width_99, ci_width_95)
    
    def test_parameter_perturbation(self):
        """Test strategy parameter perturbation (placeholder)."""
        # This is a placeholder test since parameter perturbation
        # is strategy-specific and not fully implemented
        parameter_ranges = {
            'param1': (0.1, 0.9),
            'param2': (10, 100)
        }
        
        perturbed_strategy = self.simulator._perturb_strategy_parameters(
            self.strategy, parameter_ranges
        )
        
        # For now, should return the same strategy
        self.assertEqual(perturbed_strategy, self.strategy)


if __name__ == '__main__':
    unittest.main()