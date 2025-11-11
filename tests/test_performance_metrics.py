"""
Tests for Comprehensive Performance Metrics Calculator

This module tests the performance metrics calculation functionality to ensure
accurate calculation of risk-adjusted returns and trading statistics.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.advanced_backtesting_engine import PerformanceMetricsCalculator
from simulation.backtesting_engine import Trade


class TestPerformanceMetricsCalculator(unittest.TestCase):
    """Test cases for PerformanceMetricsCalculator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = PerformanceMetricsCalculator()
        
        # Create sample portfolio values (growing portfolio)
        np.random.seed(42)
        initial_value = 100000
        returns = np.random.normal(0.0008, 0.02, 252)  # Daily returns for one year
        portfolio_values = [initial_value]
        
        for ret in returns:
            new_value = portfolio_values[-1] * (1 + ret)
            portfolio_values.append(new_value)
        
        self.portfolio_values = portfolio_values
        
        # Create sample trades
        self.trades = [
            Trade(
                symbol="AAPL",
                action="BUY",
                date=datetime(2020, 1, 1),
                price=100.0,
                quantity=100,
                total_value=10000.0,
                commission=10.0
            ),
            Trade(
                symbol="AAPL",
                action="SELL",
                date=datetime(2020, 1, 15),
                price=105.0,
                quantity=100,
                total_value=10500.0,
                commission=10.0
            ),
            Trade(
                symbol="MSFT",
                action="BUY",
                date=datetime(2020, 2, 1),
                price=200.0,
                quantity=50,
                total_value=10000.0,
                commission=10.0
            ),
            Trade(
                symbol="MSFT",
                action="SELL",
                date=datetime(2020, 2, 10),
                price=190.0,
                quantity=50,
                total_value=9500.0,
                commission=10.0
            )
        ]
        
        # Create benchmark returns
        self.benchmark_returns = np.random.normal(0.0005, 0.015, 252).tolist()
    
    def test_calculate_comprehensive_metrics_basic(self):
        """Test basic comprehensive metrics calculation."""
        metrics = self.calculator.calculate_comprehensive_metrics(
            portfolio_values=self.portfolio_values,
            trades=self.trades,
            risk_free_rate=0.02
        )
        
        # Check that all expected metrics are present
        expected_metrics = [
            'total_return',
            'annualized_return',
            'volatility',
            'sharpe_ratio',
            'sortino_ratio',
            'max_drawdown',
            'calmar_ratio',
            'total_trades',
            'win_rate',
            'profit_factor'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics, f"Missing metric: {metric}")
            self.assertIsInstance(metrics[metric], (int, float), f"Invalid type for {metric}")
    
    def test_calculate_comprehensive_metrics_with_benchmark(self):
        """Test metrics calculation with benchmark comparison."""
        metrics = self.calculator.calculate_comprehensive_metrics(
            portfolio_values=self.portfolio_values,
            trades=self.trades,
            benchmark_returns=self.benchmark_returns,
            risk_free_rate=0.02
        )
        
        # Check for benchmark-specific metrics
        benchmark_metrics = [
            'tracking_error',
            'information_ratio',
            'beta',
            'alpha',
            'correlation_with_benchmark'
        ]
        
        for metric in benchmark_metrics:
            if metric in metrics:
                self.assertIsInstance(metrics[metric], (int, float), f"Invalid type for {metric}")
    
    def test_calculate_trade_metrics(self):
        """Test trade-based metrics calculation."""
        trade_metrics = self.calculator._calculate_trade_metrics(self.trades)
        
        # Check structure
        expected_trade_metrics = [
            'total_trades',
            'win_rate',
            'profit_factor',
            'avg_trade_return',
            'avg_win',
            'avg_loss',
            'largest_win',
            'largest_loss',
            'consecutive_wins',
            'consecutive_losses'
        ]
        
        for metric in expected_trade_metrics:
            self.assertIn(metric, trade_metrics, f"Missing trade metric: {metric}")
            self.assertIsInstance(trade_metrics[metric], (int, float), f"Invalid type for {metric}")
        
        # Check logical constraints
        self.assertGreaterEqual(trade_metrics['win_rate'], 0)
        self.assertLessEqual(trade_metrics['win_rate'], 1)
        self.assertGreaterEqual(trade_metrics['profit_factor'], 0)
        self.assertGreaterEqual(trade_metrics['consecutive_wins'], 0)
        self.assertGreaterEqual(trade_metrics['consecutive_losses'], 0)
    
    def test_calculate_trade_metrics_empty_trades(self):
        """Test trade metrics with empty trade list."""
        trade_metrics = self.calculator._calculate_trade_metrics([])
        
        # Should return default values
        self.assertEqual(trade_metrics['total_trades'], 0)
        self.assertEqual(trade_metrics['win_rate'], 0)
        self.assertEqual(trade_metrics['profit_factor'], 0)
    
    def test_calculate_trade_metrics_single_symbol(self):
        """Test trade metrics with single symbol trades."""
        single_symbol_trades = [
            Trade(
                symbol="AAPL",
                action="BUY",
                date=datetime(2020, 1, 1),
                price=100.0,
                quantity=100,
                total_value=10000.0,
                commission=10.0
            ),
            Trade(
                symbol="AAPL",
                action="SELL",
                date=datetime(2020, 1, 15),
                price=110.0,
                quantity=100,
                total_value=11000.0,
                commission=10.0
            )
        ]
        
        trade_metrics = self.calculator._calculate_trade_metrics(single_symbol_trades)
        
        # Should calculate P&L correctly
        expected_pnl = (110.0 - 100.0) * 100  # $1000 profit
        self.assertEqual(trade_metrics['total_trades'], 1)
        self.assertEqual(trade_metrics['win_rate'], 1.0)  # 100% win rate
        self.assertAlmostEqual(trade_metrics['avg_trade_return'], expected_pnl, places=2)
    
    def test_calculate_additional_risk_metrics(self):
        """Test additional risk metrics calculation."""
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        
        risk_metrics = self.calculator._calculate_additional_risk_metrics(
            returns, self.portfolio_values
        )
        
        # Check for expected risk metrics
        expected_risk_metrics = [
            'var_95',
            'var_99',
            'cvar_95',
            'cvar_99',
            'skewness',
            'kurtosis',
            'ulcer_index',
            'recovery_factor',
            'tail_ratio'
        ]
        
        for metric in expected_risk_metrics:
            if metric in risk_metrics:
                self.assertIsInstance(risk_metrics[metric], (int, float), f"Invalid type for {metric}")
        
        # Check logical constraints
        if 'var_95' in risk_metrics and 'var_99' in risk_metrics:
            # VaR 99% should be more negative than VaR 95%
            self.assertLessEqual(risk_metrics['var_99'], risk_metrics['var_95'])
    
    def test_calculate_benchmark_metrics(self):
        """Test benchmark comparison metrics."""
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        
        benchmark_metrics = self.calculator._calculate_benchmark_metrics(
            returns, self.benchmark_returns
        )
        
        # Check for expected benchmark metrics
        expected_benchmark_metrics = [
            'tracking_error',
            'information_ratio',
            'beta',
            'alpha',
            'correlation_with_benchmark'
        ]
        
        for metric in expected_benchmark_metrics:
            if metric in benchmark_metrics:
                self.assertIsInstance(benchmark_metrics[metric], (int, float), f"Invalid type for {metric}")
        
        # Check logical constraints
        if 'correlation_with_benchmark' in benchmark_metrics:
            correlation = benchmark_metrics['correlation_with_benchmark']
            self.assertGreaterEqual(correlation, -1)
            self.assertLessEqual(correlation, 1)
    
    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        # Single portfolio value
        metrics = self.calculator.calculate_comprehensive_metrics(
            portfolio_values=[100000],
            trades=[],
            risk_free_rate=0.02
        )
        
        # Should return error
        self.assertIn('error', metrics)
    
    def test_empty_portfolio_values(self):
        """Test handling of empty portfolio values."""
        metrics = self.calculator.calculate_comprehensive_metrics(
            portfolio_values=[],
            trades=[],
            risk_free_rate=0.02
        )
        
        # Should return error
        self.assertIn('error', metrics)
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        # Create portfolio with known returns
        initial_value = 100000
        daily_return = 0.001  # 0.1% daily return
        portfolio_values = [initial_value]
        
        for i in range(252):  # One year
            new_value = portfolio_values[-1] * (1 + daily_return)
            portfolio_values.append(new_value)
        
        metrics = self.calculator.calculate_comprehensive_metrics(
            portfolio_values=portfolio_values,
            trades=[],
            risk_free_rate=0.02
        )
        
        # Sharpe ratio should be positive for consistent positive returns
        # With zero volatility, it should be infinite
        self.assertGreater(metrics['sharpe_ratio'], 0)
        # For constant positive returns with zero volatility, Sharpe ratio is infinite
        self.assertTrue(np.isinf(metrics['sharpe_ratio']) or metrics['sharpe_ratio'] > 100)
    
    def test_sortino_ratio_calculation(self):
        """Test Sortino ratio calculation."""
        # Create portfolio with mixed returns (some negative)
        initial_value = 100000
        returns = [0.01, -0.005, 0.008, -0.003, 0.012] * 50  # Mixed returns
        portfolio_values = [initial_value]
        
        for ret in returns:
            new_value = portfolio_values[-1] * (1 + ret)
            portfolio_values.append(new_value)
        
        metrics = self.calculator.calculate_comprehensive_metrics(
            portfolio_values=portfolio_values,
            trades=[],
            risk_free_rate=0.02
        )
        
        # Sortino ratio should be calculated
        self.assertIn('sortino_ratio', metrics)
        self.assertIsInstance(metrics['sortino_ratio'], (int, float))
    
    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        # Create portfolio with known drawdown
        portfolio_values = [100000, 110000, 105000, 95000, 120000]  # 13.6% drawdown
        
        metrics = self.calculator.calculate_comprehensive_metrics(
            portfolio_values=portfolio_values,
            trades=[],
            risk_free_rate=0.02
        )
        
        # Max drawdown should be approximately 13.6%
        expected_drawdown = (110000 - 95000) / 110000  # 13.6%
        self.assertAlmostEqual(metrics['max_drawdown'], expected_drawdown, places=3)
    
    def test_calmar_ratio_calculation(self):
        """Test Calmar ratio calculation."""
        # Create portfolio with positive returns and some drawdown
        portfolio_values = [100000, 110000, 105000, 115000, 120000]
        
        metrics = self.calculator.calculate_comprehensive_metrics(
            portfolio_values=portfolio_values,
            trades=[],
            risk_free_rate=0.02
        )
        
        # Calmar ratio should be positive (positive return / positive drawdown)
        if metrics['max_drawdown'] > 0:
            self.assertGreater(metrics['calmar_ratio'], 0)
    
    def test_win_rate_calculation(self):
        """Test win rate calculation with known trades."""
        # Create trades with known outcomes
        winning_trades = [
            Trade("AAPL", "BUY", datetime(2020, 1, 1), 100.0, 100, 10000.0, 10.0),
            Trade("AAPL", "SELL", datetime(2020, 1, 2), 105.0, 100, 10500.0, 10.0),  # Win
            Trade("MSFT", "BUY", datetime(2020, 1, 3), 200.0, 50, 10000.0, 10.0),
            Trade("MSFT", "SELL", datetime(2020, 1, 4), 195.0, 50, 9750.0, 10.0),   # Loss
        ]
        
        trade_metrics = self.calculator._calculate_trade_metrics(winning_trades)
        
        # Should have 50% win rate (1 win, 1 loss)
        self.assertAlmostEqual(trade_metrics['win_rate'], 0.5, places=2)
    
    def test_profit_factor_calculation(self):
        """Test profit factor calculation."""
        # Create trades with known P&L
        trades_with_pnl = [
            Trade("AAPL", "BUY", datetime(2020, 1, 1), 100.0, 100, 10000.0, 10.0),
            Trade("AAPL", "SELL", datetime(2020, 1, 2), 110.0, 100, 11000.0, 10.0),  # $1000 profit
            Trade("MSFT", "BUY", datetime(2020, 1, 3), 200.0, 50, 10000.0, 10.0),
            Trade("MSFT", "SELL", datetime(2020, 1, 4), 190.0, 50, 9500.0, 10.0),    # $500 loss
        ]
        
        trade_metrics = self.calculator._calculate_trade_metrics(trades_with_pnl)
        
        # Profit factor should be 1000 / 500 = 2.0
        self.assertAlmostEqual(trade_metrics['profit_factor'], 2.0, places=1)
    
    def test_mismatched_benchmark_length(self):
        """Test handling of mismatched benchmark length."""
        short_benchmark = self.benchmark_returns[:100]  # Shorter than portfolio
        
        metrics = self.calculator.calculate_comprehensive_metrics(
            portfolio_values=self.portfolio_values,
            trades=self.trades,
            benchmark_returns=short_benchmark,
            risk_free_rate=0.02
        )
        
        # Should handle gracefully and may or may not include benchmark metrics
        self.assertIsInstance(metrics, dict)
        self.assertNotIn('error', metrics)


if __name__ == '__main__':
    unittest.main()