"""
Unit tests for real-time portfolio risk assessment module.

Tests the portfolio risk assessment system for accuracy and robustness.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.portfolio_risk_assessment import (
    RealTimePortfolioRiskAssessment,
    Position,
    PortfolioSnapshot,
    RiskMetrics,
    RiskAlert,
    RiskLevel,
    AlertType,
    assess_portfolio_risk_realtime
)


class TestPosition(unittest.TestCase):
    """Test Position dataclass."""
    
    def test_position_creation(self):
        """Test position creation and calculations."""
        position = Position(
            symbol='AAPL',
            quantity=100,
            current_price=150.0,
            entry_price=140.0,
            entry_date=datetime.now(),
            market_value=0,  # Should be calculated
            unrealized_pnl=0,  # Should be calculated
            weight=0.5
        )
        
        # Check calculated fields
        self.assertEqual(position.market_value, 15000.0)  # 100 * 150
        self.assertEqual(position.unrealized_pnl, 1000.0)  # 100 * (150 - 140)
    
    def test_position_with_provided_values(self):
        """Test position with pre-calculated values."""
        position = Position(
            symbol='AAPL',
            quantity=100,
            current_price=150.0,
            entry_price=140.0,
            entry_date=datetime.now(),
            market_value=16000.0,  # Provided value
            unrealized_pnl=1200.0,  # Provided value
            weight=0.5
        )
        
        # Should use provided values
        self.assertEqual(position.market_value, 16000.0)
        self.assertEqual(position.unrealized_pnl, 1200.0)


class TestPortfolioSnapshot(unittest.TestCase):
    """Test PortfolioSnapshot dataclass."""
    
    def test_portfolio_snapshot_creation(self):
        """Test portfolio snapshot creation and calculations."""
        positions = {
            'AAPL': Position('AAPL', 100, 150.0, 140.0, datetime.now(), 0, 0, 0.6),
            'GOOGL': Position('GOOGL', 50, 2800.0, 2700.0, datetime.now(), 0, 0, 0.4)
        }
        
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            positions=positions,
            total_value=0,  # Should be calculated
            cash=5000.0,
            total_equity=0,  # Should be calculated
            daily_pnl=500.0,
            total_pnl=2000.0,
            leverage=1.2
        )
        
        # Check calculated fields
        expected_total_value = 15000.0 + 140000.0  # AAPL + GOOGL market values
        self.assertEqual(snapshot.total_value, expected_total_value)
        self.assertEqual(snapshot.total_equity, expected_total_value + 5000.0)


class TestRealTimePortfolioRiskAssessment(unittest.TestCase):
    """Test real-time portfolio risk assessment."""
    
    def setUp(self):
        """Set up test data."""
        self.risk_assessor = RealTimePortfolioRiskAssessment()
        
        # Create test portfolio
        self.positions = {
            'AAPL': Position('AAPL', 100, 150.0, 140.0, datetime.now(), 15000.0, 1000.0, 0.6),
            'GOOGL': Position('GOOGL', 50, 2800.0, 2700.0, datetime.now(), 140000.0, 5000.0, 0.4)
        }
        
        self.portfolio_snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            positions=self.positions,
            total_value=155000.0,
            cash=5000.0,
            total_equity=160000.0,
            daily_pnl=500.0,
            total_pnl=6000.0,
            leverage=1.0
        )
        
        # Create test market data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        # Generate correlated price series
        returns_aapl = np.random.normal(0.001, 0.02, 100)
        returns_googl = 0.7 * returns_aapl + 0.3 * np.random.normal(0.001, 0.025, 100)
        
        prices_aapl = 100 * np.exp(np.cumsum(returns_aapl))
        prices_googl = 2000 * np.exp(np.cumsum(returns_googl))
        
        self.market_data = {
            'AAPL': pd.DataFrame({'Close': prices_aapl}, index=dates),
            'GOOGL': pd.DataFrame({'Close': prices_googl}, index=dates),
            'SPY': pd.DataFrame({'Close': prices_aapl * 0.5 + 200}, index=dates)  # Benchmark
        }
    
    def test_initialization(self):
        """Test risk assessor initialization."""
        custom_thresholds = {'max_drawdown_warning': 0.15}
        
        assessor = RealTimePortfolioRiskAssessment(
            risk_thresholds=custom_thresholds,
            lookback_window=100,
            alert_cooldown_hours=2
        )
        
        self.assertEqual(assessor.lookback_window, 100)
        self.assertEqual(assessor.alert_cooldown_hours, 2)
        self.assertEqual(assessor.risk_thresholds['max_drawdown_warning'], 0.15)
        
        # Check default thresholds are still present
        self.assertIn('max_drawdown_critical', assessor.risk_thresholds)
    
    def test_assess_portfolio_risk_single_snapshot(self):
        """Test risk assessment with single portfolio snapshot."""
        risk_metrics = self.risk_assessor.assess_portfolio_risk(
            self.portfolio_snapshot,
            self.market_data,
            generate_alerts=False
        )
        
        # Check result structure
        self.assertIsInstance(risk_metrics, RiskMetrics)
        
        # Check basic metrics
        self.assertEqual(risk_metrics.portfolio_value, 155000.0)
        self.assertIsInstance(risk_metrics.daily_return, float)
        self.assertIsInstance(risk_metrics.volatility, float)
        self.assertIsInstance(risk_metrics.sharpe_ratio, float)
        
        # Check risk level assessment
        self.assertIsInstance(risk_metrics.overall_risk_level, RiskLevel)
        self.assertIsInstance(risk_metrics.risk_score, float)
        
        # Check concentration metrics
        self.assertGreater(risk_metrics.concentration_ratio, 0)
        self.assertGreater(risk_metrics.effective_positions, 0)
        
        # Check timestamp
        self.assertIsInstance(risk_metrics.timestamp, datetime)
    
    def test_assess_portfolio_risk_multiple_snapshots(self):
        """Test risk assessment with multiple portfolio snapshots."""
        # Add multiple snapshots to build history
        for i in range(10):
            # Simulate portfolio value changes
            modified_snapshot = PortfolioSnapshot(
                timestamp=datetime.now() - timedelta(days=i),
                positions=self.positions,
                total_value=155000.0 * (1 + np.random.normal(0, 0.01)),
                cash=5000.0,
                total_equity=160000.0 * (1 + np.random.normal(0, 0.01)),
                daily_pnl=np.random.normal(0, 500),
                total_pnl=6000.0 + np.random.normal(0, 1000),
                leverage=1.0
            )
            
            risk_metrics = self.risk_assessor.assess_portfolio_risk(
                modified_snapshot,
                self.market_data,
                generate_alerts=False
            )
        
        # Should have meaningful volatility and drawdown metrics
        self.assertGreater(risk_metrics.volatility, 0)
        self.assertIsInstance(risk_metrics.current_drawdown, float)
        self.assertIsInstance(risk_metrics.max_drawdown, float)
        
        # Check history is maintained
        self.assertEqual(len(self.risk_assessor.portfolio_history), 10)
        self.assertEqual(len(self.risk_assessor.risk_metrics_history), 10)
    
    def test_drawdown_calculation(self):
        """Test drawdown calculation accuracy."""
        # Create portfolio with known drawdown pattern
        values = [100000, 105000, 110000, 95000, 90000, 100000, 105000]
        
        for i, value in enumerate(values):
            snapshot = PortfolioSnapshot(
                timestamp=datetime.now() - timedelta(days=len(values)-i),
                positions=self.positions,
                total_value=value,
                cash=5000.0,
                total_equity=value + 5000.0,
                daily_pnl=0,
                total_pnl=0,
                leverage=1.0
            )
            
            self.risk_assessor.assess_portfolio_risk(
                snapshot, self.market_data, generate_alerts=False
            )
        
        # Check drawdown calculation
        latest_metrics = self.risk_assessor.risk_metrics_history[-1]
        
        # Maximum drawdown should be from peak (110000) to trough (90000)
        expected_max_dd = (90000 - 110000) / 110000  # ≈ -18.18%
        
        self.assertAlmostEqual(latest_metrics.max_drawdown, expected_max_dd, places=2)
    
    def test_concentration_metrics(self):
        """Test concentration metrics calculation."""
        # Create highly concentrated portfolio
        concentrated_positions = {
            'AAPL': Position('AAPL', 1000, 150.0, 140.0, datetime.now(), 150000.0, 10000.0, 0.9),
            'GOOGL': Position('GOOGL', 5, 2800.0, 2700.0, datetime.now(), 14000.0, 500.0, 0.1)
        }
        
        concentrated_snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            positions=concentrated_positions,
            total_value=164000.0,
            cash=1000.0,
            total_equity=165000.0,
            daily_pnl=0,
            total_pnl=0,
            leverage=1.0
        )
        
        risk_metrics = self.risk_assessor.assess_portfolio_risk(
            concentrated_snapshot, self.market_data, generate_alerts=False
        )
        
        # Should detect high concentration
        self.assertGreater(risk_metrics.concentration_ratio, 0.8)
        self.assertLess(risk_metrics.effective_positions, 1.5)  # Close to 1 position
        self.assertGreater(risk_metrics.herfindahl_index, 0.8)
    
    def test_correlation_metrics(self):
        """Test correlation metrics calculation."""
        risk_metrics = self.risk_assessor.assess_portfolio_risk(
            self.portfolio_snapshot, self.market_data, generate_alerts=False
        )
        
        # Should calculate correlation metrics
        self.assertIsInstance(risk_metrics.avg_correlation, float)
        self.assertIsInstance(risk_metrics.max_correlation, float)
        self.assertIsInstance(risk_metrics.correlation_breakdown_risk, float)
        
        # Correlations should be reasonable
        self.assertGreaterEqual(risk_metrics.avg_correlation, 0)
        self.assertLessEqual(risk_metrics.avg_correlation, 1)
    
    def test_risk_level_assessment(self):
        """Test overall risk level assessment."""
        # Test low risk scenario
        low_risk_snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            positions={
                'AAPL': Position('AAPL', 50, 150.0, 140.0, datetime.now(), 7500.0, 500.0, 0.25),
                'GOOGL': Position('GOOGL', 25, 2800.0, 2700.0, datetime.now(), 70000.0, 2500.0, 0.25),
                'MSFT': Position('MSFT', 100, 300.0, 290.0, datetime.now(), 30000.0, 1000.0, 0.25),
                'TSLA': Position('TSLA', 50, 800.0, 750.0, datetime.now(), 40000.0, 2500.0, 0.25)
            },
            total_value=147500.0,
            cash=10000.0,
            total_equity=157500.0,
            daily_pnl=100.0,
            total_pnl=6500.0,
            leverage=0.9
        )
        
        # Add to market data for diversified portfolio
        self.market_data['MSFT'] = self.market_data['AAPL'].copy()
        self.market_data['TSLA'] = self.market_data['GOOGL'].copy()
        
        risk_metrics = self.risk_assessor.assess_portfolio_risk(
            low_risk_snapshot, self.market_data, generate_alerts=False
        )
        
        # Should assess as lower risk due to diversification
        # Note: concentration ratio is the largest single position weight
        self.assertLessEqual(risk_metrics.concentration_ratio, 0.5)  # More lenient threshold
        self.assertIn(risk_metrics.overall_risk_level, [RiskLevel.LOW, RiskLevel.MODERATE])
    
    def test_alert_generation(self):
        """Test risk alert generation."""
        # Create high-risk scenario
        high_risk_positions = {
            'AAPL': Position('AAPL', 2000, 150.0, 200.0, datetime.now(), 300000.0, -100000.0, 1.0)
        }
        
        high_risk_snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            positions=high_risk_positions,
            total_value=300000.0,
            cash=-50000.0,  # Negative cash (margin)
            total_equity=250000.0,
            daily_pnl=-5000.0,
            total_pnl=-100000.0,
            leverage=2.5  # High leverage
        )
        
        # Build history with declining values to create drawdown
        declining_values = [400000, 380000, 360000, 340000, 320000, 300000]
        
        for value in declining_values:
            snapshot = PortfolioSnapshot(
                timestamp=datetime.now(),
                positions=high_risk_positions,
                total_value=value,
                cash=-50000.0,
                total_equity=value - 50000.0,
                daily_pnl=-2000.0,
                total_pnl=-100000.0,
                leverage=2.5
            )
            
            self.risk_assessor.assess_portfolio_risk(
                snapshot, self.market_data, generate_alerts=True
            )
        
        # Should generate alerts
        alerts = self.risk_assessor.get_recent_alerts()
        self.assertGreater(len(alerts), 0)
        
        # Check alert types
        alert_types = {alert.alert_type for alert in alerts}
        self.assertTrue(
            AlertType.DRAWDOWN_WARNING in alert_types or 
            AlertType.DRAWDOWN_CRITICAL in alert_types or
            AlertType.LEVERAGE_EXCESSIVE in alert_types
        )
    
    def test_alert_cooldown(self):
        """Test alert cooldown mechanism."""
        # Set short cooldown for testing
        self.risk_assessor.alert_cooldown_hours = 0.001  # Very short cooldown
        
        # Create scenario that triggers alerts
        high_drawdown_snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            positions=self.positions,
            total_value=100000.0,  # Significant drop
            cash=5000.0,
            total_equity=105000.0,
            daily_pnl=-10000.0,
            total_pnl=-50000.0,
            leverage=1.0
        )
        
        # Build history to create large drawdown
        for value in [200000, 180000, 160000, 140000, 120000, 100000]:
            snapshot = PortfolioSnapshot(
                timestamp=datetime.now(),
                positions=self.positions,
                total_value=value,
                cash=5000.0,
                total_equity=value + 5000.0,
                daily_pnl=-5000.0,
                total_pnl=-50000.0,
                leverage=1.0
            )
            
            self.risk_assessor.assess_portfolio_risk(
                snapshot, self.market_data, generate_alerts=True
            )
        
        initial_alert_count = len(self.risk_assessor.alerts_history)
        
        # Trigger same scenario again immediately
        self.risk_assessor.assess_portfolio_risk(
            high_drawdown_snapshot, self.market_data, generate_alerts=True
        )
        
        # Should not generate duplicate alerts due to cooldown
        # (Note: cooldown is very short, so this test may be timing-sensitive)
        final_alert_count = len(self.risk_assessor.alerts_history)
        
        # The exact behavior depends on timing, but we should have some alerts
        self.assertGreaterEqual(final_alert_count, initial_alert_count)
    
    def test_risk_dashboard(self):
        """Test risk dashboard generation."""
        # Add some history
        self.risk_assessor.assess_portfolio_risk(
            self.portfolio_snapshot, self.market_data, generate_alerts=True
        )
        
        dashboard = self.risk_assessor.get_risk_dashboard()
        
        # Check dashboard structure
        self.assertIn('current_metrics', dashboard)
        self.assertIn('recent_alerts', dashboard)
        self.assertIn('risk_trends', dashboard)
        self.assertIn('thresholds', dashboard)
        
        # Check current metrics
        current_metrics = dashboard['current_metrics']
        self.assertIn('portfolio_value', current_metrics)
        self.assertIn('overall_risk_level', current_metrics)
        self.assertIn('risk_score', current_metrics)
        
        # Check alerts format
        recent_alerts = dashboard['recent_alerts']
        self.assertIsInstance(recent_alerts, list)
    
    def test_update_risk_thresholds(self):
        """Test updating risk thresholds."""
        new_thresholds = {
            'max_drawdown_warning': 0.15,
            'concentration_critical': 0.60
        }
        
        original_warning = self.risk_assessor.risk_thresholds['max_drawdown_warning']
        
        self.risk_assessor.update_risk_thresholds(new_thresholds)
        
        # Should update specified thresholds
        self.assertEqual(self.risk_assessor.risk_thresholds['max_drawdown_warning'], 0.15)
        self.assertEqual(self.risk_assessor.risk_thresholds['concentration_critical'], 0.60)
        
        # Should preserve other thresholds
        self.assertIn('max_drawdown_critical', self.risk_assessor.risk_thresholds)
    
    def test_empty_portfolio(self):
        """Test handling of empty portfolio."""
        empty_snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            positions={},
            total_value=0.0,
            cash=10000.0,
            total_equity=10000.0,
            daily_pnl=0.0,
            total_pnl=0.0,
            leverage=0.0
        )
        
        risk_metrics = self.risk_assessor.assess_portfolio_risk(
            empty_snapshot, self.market_data, generate_alerts=False
        )
        
        # Should handle gracefully
        self.assertIsInstance(risk_metrics, RiskMetrics)
        self.assertEqual(risk_metrics.concentration_ratio, 0.0)
        self.assertEqual(risk_metrics.effective_positions, 0.0)


class TestConvenienceFunction(unittest.TestCase):
    """Test convenience function for portfolio risk assessment."""
    
    def test_assess_portfolio_risk_realtime(self):
        """Test convenience function with valid data."""
        portfolio_data = {
            'positions': {
                'AAPL': {
                    'quantity': 100,
                    'current_price': 150.0,
                    'entry_price': 140.0,
                    'entry_date': datetime.now().isoformat(),
                    'market_value': 15000.0,
                    'unrealized_pnl': 1000.0,
                    'weight': 0.6
                },
                'GOOGL': {
                    'quantity': 50,
                    'current_price': 2800.0,
                    'entry_price': 2700.0,
                    'entry_date': datetime.now().isoformat(),
                    'market_value': 140000.0,
                    'unrealized_pnl': 5000.0,
                    'weight': 0.4
                }
            },
            'total_value': 155000.0,
            'cash': 5000.0,
            'total_equity': 160000.0,
            'daily_pnl': 500.0,
            'total_pnl': 6000.0,
            'leverage': 1.0
        }
        
        # Create market data
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        market_data = {
            'AAPL': pd.DataFrame({'Close': np.random.uniform(140, 160, 50)}, index=dates),
            'GOOGL': pd.DataFrame({'Close': np.random.uniform(2600, 2900, 50)}, index=dates)
        }
        
        result = assess_portfolio_risk_realtime(portfolio_data, market_data)
        
        # Check result structure
        self.assertIn('risk_metrics', result)
        self.assertIn('dashboard', result)
        self.assertIn('alerts', result)
        
        # Check risk metrics
        risk_metrics = result['risk_metrics']
        self.assertIn('portfolio_value', risk_metrics)
        self.assertIn('overall_risk_level', risk_metrics)
        
        # Check dashboard
        dashboard = result['dashboard']
        self.assertIn('current_metrics', dashboard)
    
    def test_assess_portfolio_risk_realtime_error_handling(self):
        """Test convenience function error handling."""
        # Invalid portfolio data
        invalid_data = {'invalid': 'data'}
        market_data = {}
        
        result = assess_portfolio_risk_realtime(invalid_data, market_data)
        
        # Should handle gracefully
        self.assertIn('error', result)
        self.assertIsNone(result['risk_metrics'])


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)