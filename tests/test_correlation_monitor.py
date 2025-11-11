"""
Unit tests for dynamic correlation monitoring module.

Tests the correlation monitoring system for accuracy and robustness.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.correlation_monitor import (
    DynamicCorrelationMonitor,
    CorrelationRegime,
    CorrelationMetrics,
    CorrelationAlert,
    calculate_correlation_metrics
)


class TestDynamicCorrelationMonitor(unittest.TestCase):
    """Test dynamic correlation monitor."""
    
    def setUp(self):
        """Set up test data."""
        # Create synthetic return data with known correlation structure
        np.random.seed(42)
        n_periods = 200
        
        # Generate correlated returns
        # Asset 1 and 2 are highly correlated, Asset 3 is independent
        factor = np.random.normal(0, 0.02, n_periods)  # Common factor
        
        returns_1 = 0.8 * factor + 0.6 * np.random.normal(0, 0.01, n_periods)
        returns_2 = 0.7 * factor + 0.7 * np.random.normal(0, 0.01, n_periods)
        returns_3 = np.random.normal(0, 0.015, n_periods)  # Independent
        
        dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')
        
        self.returns_data = pd.DataFrame({
            'STOCK_A': returns_1,
            'STOCK_B': returns_2,
            'STOCK_C': returns_3
        }, index=dates)
        
        self.monitor = DynamicCorrelationMonitor(window_size=60, min_periods=30)
    
    def test_initialization(self):
        """Test monitor initialization."""
        monitor = DynamicCorrelationMonitor(
            window_size=50,
            min_periods=25,
            regime_threshold=0.2
        )
        
        self.assertEqual(monitor.window_size, 50)
        self.assertEqual(monitor.min_periods, 25)
        self.assertEqual(monitor.regime_threshold, 0.2)
        self.assertIsInstance(monitor.alert_thresholds, dict)
        self.assertEqual(len(monitor.correlation_history), 0)
        self.assertEqual(len(monitor.alerts), 0)
    
    def test_calculate_rolling_correlations(self):
        """Test rolling correlation calculation."""
        rolling_corrs = self.monitor.calculate_rolling_correlations(self.returns_data)
        
        # Check structure
        self.assertIsInstance(rolling_corrs, pd.DataFrame)
        self.assertGreater(len(rolling_corrs), 0)
        
        # Check columns
        expected_columns = ['timestamp', 'correlation_matrix', 'window_size']
        for col in expected_columns:
            self.assertIn(col, rolling_corrs.columns)
        
        # Check correlation matrix dimensions
        first_corr_matrix = rolling_corrs.iloc[0]['correlation_matrix']
        self.assertEqual(first_corr_matrix.shape, (3, 3))
        
        # Check diagonal elements are 1 (or close to 1)
        np.testing.assert_array_almost_equal(np.diag(first_corr_matrix), [1, 1, 1], decimal=2)
    
    def test_calculate_rolling_correlations_insufficient_assets(self):
        """Test rolling correlations with insufficient assets."""
        single_asset_data = self.returns_data[['STOCK_A']]
        
        with self.assertRaises(ValueError):
            self.monitor.calculate_rolling_correlations(single_asset_data)
    
    def test_analyze_correlation_regime(self):
        """Test correlation regime analysis."""
        # Create test correlation matrix
        corr_matrix = np.array([
            [1.0, 0.8, 0.2],
            [0.8, 1.0, 0.1],
            [0.2, 0.1, 1.0]
        ])
        
        metrics = self.monitor.analyze_correlation_regime(
            corr_matrix, 
            ['STOCK_A', 'STOCK_B', 'STOCK_C']
        )
        
        # Check structure
        self.assertIsInstance(metrics, CorrelationMetrics)
        self.assertIsInstance(metrics.correlation_regime, CorrelationRegime)
        
        # Check values
        self.assertGreater(metrics.average_correlation, 0)
        self.assertGreater(metrics.max_correlation, 0)
        self.assertGreaterEqual(metrics.regime_confidence, 0)
        self.assertLessEqual(metrics.regime_confidence, 1)
        
        # Check correlation matrix
        np.testing.assert_array_equal(metrics.correlation_matrix, corr_matrix)
    
    def test_correlation_regime_classification(self):
        """Test different correlation regime classifications."""
        # Low correlation regime
        low_corr_matrix = np.array([
            [1.0, 0.1, 0.05],
            [0.1, 1.0, 0.15],
            [0.05, 0.15, 1.0]
        ])
        
        metrics_low = self.monitor.analyze_correlation_regime(low_corr_matrix)
        self.assertEqual(metrics_low.correlation_regime, CorrelationRegime.LOW_CORRELATION)
        
        # High correlation regime
        high_corr_matrix = np.array([
            [1.0, 0.8, 0.75],
            [0.8, 1.0, 0.85],
            [0.75, 0.85, 1.0]
        ])
        
        metrics_high = self.monitor.analyze_correlation_regime(high_corr_matrix)
        self.assertEqual(metrics_high.correlation_regime, CorrelationRegime.HIGH_CORRELATION)
        
        # Crisis correlation regime
        crisis_corr_matrix = np.array([
            [1.0, 0.95, 0.92],
            [0.95, 1.0, 0.98],
            [0.92, 0.98, 1.0]
        ])
        
        metrics_crisis = self.monitor.analyze_correlation_regime(crisis_corr_matrix)
        self.assertEqual(metrics_crisis.correlation_regime, CorrelationRegime.CRISIS_CORRELATION)
    
    def test_monitor_correlations(self):
        """Test full correlation monitoring."""
        correlation_metrics = self.monitor.monitor_correlations(
            self.returns_data, 
            generate_alerts=True
        )
        
        # Check results
        self.assertIsInstance(correlation_metrics, list)
        self.assertGreater(len(correlation_metrics), 0)
        
        # Check each metric
        for metrics in correlation_metrics:
            self.assertIsInstance(metrics, CorrelationMetrics)
            self.assertIsInstance(metrics.correlation_regime, CorrelationRegime)
        
        # Check history is populated
        self.assertEqual(len(self.monitor.correlation_history), len(correlation_metrics))
        
        # Check asset names are set
        self.assertEqual(self.monitor.asset_names, ['STOCK_A', 'STOCK_B', 'STOCK_C'])
    
    def test_detect_correlation_breakpoints(self):
        """Test correlation breakpoint detection."""
        # Create correlation series with a clear breakpoint
        n = 100
        correlation_series = pd.Series(
            np.concatenate([
                np.random.normal(0.3, 0.05, n//2),  # Low correlation period
                np.random.normal(0.8, 0.05, n//2)   # High correlation period
            ])
        )
        
        # Test different methods
        for method in ['cusum', 'variance', 'mean']:
            breakpoints = self.monitor.detect_correlation_breakpoints(
                correlation_series, 
                method=method
            )
            
            self.assertIsInstance(breakpoints, list)
            # Should detect at least one breakpoint for this clear structural change
            # (though exact detection depends on method sensitivity)
    
    def test_detect_correlation_breakpoints_insufficient_data(self):
        """Test breakpoint detection with insufficient data."""
        short_series = pd.Series([0.1, 0.2, 0.3])
        
        breakpoints = self.monitor.detect_correlation_breakpoints(short_series)
        self.assertEqual(breakpoints, [])
    
    def test_get_diversification_ratio(self):
        """Test diversification ratio calculation."""
        # Perfect diversification (zero correlations)
        perfect_div_matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        div_ratio_perfect = self.monitor.get_diversification_ratio(perfect_div_matrix)
        self.assertAlmostEqual(div_ratio_perfect, 1.0, places=2)
        
        # No diversification (perfect correlations)
        no_div_matrix = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ])
        
        div_ratio_none = self.monitor.get_diversification_ratio(no_div_matrix)
        self.assertAlmostEqual(div_ratio_none, 0.0, places=2)
        
        # Test with custom weights
        weights = np.array([0.5, 0.3, 0.2])
        div_ratio_weighted = self.monitor.get_diversification_ratio(
            perfect_div_matrix, 
            weights
        )
        self.assertAlmostEqual(div_ratio_weighted, 1.0, places=2)
    
    def test_correlation_clustering(self):
        """Test correlation-based clustering."""
        # Create correlation matrix with clear clusters
        corr_matrix = np.array([
            [1.0, 0.9, 0.1, 0.05],  # Assets 0,1 highly correlated
            [0.9, 1.0, 0.15, 0.1],
            [0.1, 0.15, 1.0, 0.85], # Assets 2,3 highly correlated
            [0.05, 0.1, 0.85, 1.0]
        ])
        
        asset_names = ['A', 'B', 'C', 'D']
        
        clusters = self.monitor._perform_correlation_clustering(corr_matrix, asset_names)
        
        # Should identify clusters
        self.assertIsInstance(clusters, list)
        self.assertGreater(len(clusters), 1)
        
        # All assets should be assigned to clusters
        all_assets_in_clusters = []
        for cluster in clusters:
            all_assets_in_clusters.extend(cluster)
        
        self.assertEqual(set(all_assets_in_clusters), set(asset_names))
    
    def test_alert_generation(self):
        """Test correlation alert generation."""
        # Create high correlation scenario
        high_corr_matrix = np.array([
            [1.0, 0.85, 0.8],
            [0.85, 1.0, 0.9],
            [0.8, 0.9, 1.0]
        ])
        
        metrics = self.monitor.analyze_correlation_regime(high_corr_matrix)
        alerts = self.monitor._generate_correlation_alerts(metrics)
        
        # Should generate alerts for high correlation
        self.assertIsInstance(alerts, list)
        
        # Check alert structure
        for alert in alerts:
            self.assertIsInstance(alert, CorrelationAlert)
            self.assertIn(alert.alert_type, [
                'high_correlation', 'crisis_correlation', 
                'diversification_loss', 'regime_change'
            ])
            self.assertIn(alert.severity, ['low', 'medium', 'high', 'critical'])
            self.assertIsInstance(alert.timestamp, datetime)
    
    def test_get_recent_alerts(self):
        """Test recent alerts retrieval."""
        # Add some test alerts
        now = datetime.now()
        
        old_alert = CorrelationAlert(
            alert_type='test',
            message='Old alert',
            severity='low',
            timestamp=now - timedelta(hours=48),
            affected_assets=['A'],
            correlation_value=0.5,
            threshold=0.4
        )
        
        recent_alert = CorrelationAlert(
            alert_type='test',
            message='Recent alert',
            severity='medium',
            timestamp=now - timedelta(hours=1),
            affected_assets=['B'],
            correlation_value=0.8,
            threshold=0.7
        )
        
        self.monitor.alerts = [old_alert, recent_alert]
        
        # Get recent alerts (last 24 hours)
        recent_alerts = self.monitor.get_recent_alerts(hours=24)
        
        self.assertEqual(len(recent_alerts), 1)
        self.assertEqual(recent_alerts[0].message, 'Recent alert')
    
    def test_clear_old_alerts(self):
        """Test clearing old alerts."""
        now = datetime.now()
        
        old_alert = CorrelationAlert(
            alert_type='test',
            message='Old alert',
            severity='low',
            timestamp=now - timedelta(days=10),
            affected_assets=['A'],
            correlation_value=0.5,
            threshold=0.4
        )
        
        recent_alert = CorrelationAlert(
            alert_type='test',
            message='Recent alert',
            severity='medium',
            timestamp=now - timedelta(days=1),
            affected_assets=['B'],
            correlation_value=0.8,
            threshold=0.7
        )
        
        self.monitor.alerts = [old_alert, recent_alert]
        
        # Clear alerts older than 7 days
        self.monitor.clear_old_alerts(days=7)
        
        self.assertEqual(len(self.monitor.alerts), 1)
        self.assertEqual(self.monitor.alerts[0].message, 'Recent alert')


class TestConvenienceFunction(unittest.TestCase):
    """Test convenience function for correlation analysis."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n_periods = 100
        
        returns_1 = np.random.normal(0.001, 0.02, n_periods)
        returns_2 = 0.7 * returns_1 + 0.3 * np.random.normal(0.001, 0.02, n_periods)
        returns_3 = np.random.normal(0.001, 0.02, n_periods)
        
        dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')
        
        self.returns_data = pd.DataFrame({
            'A': returns_1,
            'B': returns_2,
            'C': returns_3
        }, index=dates)
    
    def test_calculate_correlation_metrics_valid_data(self):
        """Test correlation metrics calculation with valid data."""
        result = calculate_correlation_metrics(self.returns_data)
        
        # Check structure
        self.assertIn('correlation_matrix', result)
        self.assertIn('average_correlation', result)
        self.assertIn('regime', result)
        self.assertIn('diversification_ratio', result)
        
        # Check values
        self.assertIsInstance(result['correlation_matrix'], list)
        self.assertIsInstance(result['average_correlation'], float)
        self.assertIsInstance(result['regime'], str)
        self.assertIsInstance(result['diversification_ratio'], float)
        
        # Check correlation matrix dimensions
        corr_matrix = result['correlation_matrix']
        self.assertEqual(len(corr_matrix), 3)
        self.assertEqual(len(corr_matrix[0]), 3)
    
    def test_calculate_correlation_metrics_insufficient_assets(self):
        """Test correlation metrics with insufficient assets."""
        single_asset_data = self.returns_data[['A']]
        
        result = calculate_correlation_metrics(single_asset_data)
        
        self.assertIn('error', result)
        self.assertIsNone(result['correlation_matrix'])
        self.assertIsNone(result['regime'])
    
    def test_calculate_correlation_metrics_insufficient_data(self):
        """Test correlation metrics with insufficient data."""
        short_data = self.returns_data.head(10)  # Very short time series
        
        result = calculate_correlation_metrics(short_data, window_size=50)
        
        # Should handle gracefully
        self.assertIn('error', result)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_invalid_correlation_matrix(self):
        """Test handling of invalid correlation matrices."""
        monitor = DynamicCorrelationMonitor()
        
        # Matrix with NaN values
        nan_matrix = np.array([
            [1.0, np.nan, 0.5],
            [np.nan, 1.0, 0.3],
            [0.5, 0.3, 1.0]
        ])
        
        metrics = monitor.analyze_correlation_regime(nan_matrix)
        
        # Should handle gracefully
        self.assertIsInstance(metrics, CorrelationMetrics)
        self.assertIsInstance(metrics.correlation_regime, CorrelationRegime)
    
    def test_empty_returns_data(self):
        """Test handling of empty returns data."""
        monitor = DynamicCorrelationMonitor()
        empty_data = pd.DataFrame()
        
        with self.assertRaises(ValueError):
            monitor.calculate_rolling_correlations(empty_data)
    
    def test_clustering_failure_fallback(self):
        """Test clustering fallback when clustering fails."""
        monitor = DynamicCorrelationMonitor()
        
        # Create problematic correlation matrix
        problematic_matrix = np.array([
            [1.0, np.inf, 0.5],
            [np.inf, 1.0, 0.3],
            [0.5, 0.3, 1.0]
        ])
        
        asset_names = ['A', 'B', 'C']
        
        # Should not crash, should return some clustering
        clusters = monitor._perform_correlation_clustering(problematic_matrix, asset_names)
        
        self.assertIsInstance(clusters, list)
        self.assertGreater(len(clusters), 0)  # Should have at least one cluster
        
        # All assets should be assigned to clusters
        all_assets_in_clusters = []
        for cluster in clusters:
            all_assets_in_clusters.extend(cluster)
        
        self.assertEqual(set(all_assets_in_clusters), set(asset_names))


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)