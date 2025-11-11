import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)



class CorrelationRegime(Enum):
    LOW_CORRELATION = "low_correlation"
    MODERATE_CORRELATION = "moderate_correlation"
    HIGH_CORRELATION = "high_correlation"
    CRISIS_CORRELATION = "crisis_correlation"


@dataclass
class CorrelationMetrics:
    correlation_matrix: np.ndarray
    average_correlation: float
    max_correlation: float
    min_correlation: float
    correlation_regime: CorrelationRegime
    regime_confidence: float
    correlation_clusters: Optional[List[List[str]]] = None
    regime_change_probability: Optional[float] = None


@dataclass
class CorrelationAlert:
    alert_type: str
    message: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    timestamp: datetime
    affected_assets: List[str]
    correlation_value: float
    threshold: float


class DynamicCorrelationMonitor:
    def __init__(self,
                 window_size: int = 60,
                 min_periods: int = 30,
                 regime_threshold: float = 0.15,
                 alert_thresholds: Dict[str, float] = None):
        self.window_size = window_size
        self.min_periods = min_periods
        self.regime_threshold = regime_threshold
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'high_correlation': 0.7,
            'crisis_correlation': 0.9,
            'correlation_spike': 0.3,  # Change in correlation
            'diversification_loss': 0.8  # Average correlation threshold
        }
        
        # Historical data storage
        self.correlation_history: List[CorrelationMetrics] = []
        self.alerts: List[CorrelationAlert] = []
        self.asset_names: Optional[List[str]] = None
        
    def calculate_rolling_correlations(self,
                                     returns_data: pd.DataFrame,
                                     method: str = 'pearson') -> pd.DataFrame:
        if returns_data.empty or len(returns_data.columns) < 2:
            raise ValueError("Need at least 2 assets for correlation calculation")
        
        self.asset_names = list(returns_data.columns)
        n_assets = len(self.asset_names)
        
        # Calculate rolling correlations
        rolling_corrs = []
        
        for i in range(self.min_periods - 1, len(returns_data)):
            # Get window data
            window_start = max(0, i - self.window_size + 1)
            window_data = returns_data.iloc[window_start:i+1]
            
            if len(window_data) >= self.min_periods:
                # Calculate correlation matrix
                corr_matrix = window_data.corr(method=method).values
                
                # Handle NaN values
                corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
                
                # Store with timestamp
                rolling_corrs.append({
                    'timestamp': returns_data.index[i],
                    'correlation_matrix': corr_matrix,
                    'window_size': len(window_data)
                })
        
        return pd.DataFrame(rolling_corrs)
    
    def analyze_correlation_regime(self,
                                 correlation_matrix: np.ndarray,
                                 asset_names: List[str] = None) -> CorrelationMetrics:
        if asset_names is None:
            asset_names = self.asset_names or [f"Asset_{i}" for i in range(len(correlation_matrix))]
        
        # Extract upper triangular correlations (excluding diagonal)
        n = correlation_matrix.shape[0]
        upper_tri_indices = np.triu_indices(n, k=1)
        correlations = correlation_matrix[upper_tri_indices]
        
        # Remove NaN and infinite values
        valid_correlations = correlations[np.isfinite(correlations)]
        
        if len(valid_correlations) == 0:
            # Fallback for invalid data
            return CorrelationMetrics(
                correlation_matrix=correlation_matrix,
                average_correlation=0.0,
                max_correlation=0.0,
                min_correlation=0.0,
                correlation_regime=CorrelationRegime.LOW_CORRELATION,
                regime_confidence=0.0
            )
        
        # Calculate metrics
        avg_corr = np.mean(valid_correlations)
        max_corr = np.max(valid_correlations)
        min_corr = np.min(valid_correlations)
        
        # Determine correlation regime
        regime, confidence = self._classify_correlation_regime(valid_correlations)
        
        # Perform correlation clustering
        clusters = self._perform_correlation_clustering(correlation_matrix, asset_names)
        
        # Calculate regime change probability if we have history
        regime_change_prob = self._calculate_regime_change_probability(regime)
        
        return CorrelationMetrics(
            correlation_matrix=correlation_matrix,
            average_correlation=avg_corr,
            max_correlation=max_corr,
            min_correlation=min_corr,
            correlation_regime=regime,
            regime_confidence=confidence,
            correlation_clusters=clusters,
            regime_change_probability=regime_change_prob
        )
    
    def monitor_correlations(self,
                           returns_data: pd.DataFrame,
                           generate_alerts: bool = True) -> List[CorrelationMetrics]:
        rolling_corr_data = self.calculate_rolling_correlations(returns_data)
        rolling_corr_data = self.calculate_rolling_correlations(returns_data)
        
        correlation_metrics = []
        
        for _, row in rolling_corr_data.iterrows():
            # Analyze correlation regime
            metrics = self.analyze_correlation_regime(
                row['correlation_matrix'], 
                self.asset_names
            )
            
            # Add timestamp
            metrics.timestamp = row['timestamp']
            
            # Store in history
            self.correlation_history.append(metrics)
            correlation_metrics.append(metrics)
            
            # Generate alerts if requested
            if generate_alerts:
                alerts = self._generate_correlation_alerts(metrics)
                self.alerts.extend(alerts)
        
        return correlation_metrics
    
    def detect_correlation_breakpoints(self, 
                                     correlation_series: pd.Series,
                                     method: str = 'cusum') -> List[int]:
        """
        Detect structural breaks in correlation time series.
        
        Args:
            correlation_series: Time series of correlation values
            method: Detection method ('cusum', 'variance', 'mean')
            
        Returns:
            List of breakpoint indices
        """
        if len(correlation_series) < 20:
            return []
        
        breakpoints = []
        
        if method == 'cusum':
            # CUSUM test for mean changes
            values = correlation_series.values
            cumsum = np.cumsum(values - np.mean(values))
            
            # Find significant deviations
            threshold = 3 * np.std(cumsum)
            
            for i in range(1, len(cumsum) - 1):
                if abs(cumsum[i]) > threshold:
                    # Check if this is a local extremum
                    if (cumsum[i] > cumsum[i-1] and cumsum[i] > cumsum[i+1]) or \
                       (cumsum[i] < cumsum[i-1] and cumsum[i] < cumsum[i+1]):
                        breakpoints.append(i)
        
        elif method == 'variance':
            # Rolling variance change detection
            window = min(20, len(correlation_series) // 4)
            rolling_var = correlation_series.rolling(window=window).var()
            
            # Find significant variance changes
            var_changes = rolling_var.diff().abs()
            threshold = var_changes.quantile(0.95)
            
            breakpoints = var_changes[var_changes > threshold].index.tolist()
        
        elif method == 'mean':
            # Rolling mean change detection
            window = min(20, len(correlation_series) // 4)
            rolling_mean = correlation_series.rolling(window=window).mean()
            
            # Find significant mean changes
            mean_changes = rolling_mean.diff().abs()
            threshold = mean_changes.quantile(0.95)
            
            breakpoints = mean_changes[mean_changes > threshold].index.tolist()
        
        return breakpoints
    
    def get_diversification_ratio(self, 
                                correlation_matrix: np.ndarray,
                                weights: np.ndarray = None) -> float:
        """
        Calculate portfolio diversification ratio.
        
        Args:
            correlation_matrix: Asset correlation matrix
            weights: Portfolio weights (equal weights if None)
            
        Returns:
            Diversification ratio (higher is better diversified)
        """
        n_assets = correlation_matrix.shape[0]
        
        if weights is None:
            weights = np.ones(n_assets) / n_assets
        
        # Weighted average correlation
        weighted_corr = 0.0
        total_weight = 0.0
        
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                pair_weight = weights[i] * weights[j]
                weighted_corr += pair_weight * correlation_matrix[i, j]
                total_weight += pair_weight
        
        if total_weight > 0:
            weighted_corr /= total_weight
        
        # Diversification ratio: 1 means perfect diversification, 0 means no diversification
        diversification_ratio = 1 - abs(weighted_corr)
        
        return max(0.0, min(1.0, diversification_ratio))
    
    def _classify_correlation_regime(self, 
                                   correlations: np.ndarray) -> Tuple[CorrelationRegime, float]:
        """Classify correlation regime based on correlation distribution."""
        avg_corr = np.mean(np.abs(correlations))  # Use absolute correlations
        max_corr = np.max(np.abs(correlations))
        
        # Calculate confidence based on consistency of correlations
        std_corr = np.std(correlations)
        confidence = max(0.0, 1.0 - std_corr)  # Lower std = higher confidence
        
        # Classify regime
        if max_corr >= 0.9:
            regime = CorrelationRegime.CRISIS_CORRELATION
        elif avg_corr >= 0.7:
            regime = CorrelationRegime.HIGH_CORRELATION
        elif avg_corr >= 0.3:
            regime = CorrelationRegime.MODERATE_CORRELATION
        else:
            regime = CorrelationRegime.LOW_CORRELATION
        
        return regime, confidence
    
    def _perform_correlation_clustering(self, 
                                      correlation_matrix: np.ndarray,
                                      asset_names: List[str]) -> List[List[str]]:
        """Perform hierarchical clustering based on correlations."""
        try:
            # Convert correlation to distance matrix
            distance_matrix = 1 - np.abs(correlation_matrix)
            
            # Ensure distance matrix is valid
            np.fill_diagonal(distance_matrix, 0)
            distance_matrix = np.maximum(distance_matrix, 0)
            
            # Perform hierarchical clustering
            condensed_distances = squareform(distance_matrix)
            
            if len(condensed_distances) == 0:
                return [[name] for name in asset_names]
            
            linkage_matrix = linkage(condensed_distances, method='ward')
            
            # Form clusters (aim for 2-4 clusters)
            n_clusters = min(4, max(2, len(asset_names) // 2))
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # Group assets by cluster
            clusters = []
            for cluster_id in range(1, n_clusters + 1):
                cluster_assets = [asset_names[i] for i, label in enumerate(cluster_labels) 
                                if label == cluster_id]
                if cluster_assets:
                    clusters.append(cluster_assets)
            
            return clusters
            
        except Exception as e:
            logger.warning(f"Correlation clustering failed: {e}")
            # Return each asset as its own cluster
            return [[name] for name in asset_names]
    
    def _calculate_regime_change_probability(self, 
                                           current_regime: CorrelationRegime) -> Optional[float]:
        """Calculate probability of regime change based on history."""
        if len(self.correlation_history) < 10:
            return None
        
        # Look at recent regime history
        recent_regimes = [metrics.correlation_regime for metrics in self.correlation_history[-10:]]
        
        # Count regime changes
        regime_changes = sum(1 for i in range(1, len(recent_regimes)) 
                           if recent_regimes[i] != recent_regimes[i-1])
        
        # Calculate change probability
        change_probability = regime_changes / (len(recent_regimes) - 1)
        
        return change_probability
    
    def _generate_correlation_alerts(self, 
                                   metrics: CorrelationMetrics) -> List[CorrelationAlert]:
        """Generate alerts based on correlation metrics."""
        alerts = []
        timestamp = datetime.now()
        
        # High correlation alert
        if metrics.average_correlation >= self.alert_thresholds['high_correlation']:
            alerts.append(CorrelationAlert(
                alert_type='high_correlation',
                message=f"High average correlation detected: {metrics.average_correlation:.3f}",
                severity='medium' if metrics.average_correlation < 0.8 else 'high',
                timestamp=timestamp,
                affected_assets=self.asset_names or [],
                correlation_value=metrics.average_correlation,
                threshold=self.alert_thresholds['high_correlation']
            ))
        
        # Crisis correlation alert
        if metrics.correlation_regime == CorrelationRegime.CRISIS_CORRELATION:
            alerts.append(CorrelationAlert(
                alert_type='crisis_correlation',
                message=f"Crisis-level correlations detected: max={metrics.max_correlation:.3f}",
                severity='critical',
                timestamp=timestamp,
                affected_assets=self.asset_names or [],
                correlation_value=metrics.max_correlation,
                threshold=self.alert_thresholds['crisis_correlation']
            ))
        
        # Diversification loss alert
        if metrics.average_correlation >= self.alert_thresholds['diversification_loss']:
            alerts.append(CorrelationAlert(
                alert_type='diversification_loss',
                message=f"Significant diversification loss: avg_corr={metrics.average_correlation:.3f}",
                severity='high',
                timestamp=timestamp,
                affected_assets=self.asset_names or [],
                correlation_value=metrics.average_correlation,
                threshold=self.alert_thresholds['diversification_loss']
            ))
        
        # Regime change alert
        if (metrics.regime_change_probability is not None and 
            metrics.regime_change_probability > 0.5):
            alerts.append(CorrelationAlert(
                alert_type='regime_change',
                message=f"High probability of correlation regime change: {metrics.regime_change_probability:.3f}",
                severity='medium',
                timestamp=timestamp,
                affected_assets=self.asset_names or [],
                correlation_value=metrics.regime_change_probability,
                threshold=0.5
            ))
        
        return alerts
    
    def get_recent_alerts(self, hours: int = 24) -> List[CorrelationAlert]:
        """Get recent correlation alerts."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert.timestamp >= cutoff_time]
    
    def clear_old_alerts(self, days: int = 7):
        """Clear alerts older than specified days."""
        cutoff_time = datetime.now() - timedelta(days=days)
        self.alerts = [alert for alert in self.alerts if alert.timestamp >= cutoff_time]


def calculate_correlation_metrics(returns_data: pd.DataFrame,
                                window_size: int = 60) -> Dict[str, Any]:
    """
    Convenience function for calculating correlation metrics.
    
    Args:
        returns_data: DataFrame with returns for multiple assets
        window_size: Rolling window size
        
    Returns:
        Dictionary with correlation analysis results
    """
    monitor = DynamicCorrelationMonitor(window_size=window_size)
    
    if len(returns_data.columns) < 2:
        return {
            'error': 'Need at least 2 assets for correlation analysis',
            'correlation_matrix': None,
            'regime': None
        }
    
    try:
        # Get latest correlation metrics
        correlation_metrics = monitor.monitor_correlations(returns_data, generate_alerts=False)
        
        if not correlation_metrics:
            return {
                'error': 'Insufficient data for correlation analysis',
                'correlation_matrix': None,
                'regime': None
            }
        
        latest_metrics = correlation_metrics[-1]
        
        return {
            'correlation_matrix': latest_metrics.correlation_matrix.tolist(),
            'average_correlation': latest_metrics.average_correlation,
            'max_correlation': latest_metrics.max_correlation,
            'min_correlation': latest_metrics.min_correlation,
            'regime': latest_metrics.correlation_regime.value,
            'regime_confidence': latest_metrics.regime_confidence,
            'clusters': latest_metrics.correlation_clusters,
            'diversification_ratio': monitor.get_diversification_ratio(latest_metrics.correlation_matrix)
        }
        
    except Exception as e:
        logger.error(f"Error in correlation analysis: {e}")
        return {
            'error': f'Correlation analysis failed: {str(e)}',
            'correlation_matrix': None,
            'regime': None
        }