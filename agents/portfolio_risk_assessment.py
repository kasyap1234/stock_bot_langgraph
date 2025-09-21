"""
Real-time Portfolio Risk Assessment Module

This module implements comprehensive portfolio-level risk metrics calculation
and drawdown monitoring with alerts as part of the advanced risk assessment system.

Requirements addressed:
- 2.4 - Portfolio-level risk monitoring and assessment
- 2.5 - Drawdown monitoring and defensive measures
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import math

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Risk alert types"""
    DRAWDOWN_WARNING = "drawdown_warning"
    DRAWDOWN_CRITICAL = "drawdown_critical"
    VOLATILITY_SPIKE = "volatility_spike"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    CONCENTRATION_RISK = "concentration_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    VAR_BREACH = "var_breach"
    LEVERAGE_EXCESSIVE = "leverage_excessive"


@dataclass
class Position:
    """Individual position in portfolio"""
    symbol: str
    quantity: float
    current_price: float
    entry_price: float
    entry_date: datetime
    market_value: float
    unrealized_pnl: float
    weight: float
    
    def __post_init__(self):
        """Calculate derived fields"""
        if self.market_value == 0:
            self.market_value = self.quantity * self.current_price
        if self.unrealized_pnl == 0:
            self.unrealized_pnl = self.quantity * (self.current_price - self.entry_price)


@dataclass
class PortfolioSnapshot:
    """Portfolio state at a point in time"""
    timestamp: datetime
    positions: Dict[str, Position]
    total_value: float
    cash: float
    total_equity: float
    daily_pnl: float
    total_pnl: float
    leverage: float
    
    def __post_init__(self):
        """Calculate derived metrics"""
        if self.total_value == 0:
            self.total_value = sum(pos.market_value for pos in self.positions.values())
        if self.total_equity == 0:
            self.total_equity = self.total_value + self.cash


@dataclass
class RiskMetrics:
    """Comprehensive portfolio risk metrics"""
    # Basic metrics
    portfolio_value: float
    daily_return: float
    volatility: float
    sharpe_ratio: float
    
    # Drawdown metrics
    current_drawdown: float
    max_drawdown: float
    drawdown_duration: int
    underwater_curve: List[float]
    
    # Risk measures
    var_95: float
    var_99: float
    cvar_95: float
    expected_shortfall: float
    
    # Concentration metrics
    concentration_ratio: float
    herfindahl_index: float
    effective_positions: float
    
    # Correlation metrics
    avg_correlation: float
    max_correlation: float
    correlation_breakdown_risk: float
    
    # Advanced metrics
    beta: float
    tracking_error: float
    information_ratio: float
    calmar_ratio: float
    sortino_ratio: float
    
    # Risk level assessment
    overall_risk_level: RiskLevel
    risk_score: float
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RiskAlert:
    """Risk monitoring alert"""
    alert_type: AlertType
    severity: RiskLevel
    message: str
    timestamp: datetime
    current_value: float
    threshold: float
    recommendation: str
    affected_positions: List[str] = field(default_factory=list)


class RealTimePortfolioRiskAssessment:
    """
    Real-time portfolio risk assessment system.
    
    Monitors portfolio-level risk metrics, detects risk threshold breaches,
    and generates alerts with recommendations for risk management actions.
    """
    
    def __init__(self,
                 risk_thresholds: Dict[str, float] = None,
                 lookback_window: int = 252,
                 alert_cooldown_hours: int = 1,
                 benchmark_symbol: str = 'SPY'):
        """
        Initialize portfolio risk assessment system.
        
        Args:
            risk_thresholds: Custom risk thresholds for alerts
            lookback_window: Historical data window for calculations
            alert_cooldown_hours: Minimum hours between similar alerts
            benchmark_symbol: Benchmark for beta and tracking error calculations
        """
        # Default risk thresholds
        default_thresholds = {
            'max_drawdown_warning': 0.10,      # 10% drawdown warning
            'max_drawdown_critical': 0.20,     # 20% drawdown critical
            'volatility_spike': 2.0,           # 2x normal volatility
            'concentration_warning': 0.30,     # 30% in single position
            'concentration_critical': 0.50,    # 50% in single position
            'correlation_breakdown': 0.90,     # 90% correlation spike
            'var_breach_multiplier': 2.0,      # 2x VaR breach
            'leverage_warning': 1.5,           # 1.5x leverage
            'leverage_critical': 2.0           # 2x leverage
        }
        
        # Merge with custom thresholds if provided
        if risk_thresholds:
            default_thresholds.update(risk_thresholds)
        
        self.risk_thresholds = default_thresholds
        
        self.lookback_window = lookback_window
        self.alert_cooldown_hours = alert_cooldown_hours
        self.benchmark_symbol = benchmark_symbol
        
        # Historical data storage
        self.portfolio_history: List[PortfolioSnapshot] = []
        self.risk_metrics_history: List[RiskMetrics] = []
        self.alerts_history: List[RiskAlert] = []
        
        # Benchmark data cache
        self.benchmark_returns: Optional[pd.Series] = None
        
    def assess_portfolio_risk(self,
                            current_portfolio: PortfolioSnapshot,
                            market_data: Dict[str, pd.DataFrame],
                            generate_alerts: bool = True) -> RiskMetrics:
        """
        Assess comprehensive portfolio risk metrics.
        
        Args:
            current_portfolio: Current portfolio state
            market_data: Historical market data for positions
            generate_alerts: Whether to generate risk alerts
            
        Returns:
            Comprehensive risk metrics
        """
        # Add to portfolio history
        self.portfolio_history.append(current_portfolio)
        
        # Limit history size
        if len(self.portfolio_history) > self.lookback_window:
            self.portfolio_history = self.portfolio_history[-self.lookback_window:]
        
        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns()
        
        # Calculate basic metrics
        daily_return = portfolio_returns[-1] if len(portfolio_returns) > 0 else 0.0
        volatility = np.std(portfolio_returns) * np.sqrt(252) if len(portfolio_returns) > 1 else 0.0
        
        # Calculate Sharpe ratio
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        excess_returns = portfolio_returns - risk_free_rate / 252
        sharpe_ratio = (np.mean(excess_returns) * 252 / volatility) if volatility > 0 else 0.0
        
        # Calculate drawdown metrics
        drawdown_metrics = self._calculate_drawdown_metrics(portfolio_returns)
        
        # Calculate risk measures (VaR, CVaR)
        risk_measures = self._calculate_risk_measures(portfolio_returns)
        
        # Calculate concentration metrics
        concentration_metrics = self._calculate_concentration_metrics(current_portfolio)
        
        # Calculate correlation metrics
        correlation_metrics = self._calculate_correlation_metrics(
            current_portfolio, market_data
        )
        
        # Calculate advanced metrics
        advanced_metrics = self._calculate_advanced_metrics(
            portfolio_returns, market_data
        )
        
        # Assess overall risk level
        risk_level, risk_score = self._assess_overall_risk_level(
            drawdown_metrics, concentration_metrics, correlation_metrics, volatility
        )
        
        # Create comprehensive risk metrics
        risk_metrics = RiskMetrics(
            portfolio_value=current_portfolio.total_value,
            daily_return=daily_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            
            current_drawdown=drawdown_metrics['current_drawdown'],
            max_drawdown=drawdown_metrics['max_drawdown'],
            drawdown_duration=drawdown_metrics['drawdown_duration'],
            underwater_curve=drawdown_metrics['underwater_curve'],
            
            var_95=risk_measures['var_95'],
            var_99=risk_measures['var_99'],
            cvar_95=risk_measures['cvar_95'],
            expected_shortfall=risk_measures['expected_shortfall'],
            
            concentration_ratio=concentration_metrics['concentration_ratio'],
            herfindahl_index=concentration_metrics['herfindahl_index'],
            effective_positions=concentration_metrics['effective_positions'],
            
            avg_correlation=correlation_metrics['avg_correlation'],
            max_correlation=correlation_metrics['max_correlation'],
            correlation_breakdown_risk=correlation_metrics['breakdown_risk'],
            
            beta=advanced_metrics['beta'],
            tracking_error=advanced_metrics['tracking_error'],
            information_ratio=advanced_metrics['information_ratio'],
            calmar_ratio=advanced_metrics['calmar_ratio'],
            sortino_ratio=advanced_metrics['sortino_ratio'],
            
            overall_risk_level=risk_level,
            risk_score=risk_score
        )
        
        # Store in history
        self.risk_metrics_history.append(risk_metrics)
        
        # Generate alerts if requested
        if generate_alerts:
            alerts = self._generate_risk_alerts(risk_metrics, current_portfolio)
            self.alerts_history.extend(alerts)
        
        return risk_metrics
    
    def get_risk_dashboard(self) -> Dict[str, Any]:
        """
        Get comprehensive risk dashboard data.
        
        Returns:
            Dictionary with dashboard data for visualization
        """
        if not self.risk_metrics_history:
            return {'error': 'No risk metrics available'}
        
        latest_metrics = self.risk_metrics_history[-1]
        recent_alerts = self.get_recent_alerts(hours=24)
        
        # Risk trend analysis
        risk_trend = self._analyze_risk_trends()
        
        return {
            'current_metrics': {
                'portfolio_value': latest_metrics.portfolio_value,
                'daily_return': latest_metrics.daily_return,
                'volatility': latest_metrics.volatility,
                'sharpe_ratio': latest_metrics.sharpe_ratio,
                'current_drawdown': latest_metrics.current_drawdown,
                'max_drawdown': latest_metrics.max_drawdown,
                'var_95': latest_metrics.var_95,
                'concentration_ratio': latest_metrics.concentration_ratio,
                'overall_risk_level': latest_metrics.overall_risk_level.value,
                'risk_score': latest_metrics.risk_score
            },
            'recent_alerts': [
                {
                    'type': alert.alert_type.value,
                    'severity': alert.severity.value,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'recommendation': alert.recommendation
                }
                for alert in recent_alerts
            ],
            'risk_trends': risk_trend,
            'thresholds': self.risk_thresholds
        }
    
    def get_recent_alerts(self, hours: int = 24) -> List[RiskAlert]:
        """Get recent risk alerts."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts_history if alert.timestamp >= cutoff_time]
    
    def update_risk_thresholds(self, new_thresholds: Dict[str, float]):
        """Update risk thresholds."""
        self.risk_thresholds.update(new_thresholds)
        logger.info(f"Updated risk thresholds: {new_thresholds}")
    
    def _calculate_portfolio_returns(self) -> np.ndarray:
        """Calculate portfolio returns from history."""
        if len(self.portfolio_history) < 2:
            return np.array([])
        
        values = [snapshot.total_value for snapshot in self.portfolio_history]
        returns = np.diff(values) / values[:-1]
        
        return returns
    
    def _calculate_drawdown_metrics(self, returns: np.ndarray) -> Dict[str, Any]:
        """Calculate drawdown-related metrics."""
        if len(returns) == 0:
            return {
                'current_drawdown': 0.0,
                'max_drawdown': 0.0,
                'drawdown_duration': 0,
                'underwater_curve': []
            }
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + returns)
        
        # Calculate running maximum (peak)
        running_max = np.maximum.accumulate(cumulative_returns)
        
        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Current drawdown
        current_drawdown = drawdown[-1]
        
        # Maximum drawdown
        max_drawdown = np.min(drawdown)
        
        # Drawdown duration (days since last peak)
        drawdown_duration = 0
        for i in range(len(drawdown) - 1, -1, -1):
            if drawdown[i] < -0.001:  # 0.1% threshold
                drawdown_duration += 1
            else:
                break
        
        return {
            'current_drawdown': current_drawdown,
            'max_drawdown': max_drawdown,
            'drawdown_duration': drawdown_duration,
            'underwater_curve': drawdown.tolist()
        }
    
    def _calculate_risk_measures(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate VaR and other risk measures."""
        if len(returns) < 30:
            return {
                'var_95': 0.0,
                'var_99': 0.0,
                'cvar_95': 0.0,
                'expected_shortfall': 0.0
            }
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)  # 95% VaR
        var_99 = np.percentile(returns, 1)  # 99% VaR
        
        # Conditional VaR (Expected Shortfall)
        cvar_95 = np.mean(returns[returns <= var_95])
        expected_shortfall = cvar_95
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'expected_shortfall': expected_shortfall
        }
    
    def _calculate_concentration_metrics(self, portfolio: PortfolioSnapshot) -> Dict[str, float]:
        """Calculate portfolio concentration metrics."""
        if not portfolio.positions:
            return {
                'concentration_ratio': 0.0,
                'herfindahl_index': 0.0,
                'effective_positions': 0.0
            }
        
        # Calculate weights
        total_value = sum(pos.market_value for pos in portfolio.positions.values())
        
        if total_value <= 0:
            return {
                'concentration_ratio': 0.0,
                'herfindahl_index': 0.0,
                'effective_positions': 0.0
            }
        
        weights = [pos.market_value / total_value for pos in portfolio.positions.values()]
        
        # Concentration ratio (largest position weight)
        concentration_ratio = max(weights) if weights else 0.0
        
        # Herfindahl-Hirschman Index
        herfindahl_index = sum(w**2 for w in weights)
        
        # Effective number of positions
        effective_positions = 1.0 / herfindahl_index if herfindahl_index > 0 else 0.0
        
        return {
            'concentration_ratio': concentration_ratio,
            'herfindahl_index': herfindahl_index,
            'effective_positions': effective_positions
        }
    
    def _calculate_correlation_metrics(self, 
                                     portfolio: PortfolioSnapshot,
                                     market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate correlation-based risk metrics."""
        symbols = list(portfolio.positions.keys())
        
        if len(symbols) < 2:
            return {
                'avg_correlation': 0.0,
                'max_correlation': 0.0,
                'breakdown_risk': 0.0
            }
        
        # Calculate correlation matrix
        returns_data = {}
        for symbol in symbols:
            if symbol in market_data and 'Close' in market_data[symbol].columns:
                returns = market_data[symbol]['Close'].pct_change().dropna()
                if len(returns) >= 30:  # Minimum data requirement
                    returns_data[symbol] = returns.tail(60)  # Use last 60 days
        
        if len(returns_data) < 2:
            return {
                'avg_correlation': 0.0,
                'max_correlation': 0.0,
                'breakdown_risk': 0.0
            }
        
        # Align data
        returns_df = pd.DataFrame(returns_data).dropna()
        
        if len(returns_df) < 10:
            return {
                'avg_correlation': 0.0,
                'max_correlation': 0.0,
                'breakdown_risk': 0.0
            }
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr().values
        
        # Extract upper triangular correlations (excluding diagonal)
        n = corr_matrix.shape[0]
        upper_tri_indices = np.triu_indices(n, k=1)
        correlations = corr_matrix[upper_tri_indices]
        
        # Remove NaN values
        valid_correlations = correlations[~np.isnan(correlations)]
        
        if len(valid_correlations) == 0:
            return {
                'avg_correlation': 0.0,
                'max_correlation': 0.0,
                'breakdown_risk': 0.0
            }
        
        avg_correlation = np.mean(np.abs(valid_correlations))
        max_correlation = np.max(np.abs(valid_correlations))
        
        # Correlation breakdown risk (high correlations reduce diversification)
        breakdown_risk = np.mean(valid_correlations > 0.7)
        
        return {
            'avg_correlation': avg_correlation,
            'max_correlation': max_correlation,
            'breakdown_risk': breakdown_risk
        }
    
    def _calculate_advanced_metrics(self, 
                                  portfolio_returns: np.ndarray,
                                  market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate advanced risk metrics."""
        if len(portfolio_returns) < 30:
            return {
                'beta': 1.0,
                'tracking_error': 0.0,
                'information_ratio': 0.0,
                'calmar_ratio': 0.0,
                'sortino_ratio': 0.0
            }
        
        # Calculate beta and tracking error vs benchmark
        beta, tracking_error = self._calculate_beta_and_tracking_error(
            portfolio_returns, market_data
        )
        
        # Information ratio
        excess_returns = portfolio_returns - 0.02/252  # vs risk-free rate
        information_ratio = (np.mean(excess_returns) * 252 / 
                           (np.std(excess_returns) * np.sqrt(252))) if np.std(excess_returns) > 0 else 0.0
        
        # Calmar ratio (return / max drawdown)
        annual_return = np.mean(portfolio_returns) * 252
        drawdown_metrics = self._calculate_drawdown_metrics(portfolio_returns)
        max_drawdown = abs(drawdown_metrics['max_drawdown'])
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0.0
        
        # Sortino ratio (return / downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0.0
        sortino_ratio = (annual_return - 0.02) / downside_deviation if downside_deviation > 0 else 0.0
        
        return {
            'beta': beta,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio
        }
    
    def _calculate_beta_and_tracking_error(self, 
                                         portfolio_returns: np.ndarray,
                                         market_data: Dict[str, pd.DataFrame]) -> Tuple[float, float]:
        """Calculate beta and tracking error vs benchmark."""
        # Try to get benchmark returns
        benchmark_returns = None
        
        if self.benchmark_symbol in market_data:
            benchmark_data = market_data[self.benchmark_symbol]
            if 'Close' in benchmark_data.columns:
                benchmark_returns = benchmark_data['Close'].pct_change().dropna()
        
        if benchmark_returns is None or len(benchmark_returns) < len(portfolio_returns):
            return 1.0, 0.0  # Default values
        
        # Align returns
        min_length = min(len(portfolio_returns), len(benchmark_returns))
        port_ret = portfolio_returns[-min_length:]
        bench_ret = benchmark_returns.values[-min_length:]
        
        # Calculate beta
        covariance = np.cov(port_ret, bench_ret)[0, 1]
        benchmark_variance = np.var(bench_ret)
        
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
        
        # Calculate tracking error
        excess_returns = port_ret - bench_ret
        tracking_error = np.std(excess_returns) * np.sqrt(252)
        
        return beta, tracking_error
    
    def _assess_overall_risk_level(self, 
                                 drawdown_metrics: Dict[str, Any],
                                 concentration_metrics: Dict[str, float],
                                 correlation_metrics: Dict[str, float],
                                 volatility: float) -> Tuple[RiskLevel, float]:
        """Assess overall portfolio risk level."""
        risk_factors = []
        
        # Drawdown risk
        current_dd = abs(drawdown_metrics['current_drawdown'])
        max_dd = abs(drawdown_metrics['max_drawdown'])
        
        if current_dd > 0.20 or max_dd > 0.30:
            risk_factors.append(3)  # High risk
        elif current_dd > 0.10 or max_dd > 0.20:
            risk_factors.append(2)  # Moderate risk
        else:
            risk_factors.append(1)  # Low risk
        
        # Concentration risk
        concentration = concentration_metrics['concentration_ratio']
        
        if concentration > 0.50:
            risk_factors.append(3)  # High risk
        elif concentration > 0.30:
            risk_factors.append(2)  # Moderate risk
        else:
            risk_factors.append(1)  # Low risk
        
        # Volatility risk
        if volatility > 0.40:
            risk_factors.append(3)  # High risk
        elif volatility > 0.25:
            risk_factors.append(2)  # Moderate risk
        else:
            risk_factors.append(1)  # Low risk
        
        # Correlation risk
        avg_corr = correlation_metrics['avg_correlation']
        
        if avg_corr > 0.80:
            risk_factors.append(3)  # High risk
        elif avg_corr > 0.60:
            risk_factors.append(2)  # Moderate risk
        else:
            risk_factors.append(1)  # Low risk
        
        # Calculate overall risk score (1-4 scale)
        risk_score = np.mean(risk_factors)
        
        # Map to risk level
        if risk_score >= 2.75:
            risk_level = RiskLevel.CRITICAL
        elif risk_score >= 2.25:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 1.75:
            risk_level = RiskLevel.MODERATE
        else:
            risk_level = RiskLevel.LOW
        
        return risk_level, risk_score
    
    def _generate_risk_alerts(self, 
                            risk_metrics: RiskMetrics,
                            portfolio: PortfolioSnapshot) -> List[RiskAlert]:
        """Generate risk alerts based on threshold breaches."""
        alerts = []
        current_time = datetime.now()
        
        # Check for alert cooldown
        recent_alert_types = {
            alert.alert_type for alert in self.alerts_history
            if (current_time - alert.timestamp).total_seconds() < self.alert_cooldown_hours * 3600
        }
        
        # Drawdown alerts
        current_dd = abs(risk_metrics.current_drawdown)
        
        if (current_dd > self.risk_thresholds['max_drawdown_critical'] and
            AlertType.DRAWDOWN_CRITICAL not in recent_alert_types):
            alerts.append(RiskAlert(
                alert_type=AlertType.DRAWDOWN_CRITICAL,
                severity=RiskLevel.CRITICAL,
                message=f"Critical drawdown: {current_dd:.2%}",
                timestamp=current_time,
                current_value=current_dd,
                threshold=self.risk_thresholds['max_drawdown_critical'],
                recommendation="Consider reducing position sizes and implementing stop-losses"
            ))
        
        elif (current_dd > self.risk_thresholds['max_drawdown_warning'] and
              AlertType.DRAWDOWN_WARNING not in recent_alert_types):
            alerts.append(RiskAlert(
                alert_type=AlertType.DRAWDOWN_WARNING,
                severity=RiskLevel.HIGH,
                message=f"Drawdown warning: {current_dd:.2%}",
                timestamp=current_time,
                current_value=current_dd,
                threshold=self.risk_thresholds['max_drawdown_warning'],
                recommendation="Monitor positions closely and consider risk reduction"
            ))
        
        # Concentration alerts
        if (risk_metrics.concentration_ratio > self.risk_thresholds['concentration_critical'] and
            AlertType.CONCENTRATION_RISK not in recent_alert_types):
            alerts.append(RiskAlert(
                alert_type=AlertType.CONCENTRATION_RISK,
                severity=RiskLevel.CRITICAL,
                message=f"Critical concentration: {risk_metrics.concentration_ratio:.2%} in single position",
                timestamp=current_time,
                current_value=risk_metrics.concentration_ratio,
                threshold=self.risk_thresholds['concentration_critical'],
                recommendation="Diversify portfolio by reducing largest positions"
            ))
        
        # Volatility spike alerts
        if len(self.risk_metrics_history) > 20:
            avg_volatility = np.mean([m.volatility for m in self.risk_metrics_history[-20:]])
            if (risk_metrics.volatility > avg_volatility * self.risk_thresholds['volatility_spike'] and
                AlertType.VOLATILITY_SPIKE not in recent_alert_types):
                alerts.append(RiskAlert(
                    alert_type=AlertType.VOLATILITY_SPIKE,
                    severity=RiskLevel.HIGH,
                    message=f"Volatility spike: {risk_metrics.volatility:.2%} (avg: {avg_volatility:.2%})",
                    timestamp=current_time,
                    current_value=risk_metrics.volatility,
                    threshold=avg_volatility * self.risk_thresholds['volatility_spike'],
                    recommendation="Consider reducing leverage and position sizes"
                ))
        
        # Correlation breakdown alerts
        if (risk_metrics.max_correlation > self.risk_thresholds['correlation_breakdown'] and
            AlertType.CORRELATION_BREAKDOWN not in recent_alert_types):
            alerts.append(RiskAlert(
                alert_type=AlertType.CORRELATION_BREAKDOWN,
                severity=RiskLevel.HIGH,
                message=f"High correlation detected: {risk_metrics.max_correlation:.2%}",
                timestamp=current_time,
                current_value=risk_metrics.max_correlation,
                threshold=self.risk_thresholds['correlation_breakdown'],
                recommendation="Diversification may be compromised - review position correlations"
            ))
        
        # Leverage alerts
        if (portfolio.leverage > self.risk_thresholds['leverage_critical'] and
            AlertType.LEVERAGE_EXCESSIVE not in recent_alert_types):
            alerts.append(RiskAlert(
                alert_type=AlertType.LEVERAGE_EXCESSIVE,
                severity=RiskLevel.CRITICAL,
                message=f"Excessive leverage: {portfolio.leverage:.2f}x",
                timestamp=current_time,
                current_value=portfolio.leverage,
                threshold=self.risk_thresholds['leverage_critical'],
                recommendation="Reduce leverage immediately to manage risk"
            ))
        
        return alerts
    
    def _analyze_risk_trends(self) -> Dict[str, Any]:
        """Analyze risk trends over time."""
        if len(self.risk_metrics_history) < 10:
            return {'error': 'Insufficient data for trend analysis'}
        
        recent_metrics = self.risk_metrics_history[-30:]  # Last 30 observations
        
        # Calculate trends
        volatility_trend = np.polyfit(range(len(recent_metrics)), 
                                    [m.volatility for m in recent_metrics], 1)[0]
        
        drawdown_trend = np.polyfit(range(len(recent_metrics)), 
                                  [abs(m.current_drawdown) for m in recent_metrics], 1)[0]
        
        concentration_trend = np.polyfit(range(len(recent_metrics)), 
                                       [m.concentration_ratio for m in recent_metrics], 1)[0]
        
        return {
            'volatility_trend': 'increasing' if volatility_trend > 0.001 else 'decreasing' if volatility_trend < -0.001 else 'stable',
            'drawdown_trend': 'increasing' if drawdown_trend > 0.001 else 'decreasing' if drawdown_trend < -0.001 else 'stable',
            'concentration_trend': 'increasing' if concentration_trend > 0.001 else 'decreasing' if concentration_trend < -0.001 else 'stable',
            'overall_trend': 'improving' if (volatility_trend < 0 and drawdown_trend < 0) else 'deteriorating' if (volatility_trend > 0 or drawdown_trend > 0) else 'stable'
        }


def assess_portfolio_risk_realtime(portfolio_data: Dict[str, Any],
                                 market_data: Dict[str, pd.DataFrame],
                                 risk_thresholds: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Convenience function for real-time portfolio risk assessment.
    
    Args:
        portfolio_data: Portfolio positions and state
        market_data: Historical market data
        risk_thresholds: Custom risk thresholds
        
    Returns:
        Dictionary with risk assessment results
    """
    try:
        # Validate input data
        if not isinstance(portfolio_data, dict) or 'positions' not in portfolio_data:
            return {
                'error': 'Invalid portfolio data format',
                'risk_metrics': None,
                'dashboard': None
            }
        
        # Create portfolio snapshot
        positions = {}
        for symbol, pos_data in portfolio_data.get('positions', {}).items():
            positions[symbol] = Position(
                symbol=symbol,
                quantity=pos_data.get('quantity', 0),
                current_price=pos_data.get('current_price', 0),
                entry_price=pos_data.get('entry_price', 0),
                entry_date=datetime.fromisoformat(pos_data.get('entry_date', datetime.now().isoformat())),
                market_value=pos_data.get('market_value', 0),
                unrealized_pnl=pos_data.get('unrealized_pnl', 0),
                weight=pos_data.get('weight', 0)
            )
        
        portfolio_snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            positions=positions,
            total_value=portfolio_data.get('total_value', 0),
            cash=portfolio_data.get('cash', 0),
            total_equity=portfolio_data.get('total_equity', 0),
            daily_pnl=portfolio_data.get('daily_pnl', 0),
            total_pnl=portfolio_data.get('total_pnl', 0),
            leverage=portfolio_data.get('leverage', 1.0)
        )
        
        # Create risk assessment system
        risk_assessor = RealTimePortfolioRiskAssessment(risk_thresholds=risk_thresholds)
        
        # Assess risk
        risk_metrics = risk_assessor.assess_portfolio_risk(
            portfolio_snapshot, market_data
        )
        
        # Get dashboard data
        dashboard = risk_assessor.get_risk_dashboard()
        
        return {
            'risk_metrics': risk_metrics.__dict__,
            'dashboard': dashboard,
            'alerts': [alert.__dict__ for alert in risk_assessor.get_recent_alerts()]
        }
        
    except Exception as e:
        logger.error(f"Portfolio risk assessment failed: {e}")
        return {
            'error': f'Risk assessment failed: {str(e)}',
            'risk_metrics': None,
            'dashboard': None
        }