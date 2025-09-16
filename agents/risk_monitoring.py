"""
Risk Monitoring and Reporting Module
Provides real-time risk monitoring, alerts, historical analysis, and stress testing capabilities.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd

from config.config import RISK_TOLERANCE
from data.models import State
from .risk_management import RiskManager, RiskLevel
from .market_risk_assessment import MarketRiskAssessor, MarketRiskMetrics

# Configure logging
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class RiskAlert:
    """Risk alert data structure."""
    timestamp: datetime
    severity: AlertSeverity
    message: str
    symbol: Optional[str] = None
    metric: Optional[str] = None
    threshold: Optional[float] = None
    actual_value: Optional[float] = None
    recommendation: Optional[str] = None


@dataclass
class RiskReport:
    """Comprehensive risk report."""
    timestamp: datetime
    portfolio_value: float
    total_risk: float
    diversification_ratio: float
    max_drawdown: float
    var_95: float
    expected_shortfall: float
    stress_test_results: Dict[str, float] = field(default_factory=dict)
    alerts: List[RiskAlert] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class RiskMonitor:
    """
    Real-time risk monitoring system with alerting and reporting capabilities.
    """

    def __init__(self, risk_manager: RiskManager, market_assessor: MarketRiskAssessor):
        self.risk_manager = risk_manager
        self.market_assessor = market_assessor
        self.alerts: List[RiskAlert] = []
        self.risk_history: List[RiskReport] = []
        self.alert_thresholds = self._set_default_thresholds()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _set_default_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Set default risk alert thresholds."""
        return {
            'portfolio_drawdown': {'warning': 0.05, 'critical': 0.10, 'emergency': 0.15},
            'daily_loss': {'warning': 0.03, 'critical': 0.05, 'emergency': 0.08},
            'position_concentration': {'warning': 0.15, 'critical': 0.25, 'emergency': 0.35},
            'volatility': {'warning': 0.25, 'critical': 0.35, 'emergency': 0.50},
            'correlation': {'warning': 0.70, 'critical': 0.85, 'emergency': 0.95},
            'liquidity_risk': {'warning': 0.30, 'critical': 0.50, 'emergency': 0.70}
        }

    def monitor_risk(self, current_prices: Dict[str, float], market_data: Dict[str, pd.DataFrame]) -> List[RiskAlert]:
        """
        Monitor risk in real-time and generate alerts.

        Args:
            current_prices: Current market prices
            market_data: Historical market data

        Returns:
            List of new risk alerts
        """
        new_alerts = []

        try:
            # Update positions
            position_updates = self.risk_manager.update_positions(current_prices)

            # Assess market risk
            market_risk = self.market_assessor.assess_market_risk(market_data)

            # Check portfolio-level alerts
            portfolio_alerts = self._check_portfolio_alerts()
            new_alerts.extend(portfolio_alerts)

            # Check market risk alerts
            market_alerts = self._check_market_risk_alerts(market_risk)
            new_alerts.extend(market_alerts)

            # Check position-level alerts
            position_alerts = self._check_position_alerts(current_prices)
            new_alerts.extend(position_alerts)

            # Add new alerts to history
            self.alerts.extend(new_alerts)

            # Log alerts
            for alert in new_alerts:
                self.logger.log(
                    self._get_log_level(alert.severity),
                    f"Risk Alert [{alert.severity.value}]: {alert.message}"
                )

            return new_alerts

        except Exception as e:
            self.logger.error(f"Error in risk monitoring: {e}")
            return []

    def generate_risk_report(self, market_data: Dict[str, pd.DataFrame]) -> RiskReport:
        """
        Generate comprehensive risk report.

        Args:
            market_data: Historical market data

        Returns:
            RiskReport object
        """
        try:
            # Get current risk metrics
            risk_metrics = self.risk_manager.get_risk_metrics()
            market_risk = self.market_assessor.assess_market_risk(market_data)

            # Calculate advanced risk metrics
            var_95, expected_shortfall = self._calculate_var_metrics(market_data)
            diversification_ratio = self._calculate_diversification_ratio(market_data)

            # Run stress tests
            stress_results = self._run_stress_tests(market_data)

            # Generate recommendations
            recommendations = self._generate_recommendations(risk_metrics, market_risk)

            # Create report
            report = RiskReport(
                timestamp=datetime.now(),
                portfolio_value=risk_metrics['portfolio_value'],
                total_risk=self._calculate_total_risk(market_data),
                diversification_ratio=diversification_ratio,
                max_drawdown=risk_metrics['max_drawdown'],
                var_95=var_95,
                expected_shortfall=expected_shortfall,
                stress_test_results=stress_results,
                alerts=self.alerts[-10:],  # Last 10 alerts
                recommendations=recommendations
            )

            # Add to history
            self.risk_history.append(report)

            return report

        except Exception as e:
            self.logger.error(f"Error generating risk report: {e}")
            return RiskReport(
                timestamp=datetime.now(),
                portfolio_value=0.0,
                total_risk=0.0,
                diversification_ratio=0.0,
                max_drawdown=0.0,
                var_95=0.0,
                expected_shortfall=0.0
            )

    def _check_portfolio_alerts(self) -> List[RiskAlert]:
        """Check portfolio-level risk alerts."""
        alerts = []
        risk_metrics = self.risk_manager.get_risk_metrics()

        # Check drawdown
        drawdown = risk_metrics['max_drawdown']
        thresholds = self.alert_thresholds['portfolio_drawdown']
        if drawdown > thresholds['emergency']:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                severity=AlertSeverity.EMERGENCY,
                message=f"Portfolio drawdown exceeded emergency threshold: {drawdown:.1%}",
                metric="drawdown",
                threshold=thresholds['emergency'],
                actual_value=drawdown,
                recommendation="Consider portfolio rebalancing or risk reduction"
            ))
        elif drawdown > thresholds['critical']:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                severity=AlertSeverity.CRITICAL,
                message=f"Portfolio drawdown exceeded critical threshold: {drawdown:.1%}",
                metric="drawdown",
                threshold=thresholds['critical'],
                actual_value=drawdown,
                recommendation="Monitor closely and consider position adjustments"
            ))

        # Check daily loss
        daily_loss = abs(risk_metrics['daily_pnl'])
        thresholds = self.alert_thresholds['daily_loss']
        if daily_loss > thresholds['emergency']:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                severity=AlertSeverity.EMERGENCY,
                message=f"Daily loss exceeded emergency threshold: {daily_loss:.1%}",
                metric="daily_loss",
                threshold=thresholds['emergency'],
                actual_value=daily_loss,
                recommendation="Immediate portfolio risk reduction required"
            ))

        return alerts

    def _check_market_risk_alerts(self, market_risk: MarketRiskMetrics) -> List[RiskAlert]:
        """Check market risk alerts."""
        alerts = []

        # Check volatility
        volatility = market_risk.market_volatility
        thresholds = self.alert_thresholds['volatility']
        if volatility > thresholds['emergency']:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                severity=AlertSeverity.EMERGENCY,
                message=f"Market volatility exceeded emergency threshold: {volatility:.1%}",
                metric="volatility",
                threshold=thresholds['emergency'],
                actual_value=volatility,
                recommendation="Reduce position sizes and implement wider stops"
            ))

        # Check correlation risk
        correlation = market_risk.correlation_risk
        thresholds = self.alert_thresholds['correlation']
        if correlation > thresholds['critical']:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                severity=AlertSeverity.CRITICAL,
                message=f"Asset correlation exceeded critical threshold: {correlation:.1%}",
                metric="correlation",
                threshold=thresholds['critical'],
                actual_value=correlation,
                recommendation="Diversify portfolio across uncorrelated assets"
            ))

        return alerts

    def _check_position_alerts(self, current_prices: Dict[str, float]) -> List[RiskAlert]:
        """Check position-level risk alerts."""
        alerts = []
        risk_metrics = self.risk_manager.get_risk_metrics()

        # Check position concentrations
        concentrations = risk_metrics['concentration_risk']
        thresholds = self.alert_thresholds['position_concentration']

        for symbol, concentration in concentrations.items():
            if concentration > thresholds['emergency']:
                alerts.append(RiskAlert(
                    timestamp=datetime.now(),
                    severity=AlertSeverity.EMERGENCY,
                    message=f"Position concentration exceeded emergency threshold for {symbol}: {concentration:.1%}",
                    symbol=symbol,
                    metric="concentration",
                    threshold=thresholds['emergency'],
                    actual_value=concentration,
                    recommendation="Reduce position size immediately"
                ))

        return alerts

    def _calculate_var_metrics(self, market_data: Dict[str, pd.DataFrame]) -> Tuple[float, float]:
        """Calculate Value at Risk and Expected Shortfall."""
        try:
            if not market_data:
                return 0.0, 0.0

            # Combine all asset returns for portfolio VaR
            returns_list = []
            for df in market_data.values():
                if len(df) >= 30:
                    returns = df['Close'].pct_change().dropna()
                    returns_list.append(returns)

            if not returns_list:
                return 0.0, 0.0

            # Simple portfolio returns (equal weighted for simplicity)
            min_length = min(len(r) for r in returns_list)
            portfolio_returns = np.mean([r.tail(min_length).values for r in returns_list], axis=0)

            # Calculate VaR (95% confidence)
            var_95 = np.percentile(portfolio_returns, 5)

            # Calculate Expected Shortfall (CVaR)
            tail_losses = portfolio_returns[portfolio_returns <= var_95]
            expected_shortfall = np.mean(tail_losses) if len(tail_losses) > 0 else var_95

            return abs(var_95), abs(expected_shortfall)

        except Exception as e:
            self.logger.error(f"Error calculating VaR metrics: {e}")
            return 0.0, 0.0

    def _calculate_diversification_ratio(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate portfolio diversification ratio."""
        try:
            if len(market_data) < 2:
                return 1.0

            # Calculate individual volatilities
            volatilities = []
            for df in market_data.values():
                if len(df) >= 30:
                    returns = df['Close'].pct_change().dropna()
                    vol = returns.std()
                    volatilities.append(vol)

            if not volatilities:
                return 1.0

            # Calculate portfolio volatility (simplified)
            portfolio_vol = np.sqrt(np.mean(np.array(volatilities) ** 2))

            # Calculate weighted average volatility
            avg_vol = np.mean(volatilities)

            # Diversification ratio
            return avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0

        except Exception as e:
            self.logger.error(f"Error calculating diversification ratio: {e}")
            return 1.0

    def _calculate_total_risk(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate total portfolio risk."""
        try:
            var_95, _ = self._calculate_var_metrics(market_data)
            risk_metrics = self.risk_manager.get_risk_metrics()

            # Combine multiple risk measures
            total_risk = (
                0.4 * var_95 +  # VaR contribution
                0.3 * risk_metrics['max_drawdown'] +  # Drawdown contribution
                0.3 * risk_metrics['total_exposure']  # Exposure contribution
            )

            return total_risk

        except Exception as e:
            self.logger.error(f"Error calculating total risk: {e}")
            return 0.0

    def _run_stress_tests(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Run portfolio stress tests."""
        stress_scenarios = {
            'market_crash': -0.20,  # 20% market drop
            'high_volatility': 0.15,  # 15% volatility increase
            'liquidity_crisis': -0.10,  # 10% liquidity shock
            'sector_crisis': -0.25  # 25% sector-specific drop
        }

        results = {}
        try:
            for scenario, shock in stress_scenarios.items():
                # Simplified stress test - in practice would be more sophisticated
                portfolio_impact = shock * 0.8  # Assume 80% correlation
                results[scenario] = portfolio_impact

        except Exception as e:
            self.logger.error(f"Error running stress tests: {e}")

        return results

    def _generate_recommendations(self, risk_metrics: Dict, market_risk: MarketRiskMetrics) -> List[str]:
        """Generate risk management recommendations."""
        recommendations = []

        try:
            # Portfolio concentration recommendations
            concentrations = risk_metrics.get('concentration_risk', {})
            high_concentration = [s for s, c in concentrations.items() if c > 0.20]
            if high_concentration:
                recommendations.append(f"Reduce concentration in: {', '.join(high_concentration)}")

            # Market regime recommendations
            if market_risk.regime == market_risk.regime.BEAR:
                recommendations.append("Consider defensive positioning in bear market")
            elif market_risk.regime == market_risk.regime.HIGH_VOLATILITY:
                recommendations.append("Implement wider stop-losses and reduce position sizes")

            # Risk limit recommendations
            if risk_metrics.get('max_drawdown', 0) > 0.10:
                recommendations.append("Portfolio drawdown exceeds 10% - consider rebalancing")

            # Default recommendations
            if not recommendations:
                recommendations.append("Maintain current risk management practices")

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            recommendations = ["Monitor portfolio risk metrics closely"]

        return recommendations

    def _get_log_level(self, severity: AlertSeverity) -> int:
        """Convert alert severity to logging level."""
        mapping = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.CRITICAL: logging.ERROR,
            AlertSeverity.EMERGENCY: logging.CRITICAL
        }
        return mapping.get(severity, logging.WARNING)

    def get_alert_summary(self, hours: int = 24) -> Dict[str, int]:
        """Get summary of alerts in the last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_alerts = [a for a in self.alerts if a.timestamp > cutoff]

        summary = {
            'total': len(recent_alerts),
            'info': len([a for a in recent_alerts if a.severity == AlertSeverity.INFO]),
            'warning': len([a for a in recent_alerts if a.severity == AlertSeverity.WARNING]),
            'critical': len([a for a in recent_alerts if a.severity == AlertSeverity.CRITICAL]),
            'emergency': len([a for a in recent_alerts if a.severity == AlertSeverity.EMERGENCY])
        }

        return summary