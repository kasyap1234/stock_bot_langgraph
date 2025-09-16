"""
Performance analysis utilities for trading strategies and portfolio metrics.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta

from utils.general_utils import safe_divide, calculate_returns, format_percentage, format_currency
from data.models import State

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Comprehensive performance analyzer for trading strategies.
    Calculates various risk and return metrics.
    """

    def __init__(self, risk_free_rate: float = 0.065, target_return: float = 0.0):
        """
        Initialize performance analyzer.

        Args:
            risk_free_rate: Annual risk-free rate (default 6.5% for India)
            target_return: Target annual return for comparison
        """
        self.risk_free_rate = risk_free_rate
        self.target_return = target_return

    def analyze_strategy_performance(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze comprehensive strategy performance metrics.

        Args:
            backtest_results: Results from backtest simulation

        Returns:
            Dictionary with detailed performance analysis
        """
        if "error" in backtest_results:
            return {"error": "Cannot analyze performance with invalid backtest data"}

        try:
            analysis = {}

            # Basic metrics
            analysis["basic_metrics"] = self._calculate_basic_metrics(backtest_results)

            # Risk metrics
            analysis["risk_metrics"] = self._calculate_risk_metrics(backtest_results)

            # Return metrics
            analysis["return_metrics"] = self._calculate_return_metrics(backtest_results)

            # Risk-adjusted metrics
            analysis["risk_adjusted_metrics"] = self._calculate_risk_adjusted_metrics(backtest_results)

            # Performance rating and insights
            analysis["performance_rating"] = self._generate_performance_rating(analysis)
            analysis["insights"] = self._generate_performance_insights(analysis)

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
            return {"error": str(e)}

    def _calculate_basic_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate basic performance metrics."""
        return {
            "total_return": results.get("total_return", 0),
            "annualized_return": results.get("annualized_return", 0),
            "volatility": results.get("volatility", 0),
            "max_drawdown": results.get("max_drawdown", 0),
            "sharpe_ratio": results.get("sharpe_ratio", 0),
            "win_rate": results.get("win_rate", 0),
            "total_trades": results.get("total_trades", 0),
            "final_portfolio_value": results.get("final_portfolio_value", 0),
            "total_commission": results.get("total_commission", 0)
        }

    def _calculate_risk_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics."""
        portfolio_history = results.get("portfolio_history", [])
        if not portfolio_history:
            return {"error": "No portfolio history available"}

        # Daily returns
        daily_returns = calculate_returns(portfolio_history)
        daily_returns = np.array(daily_returns)

        if len(daily_returns) == 0:
            return {"error": "Insufficient data for risk calculations"}

        # Value at Risk (95%, 99% confidence)
        var_95 = np.percentile(daily_returns, 5)
        var_99 = np.percentile(daily_returns, 1)

        # Expected Shortfall (Conditional VaR)
        filtered_95 = daily_returns[daily_returns <= var_95]
        es_95 = np.mean(filtered_95) if len(filtered_95) > 0 else 0
        filtered_99 = daily_returns[daily_returns <= var_99]
        es_99 = np.mean(filtered_99) if len(filtered_99) > 0 else 0

        # Downside deviation (only negative returns)
        downside_returns = [r for r in daily_returns if r < 0]
        downside_deviation = np.std(downside_returns) if downside_returns else 0

        # Sortino ratio (reward per unit of downside risk)
        excess_returns = [(r - self.risk_free_rate/252) for r in daily_returns]
        sortino = safe_divide(np.mean(excess_returns), downside_deviation) if downside_deviation > 0 else 0

        return {
            "value_at_risk_95": var_95,
            "value_at_risk_99": var_99,
            "expected_shortfall_95": es_95,
            "expected_shortfall_99": es_99,
            "downside_deviation": downside_deviation,
            "sortino_ratio": sortino,
            "largest_win": max(daily_returns) if len(daily_returns) > 0 else 0,
            "largest_loss": min(daily_returns) if len(daily_returns) > 0 else 0
        }

    def _calculate_return_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate return-based metrics."""
        portfolio_history = results.get("portfolio_history", [])
        if not portfolio_history:
            return {"error": "No portfolio history available"}

        # Overall statistics
        initial_value = portfolio_history[0] if portfolio_history else 0
        final_value = portfolio_history[-1] if portfolio_history else 0

        total_return = safe_divide(final_value - initial_value, initial_value)

        # Best and worst periods
        daily_returns = calculate_returns(portfolio_history)
        daily_returns = np.array(daily_returns)
        cumulative_returns = np.cumsum(daily_returns)

        best_day = max(daily_returns) if len(daily_returns) > 0 else 0
        worst_day = min(daily_returns) if len(daily_returns) > 0 else 0
        best_month_return = self._calculate_period_return(daily_returns, 21)
        best_quarter_return = self._calculate_period_return(daily_returns, 63)

        # Recovery metrics
        recovery_time = self._calculate_recovery_time(cumulative_returns)

        # Positive return days percentage
        positive_days_pct = safe_divide(sum(1 for r in daily_returns if r > 0), len(daily_returns)) * 100

        return {
            "total_return": total_return,
            "best_single_day": best_day,
            "worst_single_day": worst_day,
            "best_month_return": best_month_return,
            "worst_month_return": self._calculate_period_return(daily_returns, 21, worst=True),
            "recovery_time_days": recovery_time,
            "positive_days_percentage": positive_days_pct,
            "average_win": np.mean([r for r in daily_returns if r > 0]) if len(daily_returns) > 0 else 0,
            "average_loss": np.mean([r for r in daily_returns if r < 0]) if len(daily_returns) > 0 else 0
        }

    def _calculate_period_return(self, daily_returns: List[float], period_days: int, worst: bool = False) -> float:
        """Calculate best/worst return for a given period."""
        if len(daily_returns) < period_days:
            return 0

        period_returns = []
        for i in range(len(daily_returns) - period_days + 1):
            period_sum = sum(daily_returns[i:i + period_days])
            period_returns.append(period_sum)

        if worst:
            return min(period_returns) if period_returns else 0
        else:
            return max(period_returns) if period_returns else 0

    def _calculate_recovery_time(self, cumulative_returns: List[float]) -> int:
        """Calculate time needed to recover from drawdowns."""
        if not cumulative_returns.any():
            return 0

        peak = 0
        recovery_days = 0
        max_recovery_time = 0

        for i, value in enumerate(cumulative_returns):
            if value > peak:
                peak = value
            else:
                # Calculate recovery time from this point
                recovery_start = i
                for j in range(i, len(cumulative_returns)):
                    if cumulative_returns[j] >= peak:
                        current_recovery = j - recovery_start
                        max_recovery_time = max(max_recovery_time, current_recovery)
                        break

        return max_recovery_time

    def _calculate_risk_adjusted_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk-adjusted performance metrics."""
        total_return = results.get("total_return", 0)
        volatility = results.get("volatility", 0)
        max_drawdown = results.get("max_drawdown", 0)

        # Calmar ratio (annual return per unit of max drawdown)
        calmar = safe_divide(total_return, abs(max_drawdown)) if max_drawdown != 0 else 0

        # Omega ratio (ratio of gains to losses)
        portfolio_history = results.get("portfolio_history", [])
        daily_returns = calculate_returns(portfolio_history)
        daily_returns = np.array(daily_returns)

        if len(daily_returns) > 0:
            omega = self._calculate_omega_ratio(daily_returns, self.risk_free_rate/252)
        else:
            omega = 0

        # Information ratio (active return per unit of tracking error)
        # Using market return as benchmark (simplified)
        benchmark_return = 0.08  # Assuming 8% market return
        tracking_error = abs(total_return - benchmark_return)
        information_ratio = safe_divide(total_return - benchmark_return, tracking_error)

        return {
            "calmar_ratio": calmar,
            "omega_ratio": omega,
            "information_ratio": information_ratio,
            "excess_return": total_return - self.target_return,
            "tracking_error": tracking_error
        }

    def _calculate_omega_ratio(self, returns: List[float], threshold: float = 0) -> float:
        """Calculate Omega ratio."""
        if len(returns) == 0:
            return 0

        gains = sum(r for r in returns if r > threshold)
        losses = abs(sum(r for r in returns if r < threshold))

        return gains / losses if losses > 0 else 0

    def _generate_performance_rating(self, analysis: Dict[str, Any]) -> str:
        """Generate overall performance rating."""
        try:
            basic = analysis.get("basic_metrics", {})
            risk_adj = analysis.get("risk_adjusted_metrics", {})

            sharpe = basic.get("sharpe_ratio", 0)
            calmar = risk_adj.get("calmar_ratio", 0)
            win_rate = basic.get("win_rate", 0)
            max_dd = basic.get("max_drawdown", 0)

            score = 0

            # Sharpe ratio scoring
            if sharpe > 2: score += 4
            elif sharpe > 1.5: score += 3
            elif sharpe > 1: score += 2
            elif sharpe > 0.5: score += 1

            # Calmar ratio scoring
            if calmar > 1: score += 3
            elif calmar > 0.5: score += 2
            elif calmar > 0.25: score += 1

            # Win rate scoring
            if win_rate > 60: score += 3
            elif win_rate > 55: score += 2
            elif win_rate > 50: score += 1

            # Max drawdown penalty
            if max_dd < 0.1: score += 2
            elif max_dd < 0.15: score += 1

            # Rating mapping
            if score >= 10: return "Excellent"
            elif score >= 7: return "Very Good"
            elif score >= 5: return "Good"
            elif score >= 3: return "Fair"
            else: return "Poor"

        except Exception as e:
            logger.error(f"Error generating performance rating: {e}")
            return "Unable to Rate"

    def _generate_performance_insights(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable insights from performance analysis."""
        insights = []

        try:
            basic = analysis.get("basic_metrics", {})
            risk = analysis.get("risk_metrics", {})
            risk_adj = analysis.get("risk_adjusted_metrics", {})

            sharpe = basic.get("sharpe_ratio", 0)
            max_dd = basic.get("max_drawdown", 0)
            win_rate = basic.get("win_rate", 0)
            var_95 = risk.get("value_at_risk_95", 0)
            calmar = risk_adj.get("calmar_ratio", 0)

            # Sharpe ratio insights
            if sharpe < 1:
                insights.append("Strategy shows poor risk-adjusted returns. Consider reducing volatility or improving returns.")
            elif sharpe > 2:
                insights.append("Excellent risk-adjusted returns. Strategy efficiently converts risk into returns.")

            # Drawdown insights
            if max_dd > 0.2:
                insights.append(".20%")
            elif max_dd > 0.15:
                insights.append(".15%")

            # Win rate insights
            if win_rate < 50:
                insights.append("Win rate below 50%. Consider improving entry/exit strategy or risk management.")
            elif win_rate > 70:
                insights.append("Excellent win rate above 70%. Strategy shows consistent profitability.")

            # VaR insights
            if abs(var_95) > 0.05:
                insights.append("High daily value at risk. Consider position sizing or stop-loss rules.")

            # Calmar ratio insights
            if calmar > 0.5:
                insights.append("Strong Calmar ratio indicates good returns per unit of drawdown risk.")

            # Overall performance summary
            rating = analysis.get("performance_rating", "")
            if rating == "Excellent":
                insights.append("Strategy performance is excellent across multiple metrics.")
            elif rating == "Poor":
                insights.append("Strategy needs significant improvement. Consider fundamental changes.")

            if not insights:
                insights.append("Strategy shows balanced performance with room for optimization.")

        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            insights.append("Unable to generate detailed insights due to analysis errors.")

        return insights


def compare_strategies(strategy_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare multiple trading strategies.

    Args:
        strategy_results: List of backtest results for different strategies

    Returns:
        Comparison analysis
    """
    if len(strategy_results) < 2:
        return {"error": "Need at least 2 strategies for comparison"}

    try:
        comparison = {}

        # Extract key metrics for each strategy
        metrics_comparison = {}
        for i, results in enumerate(strategy_results):
            strategy_name = results.get("strategy_name", f"Strategy_{i+1}")
            metrics_comparison[strategy_name] = {
                "total_return": results.get("total_return", 0),
                "sharpe_ratio": results.get("sharpe_ratio", 0),
                "max_drawdown": results.get("max_drawdown", 0),
                "win_rate": results.get("win_rate", 0),
                "volatility": results.get("volatility", 0)
            }

        comparison["metrics_comparison"] = metrics_comparison

        # Find best and worst strategies by different metrics
        best_by_return = max(metrics_comparison.items(), key=lambda x: x[1]["total_return"])
        best_by_sharpe = max(metrics_comparison.items(), key=lambda x: x[1]["sharpe_ratio"])
        best_by_drawdown = min(metrics_comparison.items(), key=lambda x: x[1]["max_drawdown"])

        comparison["best_strategies"] = {
            "highest_return": best_by_return[0],
            "best_sharpe_ratio": best_by_sharpe[0],
            "lowest_drawdown": best_by_drawdown[0]
        }

        return comparison

    except Exception as e:
        logger.error(f"Error comparing strategies: {e}")
        return {"error": str(e)}