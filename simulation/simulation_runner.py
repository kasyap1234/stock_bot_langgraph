

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from data.models import State
from .backtesting_engine import BacktestingEngine

logger = logging.getLogger(__name__)

# Defaults for standalone execution
SIMULATION_DAYS = 252
TRADE_LIMIT = 100


from config.config import WALK_FORWARD_ENABLED
from data.apis import get_stock_history
from simulation.advanced_backtesting_engine import WalkForwardOptimizer
from simulation.backtesting_engine import BacktestingEngine
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def run_walk_forward_rsi_analysis(
    symbol: str = "RELIANCE.NS",
    total_period: str = "2y",
    train_ratio: float = 0.8,
    test_ratio: float = 0.2,
    step_days: int = 50,
    initial_capital: float = 1000000.0,
    commission_rate: float = 0.001
) -> Dict[str, Any]:
    """
    Run walk-forward analysis for RSI strategy on the given symbol.
    Uses expanding training windows with fixed test windows, stepping by step_days.
    Tunes RSI threshold on train data, tests on out-of-sample test data.
    Returns average out-of-sample metrics.
    """
    try:
        # Fetch historical data
        logger.info(f"Fetching {total_period} data for {symbol}")
        raw_data = get_stock_history(symbol, period=total_period, interval="1d")
        if not raw_data:
            return {"error": f"No data fetched for {symbol}"}

        # Convert to DataFrame
        df_list = []
        for record in raw_data:
            df_list.append({
                'Date': pd.to_datetime(record['date']),
                'Open': record['open'],
                'High': record['high'],
                'Low': record['low'],
                'Close': record['close'],
                'Volume': record['volume']
            })
        df = pd.DataFrame(df_list)
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        # Filter trading days (weekdays)
        df = df[df.index.weekday < 5]
        trading_days = len(df)
        logger.info(f"Loaded {trading_days} trading days for {symbol}")

        if trading_days < 200:  # Minimum for meaningful walk-forward
            return {"error": f"Insufficient data: {trading_days} days"}

        # Calculate window sizes (in trading days)
        min_train_days = int(100 * train_ratio)  # Min 80 days train
        test_days = int(50 * test_ratio)  # ~10 days test, but adjust to step
        if test_days < 20:
            test_days = 20  # Ensure reasonable test size

        engine = BacktestingEngine(
            initial_capital=initial_capital,
            commission_rate=commission_rate
        )

        oos_metrics = {
            'sharpe_ratios': [],
            'win_rates': [],
            'max_drawdowns': [],
            'total_returns': [],
            'num_windows': 0
        }

        # Walk-forward: expanding train, rolling test
        for start_idx in range(min_train_days, trading_days - test_days, step_days):
            train_end_idx = start_idx
            test_end_idx = min(start_idx + test_days, trading_days)

            train_df = df.iloc[:train_end_idx]
            test_df = df.iloc[start_idx:test_end_idx]

            if len(train_df) < min_train_days or len(test_df) < 10:
                continue

            # Tune RSI threshold on train data
            tuned_threshold = engine.tune_rsi_threshold(
                {symbol: train_df},
                num_windows=5,  # Fewer windows for tuning
                candidates=[25, 30, 35, 40, 45]
            )
            logger.info(f"Window {start_idx}: Tuned RSI threshold = {tuned_threshold:.1f}")

            # Run backtest on test data with tuned threshold
            stock_data_test = {symbol: test_df}
            test_start = test_df.index[0]
            test_end = test_df.index[-1]

            results = engine.run_backtest(
                None,  # No recommendations, use RSI
                stock_data_test,
                test_start,
                test_end,
                rsi_buy_threshold=tuned_threshold
            )

            if 'error' not in results and results.get('total_trades', 0) > 0:
                oos_metrics['sharpe_ratios'].append(results.get('sharpe_ratio', 0))
                oos_metrics['win_rates'].append(results.get('win_rate', 0))
                oos_metrics['max_drawdowns'].append(results.get('max_drawdown', 0))
                oos_metrics['total_returns'].append(results.get('total_return', 0))
                oos_metrics['num_windows'] += 1
                logger.info(f"Window {start_idx}-{test_end_idx}: Sharpe={results.get('sharpe_ratio') or 0:.2f}, "
                            f"Win Rate={results.get('win_rate') or 0:.1%}, MDD={results.get('max_drawdown') or 0:.1%}")

        if oos_metrics['num_windows'] == 0:
            return {"error": "No valid walk-forward windows completed"}

        # Calculate averages
        avg_metrics = {
            'avg_sharpe': np.mean(oos_metrics['sharpe_ratios']),
            'avg_win_rate': np.mean(oos_metrics['win_rates']),
            'avg_max_drawdown': np.mean(oos_metrics['max_drawdowns']),
            'avg_total_return': np.mean(oos_metrics['total_returns']),
            'num_windows': oos_metrics['num_windows'],
            'std_sharpe': np.std(oos_metrics['sharpe_ratios']),
            'reliance_data_points': trading_days
        }

        logger.info(f"Walk-forward analysis complete for {symbol}: "
                   f"Avg Sharpe={avg_metrics['avg_sharpe']:.2f}, "
                   f"Avg Win Rate={avg_metrics['avg_win_rate']:.1%}, "
                   f"Avg MDD={avg_metrics['avg_max_drawdown']:.1%}")

        return avg_metrics

    except Exception as e:
        logger.error(f"Walk-forward analysis failed for {symbol}: {e}")
        return {"error": str(e)}


def run_trading_simulation(
    state: State,
    initial_capital: float = 1000000.0,
    backtest_days: int = SIMULATION_DAYS,
    max_trades: int = TRADE_LIMIT,
    commission_rate: float = 0.001,
    rsi_buy_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """SimulationActor: Run trading simulation."""
    logger.info("SimulationActor started")
    
    try:
        stock_data = state.get("stock_data", {})
        final_recommendations = state.get("final_recommendation", {})

        if not stock_data:
            return {"error": "No stock data available for simulation"}

        if not final_recommendations:
            return {"error": "No recommendations available for simulation"}

        logger.info(f"SimulationActor processing {len(stock_data)} symbols")

        # Walk-forward optimization if enabled
        walk_forward_results = {}
        if WALK_FORWARD_ENABLED and stock_data:
            logger.info("Running walk-forward optimization...")
            optimizer = WalkForwardOptimizer()
            walk_forward_results = optimizer.run_optimization(
                stock_data,
                list(stock_data.keys()),
                initial_capital
            )
            # Log aggregated OOS metrics
            for symbol, wf_res in walk_forward_results.items():
                if 'aggregated_oos' in wf_res and 'avg_sharpe' in wf_res['aggregated_oos']:
                    oos = wf_res['aggregated_oos']
                    logger.info(f"SimulationActor: {symbol} Walk-Forward OOS: Avg Sharpe={oos['avg_sharpe']:.2f}, "
                               f"Avg Win Rate={oos['avg_win_rate']:.1%}, Avg Drawdown={oos['avg_drawdown']:.1%}, "
                               f"Avg Return={oos['avg_returns']:.1%}, Periods={wf_res['num_periods']}, "
                               f"Target Met={oos['oos_win_rate_target_met']}")
                else:
                    logger.warning(f"SimulationActor: {symbol} Walk-Forward failed: {wf_res.get('error', 'Unknown error')}")

        # Initialize backtesting engine
        engine = BacktestingEngine(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            max_position_size=0.1  # 10% max position
        )

        # Determine simulation period
        latest_date = None
        for df in stock_data.values():
            if hasattr(df, 'index') and not df.empty:
                df_date = df.index.max()
                if latest_date is None or df_date > latest_date:
                    latest_date = df_date

        if latest_date is None:
            return {"error": "No valid dates in stock data"}

        # Ensure start_date is within available data range
        min_date = min((df.index.min() for df in stock_data.values() if not df.empty), default=latest_date)
        start_date = latest_date - timedelta(days=backtest_days)
        start_date = max(start_date, min_date)
        end_date = latest_date

        # Format recommendations for backtesting engine
        formatted_recommendations = {}
        for symbol, rec in final_recommendations.items():
            if isinstance(rec, dict) and 'action' in rec:
                formatted_recommendations[symbol] = rec

        # Run multi-period backtests
        period_days = backtest_days // 4  # e.g., quarterly
        period_results = []
        for i in range(4):
            period_start = start_date + timedelta(days=i * period_days)
            period_end = start_date + timedelta(days=(i + 1) * period_days) if i < 3 else end_date
            period_stock_data = {s: df[(df.index >= period_start) & (df.index <= period_end)] for s, df in stock_data.items() if not df.empty}
            if period_stock_data:
                period_backtest = engine.run_backtest(
                    formatted_recommendations,
                    period_stock_data,
                    period_start,
                    period_end,
                    rsi_buy_threshold=rsi_buy_threshold
                )
                if 'error' not in period_backtest:
                    period_results.append(period_backtest)
                    logger.info(f"SimulationActor: Period {i+1} ({period_start.date()} to {period_end.date()}): Sharpe {period_backtest.get('sharpe_ratio', 0):.2f}, Win Rate {period_backtest.get('win_rate', 0):.2%}, Max Drawdown {period_backtest.get('max_drawdown', 0):.2%}")
        if period_results:
            avg_sharpe = sum(r.get('sharpe_ratio', 0) for r in period_results) / len(period_results)
            avg_win = sum(r.get('win_rate', 0) for r in period_results) / len(period_results)
            avg_drawdown = sum(r.get('max_drawdown', 0) for r in period_results) / len(period_results)
            backtest_results = {
                **period_results[-1],  # Use last period as base
                'tuned_rsi_threshold': rsi_buy_threshold,
                'averaged_sharpe_ratio': avg_sharpe,
                'averaged_win_rate': avg_win,
                'averaged_max_drawdown': avg_drawdown,
                'period_results': period_results
            }
            logger.info(f"SimulationActor: Averaged metrics: Sharpe {avg_sharpe:.2f}, Win Rate {avg_win:.2%}, Max Drawdown {avg_drawdown:.2%}")
        else:
            backtest_results = {"error": "No valid periods for backtest"}

        if "error" in backtest_results:
            return backtest_results

        # Add additional analysis
        analysis = _analyze_simulation_results(backtest_results, state)

        # Combine results
        results = {
            **backtest_results,
            "walk_forward_results": walk_forward_results,
            "simulation_analysis": analysis,
            "simulation_metadata": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "initial_capital": initial_capital,
                "final_capital": backtest_results.get("final_portfolio_value", initial_capital),
                "stocks_simulated": len(stock_data),
                "total_trading_days": backtest_days
            }
        }

        symbols = list(stock_data.keys())
        logger.info(f"SimulationActor completed for {len(symbols)} symbols: {symbols}")
        logger.info("SimulationActor completed")
        return results

    except Exception as e:
        logger.error(f"SimulationActor failed: {e}")
        return {"error": str(e)}


def _analyze_simulation_results(results: Dict[str, Any], state: State) -> Dict[str, Any]:
    
    try:
        analysis = {}

        # Risk-adjusted returns analysis
        total_return = results.get("total_return", 0)
        sharpe = results.get("sharpe_ratio", 0)
        max_drawdown = results.get("max_drawdown", 0)

        if total_return > 0:
            # Calmar ratio (return per unit of max drawdown)
            calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
            analysis["calmar_ratio"] = calmar_ratio

            # Risk reward assessment
            if sharpe > 1.5:
                risk_assessment = "Excellent risk-adjusted returns"
            elif sharpe > 1.0:
                risk_assessment = "Good risk-adjusted returns"
            elif sharpe > 0.5:
                risk_assessment = "Moderate risk-adjusted returns"
            else:
                risk_assessment = "Poor risk-adjusted returns"

            analysis["risk_assessment"] = risk_assessment
            analysis["performance_rating"] = _get_performance_rating(total_return, sharpe, max_drawdown)
        else:
            analysis["performance_rating"] = "Poor"

        # Trading efficiency analysis
        total_trades = results.get("total_trades", 0)
        win_rate = results.get("win_rate", 0)
        total_commission = results.get("total_commission", 0)
        final_value = results.get("final_portfolio_value", 0)

        if total_trades > 0:
            # Commission impact
            commission_impact = total_commission / final_value if final_value > 0 else 0
            analysis["commission_impact"] = commission_impact

            # Trading frequency assessment
            trading_days = results.get("simulation_metadata", {}).get("total_trading_days", 252)
            trades_per_month = (total_trades / trading_days) * 21

            if trades_per_month > 20:
                frequency = "Very high"
            elif trades_per_month > 10:
                frequency = "High"
            elif trades_per_month > 5:
                frequency = "Moderate"
            elif trades_per_month > 2:
                frequency = "Low"
            else:
                frequency = "Very low"

            analysis["trading_frequency"] = frequency
            analysis["trades_per_month"] = trades_per_month

        # Strategy effectiveness based on original recommendations
        final_recommendations = state.get("final_recommendation", {})
        analysis["recommendation_summary"] = _summarize_recommendations(final_recommendations)

        return analysis

    except Exception as e:
        logger.warning(f"Analysis failed: {e}")
        return {"error": str(e)}


def _get_performance_rating(total_return: float, sharpe: float, max_drawdown: float) -> str:
    
    rating_score = 0

    # Return component
    if total_return > 0.5:
        rating_score += 3
    elif total_return > 0.2:
        rating_score += 2
    elif total_return > 0:
        rating_score += 1

    # Sharpe ratio component
    if sharpe > 1.5:
        rating_score += 3
    elif sharpe > 1.0:
        rating_score += 2
    elif sharpe > 0.5:
        rating_score += 1

    # Drawdown component (lower drawdown is better)
    if max_drawdown < 0.1:
        rating_score += 3
    elif max_drawdown < 0.2:
        rating_score += 2
    elif max_drawdown < 0.3:
        rating_score += 1

    # Convert to rating
    if rating_score >= 7:
        return "Excellent"
    elif rating_score >= 5:
        return "Good"
    elif rating_score >= 3:
        return "Fair"
    else:
        return "Poor"


def _summarize_recommendations(recommendations: Dict[str, Dict]) -> Dict[str, Any]:
    
    summary = {
        "total_recommendations": len(recommendations),
        "buy_count": 0,
        "sell_count": 0,
        "hold_count": 0,
        "top_picks": [],
        "symbols_recommended": []
    }

    confidence_threshold = 0.6  # Only count high-confidence recommendations

    for symbol, rec in recommendations.items():
        if isinstance(rec, dict):
            action = rec.get('action', 'HOLD')
            confidence = rec.get('confidence', 0)

            summary["symbols_recommended"].append(symbol)

            if confidence >= confidence_threshold:
                if action.upper() == 'BUY':
                    summary["buy_count"] += 1
                elif action.upper() == 'SELL':
                    summary["sell_count"] += 1
                else:
                    summary["hold_count"] += 1

    # Identify top picks (based on confidence)
    high_confidence_picks = [
        (symbol, rec.get('confidence', 0))
        for symbol, rec in recommendations.items()
        if isinstance(rec, dict) and rec.get('confidence', 0) >= 0.7
    ]

    high_confidence_picks.sort(key=lambda x: x[1], reverse=True)
    summary["top_picks"] = high_confidence_picks[:3]  # Top 3

    return summary


def validate_simulation_state(state: State) -> bool:
    
    required_keys = ["stock_data", "final_recommendation"]

    # Check required keys exist
    for key in required_keys:
        if key not in state or not state[key]:
            logger.warning(f"Missing required data: {key}")
            return False

    # Check stock data quality
    stock_data = state.get("stock_data", {})
    if len(stock_data) == 0:
        logger.warning("No stock data available")
        return False

    # Check data completeness
    for symbol, df in stock_data.items():
        if df is None or df.empty:
            logger.warning(f"Incomplete data for {symbol}")
            return False
        if len(df) < 30:  # Need at least a month of data
            logger.warning(f"Insufficient historical data for {symbol}")
            return False

    # Check recommendations quality
    final_recommendations = state.get("final_recommendation", {})
    if len(final_recommendations) == 0:
        logger.warning("No recommendations available")
        return False

    logger.info("Simulation state validation passed")
    return True


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    results = run_walk_forward_rsi_analysis()
    print("Walk-Forward RSI Analysis Results for Reliance:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")