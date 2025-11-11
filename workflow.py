import datetime
import logging
from typing import Any, Dict, Optional

import numpy as np
from langgraph.graph import END, START, StateGraph

from agents import (
    advanced_ml_agent,
    data_fetcher_agent,
    fundamental_analysis_agent,
    macro_analysis_agent,
    market_regime_detection_agent,
    neural_network_agent,
    risk_assessment_agent,
    sentiment_analysis_agent,
    technical_analysis_agent,
)
from agents.feature_engineering import feature_engineering_agent
from analysis import PerformanceAnalyzer
from config.constants import DEFAULT_STOCKS
from config.trading_config import WALK_FORWARD_ENABLED
from data.models import State
from processing.parallel_analysis_engine import run_analyses_in_parallel
from recommendation import final_recommendation_agent
from simulation import run_trading_simulation
from simulation.advanced_backtesting_engine import WalkForwardOptimizer

logger = logging.getLogger(__name__)


def create_initial_state() -> State:
    return {
        "stock_data": {},
        "technical_signals": {},
        "fundamental_analysis": {},
        "sentiment_scores": {},
        "risk_metrics": {},
        "macro_scores": {},
        "final_recommendation": {},
        "simulation_results": {},
        "performance_analysis": {},
        "failed_stocks": [],
        "data_valid": False,
        "validation_errors": [],
        "backtest": False,
        "analysis_failure_count": 0,
    }


def validation_node(state: State) -> Dict[str, Any]:
    stock_data = state.get("stock_data", {})
    failed_stocks = state.get("failed_stocks", [])
    symbols = list(stock_data.keys()) + [f["symbol"] for f in failed_stocks]

    validation_errors = []
    data_valid = len(stock_data) > 0 and len(failed_stocks) < len(symbols)

    if not data_valid:
        if len(stock_data) == 0:
            validation_errors.append("No stock data fetched successfully")
        if len(failed_stocks) == len(symbols):
            validation_errors.append("All stocks failed to fetch data")

    logger.info(
        f"Data validation: valid={data_valid}, successful={len(stock_data)}/{len(symbols)}, errors={len(validation_errors)}"
    )

    return {"data_valid": data_valid, "validation_errors": validation_errors}


def should_proceed_to_analyses(state: State) -> str:
    if state.get("data_valid", False) or state.get("backtest", False):
        return "analyses_hub"
    logger.warning(
        f"Skipping analyses due to invalid data: {state.get('validation_errors', [])}"
    )
    return END


def analyses_hub(state: State) -> State:
    logger.info("Analyses hub: routing to parallel actors")
    analyses_to_run = {
        "technical_signals": log_technical_analysis,
        "fundamental_analysis": log_fundamental_analysis,
        "sentiment_scores": log_sentiment_analysis,
        "macro_scores": log_macro_analysis,
    }

    parallel_results = run_analyses_in_parallel(state, analyses_to_run)

    # Create a new state object with the combined results
    new_state = state.copy()
    new_state.update(parallel_results)

    return new_state


def log_technical_analysis(state: State) -> Dict[str, Any]:
    start_time = datetime.datetime.now()
    logger.info(f"Technical analysis started at {start_time}")
    try:
        result = technical_analysis_agent(state)
        end_time = datetime.datetime.now()
        logger.info(
            f"Technical analysis completed at {end_time} (duration: {end_time - start_time})"
        )
        return result
    except Exception as e:
        failure_count = state.get("analysis_failure_count", 0) + 1
        if failure_count > 3:
            raise RuntimeError(
                f"Circuit breaker activated: too many analysis failures ({failure_count})"
            )
        logger.exception(f"Technical analysis failed: {e}")
        end_time = datetime.datetime.now()
        logger.info(
            f"Technical analysis failed at {end_time} (duration: {end_time - start_time})"
        )
        return {
            "analysis_failure_count": failure_count,
            "technical_signals": {"error": str(e)},
        }


def log_fundamental_analysis(state: State) -> Dict[str, Any]:
    start_time = datetime.datetime.now()
    logger.info(f"Fundamental analysis started at {start_time}")
    try:
        result = fundamental_analysis_agent(state)
        end_time = datetime.datetime.now()
        logger.info(
            f"Fundamental analysis completed at {end_time} (duration: {end_time - start_time})"
        )
        return result
    except Exception as e:
        failure_count = state.get("analysis_failure_count", 0) + 1
        if failure_count > 3:
            raise RuntimeError(
                f"Circuit breaker activated: too many analysis failures ({failure_count})"
            )
        logger.exception(f"Fundamental analysis failed: {e}")
        end_time = datetime.datetime.now()
        logger.info(
            f"Fundamental analysis failed at {end_time} (duration: {end_time - start_time})"
        )
        return {
            "analysis_failure_count": failure_count,
            "fundamental_analysis": {"error": str(e)},
        }


def log_sentiment_analysis(state: State) -> Dict[str, Any]:
    start_time = datetime.datetime.now()
    logger.info(f"Sentiment analysis started at {start_time}")
    try:
        result = sentiment_analysis_agent(state)
        end_time = datetime.datetime.now()
        logger.info(
            f"Sentiment analysis completed at {end_time} (duration: {end_time - start_time})"
        )
        return result
    except Exception as e:
        failure_count = state.get("analysis_failure_count", 0) + 1
        if failure_count > 3:
            raise RuntimeError(
                f"Circuit breaker activated: too many analysis failures ({failure_count})"
            )
        logger.exception(f"Sentiment analysis failed: {e}")
        end_time = datetime.datetime.now()
        logger.info(
            f"Sentiment analysis failed at {end_time} (duration: {end_time - start_time})"
        )
        return {
            "analysis_failure_count": failure_count,
            "sentiment_scores": {"error": str(e)},
        }


def log_macro_analysis(state: State) -> Dict[str, Any]:
    start_time = datetime.datetime.now()
    logger.info(f"Macro analysis started at {start_time}")
    try:
        result = macro_analysis_agent(state)
        end_time = datetime.datetime.now()
        logger.info(
            f"Macro analysis completed at {end_time} (duration: {end_time - start_time})"
        )
        return result
    except Exception as e:
        failure_count = state.get("analysis_failure_count", 0) + 1
        if failure_count > 3:
            raise RuntimeError(
                f"Circuit breaker activated: too many analysis failures ({failure_count})"
            )
        logger.exception(f"Macro analysis failed: {e}")
        end_time = datetime.datetime.now()
        logger.info(
            f"Macro analysis failed at {end_time} (duration: {end_time - start_time})"
        )
        return {
            "analysis_failure_count": failure_count,
            "macro_scores": {"error": str(e)},
        }


def simulation_node(state: State) -> Dict[str, Any]:
    logger.info("SimulationActor started")
    try:
        rsi_threshold = state.get("rsi_threshold")
        if rsi_threshold is not None:
            simulation_results = run_trading_simulation(
                state, rsi_buy_threshold=rsi_threshold
            )
            mode = "basic RSI"
        else:
            simulation_results = run_trading_simulation(state)
            mode = "enhanced"
        logger.info(f"SimulationActor completed ({mode} mode)")
        return {"simulation_results": simulation_results}
    except Exception as e:
        logger.error(f"SimulationActor failed: {e}")
        return {"simulation_results": {"error": str(e)}}


def performance_node(state: State) -> Dict[str, Any]:
    simulation_results = state.get("simulation_results", {})
    if "error" not in simulation_results:
        try:
            analyzer = PerformanceAnalyzer()
            performance_analysis = analyzer.analyze_strategy_performance(
                simulation_results
            )
            logger.info("Performance analysis completed")
            return {"performance_analysis": performance_analysis}
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return {"performance_analysis": {"error": str(e)}}
    else:
        return {"performance_analysis": {"error": "No simulation results"}}


def should_run_walk_forward_and_simulate(state: State) -> str:
    if state.get("backtest", False):
        return "walk_forward"
    return END


def walk_forward_node(state: State) -> Dict[str, Any]:
    try:
        if "stock_data" not in state or not state["stock_data"]:
            logger.warning("No stock data for walk-forward optimization")
            return {"walk_forward_results": {}}

        if WALK_FORWARD_ENABLED:
            optimizer = WalkForwardOptimizer()
            results = optimizer.run_optimization(
                state["stock_data"], list(state["stock_data"].keys())
            )
            logger.info("Walk-forward optimization completed")
        else:
            from simulation.backtesting_engine import BacktestingEngine

            engine = BacktestingEngine(initial_capital=100000)
            results = engine.walk_forward_validation(state["stock_data"])
            logger.info("Basic walk-forward validation completed")

        return {"walk_forward_results": results}
    except Exception as e:
        logger.error(f"Walk-forward optimization failed: {e}")
        return {"walk_forward_results": {"error": str(e)}}


def should_simulate(state: State):
    logger.info(
        f"should_simulate: backtest={state.get('backtest', False)}, recommendations={state.get('final_recommendation', {})}"
    )
    if state.get("backtest", False):
        return "simulation"
    recommendations = state.get("final_recommendation", {})
    for rec in recommendations.values():
        if isinstance(rec, dict) and rec.get("action") in ["BUY", "SELL"]:
            return "simulation"
    return END


def build_workflow_graph(
    stocks_to_analyze: Optional[list] = None, period: str = "5y"
) -> StateGraph:
    try:
        graph = StateGraph(State)

        default_stocks = stocks_to_analyze or DEFAULT_STOCKS
        graph.add_node(
            "data_fetcher",
            lambda state: data_fetcher_agent(state, default_stocks, period=period),
        )
        graph.add_node("validation", validation_node)
        graph.add_node("analyses_hub", analyses_hub)
        graph.add_node("feature_engineering", feature_engineering_agent)
        graph.add_node("advanced_ml", advanced_ml_agent)
        graph.add_node("neural_network", neural_network_agent)

        def log_risk_assessment(state: State) -> Dict[str, Any]:
            start_time = datetime.datetime.now()
            logger.info(f"Risk assessment started at {start_time}")
            try:
                result = risk_assessment_agent(state)
                end_time = datetime.datetime.now()
                logger.info(
                    f"Risk assessment completed at {end_time} (duration: {end_time - start_time})"
                )
                return result
            except Exception as e:
                failure_count = state.get("analysis_failure_count", 0) + 1
                if failure_count > 3:
                    raise RuntimeError(
                        f"Circuit breaker activated: too many analysis failures ({failure_count})"
                    )
                logger.exception(f"Risk assessment failed: {e}")
                end_time = datetime.datetime.now()
                logger.info(
                    f"Risk assessment failed at {end_time} (duration: {end_time - start_time})"
                )
                return {
                    "analysis_failure_count": failure_count,
                    "risk_metrics": {"error": str(e)},
                }

        graph.add_node("risk_assessment", log_risk_assessment)
        graph.add_node("final_recommendation", final_recommendation_agent)
        graph.add_node("simulation", simulation_node)
        graph.add_node("performance", performance_node)
        graph.add_node("walk_forward", walk_forward_node)

        graph.add_edge(START, "data_fetcher")
        graph.add_edge("data_fetcher", "validation")
        graph.add_conditional_edges(
            "validation",
            should_proceed_to_analyses,
            {"analyses_hub": "analyses_hub", END: END},
        )
        graph.add_edge("analyses_hub", "feature_engineering")
        graph.add_edge("feature_engineering", "advanced_ml")
        graph.add_edge("advanced_ml", "neural_network")
        graph.add_edge("neural_network", "risk_assessment")
        graph.add_edge("risk_assessment", "final_recommendation")

        graph.add_conditional_edges(
            "final_recommendation",
            should_run_walk_forward_and_simulate,
            {"walk_forward": "walk_forward", END: END},
        )
        graph.add_edge("walk_forward", "simulation")

        graph.add_edge("simulation", "performance")
        graph.add_edge("performance", END)

        compiled_graph = graph.compile()
        logger.info("Trading workflow graph compiled successfully")
        return compiled_graph

    except Exception as e:
        logger.error(f"Failed to build workflow graph: {e}")
        raise ValueError("Workflow graph compilation failed") from e


def invoke_workflow(
    stocks_to_analyze: Optional[list] = None,
    backtest: bool = False,
    basic: bool = False,
    period: str = "5y",
    walk_forward_enabled: bool = True,
) -> Optional[State]:
    try:
        logger.info("Starting trading analysis workflow")

        if not walk_forward_enabled:
            global WALK_FORWARD_ENABLED
            WALK_FORWARD_ENABLED = False
            logger.info("Walk-forward disabled via flag")

        graph = build_workflow_graph(stocks_to_analyze, period)

        initial_state = create_initial_state()

        if backtest:
            initial_state["backtest"] = True
            logger.info("Backtest flag set to True in initial_state")
        if basic:
            initial_state["rsi_threshold"] = 30.0
            logger.info("Basic RSI mode enabled with threshold 30")
        final_state = graph.invoke(initial_state)
        final_state["backtest"] = backtest
        logger.info(f"Final state backtest flag: {final_state.get('backtest', False)}")

        stock_data = final_state.get("stock_data", {})
        failed_stocks = final_state.get("failed_stocks", [])
        if not stock_data:
            logger.warning(
                f"No stocks were successfully fetched. Failed stocks: {len(failed_stocks)}"
            )
            if failed_stocks:
                for failed in failed_stocks:
                    logger.warning(
                        f"Failed to fetch {failed['symbol']}: {failed['error']}"
                    )
            return None

        if backtest and (
            not final_state.get("simulation_results")
            or not final_state["simulation_results"]
            or "error" in final_state.get("simulation_results", {})
        ):
            logger.info("Running backtest simulation separately")
            sim_state = simulation_node(final_state)
            final_state.update(sim_state)
            perf_state = performance_node(final_state)
            final_state.update(perf_state)

        return final_state

    except Exception as e:
        logger.error(f"Analysis workflow failed: {e}")
        raise RuntimeError("Trading analysis failed") from e
