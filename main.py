
import os
from dotenv import load_dotenv

load_dotenv()


import logging
from typing import Optional, Dict, Any, Tuple

from langgraph.graph import StateGraph, START, END
from data.quality_validator import validate_data
import numpy as np
import datetime

import argparse

from config.config import GROQ_API_KEY, MODEL_NAME, DEFAULT_STOCKS, NIFTY_50_STOCKS, TOP_N_RECOMMENDATIONS
from data.models import State
from agents import (
    data_fetcher_agent,
    technical_analysis_agent,
    fundamental_analysis_agent,
    sentiment_analysis_agent,
    risk_assessment_agent,
    macro_analysis_agent
)
from recommendation import final_recommendation_agent
from config.config import WALK_FORWARD_ENABLED
from simulation import run_trading_simulation
from simulation.advanced_backtesting_engine import WalkForwardOptimizer
from analysis import PerformanceAnalyzer
from utils import setup_logging, graceful_shutdown


setup_logging()
logger = logging.getLogger(__name__)

llm: Optional[Any] = None
if GROQ_API_KEY and GROQ_API_KEY != "demo":
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=MODEL_NAME, api_key=GROQ_API_KEY)
        logger.info("LLM initialized successfully")
    except Exception as e:
        logger.warning(f"LLM initialization failed: {e}")
        logger.info("Proceeding with rule-based analysis only")
else:
    logger.info("LLM not configured, using rule-based recommendations")


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
        "backtest": False
    }


def validation_node(state: State) -> Dict[str, Any]:
    """Validate fetched data and determine if analysis should proceed."""
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
    
    logger.info(f"Data validation: valid={data_valid}, successful={len(stock_data)}/{len(symbols)}, errors={len(validation_errors)}")
    
    return {
        "data_valid": data_valid,
        "validation_errors": validation_errors
    }


def should_proceed_to_analyses(state: State) -> str:
    """Conditional router: proceed to analyses if data is valid or backtest enabled, else end."""
    if state.get("data_valid", False) or state.get("backtest", False):
        return "analyses_hub"
    logger.warning(f"Skipping analyses due to invalid data: {state.get('validation_errors', [])}")
    return END

def print_analysis_header() -> None:
    
    print(f"\n📊 Top {TOP_N_RECOMMENDATIONS} Trading Recommendations (ranked by confidence):")
    print("=" * 50)

def print_recommendations(final_state: State) -> None:
    
    recommendations = final_state.get("final_recommendation", {})
    failed_stocks = final_state.get("failed_stocks", [])

    if not recommendations:
        print("No recommendations generated.")
        if failed_stocks:
            print("\n⚠️ The following stocks could not be analyzed due to data unavailability:")
            for failed in failed_stocks:
                symbol = failed.get("symbol", "Unknown")
                error = failed.get("error", "Unknown error")
                print(f"  • {symbol}: {error}")
        return

    # Check if this is a multi-stock analysis with ranking
    buy_ranking = final_state.get("buy_ranking")
    ranking_reasoning = final_state.get("ranking_reasoning")
    all_recommendations = final_state.get("all_recommendations", {})

    if buy_ranking and len(buy_ranking) > 0:
        # Multi-stock analysis with BUY ranking
        print("\n" + "="*80)
        print("🎯 MULTI-STOCK ANALYSIS RESULTS")
        print("="*80)

        # Show ranking reasoning
        if ranking_reasoning:
            print(f"\n📊 Ranking Summary: {ranking_reasoning}")

        # Highlight top BUY candidate prominently
        top_candidate = final_state.get("top_buy_candidate")
        if top_candidate:
            print(f"\n🏆 TOP BUY CANDIDATE:")
            print("-" * 40)
            for symbol, rec in top_candidate.items():
                action = rec.get("action", "UNKNOWN")
                confidence = rec.get("confidence", 0)
                reasoning = rec.get("reasoning", "No reasoning provided")
                print(f"⭐ {symbol}: {action} (Confidence: {confidence:.1%})")
                print(f"   📝 {reasoning}")
                print()

        # Display complete BUY ranking table
        print("\n📈 BUY RANKING TABLE:")
        print("-" * 70)
        print(f"{'Rank':<5} {'Symbol':<12} {'Confidence':<12} {'Composite Score':<15}")
        print("-" * 70)

        rank = 1
        for item in buy_ranking:
            symbol = item.get("symbol", "UNKNOWN")
            confidence = item.get("confidence", 0)
            composite_score = item.get("composite_score", 0)
            marker = "👑" if rank == 1 else f"{rank}."
            print(f"{marker:<5} {symbol:<12} {confidence:<12.1%} {composite_score:<15.2f}")
            rank += 1

        # Include individual stock recommendations for reference
        if all_recommendations:
            print(f"\n📋 ALL INDIVIDUAL RECOMMENDATIONS ({len(all_recommendations)} stocks):")
            print("-" * 70)
            for symbol, rec in all_recommendations.items():
                action = rec.get("action", "UNKNOWN")
                confidence = rec.get("confidence", 0)
                reasoning = rec.get("reasoning", "No reasoning provided")
                if len(reasoning) > 60:
                    reasoning = reasoning[:57] + "..."
                print(f"{symbol:<12} {action:<6} {confidence:<12.1%} {reasoning}")

    else:
        # Single stock analysis - maintain existing format
        print_analysis_header()
        for symbol, rec in recommendations.items():
            action = rec.get("action", "UNKNOWN")
            reasoning = rec.get("reasoning", "No reasoning provided")
            confidence = rec.get("confidence", 0)
            print(f"{symbol}: {action} (Confidence: {confidence}%)")
            print(f"  Reasoning: {reasoning}")

def print_simulation_results(simulation_results: Dict[str, Any],
                           performance_analysis: Dict[str, Any]) -> None:
    
    if not simulation_results or "error" in simulation_results:
        error_msg = simulation_results.get("error", "Unknown error") if simulation_results else "No results"
        print(f"\n💰 Simulation failed: {error_msg}")
        return

    print("\n💰 Portfolio Simulation Results:")
    print("=" * 50)

    # Portfolio metrics with None checks
    final_value = float(simulation_results.get('final_portfolio_value', 0) or 0)
    total_return = float(simulation_results.get('total_return', 0) or 0)
    max_drawdown = float(simulation_results.get('max_drawdown', 0) or 0)

    print(f"Final Portfolio Value: ₹{final_value:,.0f}")
    print(f"Total Return: {total_return:.2%}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")

    # Performance rating
    rating = performance_analysis.get('performance_rating', 'Unknown') or 'Unknown'
    print(f"\n🏆 Performance Rating: {rating}")

    # Key insights
    insights = performance_analysis.get("insights", []) or []
    if insights:
        print("\n💡 Key Insights:")
        for insight in insights[:3]:  # Show top 3 insights
            print(f"• {insight}")


def simulation_node(state: State) -> Dict[str, Any]:
    """SimulationActor: Run trading simulation."""
    logger.info("SimulationActor started")
    try:
        rsi_threshold = state.get('rsi_threshold')
        if rsi_threshold is not None:
            simulation_results = run_trading_simulation(state, rsi_buy_threshold=rsi_threshold)
            mode = "basic RSI"
        else:
            simulation_results = run_trading_simulation(state)
            mode = "enhanced"
        logger.info(f"SimulationActor completed ({mode} mode)")
        return {"simulation_results": simulation_results}
    except Exception as e:
        logger.error(f"SimulationActor failed: {e}")
        return {"simulation_results": {"error": str(e)}}


def analyses_hub(state: State) -> State:
    """No-op hub to fan out to parallel analyses."""
    logger.info("Analyses hub: routing to parallel actors")
    return state


def log_technical_analysis(state: State) -> Dict[str, Any]:
    start_time = datetime.datetime.now()
    logger.info(f"Technical analysis started at {start_time}")
    result = {}
    try:
        result = technical_analysis_agent(state)
    except Exception as e:
        logger.error(f"Technical analysis failed: {e}")
    end_time = datetime.datetime.now()
    logger.info(f"Technical analysis completed at {end_time} (duration: {end_time - start_time})")
    return result


def log_fundamental_analysis(state: State) -> Dict[str, Any]:
    start_time = datetime.datetime.now()
    logger.info(f"Fundamental analysis started at {start_time}")
    result = {}
    try:
        result = fundamental_analysis_agent(state)
    except Exception as e:
        logger.error(f"Fundamental analysis failed: {e}")
    end_time = datetime.datetime.now()
    logger.info(f"Fundamental analysis completed at {end_time} (duration: {end_time - start_time})")
    return result


def log_sentiment_analysis(state: State) -> Dict[str, Any]:
    start_time = datetime.datetime.now()
    logger.info(f"Sentiment analysis started at {start_time}")
    result = {}
    try:
        result = sentiment_analysis_agent(state)
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
    end_time = datetime.datetime.now()
    logger.info(f"Sentiment analysis completed at {end_time} (duration: {end_time - start_time})")
    return result


def log_macro_analysis(state: State) -> Dict[str, Any]:
    start_time = datetime.datetime.now()
    logger.info(f"Macro analysis started at {start_time}")
    result = {}
    try:
        result = macro_analysis_agent(state)
    except Exception as e:
        logger.error(f"Macro analysis failed: {e}")
    end_time = datetime.datetime.now()
    logger.info(f"Macro analysis completed at {end_time} (duration: {end_time - start_time})")
    return result


def performance_node(state: State) -> Dict[str, Any]:
    
    simulation_results = state.get("simulation_results", {})
    if "error" not in simulation_results:
        try:
            analyzer = PerformanceAnalyzer()
            performance_analysis = analyzer.analyze_strategy_performance(simulation_results)
            logger.info("Performance analysis completed")
            return {"performance_analysis": performance_analysis}
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return {"performance_analysis": {"error": str(e)}}
    else:
        return {"performance_analysis": {"error": "No simulation results"}}


def should_run_walk_forward_and_simulate(state: State) -> str:
    """Conditional: run walk-forward then simulation if backtest enabled."""
    if state.get('backtest', False):
        return "walk_forward"
    return END


def walk_forward_node(state: State) -> Dict[str, Any]:
    """Run walk-forward optimization on stock data before backtest."""
    try:
        if 'stock_data' not in state or not state['stock_data']:
            logger.warning("No stock data for walk-forward optimization")
            return {"walk_forward_results": {}}
        
        if WALK_FORWARD_ENABLED:
            optimizer = WalkForwardOptimizer()
            results = optimizer.run_optimization(
                state['stock_data'],
                list(state['stock_data'].keys())
            )
            logger.info("Walk-forward optimization completed")
        else:
            # Fallback to basic validation if disabled
            from simulation.backtesting_engine import BacktestingEngine
            engine = BacktestingEngine(initial_capital=100000)
            results = engine.walk_forward_validation(state['stock_data'])
            logger.info("Basic walk-forward validation completed")
        
        return {"walk_forward_results": results}
    except Exception as e:
        logger.error(f"Walk-forward optimization failed: {e}")
        return {"walk_forward_results": {"error": str(e)}}


def print_walk_forward_results(final_state: State) -> None:
    """Print walk-forward optimization results, focusing on RELIANCE.NS and TATAMOTORS.NS."""
    results = final_state.get("walk_forward_results", {})
    if not results or "error" in results:
        if isinstance(results, dict) and "error" in results:
            print(f"\n🔄 Walk-Forward Optimization failed: {results['error']}")
        return
    
    print("\n🔄 Walk-Forward Optimization Results:")
    print("=" * 80)
    
    target_symbols = ["RELIANCE.NS", "TATAMOTORS.NS"]
    for symbol in target_symbols:
        if symbol in results:
            wf_res = results[symbol]
            oos = wf_res.get('aggregated_oos', {})
            if 'avg_sharpe' in oos:
                avg_sharpe = oos.get('avg_sharpe', 0.0) or 0.0
                std_sharpe = oos.get('std_sharpe', 0.0) or 0.0
                avg_win_rate = oos.get('avg_win_rate', 0.0) or 0.0
                avg_drawdown = oos.get('avg_drawdown', 0.0) or 0.0
                avg_returns = oos.get('avg_returns', 0.0) or 0.0
                total_oos_trades = oos.get('total_oos_trades', 0) or 0
                target_met = oos.get('oos_win_rate_target_met', False)
                print(f"\n{symbol} ({wf_res['num_periods']} periods):")
                print(f"  OOS Avg Sharpe: {avg_sharpe:.2f} (Std: {std_sharpe:.2f})")
                print(f"  OOS Avg Win Rate: {avg_win_rate:.1%}")
                print(f"  OOS Avg Max Drawdown: {avg_drawdown:.1%}")
                print(f"  OOS Avg Return: {avg_returns:.1%}")
                print(f"  Total OOS Trades: {total_oos_trades}")
                print(f"  OOS Win Rate Target (>50%): {'✅' if target_met else '❌'}")
                
                # Log sample period details
                periods = wf_res.get('periods', [])
                if periods:
                    print("  Sample Periods:")
                    for p in periods[:2]:  # First 2 periods
                        is_sharpe = p['is_metrics'].get('sharpe', 0.0) or 0.0
                        oos_sharpe = p['oos_metrics'].get('sharpe', 0.0) or 0.0
                        print(f"    Period {p['period_id']}: IS {p['is_dates']} | OOS {p['oos_dates']} | "
                              f"IS Sharpe={is_sharpe:.2f} | OOS Sharpe={oos_sharpe:.2f}")
            else:
                print(f"{symbol}: No OOS metrics available")
        else:
            print(f"{symbol}: No results available")
    
    # Overall summary
    all_symbols = [s for s in results if s in target_symbols and 'aggregated_oos' in results[s]]
    if all_symbols:
        avg_oos_sharpe = np.mean([results[s]['aggregated_oos']['avg_sharpe'] for s in all_symbols])
        avg_oos_win = np.mean([results[s]['aggregated_oos']['avg_win_rate'] for s in all_symbols])
        print(f"\nOverall OOS (across {len(all_symbols)} symbols):")
        print(f"  Avg Sharpe: {avg_oos_sharpe:.2f}, Avg Win Rate: {avg_oos_win:.1%}")


def should_simulate(state: State):
    
    logger.info(f"should_simulate: backtest={state.get('backtest', False)}, recommendations={state.get('final_recommendation', {})}")
    if state.get('backtest', False):
        return "simulation"
    recommendations = state.get("final_recommendation", {})
    for rec in recommendations.values():
        if isinstance(rec, dict) and rec.get("action") in ["BUY", "SELL"]:
            return "simulation"
    return END


def build_workflow_graph(stocks_to_analyze: Optional[list] = None, period: str = "5y") -> StateGraph:
    
    try:
        # Create graph with state schema
        graph = StateGraph(State)

        # Add analysis nodes with configurable stock symbols
        default_stocks = stocks_to_analyze or DEFAULT_STOCKS
        graph.add_node("data_fetcher",
                      lambda state: data_fetcher_agent(state, default_stocks, period=period))
        graph.add_node("validation", validation_node)
        graph.add_node("analyses_hub", analyses_hub)
        graph.add_node("technical_analysis", log_technical_analysis)
        graph.add_node("fundamental_analysis", log_fundamental_analysis)
        graph.add_node("sentiment_analysis", log_sentiment_analysis)
        graph.add_node("macro_analysis", log_macro_analysis)
        def log_risk_assessment(state: State) -> Dict[str, Any]:
            start_time = datetime.datetime.now()
            logger.info(f"Risk assessment started at {start_time}")
            result = {}
            try:
                result = risk_assessment_agent(state)
            except Exception as e:
                logger.error(f"Risk assessment failed: {e}")
            end_time = datetime.datetime.now()
            logger.info(f"Risk assessment completed at {end_time} (duration: {end_time - start_time})")
            return result

        graph.add_node("risk_assessment", log_risk_assessment)
        graph.add_node("final_recommendation", final_recommendation_agent)
        graph.add_node("simulation", simulation_node)
        graph.add_node("performance", performance_node)
        graph.add_node("walk_forward", walk_forward_node)

        # Define workflow: START -> data_fetcher -> validation -> conditional to analyses_hub -> parallel analyses -> risk_assessment -> final_recommendation
        graph.add_edge(START, "data_fetcher")
        graph.add_edge("data_fetcher", "validation")
        graph.add_conditional_edges(
            "validation",
            should_proceed_to_analyses,
            {"analyses_hub": "analyses_hub", END: END}
        )
        graph.add_edge("analyses_hub", "technical_analysis")
        graph.add_edge("analyses_hub", "fundamental_analysis")
        graph.add_edge("analyses_hub", "sentiment_analysis")
        graph.add_edge("analyses_hub", "macro_analysis")
        graph.add_edge("technical_analysis", "risk_assessment")
        graph.add_edge("fundamental_analysis", "risk_assessment")
        graph.add_edge("sentiment_analysis", "risk_assessment")
        graph.add_edge("macro_analysis", "risk_assessment")
        graph.add_edge("risk_assessment", "final_recommendation")
        
        # Conditional: walk-forward then simulation if backtest enabled
        graph.add_conditional_edges(
            "final_recommendation",
            should_run_walk_forward_and_simulate,
            {"walk_forward": "walk_forward", END: END}
        )
        graph.add_edge("walk_forward", "simulation")
        
        # Simulation flow
        graph.add_edge("simulation", "performance")
        graph.add_edge("performance", END)

        # Compile for execution
        compiled_graph = graph.compile()
        logger.info("Trading workflow graph compiled successfully")
        return compiled_graph

    except Exception as e:
        logger.error(f"Failed to build workflow graph: {e}")
        raise ValueError("Workflow graph compilation failed") from e


def run_analysis_and_simulation(stocks_to_analyze: Optional[list] = None, backtest: bool = False, basic: bool = False, period: str = "5y", walk_forward_enabled: bool = True) -> Tuple[Optional[State], Optional[Dict], Optional[Dict]]:
    
    try:
        logger.info("Starting trading analysis workflow")
    
        if not walk_forward_enabled:
            global WALK_FORWARD_ENABLED
            WALK_FORWARD_ENABLED = False
            logger.info("Walk-forward disabled via flag")
    
        # Build workflow
        graph = build_workflow_graph(stocks_to_analyze, period)

        # Initialize state
        initial_state = create_initial_state()

        if backtest:
            initial_state['backtest'] = True
            logger.info("Backtest flag set to True in initial_state")
        if basic:
            initial_state['rsi_threshold'] = 30.0
            logger.info("Basic RSI mode enabled with threshold 30")
        # Execute full workflow
        final_state = graph.invoke(initial_state)
        final_state["backtest"] = backtest
        logger.info(f"Final state backtest flag: {final_state.get('backtest', False)}")

        # Check if any stocks were successfully fetched
        stock_data = final_state.get("stock_data", {})
        failed_stocks = final_state.get("failed_stocks", [])
        if not stock_data:
            logger.warning(f"No stocks were successfully fetched. Failed stocks: {len(failed_stocks)}")
            if failed_stocks:
                for failed in failed_stocks:
                    logger.warning(f"Failed to fetch {failed['symbol']}: {failed['error']}")
            return None, None, None

        # If backtest requested but simulation not run, run it separately
        if backtest and (not final_state.get('simulation_results') or not final_state['simulation_results'] or 'error' in final_state.get('simulation_results', {})):
            logger.info("Running backtest simulation separately")
            sim_state = simulation_node(final_state)
            final_state.update(sim_state)
            perf_state = performance_node(final_state)
            final_state.update(perf_state)

        # Display walk-forward results if performed
        print_walk_forward_results(final_state)

        # Display recommendations
        print_recommendations(final_state)

        # Display simulation results if performed
        simulation_results = final_state.get("simulation_results")
        performance_analysis = final_state.get("performance_analysis")
        if simulation_results and "error" not in simulation_results:
            print_simulation_results(simulation_results, performance_analysis or {})
        elif simulation_results:
            print(f"\n💰 Simulation failed: {simulation_results.get('error', 'Unknown error')}")

        return final_state, simulation_results, performance_analysis

    except Exception as e:
        logger.error(f"Analysis workflow failed: {e}")
        raise RuntimeError("Trading analysis failed") from e


if __name__ == "__main__":
    
    # Setup graceful shutdown for clean termination
    graceful_shutdown()

    parser = argparse.ArgumentParser(description="Stock Trading Bot")
    parser.add_argument("--ticker", nargs="+", help="Stock symbols to analyze (single or multiple, e.g., RELIANCE.NS or RELIANCE.NS TCS.NS)")
    parser.add_argument("--tickers", help="Comma-separated stock symbols to analyze (e.g., RELIANCE.NS,TCS.NS,INFY.NS)")
    parser.add_argument("--period", default="5y", help="Period for data fetch")
    parser.add_argument("--walk_forward_enabled", action="store_true", default=True, help="Enable walk-forward optimization")
    parser.add_argument("--nifty50", action="store_true", help="Analyze NIFTY 50 stocks")
    parser.add_argument("--backtest", action="store_true", help="Run backtest simulation")
    parser.add_argument("--basic", action="store_true", help="Use basic RSI strategy for backtest (baseline)")
    args = parser.parse_args()

    try:
        # Determine stocks to analyze
        if args.tickers:
            # Parse comma-separated tickers
            stocks_to_analyze = [ticker.strip() for ticker in args.tickers.split(',') if ticker.strip()]
            logger.info(f"Using comma-separated tickers: {stocks_to_analyze}")
        else:
            stocks_to_analyze = args.ticker

        if args.nifty50 and not stocks_to_analyze:
            stocks_to_analyze = NIFTY_50_STOCKS
            logger.info(f"Using NIFTY 50 stocks ({len(NIFTY_50_STOCKS)} symbols)")

        # Execute main analysis
        analysis_state, simulation_results, performance_analysis = run_analysis_and_simulation(
            stocks_to_analyze, args.backtest, args.basic, args.period, args.walk_forward_enabled
        )
        logger.info("Trading analysis completed successfully")

    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
        print("\n⚠️ Execution interrupted. Analysis may be incomplete.")
    except Exception as e:
        logger.error(f"Unexpected application error: {e}")
        raise
    finally:
        logger.info("Application execution finished")