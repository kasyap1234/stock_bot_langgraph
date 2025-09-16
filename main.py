
import os
from dotenv import load_dotenv

load_dotenv()


import logging
from typing import Optional, Dict, Any, Tuple

from langgraph.graph import StateGraph, START, END

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
from recommendation import final_recommendation_agent, backtest_interpretation_agent
from simulation import run_trading_simulation
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
        "failed_stocks": []
    }

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

    # Portfolio metrics
    final_value = simulation_results.get('final_portfolio_value', 0)
    total_return = simulation_results.get('total_return', 0)
    max_drawdown = simulation_results.get('max_drawdown', 0)

    print(f"Final Portfolio Value: ₹{final_value:,.0f}")
    print(f"Total Return: {total_return:.2%}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")

    # Performance rating
    rating = performance_analysis.get('performance_rating', 'Unknown')
    print(f"\n🏆 Performance Rating: {rating}")

    # Key insights
    insights = performance_analysis.get("insights", [])
    if insights:
        print("\n💡 Key Insights:")
        for insight in insights[:3]:  # Show top 3 insights
            print(f"• {insight}")


def simulation_node(state: State) -> Dict[str, Any]:
    
    try:
        rsi_threshold = state.get('rsi_threshold')
        simulation_results = run_trading_simulation(state, rsi_buy_threshold=rsi_threshold)
        mode = "basic RSI" if rsi_threshold else "enhanced"
        logger.info(f"{mode.capitalize()} simulation completed")
        return {"simulation_results": simulation_results}
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        return {"simulation_results": {"error": str(e)}}


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


def should_simulate(state: State):
    
    logger.info(f"should_simulate: backtest={state.get('backtest', False)}, recommendations={state.get('final_recommendation', {})}")
    if state.get('backtest', False):
        return "simulation"
    recommendations = state.get("final_recommendation", {})
    for rec in recommendations.values():
        if isinstance(rec, dict) and rec.get("action") in ["BUY", "SELL"]:
            return "simulation"
    return END


def build_workflow_graph(stocks_to_analyze: Optional[list] = None) -> StateGraph:
    
    try:
        # Create graph with state schema
        graph = StateGraph(State)

        # Add analysis nodes with configurable stock symbols
        default_stocks = stocks_to_analyze or DEFAULT_STOCKS
        graph.add_node("data_fetcher",
                      lambda state: data_fetcher_agent(state, default_stocks))
        graph.add_node("technical_analysis", technical_analysis_agent)
        graph.add_node("fundamental_analysis", fundamental_analysis_agent)
        graph.add_node("sentiment_analysis", sentiment_analysis_agent)
        graph.add_node("macro_analysis", macro_analysis_agent)
        graph.add_node("risk_assessment", risk_assessment_agent)
        graph.add_node("final_recommendation", final_recommendation_agent)
        graph.add_node("simulation", simulation_node)
        graph.add_node("performance", performance_node)

        # Define workflow: START -> data_fetcher -> parallel analyses -> risk_assessment -> final_recommendation
        graph.add_edge(START, "data_fetcher")
        graph.add_edge("data_fetcher", "technical_analysis")
        graph.add_edge("data_fetcher", "fundamental_analysis")
        graph.add_edge("data_fetcher", "sentiment_analysis")
        graph.add_edge("data_fetcher", "macro_analysis")
        graph.add_edge("technical_analysis", "risk_assessment")
        graph.add_edge("fundamental_analysis", "risk_assessment")
        graph.add_edge("sentiment_analysis", "risk_assessment")
        graph.add_edge("macro_analysis", "risk_assessment")
        graph.add_edge("risk_assessment", "final_recommendation")
        
        # Conditional edge after final_recommendation
        graph.add_conditional_edges(
            "final_recommendation",
            should_simulate,
            {"simulation": "simulation", END: END}
        )
        
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


def run_analysis_and_simulation(stocks_to_analyze: Optional[list] = None, backtest: bool = False, basic: bool = False) -> Tuple[Optional[State], Optional[Dict], Optional[Dict]]:
    
    try:
        logger.info("Starting trading analysis workflow")

        # Build workflow
        graph = build_workflow_graph(stocks_to_analyze)

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

        if not final_state:
            logger.error("Analysis workflow returned empty state")
            return None, None, None

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
            stocks_to_analyze, args.backtest, args.basic
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