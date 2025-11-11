import os
from dotenv import load_dotenv

load_dotenv()

import logging
import json
import argparse

from config.api_config import GROQ_API_KEY, MODEL_NAME, GROQ_TOOL_CHOICE, MAX_TOOL_CALLS, DISCLAIMER_TEXT
from config.constants import DEFAULT_STOCKS
from config.trading_config import NIFTY_50_STOCKS
from tools import (
    fetch_stock_price,
    compute_technical_indicators,
    get_sentiment,
    validate_indian_stock,
    get_tool_schemas,
    web_search,
    stock_search
)
from workflow import invoke_workflow
from output import render_recommendations, render_simulation, render_walk_forward
from utils import setup_logging, graceful_shutdown
from utils.config_validator import validate_and_warn

setup_logging()
logger = logging.getLogger(__name__)

llm = None
groq_client = None
tool_functions = {
    "fetch_stock_price": fetch_stock_price,
    "compute_technical_indicators": compute_technical_indicators,
    "get_sentiment": get_sentiment,
    "validate_indian_stock": validate_indian_stock,
    "web_search": web_search,
    "stock_search": stock_search,
}
if GROQ_API_KEY and GROQ_API_KEY != "demo":
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq client initialized successfully")
    except Exception as e:
        logger.warning(f"Groq client initialization failed: {e}")
        logger.info("Proceeding with rule-based analysis only")
else:
    logger.info("Groq not configured, using rule-based recommendations")

def run_cli():
    graceful_shutdown()

    parser = argparse.ArgumentParser(description="StockBot: Indian Stock Analysis")
    parser.add_argument("--ticker", nargs="+", help="Stock symbols to analyze (single or multiple, e.g., RELIANCE.NS or RELIANCE.NS TCS.NS)")
    parser.add_argument("--tickers", help="Comma-separated stock symbols to analyze (e.g., RELIANCE.NS,TCS.NS,INFY.NS)")
    parser.add_argument("--period", default="5y", help="Period for data fetch")
    parser.add_argument("--walk_forward_enabled", action="store_true", default=True, help="Enable walk-forward optimization")
    parser.add_argument("--nifty50", action="store_true", help="Analyze NIFTY 50 stocks")
    parser.add_argument("--backtest", action="store_true", help="Run backtest simulation")
    parser.add_argument("--basic", action="store_true", help="Use basic RSI strategy for backtest (baseline)")
    parser.add_argument("--interactive", action="store_true", help="Interactive Groq tool mode (default if no other flags)")
    parser.add_argument("--skip-validation", action="store_true", help="Skip configuration validation at startup")
    parser.epilog = "Run without flags for interactive mode, or use --ticker for batch."
    args = parser.parse_args()
    
    # Validate configuration unless skipped
    if not args.skip_validation:
        logger.info("Validating configuration...")
        validate_and_warn()
    else:
        logger.warning("Skipping configuration validation (--skip-validation flag set)")

    if not args.interactive and not (args.ticker or args.tickers or args.nifty50 or args.backtest or args.basic):
        args.interactive = True

    try:
        if args.interactive:
            print("StockBot Interactive Mode. Type 'exit' to quit.")
            if not groq_client:
                print("Groq client not configured for interactive mode.")
            else:
                while True:
                    query = input("You: ")
                    if query.lower() == 'exit':
                        break
                    try:
                        messages = [{"role": "user", "content": query}]
                        tool_call_count = 0
                        while True:
                            response = groq_client.chat.completions.create(
                                model=MODEL_NAME,
                                messages=messages,
                                tools=get_tool_schemas(),
                                tool_choice="auto",
                                max_tokens=1024
                            )
                            message = response.choices[0].message
                            if message.tool_calls:
                                tool_calls = message.tool_calls
                                for tool_call in tool_calls:
                                    func_name = tool_call.function.name
                                    try:
                                        args_dict = json.loads(tool_call.function.arguments)
                                        result = tool_functions[func_name](**args_dict)
                                        messages.append({
                                            "role": "tool",
                                            "tool_call_id": tool_call.id,
                                            "name": func_name,
                                            "content": json.dumps(result)
                                        })
                                    except json.JSONDecodeError as e:
                                        logger.error(f"Invalid tool arguments JSON: {e}")
                                        messages.append({
                                            "role": "tool",
                                            "tool_call_id": tool_call.id,
                                            "name": func_name,
                                            "content": json.dumps({"error": "Invalid arguments"})
                                        })
                                    except Exception as e:
                                        logger.error(f"Tool execution error: {e}")
                                        messages.append({
                                            "role": "tool",
                                            "tool_call_id": tool_call.id,
                                            "name": func_name,
                                            "content": json.dumps({"error": str(e)})
                                        })
                                tool_call_count += 1
                                if tool_call_count >= MAX_TOOL_CALLS:
                                    break
                            else:
                                final_response = message.content
                                print("StockBot:", final_response)
                                print(DISCLAIMER_TEXT)
                                break
                    except Exception as e:
                        print(f"Error: {e}")
        else:
            if args.tickers:
                stocks_to_analyze = [ticker.strip() for ticker in args.tickers.split(',') if ticker.strip()]
                logger.info(f"Using comma-separated tickers: {stocks_to_analyze}")
            else:
                stocks_to_analyze = args.ticker

            if args.nifty50 and not stocks_to_analyze:
                stocks_to_analyze = NIFTY_50_STOCKS
                logger.info(f"Using NIFTY 50 stocks ({len(NIFTY_50_STOCKS)} symbols)")

            analysis_state = invoke_workflow(
                stocks_to_analyze, args.backtest, args.basic, args.period, args.walk_forward_enabled
            )
            logger.info("Trading analysis completed successfully")

            if analysis_state:
                render_walk_forward(analysis_state)
                render_recommendations(analysis_state)
                simulation_results = analysis_state.get("simulation_results")
                if simulation_results and "error" not in simulation_results:
                    render_simulation(analysis_state)
                elif simulation_results:
                    print(f"\nSimulation failed: {simulation_results.get('error', 'Unknown error')}")

    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
        print("\nExecution interrupted. Analysis may be incomplete.")
    except Exception as e:
        logger.error(f"Unexpected application error: {e}")
        raise
    finally:
        logger.info("Application execution finished")