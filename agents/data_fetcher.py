"""
Data fetching agent for stock market data.
Handles collection of historical stock data using yfinance.
"""

import logging
import asyncio
from functools import lru_cache
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import concurrent.futures

from config.config import DEFAULT_STOCKS, MAX_WORKERS
from data.models import State
from data.real_time_data import real_time_manager

# Configure logging
logger = logging.getLogger(__name__)


@lru_cache(maxsize=128)
def _get_stock_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """
    Fetch historical stock data for given symbol.

    Args:
        symbol: Stock symbol (e.g., "RELIANCE.NS")
        period: Time period (e.g., "1y", "2y", "5y")

    Returns:
        pandas DataFrame with OHLCV data
    """
    try:
        from yahooquery import Ticker
        ticker = Ticker(symbol)
        data = ticker.history(period=period)

        if data.empty:
            raise Exception("No data from API")

        # Handle MultiIndex if present (e.g., when yahooquery returns tuples in index)
        if isinstance(data.index, pd.MultiIndex):
            data = data.reset_index()
            data.set_index('date', inplace=True)

        # Convert to UTC and make tz-naive to avoid timezone mismatch errors
        data.index = pd.to_datetime(data.index, utc=True).tz_localize(None)
        data = data.sort_index()  # Sort by date to ensure proper indexing

        logger.info(f"Fetched real data for {symbol}: {len(data)} rows")
        return data
    except Exception as e:
        logger.error(f"Failed to fetch real data for {symbol}: {e}")
        raise  # Raise to prevent fallback; handle in caller if needed


def data_fetcher_agent(state: State, symbols: List[str] = None, real_time: bool = False,
                      sources: List[str] = None, interval: int = 60) -> State:
    """
    Data fetching agent for the LangGraph workflow.
    Uses parallel fetching for improved performance with multiple stocks.
    Supports real-time streaming when real_time=True.

    Args:
        state: Current workflow state
        symbols: List of stock symbols to fetch (uses DEFAULT_STOCKS if None)
        real_time: Whether to enable real-time streaming
        sources: List of data sources to use for real-time (default: ['yahoo', 'alpha_vantage'])
        interval: Polling interval in seconds for real-time data

    Returns:
        Updated state with stock data and failed stocks list
    """
    logging.info(f"Starting data fetcher agent (real_time={real_time})")

    if symbols is None:
        symbols = DEFAULT_STOCKS

    if real_time:
        # Initialize real-time data structure
        if 'real_time_data' not in state:
            state['real_time_data'] = {}
        if 'real_time_active' not in state:
            state['real_time_active'] = False

        # Start real-time streaming
        if not state.get('real_time_active', False):
            if sources is None:
                sources = ['yahoo', 'alpha_vantage']

            # Create event loop if not exists
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Start streaming in background
            loop.create_task(real_time_manager.start_streaming(symbols, sources, interval))

            # Add callback to update state
            def update_state_callback(data):
                from config.config import REAL_TIME_MAX_UPDATES
                symbol = data.get('symbol')
                if symbol:
                    if symbol not in state['real_time_data']:
                        state['real_time_data'][symbol] = []
                    state['real_time_data'][symbol].append(data)
                    # Keep only last N updates per symbol
                    if len(state['real_time_data'][symbol]) > REAL_TIME_MAX_UPDATES:
                        state['real_time_data'][symbol] = state['real_time_data'][symbol][-REAL_TIME_MAX_UPDATES:]

            real_time_manager.add_callback(update_state_callback)
            state['real_time_active'] = True
            logger.info(f"Real-time streaming started for {len(symbols)} symbols")

        return state

    # Original historical data fetching logic
    stock_data = {}
    failed_stocks = []

    # Use ThreadPoolExecutor for parallel data fetching
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(symbols), MAX_WORKERS)) as executor:
        # Submit all fetch tasks
        future_to_symbol = {executor.submit(_get_stock_data, symbol, "1y"): symbol for symbol in symbols}

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                df = future.result()
                stock_data[symbol] = df
                logger.info(f"Successfully fetched {len(df)} rows for {symbol}")
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
                failed_stocks.append({"symbol": symbol, "error": str(e)})
                continue

    # Log summary of fetch results
    successful_count = len(stock_data)
    failed_count = len(failed_stocks)
    total_count = len(symbols)
    logger.info(f"Data fetch summary: {successful_count}/{total_count} stocks successful, {failed_count} failed")

    state["stock_data"] = stock_data
    state["failed_stocks"] = failed_stocks
    return state


def stop_real_time_streaming(state: State, symbols: List[str] = None) -> State:
    """
    Stop real-time streaming for specified symbols or all if None.

    Args:
        state: Current workflow state
        symbols: List of symbols to stop streaming for (all if None)

    Returns:
        Updated state
    """
    if not state.get('real_time_active', False):
        logger.info("Real-time streaming is not active")
        return state

    # Create event loop if not exists
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Stop streaming
    loop.create_task(real_time_manager.stop_streaming(symbols))

    if symbols is None:
        state['real_time_active'] = False
        logger.info("Stopped all real-time streaming")
    else:
        logger.info(f"Stopped real-time streaming for {len(symbols)} symbols")

    return state