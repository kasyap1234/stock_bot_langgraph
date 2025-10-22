

import logging
import asyncio
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import concurrent.futures
import os
import json
from datetime import datetime, timedelta

from config.constants import DEFAULT_STOCKS, MAX_WORKERS, REAL_TIME_MAX_UPDATES
from data.models import State
from data.real_time_data import real_time_manager

logger = logging.getLogger(__name__)




def data_fetcher_agent(state: State, symbols: List[str] = None, period: str = "5y", real_time: bool = False,
                      sources: List[str] = None, interval: int = 60) -> State:
    
    logger.info(f"Starting data fetcher agent (real_time={real_time})")

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
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Start streaming in background
            loop.create_task(real_time_manager.start_streaming(symbols, sources, interval))

            # Add callback to update state
            def update_state_callback(data):
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

    from data.apis import UnifiedDataFetcher
    fetcher = UnifiedDataFetcher(symbols[0] if symbols else "") # Initialize with a dummy symbol
    batch_results = fetcher.get_batch_historical_data(symbols, period)

    for symbol, data in batch_results.items():
        if data:
            stock_data[symbol] = data
            logger.info(f"Successfully fetched {len(data)} rows for {symbol}")
        else:
            logger.error(f"Failed to fetch data for {symbol}")
            failed_stocks.append({"symbol": symbol, "error": "Failed to fetch data"})


    # Log summary of fetch results
    successful_count = len(stock_data)
    failed_count = len(failed_stocks)
    total_count = len(symbols)
    logger.info(f"DataActor completed: {successful_count}/{total_count} stocks successful, {failed_count} failed")

    return {"stock_data": stock_data, "failed_stocks": failed_stocks}


def stop_real_time_streaming(state: State, symbols: List[str] = None) -> State:
    
    if not state.get('real_time_active', False):
        logger.info("Real-time streaming is not active")
        return state

    # Create event loop if not exists
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Stop streaming
    loop.create_task(real_time_manager.stop_streaming(symbols))

    if symbols is None:
        state['real_time_active'] = False
        logger.info("DataActor stopped all real-time streaming")
    else:
        logger.info(f"DataActor stopped real-time streaming for {len(symbols)} symbols")

    return state