"""
Simulation and backtesting module for trading strategies.
"""

from .backtesting_engine import BacktestingEngine
from .simulation_runner import run_trading_simulation, validate_simulation_state

__all__ = [
    "BacktestingEngine",
    "run_trading_simulation",
    "validate_simulation_state"
]