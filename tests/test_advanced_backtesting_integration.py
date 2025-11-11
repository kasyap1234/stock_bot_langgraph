"""
Integration Tests for Advanced Backtesting Framework

This module tests the complete advanced backtesting framework integration
to ensure all components work together correctly.
"""

import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.advanced_backtesting_engine import AdvancedBacktestingEngine
from simulation.trading_strategies import BaseStrategy, StrategyConfig, TradingSignal


class MockStrategy(BaseStrategy):
    """Mock strategy for testing purposes."""
    
    def __init__(self):
        pass