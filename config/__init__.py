"""
Configuration Module

This module contains configuration settings and parameters for the
automated trading system. It includes API keys, model parameters,
risk thresholds, and other system-wide settings.

Main Components:
- API configuration for external services (Yahoo Finance, Alpha Vantage, etc.)
- Model parameters for technical analysis and machine learning
- Risk management thresholds and limits
- Default stock lists and market parameters
- Database and logging configuration

The configuration is centralized to ensure consistency across
all modules and to make it easy to adjust system behavior.

Usage:
    from config.config import DEFAULT_STOCKS, RISK_TOLERANCE, API_KEYS
"""

from .config import *

__all__ = [
    # API Configuration
    "GROQ_API_KEY",
    "MODEL_NAME",
    "YAHOO_FINANCE_API_KEY",
    "ALPHA_VANTAGE_API_KEY",
    "NEWS_API_KEY",

    # Stock Lists
    "DEFAULT_STOCKS",
    "NIFTY_50_STOCKS",
    "SENSEX_STOCKS",

    # Model Parameters
    "TECHNICAL_ANALYSIS_PERIODS",
    "RSI_OVERSOLD_THRESHOLD",
    "RSI_OVERBOUGHT_THRESHOLD",
    "MACD_FAST_PERIOD",
    "MACD_SLOW_PERIOD",
    "MACD_SIGNAL_PERIOD",
    "BOLLINGER_BANDS_PERIOD",
    "BOLLINGER_BANDS_STD_DEV",

    # Risk Management
    "RISK_TOLERANCE",
    "MAX_POSITION_SIZE",
    "STOP_LOSS_PERCENTAGE",
    "TAKE_PROFIT_PERCENTAGE",
    "MAX_DRAWDOWN_LIMIT",
    "VOLATILITY_THRESHOLD",

    # Analysis Settings
    "MIN_DATA_POINTS",
    "DEFAULT_ANALYSIS_PERIOD",
    "BACKTEST_PERIOD_YEARS",
    "TRAINING_DATA_RATIO",
    "VALIDATION_DATA_RATIO",
    "TEST_DATA_RATIO",

    # Logging Configuration
    "LOG_LEVEL",
    "LOG_FILE_PATH",
    "LOG_MAX_SIZE",
    "LOG_BACKUP_COUNT",
]