"""
Configuration Validator
Validates API keys and configuration at startup to ensure accurate analysis
"""

import logging
import os
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class ConfigValidationResult:
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
    
    def add_error(self, message: str):
        self.errors.append(message)
        logger.error(f"Config Error: {message}")
    
    def add_warning(self, message: str):
        self.warnings.append(message)
        logger.warning(f"Config Warning: {message}")
    
    def add_info(self, message: str):
        self.info.append(message)
        logger.info(f"Config Info: {message}")
    
    def is_valid(self) -> bool:
        return len(self.errors) == 0
    
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0
    
    def print_summary(self):
        print("\n" + "="*80)
        print("CONFIGURATION VALIDATION SUMMARY")
        print("="*80)
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for err in self.errors:
                print(f"  - {err}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warn in self.warnings:
                print(f"  - {warn}")
        
        if self.info:
            print(f"\n✓  INFO ({len(self.info)}):")
            for info_msg in self.info:
                print(f"  - {info_msg}")
        
        print("\n" + "="*80)
        
        if not self.is_valid():
            print("⚠️  Configuration has ERRORS - stock recommendations may be inaccurate!")
        elif self.has_warnings():
            print("⚠️  Configuration has WARNINGS - some features may not work optimally")
        else:
            print("✓  Configuration is valid - all systems operational")
        
        print("="*80 + "\n")


def validate_configuration() -> ConfigValidationResult:
    """
    Validate all configuration settings and API keys
    Returns validation result with errors, warnings, and info
    """
    result = ConfigValidationResult()
    
    # Validate API Keys
    _validate_api_keys(result)
    
    # Validate Data Sources
    _validate_data_sources(result)
    
    # Validate ML Dependencies
    _validate_ml_dependencies(result)
    
    # Validate Directory Structure
    _validate_directories(result)
    
    return result


def _validate_api_keys(result: ConfigValidationResult):
    """Validate API keys are configured"""
    from config.api_config import (
        ALPHA_VANTAGE_API_KEY,
        TWITTER_BEARER_TOKEN,
        NEWS_API_KEY,
        GROQ_API_KEY
    )
    
    # Alpha Vantage (for fundamental data)
    if not ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY in ["demo", "9YM9MF6IN0GJMMCO"]:
        result.add_warning(
            "ALPHA_VANTAGE_API_KEY not configured or using demo key - "
            "fundamental analysis may use fallback data sources"
        )
    else:
        result.add_info("Alpha Vantage API key configured ✓")
    
    # Twitter (for sentiment analysis)
    if not TWITTER_BEARER_TOKEN or TWITTER_BEARER_TOKEN == "":
        result.add_warning(
            "TWITTER_BEARER_TOKEN not configured - "
            "sentiment analysis will use news only (no Twitter data)"
        )
    else:
        result.add_info("Twitter API key configured ✓")
    
    # News API (for sentiment analysis)
    if not NEWS_API_KEY or NEWS_API_KEY == "":
        result.add_warning(
            "NEWS_API_KEY not configured - "
            "sentiment analysis will use web scraping fallback"
        )
    else:
        result.add_info("News API key configured ✓")
    
    # Groq (for LLM-based analysis)
    if not GROQ_API_KEY or GROQ_API_KEY == "demo":
        result.add_warning(
            "GROQ_API_KEY not configured - "
            "using rule-based decision making instead of LLM"
        )
    else:
        result.add_info("Groq API key configured ✓")


def _validate_data_sources(result: ConfigValidationResult):
    """Validate data source availability"""
    # Check yahooquery
    try:
        from yahooquery import Ticker
        result.add_info("yahooquery library available ✓")
    except ImportError:
        result.add_error(
            "yahooquery not installed - primary data source unavailable. "
            "Run: pip install yahooquery"
        )
    
    # Check pandas_datareader
    try:
        import pandas_datareader
        result.add_info("pandas_datareader available ✓")
    except ImportError:
        result.add_warning(
            "pandas_datareader not installed - some data sources unavailable"
        )


def _validate_ml_dependencies(result: ConfigValidationResult):
    """Validate ML library availability"""
    # Check scikit-learn
    try:
        import sklearn
        result.add_info(f"scikit-learn {sklearn.__version__} available ✓")
    except ImportError:
        result.add_error(
            "scikit-learn not installed - ML models unavailable. "
            "Run: pip install scikit-learn"
        )
    
    # Check TA-Lib (optional but recommended)
    try:
        import talib
        result.add_info("TA-Lib available ✓ (enhanced technical analysis)")
    except ImportError:
        result.add_warning(
            "TA-Lib not installed - using basic technical analysis fallback. "
            "For better accuracy, install TA-Lib"
        )
    
    # Check TensorFlow (optional for neural networks)
    try:
        import tensorflow as tf
        result.add_info(f"TensorFlow {tf.__version__} available ✓ (neural networks)")
    except ImportError:
        result.add_warning(
            "TensorFlow not installed - neural network models disabled"
        )
    
    # Check XGBoost (optional for advanced ML)
    try:
        import xgboost as xgb
        result.add_info(f"XGBoost {xgb.__version__} available ✓")
    except ImportError:
        result.add_warning(
            "XGBoost not installed - advanced ML models limited"
        )
    
    # Check VADER Sentiment
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        result.add_info("VADER Sentiment available ✓")
    except ImportError:
        result.add_warning(
            "VADER Sentiment not installed - using basic sentiment analysis. "
            "Run: pip install vaderSentiment"
        )


def _validate_directories(result: ConfigValidationResult):
    """Validate required directories exist"""
    from config.constants import MODEL_DIR
    
    # Check model directory
    if not os.path.exists(MODEL_DIR):
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            result.add_info(f"Created model directory: {MODEL_DIR}")
        except Exception as e:
            result.add_error(f"Cannot create model directory {MODEL_DIR}: {e}")
    else:
        result.add_info(f"Model directory exists: {MODEL_DIR}")
    
    # Check log directory
    log_dir = "logs"
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
            result.add_info(f"Created log directory: {log_dir}")
        except Exception as e:
            result.add_error(f"Cannot create log directory {log_dir}: {e}")
    else:
        result.add_info(f"Log directory exists: {log_dir}")
    
    # Check cache directory
    cache_dir = "data/cache"
    if not os.path.exists(cache_dir):
        try:
            os.makedirs(cache_dir, exist_ok=True)
            result.add_info(f"Created cache directory: {cache_dir}")
        except Exception as e:
            result.add_warning(f"Cannot create cache directory {cache_dir}: {e}")


def validate_and_warn():
    """
    Main entry point for configuration validation
    Validates configuration and prints summary
    """
    result = validate_configuration()
    result.print_summary()
    
    if not result.is_valid():
        print("\n⚠️  WARNING: Configuration errors detected!")
        print("    Stock recommendations may be inaccurate or incomplete.")
        print("    Please fix the errors above before proceeding.\n")
        
        response = input("Continue anyway? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Exiting...")
            exit(1)
    
    return result


if __name__ == "__main__":
    validate_and_warn()
