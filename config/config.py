from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field
import os
from typing import List

import logging

logger = logging.getLogger(__name__)

load_dotenv()

class Settings(BaseSettings):
    alpha_vantage_api_key: str = Field(default="", env="ALPHA_VANTAGE_API_KEY")
    groq_api_key: str = Field(default="", env="GROQ_API_KEY")
    fred_api_key: str = Field(default="", env="FRED_API_KEY")
    news_api_key: str = Field(default="", env="NEWS_API_KEY")
    twitter_bearer_token: str = Field(default="", env="TWITTER_BEARER_TOKEN")
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    secret_key: str = Field(default="", env="SECRET_KEY")
    groq_model_name: str = Field(default="moonshotai/kimi-k2-instruct-0905", env="GROQ_MODEL_NAME")
    temperature: float = 0.7
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    risk_tolerance: float = Field(default=0.20, env="RISK_TOLERANCE")
    max_positions: float = Field(default=0.15, env="MAX_POSITIONS")
    max_portfolio_drawdown: float = Field(default=0.20, env="MAX_PORTFOLIO_DRAWDOWN")
    max_daily_loss: float = Field(default=0.08, env="MAX_DAILY_LOSS")
    max_position_size_pct: float = Field(default=0.15, env="MAX_POSITION_SIZE_PCT")
    max_sector_exposure: float = Field(default=0.30, env="MAX_SECTOR_EXPOSURE")
    kelly_fraction: float = Field(default=0.75, env="KELLY_FRACTION")
    risk_free_rate: float = Field(default=0.065, env="RISK_FREE_RATE")
    atr_period: int = Field(default=14, env="ATR_PERIOD")
    trailing_stop_pct: float = Field(default=0.08, env="TRAILING_STOP_PCT")
    time_exit_days: int = Field(default=45, env="TIME_EXIT_DAYS")
    profit_target_levels: List[float] = Field(default_factory=lambda: [0.08, 0.15, 0.25])
    confirmation_threshold: int = 2
    ensemble_threshold: float = 0.15
    trend_strength_threshold: float = 0.5
    probability_threshold: float = 0.52
    backtest_validation_threshold: float = 0.45
    top_n_recommendations: int = 10
    max_workers: int = 10
    rsi_overbought: int = 65
    rsi_oversold: int = 35
    india_specific_params: dict = {
        'RSI_OVERSOLD': 25,
        'RSI_OVERBOUGHT': 75,
        'MACD_FAST': 10,
        'MACD_SLOW': 22,
        'MACD_SIGNAL': 9
    }
    ichimoku_periods: dict = {
        'tenkan_sen': 9,
        'kijun_sen': 26,
        'senkou_span_b': 52,
        'chikou_span': 26
    }
    fib_levels: List[float] = [0.236, 0.382, 0.5, 0.618, 0.786]
    support_resistance_periods: dict = {
        'short_term': 20,
        'medium_term': 50,
        'long_term': 200
    }
    ml_model_params: dict = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42
    }
    adaptive_thresholds: dict = {
        'high_volatility': 0.05,
        'medium_volatility': 0.02,
        'low_volatility': 0.01
    }
    grid_search_params: dict = {
        'rsi_period': [9, 14, 21],
        'macd_fast': [8, 12, 26],
        'macd_slow': [17, 26, 50],
        'stoch_k': [5, 14, 21],
        'stoch_d': [3, 5, 9]
    }
    monte_carlo_simulations: int = 1000
    monte_carlo_horizon: int = 252
    vpvr_bins: int = 50
    visible_range: int = 200
    volume_percentile: float = 0.7
    ha_confluence_bars: int = 3
    ha_tfs: List[str] = ['1h', '4h', 'daily']
    garch_p: int = 1
    garch_q: int = 1
    forecast_horizon: int = 5
    harmonic_patterns: List[str] = ['gartley', 'butterfly']
    tolerance: float = 0.05
    lookback: int = 100
    min_confidence: float = 0.7
    hmm_states: int = 3
    hmm_iter: int = 100
    lstm_epochs: int = 50
    lstm_batch: int = 32
    lstm_window: int = 60
    lstm_features: int = 20
    mc_va_paths: int = 10000
    va_confidence: List[float] = [0.95, 0.99]
    stress_scenarios: List[float] = [1.2, 1.5]
    default_period: str = "5y"
    default_interval: str = "1d"
    simulation_days: int = 252
    walk_forward_enabled: bool = True
    trade_limit: int = 50
    api_rate_limit_delay: float = 60 / 5
    request_timeout: int = 10
    debug_recommendation_logging: bool = True
    enable_advanced_tech: bool = True
    data_dir: str = "data"
    model_dir: str = "models"
    real_time_enabled: bool = os.getenv("REAL_TIME_ENABLED", "false").lower() == "true"
    real_time_sources: List[str] = os.getenv("REAL_TIME_SOURCES", "yahoo,alpha_vantage").split(",")
    real_time_interval: int = int(os.getenv("REAL_TIME_INTERVAL", "60"))
    real_time_max_updates: int = int(os.getenv("REAL_TIME_MAX_UPDATES", "100"))
    data_source_priorities: dict = {
        'yahoo': 10,
        'alpha_vantage': 8,
        'newsapi': 6,
        'fred': 5,
        'moneycontrol': 7,
        'bse': 7
    }
    api_rate_limits: dict = {
        'yahoo': 60,
        'alpha_vantage': 5,
        'newsapi': 100,
        'fred': 120,
        'moneycontrol': 30,
        'bse': 30
    }
    results_dir: str = "results"
    default_stocks: List[str] = ["RELIANCE.NS", "TCS.NS"]
    nifty_50_stocks: List[str] = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
        "HINDUNILVR.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS",
        "MARUTI.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "HCLTECH.NS", "WIPRO.NS",
        "ULTRACEMCO.NS", "NESTLEIND.NS", "POWERGRID.NS", "NTPC.NS", "ONGC.NS",
        "COALINDIA.NS", "GRASIM.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "ADANIPORTS.NS",
        "SHREECEM.NS", "BAJAJ-AUTO.NS", "TITAN.NS", "HEROMOTOCO.NS", "DRREDDY.NS",
        "SUNPHARMA.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS", "INDUSINDBK.NS",
        "HDFCLIFE.NS", "SBILIFE.NS", "BRITANNIA.NS", "TECHM.NS", "EICHERMOT.NS",
        "BPCL.NS", "UPL.NS", "M&M.NS", "TATACONSUM.NS", "ASIANPAINT.NS",
        "PIDILITIND.NS", "NMDC.NS", "GAIL.NS", "VEDL.NS"
    ]
    log_format: str = '%(asctime)s - %(levelname)s - %(message)s'
    yahoo_finance_api_key: str = ""  # Usually not required

    class Config:
        env_file = ".env"

settings = Settings()

def validate_config() -> None:
    """Validate that all required configuration values are set."""
    required_keys = [
        "alpha_vantage_api_key",
        "openai_api_key",
        "secret_key"
    ]
    missing = [key for key in required_keys if not getattr(settings, key, None)]
    if missing:
        logger.warning(f"Missing required environment variables for production: {', '.join(missing)}. Using defaults for testing/development.")
    else:
        logger.info("Configuration validation passed")

# FIXED: Call validation on import
validate_config()

# Top-level constants for compatibility with legacy code
DEFAULT_STOCKS = settings.default_stocks
MAX_WORKERS = settings.max_workers
REAL_TIME_MAX_UPDATES = settings.real_time_max_updates
ALPHA_VANTAGE_API_KEY = settings.alpha_vantage_api_key
FRED_API_KEY = settings.fred_api_key
NEWS_API_KEY = settings.news_api_key
MODEL_NAME = settings.groq_model_name
GROQ_API_KEY = settings.groq_api_key
TEMPERATURE = settings.temperature
TOP_N_RECOMMENDATIONS = settings.top_n_recommendations
DEBUG_RECOMMENDATION_LOGGING = settings.debug_recommendation_logging
TRADE_LIMIT = settings.trade_limit
SIMULATION_DAYS = settings.simulation_days
WALK_FORWARD_ENABLED = settings.walk_forward_enabled
LOG_LEVEL = settings.log_level
LOG_FORMAT = settings.log_format
API_RATE_LIMIT_DELAY = settings.api_rate_limit_delay
RSI_OVERBOUGHT = settings.rsi_overbought
RSI_OVERSOLD = settings.rsi_oversold
NIFTY_50_STOCKS = settings.nifty_50_stocks
CONFIRMATION_THRESHOLD = settings.confirmation_threshold
ENSEMBLE_THRESHOLD = settings.ensemble_threshold
TREND_STRENGTH_THRESHOLD = settings.trend_strength_threshold
PROBABILITY_THRESHOLD = settings.probability_threshold
BACKTEST_VALIDATION_THRESHOLD = settings.backtest_validation_threshold
ADAPTIVE_THRESHOLDS = settings.adaptive_thresholds
REQUEST_TIMEOUT = settings.request_timeout
ENABLE_ADVANCED_TECH = settings.enable_advanced_tech
ICHIMOKU_PERIODS = settings.ichimoku_periods
FIB_LEVELS = settings.fib_levels
SUPPORT_RESISTANCE_PERIODS = settings.support_resistance_periods
ML_MODEL_PARAMS = settings.ml_model_params
GRID_SEARCH_PARAMS = settings.grid_search_params
MONTE_CARLO_SIMULATIONS = settings.monte_carlo_simulations
MONTE_CARLO_HORIZON = settings.monte_carlo_horizon
VPVR_BINS = settings.vpvr_bins
VISIBLE_RANGE = settings.visible_range
VOLUME_PERCENTILE = settings.volume_percentile
HA_CONFLUENCE_BARS = settings.ha_confluence_bars
HA_TFS = settings.ha_tfs
GARCH_P = settings.garch_p
GARCH_Q = settings.garch_q
FORECAST_HORIZON = settings.forecast_horizon
HARMONIC_PATTERNS = settings.harmonic_patterns
TOLERANCE = settings.tolerance
LOOKBACK = settings.lookback
MIN_CONFIDENCE = settings.min_confidence
HMM_STATES = settings.hmm_states
HMM_ITER = settings.hmm_iter
LSTM_EPOCHS = settings.lstm_epochs
LSTM_BATCH = settings.lstm_batch
LSTM_WINDOW = settings.lstm_window
LSTM_FEATURES = settings.lstm_features
MC_VA_PATHS = settings.mc_va_paths
VA_CONFIDENCE = settings.va_confidence
STRESS_SCENARIOS = settings.stress_scenarios
INDIA_SPECIFIC_PARAMS = settings.india_specific_params
TWITTER_BEARER_TOKEN = settings.twitter_bearer_token
RISK_TOLERANCE = settings.risk_tolerance
MAX_POSITIONS = settings.max_positions
MAX_PORTFOLIO_DRAWDOWN = settings.max_portfolio_drawdown
MAX_DAILY_LOSS = settings.max_daily_loss
MAX_POSITION_SIZE_PCT = settings.max_position_size_pct
MAX_SECTOR_EXPOSURE = settings.max_sector_exposure
KELLY_FRACTION = settings.kelly_fraction
RISK_FREE_RATE = settings.risk_free_rate
ATR_PERIOD = settings.atr_period
TRAILING_STOP_PCT = settings.trailing_stop_pct
TIME_EXIT_DAYS = settings.time_exit_days
PROFIT_TARGET_LEVELS = settings.profit_target_levels
MODEL_DIR = settings.model_dir