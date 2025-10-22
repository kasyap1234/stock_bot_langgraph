

import os
from typing import List, Dict, Any
from dataclasses import dataclass, field

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "9YM9MF6IN0GJMMCO")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
YAHOO_FINANCE_API_KEY = ""  # Usually not required
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "demo")

MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "moonshotai/kimi-k2-instruct-0905")  
TEMPERATURE = 0.7

DEFAULT_STOCKS: List[str] = ["RELIANCE.NS", "TCS.NS"]

NIFTY_50_STOCKS: List[str] = [
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

TOP_N_RECOMMENDATIONS = 10  # Number of top recommendations to display

MAX_WORKERS = 10  # Maximum parallel workers for data fetching

RSI_OVERBOUGHT = 65
RSI_OVERSOLD = 35

RISK_TOLERANCE = 0.30  # 30% annualized volatility threshold (increased for better returns)
MAX_POSITIONS = 0.15  # 15% of portfolio per position (increased for better capital utilization)

MAX_PORTFOLIO_DRAWDOWN = 0.20  # 20% max portfolio drawdown (increased tolerance)
MAX_DAILY_LOSS = 0.08  # 8% max daily loss (increased for more flexibility)
MAX_POSITION_SIZE_PCT = 0.15  # 15% max position size (increased for better utilization)
MAX_SECTOR_EXPOSURE = 0.30  # 30% max sector exposure (increased diversification)
KELLY_FRACTION = 0.75  # Use 75% Kelly for more aggressive position sizing
RISK_FREE_RATE = 0.065  # 6.5% Indian risk-free rate
ATR_PERIOD = 14  # ATR period for stops
TRAILING_STOP_PCT = 0.08  # 8% trailing stop (increased for more room)
TIME_EXIT_DAYS = 45  # Exit after 45 days if no movement (extended for more patience)
PROFIT_TARGET_LEVELS = [0.08, 0.15, 0.25]  # 8%, 15%, 25% profit targets (higher targets)

CONFIRMATION_THRESHOLD = 3  # Minimum indicators to agree for confirmation
ENSEMBLE_THRESHOLD = 0.2  # Threshold for ensemble signal strength
TREND_STRENGTH_THRESHOLD = 0.6  # Minimum trend strength for signals
PROBABILITY_THRESHOLD = 0.55  # Minimum probability score for signals
BACKTEST_VALIDATION_THRESHOLD = 0.5  # Minimum backtest win rate

ICHIMOKU_PERIODS = {
    'tenkan_sen': 9,    # Conversion Line
    'kijun_sen': 26,    # Base Line
    'senkou_span_b': 52, # Leading Span B
    'chikou_span': 26   # Lagging Span
}

FIB_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]  # Fibonacci retracement levels

SUPPORT_RESISTANCE_PERIODS = {
    'short_term': 20,
    'medium_term': 50,
    'long_term': 200
}

ML_MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42
}

ADAPTIVE_THRESHOLDS = {
    'high_volatility': 0.05,
    'medium_volatility': 0.02,
    'low_volatility': 0.01
}

GRID_SEARCH_PARAMS = {
    'rsi_period': [9, 14, 21],
    'macd_fast': [8, 12, 26],
    'macd_slow': [17, 26, 50],
    'stoch_k': [5, 14, 21],
    'stoch_d': [3, 5, 9]
}

MONTE_CARLO_SIMULATIONS = 1000
MONTE_CARLO_HORIZON = 252  # Trading days


VPVR_BINS = 50
VISIBLE_RANGE = 200
VOLUME_PERCENTILE = 0.7

HA_CONFLUENCE_BARS = 3
HA_TFS = ['1h', '4h', 'daily']

GARCH_P = 1
GARCH_Q = 1
FORECAST_HORIZON = 5

HARMONIC_PATTERNS = ['gartley', 'butterfly']
TOLERANCE = 0.05
LOOKBACK = 100
MIN_CONFIDENCE = 0.7

HMM_STATES = 3
HMM_ITER = 100

LSTM_EPOCHS = 50
LSTM_BATCH = 32
LSTM_WINDOW = 60
LSTM_FEATURES = 20

MC_VA_PATHS = 10000
VA_CONFIDENCE = [0.95, 0.99]
STRESS_SCENARIOS = [1.2, 1.5]

DEFAULT_PERIOD = "1y"
DEFAULT_INTERVAL = "1d"

SIMULATION_DAYS = 252  # Trading days in a year
TRADE_LIMIT = 50  # Maximum trades per simulation

API_RATE_LIMIT_DELAY = 60 / 5  # Alpha Vantage: 5 calls/minute
REQUEST_TIMEOUT = 10  # seconds

LOG_LEVEL = "DEBUG"
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

DEBUG_RECOMMENDATION_LOGGING = True
ENABLE_ADVANCED_TECH = True

DATA_DIR = "data"
MODEL_DIR = "models"
REAL_TIME_ENABLED = os.getenv("REAL_TIME_ENABLED", "false").lower() == "true"
REAL_TIME_SOURCES = os.getenv("REAL_TIME_SOURCES", "yahoo,alpha_vantage").split(",")
REAL_TIME_INTERVAL = int(os.getenv("REAL_TIME_INTERVAL", "60"))  # seconds
REAL_TIME_MAX_UPDATES = int(os.getenv("REAL_TIME_MAX_UPDATES", "100"))  # per symbol

DATA_SOURCE_PRIORITIES = {
    'yahoo': 10,      # Primary for NSE/BSE
    'alpha_vantage': 8,  # Additional metrics
    'newsapi': 6,     # Real-time news
    'fred': 5,        # Macro data
    'moneycontrol': 7, # Indian market scraping
    'bse': 7          # BSE data
}

API_RATE_LIMITS = {
    'yahoo': 60,      # Yahoo Finance (generous)
    'alpha_vantage': 5,  # Alpha Vantage free tier
    'newsapi': 100,   # NewsAPI free tier
    'fred': 120,      # FRED (generous)
    'moneycontrol': 30, # Web scraping
    'bse': 30         # Web scraping
}

DATA_DIR = "data"
MODEL_DIR = "models"
RESULTS_DIR = "results"
RESULTS_DIR = "results"

@dataclass
class Settings:
    ALPHA_VANTAGE_API_KEY: str = ALPHA_VANTAGE_API_KEY
    GROQ_API_KEY: str = GROQ_API_KEY
    YAHOO_FINANCE_API_KEY: str = YAHOO_FINANCE_API_KEY
    FRED_API_KEY: str = FRED_API_KEY
    NEWS_API_KEY: str = NEWS_API_KEY
    TWITTER_BEARER_TOKEN: str = TWITTER_BEARER_TOKEN
    OPENAI_API_KEY: str = OPENAI_API_KEY
    MODEL_NAME: str = MODEL_NAME
    TEMPERATURE: float = TEMPERATURE
    DEFAULT_STOCKS: List[str] = field(default_factory=lambda: DEFAULT_STOCKS)
    NIFTY_50_STOCKS: List[str] = field(default_factory=lambda: NIFTY_50_STOCKS)
    TOP_N_RECOMMENDATIONS: int = TOP_N_RECOMMENDATIONS
    MAX_WORKERS: int = MAX_WORKERS
    RSI_OVERBOUGHT: int = RSI_OVERBOUGHT
    RSI_OVERSOLD: int = RSI_OVERSOLD
    RISK_TOLERANCE: float = RISK_TOLERANCE
    MAX_POSITIONS: float = MAX_POSITIONS
    MAX_PORTFOLIO_DRAWDOWN: float = MAX_PORTFOLIO_DRAWDOWN
    MAX_DAILY_LOSS: float = MAX_DAILY_LOSS
    MAX_POSITION_SIZE_PCT: float = MAX_POSITION_SIZE_PCT
    MAX_SECTOR_EXPOSURE: float = MAX_SECTOR_EXPOSURE
    KELLY_FRACTION: float = KELLY_FRACTION
    RISK_FREE_RATE: float = RISK_FREE_RATE
    ATR_PERIOD: int = ATR_PERIOD
    TRAILING_STOP_PCT: float = TRAILING_STOP_PCT
    TIME_EXIT_DAYS: int = TIME_EXIT_DAYS
    PROFIT_TARGET_LEVELS: List[float] = field(default_factory=lambda: PROFIT_TARGET_LEVELS)
    CONFIRMATION_THRESHOLD: int = CONFIRMATION_THRESHOLD
    ENSEMBLE_THRESHOLD: float = ENSEMBLE_THRESHOLD
    TREND_STRENGTH_THRESHOLD: float = TREND_STRENGTH_THRESHOLD
    PROBABILITY_THRESHOLD: float = PROBABILITY_THRESHOLD
    BACKTEST_VALIDATION_THRESHOLD: float = BACKTEST_VALIDATION_THRESHOLD
    ICHIMOKU_PERIODS: Dict[str, int] = field(default_factory=lambda: ICHIMOKU_PERIODS)
    FIB_LEVELS: List[float] = field(default_factory=lambda: FIB_LEVELS)
    SUPPORT_RESISTANCE_PERIODS: Dict[str, int] = field(default_factory=lambda: SUPPORT_RESISTANCE_PERIODS)
    ML_MODEL_PARAMS: Dict[str, Any] = field(default_factory=lambda: ML_MODEL_PARAMS)
    ADAPTIVE_THRESHOLDS: Dict[str, float] = field(default_factory=lambda: ADAPTIVE_THRESHOLDS)
    GRID_SEARCH_PARAMS: Dict[str, List[int]] = field(default_factory=lambda: GRID_SEARCH_PARAMS)
    MONTE_CARLO_SIMULATIONS: int = MONTE_CARLO_SIMULATIONS
    MONTE_CARLO_HORIZON: int = MONTE_CARLO_HORIZON
    VPVR_BINS: int = VPVR_BINS
    VISIBLE_RANGE: int = VISIBLE_RANGE
    VOLUME_PERCENTILE: float = VOLUME_PERCENTILE
    HA_CONFLUENCE_BARS: int = HA_CONFLUENCE_BARS
    HA_TFS: List[str] = field(default_factory=lambda: HA_TFS)
    GARCH_P: int = GARCH_P
    GARCH_Q: int = GARCH_Q
    FORECAST_HORIZON: int = FORECAST_HORIZON
    HARMONIC_PATTERNS: List[str] = field(default_factory=lambda: HARMONIC_PATTERNS)
    TOLERANCE: float = TOLERANCE
    LOOKBACK: int = LOOKBACK
    MIN_CONFIDENCE: float = MIN_CONFIDENCE
    HMM_STATES: int = HMM_STATES
    HMM_ITER: int = HMM_ITER
    LSTM_EPOCHS: int = LSTM_EPOCHS
    LSTM_BATCH: int = LSTM_BATCH
    LSTM_WINDOW: int = LSTM_WINDOW
    LSTM_FEATURES: int = LSTM_FEATURES
    MC_VA_PATHS: int = MC_VA_PATHS
    VA_CONFIDENCE: List[float] = field(default_factory=lambda: VA_CONFIDENCE)
    STRESS_SCENARIOS: List[float] = field(default_factory=lambda: STRESS_SCENARIOS)
    DEFAULT_PERIOD: str = DEFAULT_PERIOD
    DEFAULT_INTERVAL: str = DEFAULT_INTERVAL
    SIMULATION_DAYS: int = SIMULATION_DAYS
    TRADE_LIMIT: int = TRADE_LIMIT
    API_RATE_LIMIT_DELAY: float = API_RATE_LIMIT_DELAY
    REQUEST_TIMEOUT: int = REQUEST_TIMEOUT
    LOG_LEVEL: str = LOG_LEVEL
    LOG_FORMAT: str = LOG_FORMAT
    DEBUG_RECOMMENDATION_LOGGING: bool = DEBUG_RECOMMENDATION_LOGGING
    ENABLE_ADVANCED_TECH: bool = ENABLE_ADVANCED_TECH
    DATA_DIR: str = DATA_DIR
    MODEL_DIR: str = MODEL_DIR
    REAL_TIME_ENABLED: bool = REAL_TIME_ENABLED
    REAL_TIME_SOURCES: List[str] = field(default_factory=lambda: REAL_TIME_SOURCES)
    REAL_TIME_INTERVAL: int = REAL_TIME_INTERVAL
    REAL_TIME_MAX_UPDATES: int = REAL_TIME_MAX_UPDATES
    DATA_SOURCE_PRIORITIES: Dict[str, int] = field(default_factory=lambda: DATA_SOURCE_PRIORITIES)
    API_RATE_LIMITS: Dict[str, int] = field(default_factory=lambda: API_RATE_LIMITS)
    RESULTS_DIR: str = RESULTS_DIR
    secret_key: str = os.getenv("SECRET_KEY", "default_secret_key_change_me")

settings = Settings()
settings = Settings()