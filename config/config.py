

import os
from typing import List

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

RISK_TOLERANCE = 0.25  # 25% annualized volatility threshold
MAX_POSITIONS = 0.1  # 10% of portfolio per position

MAX_PORTFOLIO_DRAWDOWN = 0.15  # 15% max portfolio drawdown
MAX_DAILY_LOSS = 0.05  # 5% max daily loss
MAX_POSITION_SIZE_PCT = 0.10  # 10% max position size
MAX_SECTOR_EXPOSURE = 0.25  # 25% max sector exposure
KELLY_FRACTION = 0.5  # Use half-Kelly for conservatism
RISK_FREE_RATE = 0.065  # 6.5% Indian risk-free rate
ATR_PERIOD = 14  # ATR period for stops
TRAILING_STOP_PCT = 0.05  # 5% trailing stop
TIME_EXIT_DAYS = 30  # Exit after 30 days if no movement
PROFIT_TARGET_LEVELS = [0.05, 0.10, 0.15]  # 5%, 10%, 15% profit targets

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