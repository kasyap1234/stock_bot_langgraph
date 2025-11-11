import os

CONSTANTS_MAPPINGS = [
    ("DEFAULT_STOCKS", "default_stocks"),
    ("MAX_WORKERS", "max_workers"),
    ("REAL_TIME_MAX_UPDATES", "real_time_max_updates"),
    ("DEBUG_RECOMMENDATION_LOGGING", "debug_recommendation_logging"),
    ("LOG_LEVEL", "log_level"),
    ("LOG_FORMAT", "log_format"),
    ("RSI_OVERBOUGHT", "rsi_overbought"),
    ("RSI_OVERSOLD", "rsi_oversold"),
    ("INDIA_SPECIFIC_PARAMS", "india_specific_params"),
    ("ICHIMOKU_PERIODS", "ichimoku_periods"),
    ("FIB_LEVELS", "fib_levels"),
    ("SUPPORT_RESISTANCE_PERIODS", "support_resistance_periods"),
    ("ENABLE_ADVANCED_TECH", "enable_advanced_tech"),
    ("MODEL_DIR", "model_dir"),
]

DEFAULT_STOCKS = ["RELIANCE.NS", "TCS.NS"]
MAX_WORKERS = 10
REAL_TIME_MAX_UPDATES = int(os.getenv("REAL_TIME_MAX_UPDATES", "100"))
DEBUG_RECOMMENDATION_LOGGING = True
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
RSI_OVERBOUGHT = 65
RSI_OVERSOLD = 35
INDIA_SPECIFIC_PARAMS = {'RSI_OVERSOLD': 25, 'RSI_OVERBOUGHT': 75, 'MACD_FAST': 10, 'MACD_SLOW': 22, 'MACD_SIGNAL': 9}
ICHIMOKU_PERIODS = {'tenkan_sen': 9, 'kijun_sen': 26, 'senkou_span_b': 52, 'chikou_span': 26}
FIB_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]
SUPPORT_RESISTANCE_PERIODS = {'short_term': 20, 'medium_term': 50, 'long_term': 200}
ENABLE_ADVANCED_TECH = True
MODEL_DIR = "models"