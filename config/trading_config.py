import os

TRADING_MAPPINGS = [
    ("RISK_TOLERANCE", "risk_tolerance"),
    ("MAX_POSITIONS", "max_positions"),
    ("MAX_PORTFOLIO_DRAWDOWN", "max_portfolio_drawdown"),
    ("MAX_DAILY_LOSS", "max_daily_loss"),
    ("MAX_POSITION_SIZE_PCT", "max_position_size_pct"),
    ("MAX_SECTOR_EXPOSURE", "max_sector_exposure"),
    ("KELLY_FRACTION", "kelly_fraction"),
    ("RISK_FREE_RATE", "risk_free_rate"),
    ("ATR_PERIOD", "atr_period"),
    ("TRAILING_STOP_PCT", "trailing_stop_pct"),
    ("TIME_EXIT_DAYS", "time_exit_days"),
    ("PROFIT_TARGET_LEVELS", "profit_target_levels"),
    ("CONFIRMATION_THRESHOLD", "confirmation_threshold"),
    ("ENSEMBLE_THRESHOLD", "ensemble_threshold"),
    ("TREND_STRENGTH_THRESHOLD", "trend_strength_threshold"),
    ("PROBABILITY_THRESHOLD", "probability_threshold"),
    ("BACKTEST_VALIDATION_THRESHOLD", "backtest_validation_threshold"),
    ("TOP_N_RECOMMENDATIONS", "top_n_recommendations"),
    ("TRADE_LIMIT", "trade_limit"),
    ("SIMULATION_DAYS", "simulation_days"),
    ("WALK_FORWARD_ENABLED", "walk_forward_enabled"),
    ("NIFTY_50_STOCKS", "nifty_50_stocks"),
]

RISK_TOLERANCE = float(os.getenv("RISK_TOLERANCE", "0.20"))
MAX_POSITIONS = float(os.getenv("MAX_POSITIONS", "0.15"))
MAX_PORTFOLIO_DRAWDOWN = float(os.getenv("MAX_PORTFOLIO_DRAWDOWN", "0.20"))
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "0.08"))
MAX_POSITION_SIZE_PCT = float(os.getenv("MAX_POSITION_SIZE_PCT", "0.15"))
MAX_SECTOR_EXPOSURE = float(os.getenv("MAX_SECTOR_EXPOSURE", "0.30"))
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.75"))
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.065"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
TRAILING_STOP_PCT = float(os.getenv("TRAILING_STOP_PCT", "0.08"))
TIME_EXIT_DAYS = int(os.getenv("TIME_EXIT_DAYS", "45"))
PROFIT_TARGET_LEVELS = [0.08, 0.15, 0.25]
CONFIRMATION_THRESHOLD = 2
ENSEMBLE_THRESHOLD = 0.15
TREND_STRENGTH_THRESHOLD = 0.5
PROBABILITY_THRESHOLD = 0.52
BACKTEST_VALIDATION_THRESHOLD = 0.45
ADAPTIVE_THRESHOLDS = {
    'rsi_buy': 30,
    'rsi_sell': 70,
    'macd_signal': 0.5,
    'bb_upper': 2.0,
    'bb_lower': -2.0
}
TOP_N_RECOMMENDATIONS = 10
TRADE_LIMIT = 50
SIMULATION_DAYS = 252
WALK_FORWARD_ENABLED = os.getenv("WALK_FORWARD_ENABLED", "true").lower() == "true"
ENABLE_ADVANCED_TECH = os.getenv("ENABLE_ADVANCED_TECH", "true").lower() == "true"
NIFTY_50_STOCKS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "HINDUNILVR.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS", "MARUTI.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "HCLTECH.NS", "WIPRO.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "POWERGRID.NS", "NTPC.NS", "ONGC.NS", "COALINDIA.NS", "GRASIM.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "ADANIPORTS.NS", "SHREECEM.NS", "BAJAJ-AUTO.NS", "TITAN.NS", "HEROMOTOCO.NS", "DRREDDY.NS", "SUNPHARMA.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS", "INDUSINDBK.NS", "HDFCLIFE.NS", "SBILIFE.NS", "BRITANNIA.NS", "TECHM.NS", "EICHERMOT.NS", "BPCL.NS", "UPL.NS", "M&M.NS", "TATACONSUM.NS", "ASIANPAINT.NS", "PIDILITIND.NS", "NMDC.NS", "GAIL.NS", "VEDL.NS"]