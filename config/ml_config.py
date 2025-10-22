import os

ML_MAPPINGS = [
    ("ML_MODEL_PARAMS", "ml_model_params"),
    ("ADAPTIVE_THRESHOLDS", "adaptive_thresholds"),
    ("GRID_SEARCH_PARAMS", "grid_search_params"),
    ("MONTE_CARLO_SIMULATIONS", "monte_carlo_simulations"),
    ("MONTE_CARLO_HORIZON", "monte_carlo_horizon"),
    ("VPVR_BINS", "vpvr_bins"),
    ("VISIBLE_RANGE", "visible_range"),
    ("VOLUME_PERCENTILE", "volume_percentile"),
    ("HA_CONFLUENCE_BARS", "ha_confluence_bars"),
    ("HA_TFS", "ha_tfs"),
    ("GARCH_P", "garch_p"),
    ("GARCH_Q", "garch_q"),
    ("FORECAST_HORIZON", "forecast_horizon"),
    ("HARMONIC_PATTERNS", "harmonic_patterns"),
    ("TOLERANCE", "tolerance"),
    ("LOOKBACK", "lookback"),
    ("MIN_CONFIDENCE", "min_confidence"),
    ("HMM_STATES", "hmm_states"),
    ("HMM_ITER", "hmm_iter"),
    ("LSTM_EPOCHS", "lstm_epochs"),
    ("LSTM_BATCH", "lstm_batch"),
    ("LSTM_WINDOW", "lstm_window"),
    ("LSTM_FEATURES", "lstm_features"),
    ("MC_VA_PATHS", "mc_va_paths"),
    ("VA_CONFIDENCE", "va_confidence"),
    ("STRESS_SCENARIOS", "stress_scenarios"),
]

ML_MODEL_PARAMS = {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'random_state': 42}
ADAPTIVE_THRESHOLDS = {'high_volatility': 0.05, 'medium_volatility': 0.02, 'low_volatility': 0.01}
GRID_SEARCH_PARAMS = {'rsi_period': [9, 14, 21], 'macd_fast': [8, 12, 26], 'macd_slow': [17, 26, 50], 'stoch_k': [5, 14, 21], 'stoch_d': [3, 5, 9]}
MONTE_CARLO_SIMULATIONS = 1000
MONTE_CARLO_HORIZON = 252
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