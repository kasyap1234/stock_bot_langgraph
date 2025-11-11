import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from agents.technical_analysis import _calculate_technical_indicators_with_retry, _calculate_technical_indicators_talib, _calculate_technical_indicators_basic, GARCHForecaster, HMMRegimeDetector, LSTMPredictor

symbol = "RELIANCE.NS"
data = yf.download(symbol, period="1y", progress=False, auto_adjust=True)
print(f"Data fetched for {symbol}: shape {data.shape}")

if data.empty:
    raise ValueError("Failed to download real data for RELIANCE.NS - cannot proceed with analysis using sample data")

print(f"Data shape: {data.shape}")
print(f"Last Close: {data['Close'].iloc[-1]:.2f}")
print(f"NaN in Close: {data['Close'].isna().sum()}")
print(f"Inf in Close: {np.isinf(data['Close']).sum()}")

# Test both paths
signals_talib = _calculate_technical_indicators_talib(data, symbol=symbol)
print("TA-Lib signals:", signals_talib)

signals_basic = _calculate_technical_indicators_basic(data, symbol=symbol)
print("Basic signals:", signals_basic)

# Manual RSI check
delta = data['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss.replace(0, 1e-9)
rsi_manual = 100 - (100 / (1 + rs))
latest_rsi = rsi_manual.iloc[-1]
print(f"Manual RSI latest: {latest_rsi:.2f} (valid: {np.isfinite(latest_rsi)})")

# Test advanced indicators
print("\n--- Testing Advanced Indicators ---")

# GARCH
try:
    garch = GARCHForecaster()
    returns = data['Close'].pct_change().dropna()
    fitted = garch.fit_garch_model(returns)
    vol = garch.forecast_volatility(returns)
    print(f"GARCH fitted: {fitted}, volatility: {vol:.4f} (valid: {np.isfinite(vol)})")
except Exception as e:
    print(f"GARCH error: {e}")

# HMM
try:
    hmm = HMMRegimeDetector()
    regime = hmm.get_hmm_signal(data)
    print(f"HMM regime: {regime}")
except Exception as e:
    print(f"HMM error: {e}")

# LSTM
try:
    lstm = LSTMPredictor()
    basic_signals = signals_basic
    trained = lstm.train_model(data, basic_signals, symbol)
    pred, conf = lstm.predict_signal(data, basic_signals)
    print(f"LSTM trained: {trained}, prediction: {pred}, confidence: {conf:.2f}")
except Exception as e:
    print(f"LSTM error: {e}")