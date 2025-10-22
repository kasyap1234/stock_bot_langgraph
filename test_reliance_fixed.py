import pandas as pd
import numpy as np
from unittest.mock import patch
from agents.technical_analysis import _calculate_technical_indicators_basic, _calculate_technical_indicators_talib, TALIB_AVAILABLE

# Mock data for RELIANCE.NS (simulate 1y daily data)
dates = pd.date_range(start='2024-01-01', periods=252, freq='B')  # Business days
np.random.seed(42)
close_prices = 2500 + np.cumsum(np.random.randn(252) * 20)  # Around Reliance price ~2500 INR
data = pd.DataFrame({
    'Open': close_prices * (1 + np.random.randn(252) * 0.01),
    'High': close_prices * (1 + np.abs(np.random.randn(252) * 0.02)),
    'Low': close_prices * (1 - np.abs(np.random.randn(252) * 0.02)),
    'Close': close_prices,
    'Volume': np.random.randint(1000000, 5000000, 252)
}, index=dates)

print(f"Mock data for RELIANCE.NS: shape {data.shape}")
print(f"Last Close: {data['Close'].iloc[-1]:.2f}")
print(f"NaN in Close: {data['Close'].isna().sum()}")
print(f"Inf in Close: {np.isinf(data['Close']).sum()}")

symbol = "RELIANCE.NS"

# Test TA-Lib path (if available)
if TALIB_AVAILABLE:
    try:
        signals_talib = _calculate_technical_indicators_talib(data, symbol=symbol)
        print("TA-Lib signals:", signals_talib)
        # Check RSI
        if 'RSI' in signals_talib:
            print(f"RSI signal: {signals_talib['RSI']}")
    except Exception as e:
        print(f"TA-Lib path error: {e}")
else:
    print("TA-Lib not available, skipping TA-Lib path")

# Force basic path
with patch('agents.technical_analysis.TALIB_AVAILABLE', False):
    try:
        signals_basic = _calculate_technical_indicators_basic(data, symbol=symbol)
        print("Basic signals:", signals_basic)
        # Check RSI
        if 'RSI' in signals_basic:
            print(f"RSI signal: {signals_basic['RSI']}")
        # Manual RSI verification
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-9)
        rsi_manual = 100 - (100 / (1 + rs))
        latest_rsi = rsi_manual.iloc[-1]
        print(f"Manual RSI latest: {latest_rsi:.2f} (finite: {np.isfinite(latest_rsi)}, range: 0-100: {0 <= latest_rsi <= 100})")
        # Check for NaN/inf in manual RSI
        print(f"NaN in manual RSI: {rsi_manual.isna().sum()}, Inf: {np.isinf(rsi_manual).sum()}")
    except Exception as e:
        print(f"Basic path error: {e}")

print("Test completed.")