import sys
sys.path.insert(0, '..')
import pandas as pd
import numpy as np
from agents.technical_analysis import TechnicalAnalysis

def test_generate_ensemble_signal():
    # Mock indicators for RELIANCE.NS - oversold conditions for BUY
    indicators = {
        'RSI': 25,  # <30 buy
        'MACD': 0.1,
        'MACD_signal': 0.05,  # > signal buy
        'Close': 2440,
        'BB_lower': 2450,  # price < lower buy
        'BB_upper': 2550,
        'STOCH_K': 15,  # <20 buy
        'WilliamsR': -85,  # <-80 buy
        'CCI': -120  # <-100 buy
    }

    ta = TechnicalAnalysis()
    result = ta.generate_ensemble_signal(indicators)

    assert result['unified_signal'] == 'BUY'
    assert result['ensemble_score'] > 0.3
    assert abs(result['confidence'] - abs(result['ensemble_score'])) < 0.01
    print(f"Test passed: {result}")

if __name__ == "__main__":
    test_generate_ensemble_signal()