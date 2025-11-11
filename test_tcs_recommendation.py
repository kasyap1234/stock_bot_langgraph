import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recommendation.final_recommendation import EnhancedRecommendationEngine, FactorType, FactorAnalysis
from data.models import State
import pandas as pd

def test_tcs_oversold():
    engine = EnhancedRecommendationEngine()
    
    # Simulate state for TCS.NS
    state = State()
    state["stock_data"] = {"TCS.NS": pd.DataFrame({
        'Date': pd.date_range(start='2024-01-01', periods=100, freq='D'),
        'open': [3500] * 100,
        'high': [3520] * 100,
        'low': [3480] * 100,
        'close': [3500] * 100,
        'volume': [1000000] * 100
    })}
    
    # Simulate oversold technical signals (RSI <30 → buy)
    state["technical_signals"] = {
        "TCS.NS": {
            "RSI_daily": "buy",  # Oversold
            "MACD_daily": "buy",
            "Stochastic_daily": "buy",
            "Bollinger_daily": "buy",
            "WilliamsR_daily": "buy"
        }
    }
    
    # Neutral other factors
    state["fundamental_analysis"] = {"TCS.NS": {"valuations": "neutral"}}
    state["sentiment_scores"] = {"TCS.NS": {"compound": 0.0}}
    state["risk_metrics"] = {"TCS.NS": {"risk_ok": True, "volatility": 0.2, "sharpe_ratio": 0.5}}
    state["macro_scores"] = {"composite": 0.0}
    state["ml_predictions"] = {"TCS.NS": {"prediction": 0.0}}
    state["nn_predictions"] = {"TCS.NS": {"prediction": 0.0}}
    state["simulation_results"] = {}
    state["backtest_results"] = {"sharpe_ratio": 0.5, "win_rate": 0.5, "max_drawdown": 0.2}
    
    try:
        factors = engine.analyze_factors("TCS.NS", state)
        market_conditions = engine.assess_market_conditions(factors)
        weights = engine.calculate_dynamic_weights(factors, market_conditions)
        decision = engine.synthesize_decision(factors, weights)
        
        print(f"Recommendation for TCS.NS (oversold simulation): {decision['action']}")
        print(f"Composite Score: {decision['composite_score']:.3f}")
        print(f"Confidence: {decision['confidence']:.3f}")
        print(f"Factors: {[f.factor_type.value for f in factors]}")
        print("Technical details:", factors[0].reasoning if factors else "No factors")
        
        if decision['action'] == 'BUY' and decision['composite_score'] > 0:
            print("SUCCESS: Properly evaluated oversold as BUY")
            return True
        else:
            print("FAIL: Did not recommend BUY for oversold conditions")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tcs_oversold()
    sys.exit(0 if success else 1)