#!/usr/bin/env python3
"""
Demo script to showcase the stock analysis system with sample data.
This demonstrates all features when real Yahoo Finance data is unavailable.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def generate_sample_stock_data(symbol: str, days: int = 252, trend: str = "bullish") -> pd.DataFrame:
    """
    Generate realistic sample stock data for demonstration.

    Args:
        symbol: Stock symbol
        days: Number of trading days
        trend: 'bullish', 'bearish', or 'neutral'

    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(hash(symbol) % 2**32)  # Consistent data per symbol

    # Base prices for different stocks
    base_prices = {
        "RELIANCE.NS": 2500,
        "TCS.NS": 3800,
        "INFY.NS": 1450,
        "HDFCBANK.NS": 1600,
    }

    base_price = base_prices.get(symbol, 1000)

    # Generate dates
    end_date = datetime.now()
    dates = pd.date_range(end=end_date, periods=days, freq='B')

    # Generate price series with trend
    returns = np.random.normal(0.001 if trend == "bullish" else -0.001 if trend == "bearish" else 0,
                                0.02, days)

    # Add some momentum patterns
    for i in range(10, days):
        if i % 20 == 0:  # Periodic momentum
            returns[i:i+10] += 0.005 if trend != "bearish" else -0.005

    prices = base_price * (1 + returns).cumprod()

    # Generate OHLCV data
    data = pd.DataFrame({
        'Open': prices * (1 + np.random.uniform(-0.01, 0.01, days)),
        'High': prices * (1 + np.random.uniform(0, 0.02, days)),
        'Low': prices * (1 + np.random.uniform(-0.02, 0, days)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, days),
    }, index=dates)

    # Ensure High is highest and Low is lowest
    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)

    return data


def run_demo_analysis():
    """Run demonstration analysis on sample stocks"""

    print("=" * 80)
    print("STOCK SWING TRADE RECOMMENDER - DEMONSTRATION")
    print("=" * 80)
    print()
    print("NOTE: Using sample data for demonstration due to Yahoo Finance limitations")
    print("      In production with proper network access, this fetches real-time data")
    print()

    # Import after setting up path
    from data.models import State
    from agents.technical_analysis import technical_analysis_agent
    from agents.risk_assessment import risk_assessment_agent
    from recommendation.final_recommendation import final_recommendation_agent

    # Stocks to analyze
    stocks = ["RELIANCE.NS", "TCS.NS"]

    # Generate sample data
    print(f"Generating sample data for {len(stocks)} stocks...")
    stock_data = {}
    for symbol in stocks:
        trend = "bullish" if symbol == "RELIANCE.NS" else "neutral"
        stock_data[symbol] = generate_sample_stock_data(symbol, days=252, trend=trend)
        print(f"  ✓ {symbol}: {len(stock_data[symbol])} days of data")

    print()

    # Create state as dict (TypedDict doesn't need instantiation)
    state = {
        'stock_data': stock_data,
        'technical_signals': {},
        'fundamental_analysis': {},
        'sentiment_scores': {},
        'macro_scores': {},
        'risk_metrics': {},
        'final_recommendation': {},
        'simulation_results': {},
        'performance_analysis': {},
        'buy_ranking': [],
        'top_buy_candidate': {},
        'failed_stocks': [],
        'real_time_data': {},
    }

    # Add mock fundamental and sentiment data FIRST
    print("Adding fundamental and sentiment data...")
    for symbol in stocks:
        state['fundamental_analysis'][symbol] = {
            'pe_ratio': 25.0 if symbol == "RELIANCE.NS" else 28.0,
            'pb_ratio': 2.5 if symbol == "RELIANCE.NS" else 8.0,
            'market_cap': 16800000000000 if symbol == "RELIANCE.NS" else 12500000000000,
            'score': 7.5 if symbol == "RELIANCE.NS" else 7.0,
        }

        state['sentiment_scores'][symbol] = {
            'overall_sentiment': 0.75 if symbol == "RELIANCE.NS" else 0.65,
            'news_count': 24 if symbol == "RELIANCE.NS" else 18,
            'score': 75 if symbol == "RELIANCE.NS" else 65,
        }

    state['macro_scores'] = {'composite_score': 7.0}
    print(f"  ✓ Data added for {len(stocks)} stocks")
    print()

    # Run technical analysis
    print("Running technical analysis...")
    state = technical_analysis_agent(state)
    print(f"  ✓ Technical analysis complete for {len(state.get('technical_signals', {}))} stocks")
    print()

    # Run risk assessment
    print("Running risk assessment...")
    state = risk_assessment_agent(state)
    print(f"  ✓ Risk assessment complete for {len(state.get('risk_metrics', {}))} stocks")
    print()

    # Run final recommendation
    print("Generating final recommendations...")
    state = final_recommendation_agent(state)
    print(f"  ✓ Recommendations generated")
    print()

    # Display results
    print("=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)
    print()

    for symbol in stocks:
        print(f"\n{'─' * 80}")
        print(f"STOCK: {symbol}")
        print('─' * 80)

        # Get recommendation
        rec = state['final_recommendation'].get(symbol, {})
        if not rec:
            print("  No recommendation available")
            continue

        # Display recommendation
        action = rec.get('action', 'HOLD')
        confidence = rec.get('confidence', 0.0)

        print(f"\n📊 RECOMMENDATION")
        print(f"   Action: {action}")
        print(f"   Confidence: {confidence:.1f}%")

        if 'entry_price' in rec:
            print(f"\n💰 PRICE LEVELS")
            print(f"   Entry: ₹{rec.get('entry_price', 0):.2f}")
            print(f"   Target: ₹{rec.get('target_price', 0):.2f} (+{rec.get('target_pct', 0):.1f}%)")
            print(f"   Stop Loss: ₹{rec.get('stop_loss', 0):.2f} ({rec.get('stop_loss_pct', 0):.1f}%)")

        if 'composite_score' in rec:
            print(f"\n📈 SCORES")
            print(f"   Composite Score: {rec.get('composite_score', 0):.1f}/10")
            print(f"   Technical: {rec.get('technical_score', 0):.1f}/10")
            print(f"   Fundamental: {rec.get('fundamental_score', 0):.1f}/10")
            print(f"   Sentiment: {rec.get('sentiment_score', 0):.1f}/10")

        # Technical signals
        if symbol in state['technical_signals']:
            tech = state['technical_signals'][symbol]
            print(f"\n📉 TECHNICAL INDICATORS")
            if 'rsi' in tech:
                print(f"   RSI: {tech['rsi']:.1f}")
            if 'macd_signal' in tech:
                print(f"   MACD: {tech['macd_signal']}")
            if 'trend' in tech:
                print(f"   Trend: {tech['trend']}")

        # Risk metrics
        if symbol in state['risk_metrics']:
            risk = state['risk_metrics'][symbol]
            print(f"\n⚠️  RISK METRICS")
            if 'volatility' in risk:
                print(f"   Volatility: {risk['volatility']*100:.1f}%")
            if 'sharpe_ratio' in risk:
                print(f"   Sharpe Ratio: {risk['sharpe_ratio']:.2f}")
            if 'position_size_pct' in risk:
                print(f"   Position Size: {risk['position_size_pct']:.1f}% of portfolio")

    # Multi-stock ranking
    if state.get('buy_ranking'):
        print(f"\n\n{'=' * 80}")
        print("BUY RECOMMENDATIONS RANKING")
        print('=' * 80)
        print()

        for i, rec in enumerate(state['buy_ranking'], 1):
            symbol = rec.get('symbol', 'Unknown')
            score = rec.get('score', 0)
            confidence = rec.get('confidence', 0)

            stars = '⭐' * min(5, int(score / 2))
            print(f"Rank {i}: {symbol} {stars}")
            print(f"  Score: {score:.1f}/10 | Confidence: {confidence:.1f}%")
            if 'target_pct' in rec:
                print(f"  Potential: +{rec['target_pct']:.1f}%")
            print()

        if state.get('top_buy_candidate'):
            top = state['top_buy_candidate']
            print(f"🏆 TOP CANDIDATE: {top.get('symbol', 'N/A')}")
            print(f"   This is the strongest opportunity based on multi-factor analysis")

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("💡 Key Features Demonstrated:")
    print("   ✓ Technical analysis with 40+ indicators")
    print("   ✓ Risk assessment and position sizing")
    print("   ✓ Multi-factor recommendation engine")
    print("   ✓ Composite scoring across all factors")
    print("   ✓ Multi-stock ranking and comparison")
    print()
    print("⚠️  Note: This demo uses sample data. In production:")
    print("   • Real-time data from Yahoo Finance")
    print("   • Live news sentiment analysis")
    print("   • Current macroeconomic indicators")
    print("   • AI-powered reasoning with Groq LLM")
    print("   • Historical backtesting on real data")
    print()
    print("🚀 The system is fully functional and ready for real analysis")
    print("   when Yahoo Finance access is available!")
    print()


if __name__ == "__main__":
    try:
        run_demo_analysis()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
