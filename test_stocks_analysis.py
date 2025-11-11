#!/usr/bin/env python3
"""
Test script to run enhanced technical analysis on Indian stocks
"""

import json
import pandas as pd
from datetime import datetime
from agents.enhanced_technical_analysis import EnhancedTechnicalAnalysisEngine
from data.apis import UnifiedDataFetcher

def load_stock_data(symbol):
    """Load stock data from cache or fetch fresh"""
    cache_file = f"data/cache/{symbol}_1y.json"
    try:
        with open(cache_file, 'r') as f:
            data = json.load(f)

        # Handle different cache formats
        if isinstance(data, dict) and 'data' in data:
            df = pd.DataFrame(data['data'])
        else:
            df = pd.DataFrame(data)

        df['Date'] = pd.to_datetime(df['date'])
        df.set_index('Date', inplace=True)

        # Rename columns to match expected format
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })

        return df
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        print(f"Cache file not found or corrupted for {symbol}, fetching fresh data...")
        try:
            fetcher = UnifiedDataFetcher(symbol)
            historical_data = fetcher.get_historical_data(period="1y", interval="1d")

            # Convert to DataFrame
            df = pd.DataFrame([{
                'Date': item['date'],
                'Open': item['open'],
                'High': item['high'],
                'Low': item['low'],
                'Close': item['close'],
                'Volume': item['volume']
            } for item in historical_data])

            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

            return df
        except Exception as e:
            print(f"Failed to fetch data for {symbol}: {e}")
            return None

def analyze_stock(symbol):
    """Analyze a single stock"""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {symbol}")
    print(f"{'='*60}")

    # Load data
    df = load_stock_data(symbol)
    if df is None or len(df) < 50:
        print(f"Insufficient data for {symbol}")
        return

    print(f"Data loaded: {len(df)} days from {df.index.min().date()} to {df.index.max().date()}")
    print(f"Price range: ₹{df['Close'].min():.2f} - ₹{df['Close'].max():.2f}")

    # Initialize analysis engine
    engine = EnhancedTechnicalAnalysisEngine()

    # Run analysis
    try:
        result = engine.analyze_with_quality_control(df, symbol)

        # Print results
        print("\n📊 MARKET CONTEXT:")
        market_ctx = result.get('market_context', {})
        print(f"  • Volatility: {market_ctx.get('volatility', 'N/A'):.4f}")
        print(f"  • Trend Strength: {market_ctx.get('trend_strength', 'N/A'):.4f}")
        print(f"  • Market Regime: {market_ctx.get('regime', 'unknown')}")

        print("\n📈 SIGNALS GENERATED:")
        signals = result.get('signals', {})
        if signals:
            for indicator, signal in signals.items():
                direction = signal.direction.upper()
                strength = signal.strength
                confidence = signal.confidence
                emoji = "🟢" if direction == "BUY" else "🔴" if direction == "SELL" else "🟡"
                print(f"  {emoji} {indicator}: {direction} (Strength: {strength:.2f}, Confidence: {confidence:.2f})")
        else:
            print("  No signals generated")

        print("\n⚙️ QUALITY METRICS:")
        quality = result.get('quality_metrics', {})
        print(f"  • Overall Quality: {quality.get('overall_quality', 'N/A'):.2f}")
        print(f"  • Signal Count: {quality.get('signal_count', 0)}")
        print(f"  • High Quality Signals: {quality.get('high_quality_signals', 0)}")

        print("\n🎯 RECOMMENDATION:")
        # Simple recommendation logic based on signals
        buy_signals = sum(1 for s in signals.values() if s.direction == 'buy')
        sell_signals = sum(1 for s in signals.values() if s.direction == 'sell')

        if buy_signals > sell_signals:
            print("  🟢 BULLISH: More buy signals than sell signals")
        elif sell_signals > buy_signals:
            print("  🔴 BEARISH: More sell signals than buy signals")
        else:
            print("  🟡 NEUTRAL: Balanced buy/sell signals")

    except Exception as e:
        print(f"❌ Error analyzing {symbol}: {e}")

def main():
    """Main function"""
    print("🚀 STOCK TECHNICAL ANALYSIS TEST")
    print("Testing Enhanced Technical Analysis Engine on Indian Stocks")

    # Stocks to analyze
    stocks = ['RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS']  # Using INFY instead of ICICI since we don't have ICICI data

    for stock in stocks:
        analyze_stock(stock)

    print(f"\n{'='*60}")
    print("✅ ANALYSIS COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()