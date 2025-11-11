#!/usr/bin/env python3
"""
Direct test script for RELIANCE, INFY, and CIPLA stocks
"""

from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf


def calculate_rsi(prices, window=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    exp1 = prices.ewm(span=fast).mean()
    exp2 = prices.ewm(span=slow).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=signal).mean()
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram


def analyze_stock(symbol):
    """Analyze a single stock"""
    try:
        print(f"\n{'=' * 60}")
        print(f"Analyzing {symbol}")
        print(f"{'=' * 60}")

        # Fetch data
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1y")

        if data.empty:
            print(f"❌ No data available for {symbol}")
            return None

        # Basic info
        latest = data.iloc[-1]
        print(f"📈 Current Price: ₹{latest['Close']:.2f}")
        print(f"📊 Volume: {latest['Volume']:,}")
        print(
            f"📅 Data Period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}"
        )
        print(f"📏 Total Days: {len(data)}")

        # Technical Indicators
        rsi = calculate_rsi(data["Close"])
        macd, macd_signal, macd_histogram = calculate_macd(data["Close"])

        current_rsi = rsi.iloc[-1]
        current_macd = macd.iloc[-1]
        current_macd_signal = macd_signal.iloc[-1]
        current_macd_histogram = macd_histogram.iloc[-1]

        print(f"\n🔍 Technical Analysis:")
        print(f"  RSI (14): {current_rsi:.2f}")

        # RSI Analysis
        if current_rsi > 70:
            rsi_signal = "OVERBOUGHT 🔴"
        elif current_rsi < 30:
            rsi_signal = "OVERSOLD 🟢"
        else:
            rsi_signal = "NEUTRAL 🟡"
        print(f"  RSI Signal: {rsi_signal}")

        print(f"  MACD: {current_macd:.2f}")
        print(f"  MACD Signal: {current_macd_signal:.2f}")
        print(f"  MACD Histogram: {current_macd_histogram:.2f}")

        # MACD Analysis
        if current_macd > current_macd_signal and current_macd_histogram > 0:
            macd_signal = "BULLISH 🟢"
        elif current_macd < current_macd_signal and current_macd_histogram < 0:
            macd_signal = "BEARISH 🔴"
        else:
            macd_signal = "NEUTRAL 🟡"
        print(f"  MACD Signal: {macd_signal}")

        # Moving averages
        sma_20 = data["Close"].rolling(20).mean().iloc[-1]
        sma_50 = data["Close"].rolling(50).mean().iloc[-1]

        print(f"\n📈 Moving Averages:")
        print(f"  20-day SMA: ₹{sma_20:.2f}")
        print(f"  50-day SMA: ₹{sma_50:.2f}")
        print(f"  Current vs 20-SMA: {((latest['Close'] / sma_20 - 1) * 100):+.2f}%")
        print(f"  Current vs 50-SMA: {((latest['Close'] / sma_50 - 1) * 100):+.2f}%")

        # Price change
        prev_close = data["Close"].iloc[-2]
        daily_change = (latest["Close"] / prev_close - 1) * 100
        weekly_return = (
            ((latest["Close"] / data["Close"].iloc[-6] - 1) * 100)
            if len(data) >= 6
            else 0
        )
        monthly_return = (
            ((latest["Close"] / data["Close"].iloc[-21] - 1) * 100)
            if len(data) >= 21
            else 0
        )

        print(f"\n📊 Performance:")
        print(f"  Daily Change: {daily_change:+.2f}%")
        print(f"  1-Week Return: {weekly_return:+.2f}%")
        print(f"  1-Month Return: {monthly_return:+.2f}%")

        # Volatility
        returns = data["Close"].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
        print(f"  Annualized Volatility: {volatility:.2f}%")

        # Volume analysis
        avg_volume = data["Volume"].rolling(20).mean().iloc[-1]
        current_volume_ratio = latest["Volume"] / avg_volume

        print(f"\n📈 Volume Analysis:")
        print(f"  Current Volume: {latest['Volume']:,}")
        print(f"  20-day Avg Volume: {avg_volume:,.0f}")
        print(f"  Volume Ratio: {current_volume_ratio:.2f}x")

        if current_volume_ratio > 1.5:
            volume_signal = "HIGH VOLUME 📈"
        elif current_volume_ratio < 0.7:
            volume_signal = "LOW VOLUME 📉"
        else:
            volume_signal = "NORMAL VOLUME 📊"
        print(f"  Volume Signal: {volume_signal}")

        # Overall signal
        signals = []
        if current_rsi < 30:
            signals.append("RSI Oversold")
        elif current_rsi > 70:
            signals.append("RSI Overbought")

        if current_macd > current_macd_signal:
            signals.append("MACD Bullish")
        else:
            signals.append("MACD Bearish")

        if latest["Close"] > sma_20:
            signals.append("Above 20-SMA")
        else:
            signals.append("Below 20-SMA")

        print(f"\n🎯 Trading Signals:")
        for signal in signals:
            print(f"  • {signal}")

        return {
            "symbol": symbol,
            "current_price": latest["Close"],
            "rsi": current_rsi,
            "macd_signal": macd_signal,
            "daily_change": daily_change,
            "signals": signals,
        }

    except Exception as e:
        print(f"❌ Error analyzing {symbol}: {e}")
        return None


def main():
    print("🚀 Stock Analysis: RELIANCE, INFY, CIPLA")
    print("=" * 60)

    symbols = ["RELIANCE.NS", "INFY.NS", "CIPLA.NS"]
    results = []

    for symbol in symbols:
        result = analyze_stock(symbol)
        if result:
            results.append(result)

    # Summary
    print(f"\n{'=' * 60}")
    print("📋 SUMMARY")
    print(f"{'=' * 60}")

    for result in results:
        symbol = result["symbol"]
        price = result["current_price"]
        change = result["daily_change"]
        rsi = result["rsi"]

        print(f"\n{symbol.replace('.NS', '')}:")
        print(f"  Price: ₹{price:.2f} ({change:+.2f}%)")
        print(f"  RSI: {rsi:.1f}")
        print(f"  Key Signals: {', '.join(result['signals'][:2])}")

    # Overall market sentiment
    avg_rsi = np.mean([r["rsi"] for r in results])
    positive_changes = sum(1 for r in results if r["daily_change"] > 0)

    print(f"\n🌐 Market Sentiment:")
    print(f"  Average RSI: {avg_rsi:.1f}")
    print(f"  Stocks Up: {positive_changes}/{len(results)}")

    if avg_rsi < 40:
        market_sentiment = "OVERSOLD (Potentially Bullish) 🟢"
    elif avg_rsi > 60:
        market_sentiment = "OVERBOUGHT (Potentially Bearish) 🔴"
    else:
        market_sentiment = "NEUTRAL 🟡"

    print(f"  Overall Sentiment: {market_sentiment}")

    print(f"\n💡 Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
