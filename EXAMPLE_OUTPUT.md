# Example Stock Analysis Output

This document shows example outputs from the Stock Swing Trade Recommender system for different scenarios.

## Table of Contents
- [Single Stock Analysis](#single-stock-analysis)
- [Multi-Stock Ranking](#multi-stock-ranking)
- [Analysis with Backtesting](#analysis-with-backtesting)
- [Understanding the Scores](#understanding-the-scores)

---

## Single Stock Analysis

Command:
```bash
uv run main.py --ticker RELIANCE.NS
```

### Sample Output:

```
================================================================================
                    STOCK ANALYSIS STARTING
================================================================================

Analyzing: RELIANCE.NS
Time: 2024-01-15 10:30:45 IST

--------------------------------------------------------------------------------
[1/6] FETCHING DATA
--------------------------------------------------------------------------------
✓ Successfully fetched data for RELIANCE.NS
  - Historical data: 252 trading days
  - Latest price: ₹2,485.30
  - Volume: 8,234,567 shares

--------------------------------------------------------------------------------
[2/6] TECHNICAL ANALYSIS
--------------------------------------------------------------------------------

Trend Analysis:
  • Primary Trend: BULLISH (ADX: 28.5 - Strong trend)
  • Price vs MA50: +3.2% (Bullish)
  • Price vs MA200: +8.7% (Strong Bullish)
  • Ichimoku Cloud: Price above cloud (Bullish)

Momentum Indicators:
  • RSI(14): 45.3 - Neutral (Not overbought/oversold)
  • MACD: Bullish crossover detected 2 days ago
  • Stochastic: 42.5 - Neutral zone
  • Williams %R: -58.2 - Neutral

Volatility & Support/Resistance:
  • ATR(14): 82.5 (Moderate volatility)
  • Bollinger Bands: Middle band (neutral positioning)
  • Support Levels: ₹2,450, ₹2,420, ₹2,380
  • Resistance Levels: ₹2,550, ₹2,620, ₹2,680

Volume Analysis:
  • Volume Trend: Above average (+15%)
  • OBV: Rising (Accumulation)
  • VWAP: ₹2,478 (Price above VWAP - Bullish)

ML-Based Signals:
  • LSTM Prediction: 5-day target: ₹2,587 (+4.1%)
  • HMM Regime: TRENDING (Confidence: 78%)
  • Random Forest Signal: BUY (Probability: 72%)

Multi-Timeframe Analysis:
  • Daily: Bullish (Score: 7.5/10)
  • Weekly: Bullish (Score: 8.0/10)
  • Monthly: Neutral to Bullish (Score: 6.5/10)

Technical Score: 8.2/10

--------------------------------------------------------------------------------
[3/6] FUNDAMENTAL ANALYSIS
--------------------------------------------------------------------------------

Valuation Metrics:
  • P/E Ratio: 24.3 (Sector avg: 26.1) ✓
  • P/B Ratio: 2.8 (Sector avg: 3.2) ✓
  • Market Cap: ₹16.8 trillion
  • Dividend Yield: 0.4%

Financial Health:
  • Revenue Growth: +12.3% YoY
  • Profit Margin: 8.7%
  • Earnings Per Share: ₹102.3
  • Debt to Equity: 0.45 (Healthy)

Earnings Trend:
  • Last Quarter: Beat estimates by 5%
  • Next Earnings: March 2024 (expected)
  • Earnings Growth: Positive trend

Fundamental Score: 7.5/10

--------------------------------------------------------------------------------
[4/6] SENTIMENT ANALYSIS
--------------------------------------------------------------------------------

News Sentiment (Last 7 days):
  • Total Articles: 24
  • Positive: 15 (62.5%)
  • Neutral: 7 (29.2%)
  • Negative: 2 (8.3%)
  • Overall Score: 0.72 (Positive)

Recent Headlines:
  ✓ "Reliance announces new expansion in retail sector"
  ✓ "JIO subscriber base crosses 450 million"
  ✓ "Analysts upgrade target price to ₹2,800"
  ⚠ "Regulatory concerns in telecom sector"

Market Sentiment:
  • Analyst Recommendations: 18 BUY, 5 HOLD, 2 SELL
  • Price Target: ₹2,745 (avg of 25 analysts)
  • Institutional Holdings: 68% (Increasing)

Sentiment Score: 72/100

--------------------------------------------------------------------------------
[5/6] RISK ASSESSMENT
--------------------------------------------------------------------------------

Volatility Metrics:
  • Historical Volatility: 18.2% (annualized)
  • Implied Volatility: 19.5%
  • Volatility Rank: Medium

Risk-Adjusted Returns:
  • Sharpe Ratio (6M): 1.8 (Good)
  • Sortino Ratio: 2.4 (Excellent)
  • Beta: 0.95 (Market-like volatility)

Risk Measures:
  • Value at Risk (95%): ₹124/share or 5.0%
  • Max Drawdown (6M): -12.3%
  • Recovery Time: 18 days (average)

Position Sizing:
  • Recommended: 4-5% of portfolio
  • Stop Loss: ₹2,420 (2.6% below entry)
  • Risk per Trade: 1% of portfolio

Risk Score: Medium (Acceptable for swing trading)

--------------------------------------------------------------------------------
[6/6] MACRO ECONOMIC FACTORS
--------------------------------------------------------------------------------

Economic Indicators:
  • GDP Growth: 7.2% (Strong)
  • Inflation (CPI): 5.4% (Moderate)
  • Interest Rate: 6.5% (Stable)
  • FII Activity: Net buying (+₹2,340 cr this week)

Market Environment:
  • Nifty 50 Trend: Bullish (+8% YTD)
  • Sector Performance: Energy sector outperforming
  • Global Markets: Positive (US markets +2% this week)

Macro Score: 7.0/10

================================================================================
                    FINAL RECOMMENDATION
================================================================================

┌─────────────────────────────────────────────────────────────────────────┐
│                         RECOMMENDATION: BUY                             │
│                      CONFIDENCE: 87.5% (HIGH)                          │
└─────────────────────────────────────────────────────────────────────────┘

Entry Details:
  Entry Price Range: ₹2,475 - ₹2,495
  Current Price: ₹2,485.30
  Target Price: ₹2,650.00 (+6.6%)
  Stop Loss: ₹2,420.00 (-2.6%)

Trade Setup:
  Risk/Reward Ratio: 1:2.5 (Excellent)
  Position Size: 4-5% of portfolio
  Time Horizon: 7-10 trading days
  Strategy: Momentum + Breakout

Composite Score Breakdown:
  ┌──────────────────────────────────┬───────┬─────────┐
  │ Factor                           │Weight │  Score  │
  ├──────────────────────────────────┼───────┼─────────┤
  │ Technical Analysis               │  30%  │  8.2/10 │
  │ Sentiment Analysis               │  20%  │  7.2/10 │
  │ Fundamental Analysis             │  15%  │  7.5/10 │
  │ Risk Assessment                  │  15%  │  7.8/10 │
  │ Macro Economic Factors           │  10%  │  7.0/10 │
  │ ML Predictions                   │   5%  │  7.8/10 │
  │ Market Conditions                │   5%  │  8.0/10 │
  ├──────────────────────────────────┼───────┼─────────┤
  │ OVERALL COMPOSITE SCORE          │ 100%  │  7.8/10 │
  └──────────────────────────────────┴───────┴─────────┘

Market Conditions Assessment:
  • Volatility Regime: NORMAL (Not extreme)
  • Trend Strength: STRONG (ADX > 25)
  • Market Sentiment: POSITIVE (Risk-on environment)
  • Risk Environment: MODERATE (Acceptable)

LLM-Enhanced Reasoning:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RELIANCE.NS presents a compelling swing trade opportunity based on multiple
confluent factors:

1. TECHNICAL SETUP: The stock shows strong technical momentum with a bullish
   MACD crossover occurring just 2 days ago, combined with price trading above
   key moving averages (MA50 and MA200). The ADX reading of 28.5 confirms
   strong trending behavior, while RSI at 45.3 provides ample room for upward
   movement without reaching overbought territory.

2. VOLUME CONFIRMATION: Above-average volume (+15%) and rising On-Balance
   Volume (OBV) indicate institutional accumulation, validating the bullish
   price action. The price trading above VWAP further confirms buyer strength.

3. SUPPORT/RESISTANCE STRUCTURE: The stock has established solid support at
   ₹2,450 with immediate resistance at ₹2,550. A breakout above ₹2,550 could
   trigger momentum towards the ₹2,650 target level.

4. FUNDAMENTAL BACKING: Attractive valuation with P/E of 24.3 (below sector
   average of 26.1) combined with strong revenue growth (+12.3% YoY) and
   positive earnings surprises provides fundamental support for the technical
   setup.

5. POSITIVE SENTIMENT: 62.5% positive news coverage with multiple analyst
   upgrades and a consensus target price of ₹2,745 suggests strong market
   conviction.

6. MACRO TAILWINDS: Favorable macroeconomic environment with robust GDP growth
   (7.2%) and net FII buying supports risk-on sentiment in Indian equities.

ENTRY STRATEGY:
Consider entering on any dip to ₹2,475-₹2,480 level or on a breakout above
₹2,500 with strong volume. The tight stop loss at ₹2,420 provides good risk
management with a favorable 1:2.5 risk/reward ratio.

TIME HORIZON:
This setup targets 5-7% gains over 7-10 trading days, aligning with swing
trading parameters. Monitor daily for any breakdown below ₹2,450 support.

RISK FACTORS TO WATCH:
• Broader market correction could impact momentum
• Regulatory news in telecom sector
• Crude oil price volatility (key input cost)
• Technical invalidation below ₹2,420 support

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Key Levels to Watch:
  📈 Targets: ₹2,550 (first), ₹2,650 (main), ₹2,750 (extended)
  📉 Stop Loss: ₹2,420 (firm)
  ⚠️  Critical Support: ₹2,450 (watch closely)

Action Items:
  1. ✓ Entry can be initiated at current levels
  2. ⚠ Set stop loss at ₹2,420 (mandatory)
  3. 📊 Monitor volume at ₹2,550 resistance
  4. 🎯 Book 50% profits at ₹2,600, let 50% run to ₹2,650
  5. 📅 Review position in 7 days or on ±5% move

================================================================================
                         ANALYSIS COMPLETE
================================================================================

Analysis Time: 45.3 seconds
Timestamp: 2024-01-15 10:31:30 IST

⚠️  DISCLAIMER: This is an AI-generated recommendation for educational and
    research purposes only. Not financial advice. Always do your own research
    and consult a financial advisor before making investment decisions.
```

---

## Multi-Stock Ranking

Command:
```bash
uv run main.py --tickers RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS
```

### Sample Output:

```
================================================================================
                    MULTI-STOCK ANALYSIS & RANKING
================================================================================

Analyzing 4 stocks: RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS

[Processing in parallel...]

✓ RELIANCE.NS - Analysis complete
✓ TCS.NS - Analysis complete
✓ INFY.NS - Analysis complete
✓ HDFCBANK.NS - Analysis complete

================================================================================
                         BUY RECOMMENDATIONS RANKING
================================================================================

Rank 1: RELIANCE.NS ⭐⭐⭐⭐⭐
  Action: BUY | Confidence: 87.5%
  Entry: ₹2,485 | Target: ₹2,650 (+6.6%) | Stop: ₹2,420
  Score: 7.8/10
  Rationale: Strong technical setup, bullish MACD crossover, positive sentiment

Rank 2: TCS.NS ⭐⭐⭐⭐
  Action: BUY | Confidence: 78.2%
  Entry: ₹3,825 | Target: ₹3,980 (+4.0%) | Stop: ₹3,760
  Score: 7.2/10
  Rationale: IT sector strength, client addition news, healthy fundamentals

Rank 3: HDFCBANK.NS ⭐⭐⭐
  Action: HOLD | Confidence: 62.5%
  Entry: N/A | Current: ₹1,585
  Score: 6.5/10
  Rationale: Consolidating, wait for breakout above ₹1,620

Rank 4: INFY.NS ⭐⭐⭐
  Action: HOLD | Confidence: 58.3%
  Entry: N/A | Current: ₹1,445
  Score: 6.2/10
  Rationale: Weak momentum, below MA50, need confirmation

================================================================================
                         TOP BUY CANDIDATE
================================================================================

🏆 RELIANCE.NS is the strongest buy opportunity

Reasons:
  1. Highest composite score (7.8/10)
  2. Best technical setup with multiple confirmations
  3. Strong institutional buying
  4. Attractive risk/reward ratio (1:2.5)
  5. Positive news flow and sentiment

Suggested Allocation:
  If allocating across multiple stocks:
  • RELIANCE.NS: 50% of capital (primary position)
  • TCS.NS: 30% of capital (secondary position)
  • Hold remaining 20% for other opportunities
```

---

## Analysis with Backtesting

Command:
```bash
uv run main.py --ticker RELIANCE.NS --backtest
```

### Additional Output Section:

```
================================================================================
                    BACKTESTING RESULTS
================================================================================

Strategy: Enhanced Momentum + Mean Reversion Ensemble
Period: Last 6 months (126 trading days)
Initial Capital: ₹1,00,000

Performance Summary:
┌─────────────────────────────────────────┬──────────────┐
│ Metric                                  │    Value     │
├─────────────────────────────────────────┼──────────────┤
│ Total Return                            │   +18.5%     │
│ Annualized Return                       │   +39.2%     │
│ Buy & Hold Return                       │   +12.3%     │
│ Alpha (vs Buy & Hold)                   │   +6.2%      │
├─────────────────────────────────────────┼──────────────┤
│ Total Trades                            │      28      │
│ Winning Trades                          │      19      │
│ Losing Trades                           │       9      │
│ Win Rate                                │    67.9%     │
├─────────────────────────────────────────┼──────────────┤
│ Avg Profit per Trade                    │    +2.8%     │
│ Avg Loss per Trade                      │    -1.2%     │
│ Profit Factor                           │     2.3      │
│ Expectancy                              │    +1.6%     │
├─────────────────────────────────────────┼──────────────┤
│ Maximum Drawdown                        │    -8.2%     │
│ Avg Drawdown                            │    -3.1%     │
│ Recovery Time (Avg)                     │   12 days    │
├─────────────────────────────────────────┼──────────────┤
│ Sharpe Ratio                            │     2.1      │
│ Sortino Ratio                           │     3.2      │
│ Calmar Ratio                            │     4.8      │
├─────────────────────────────────────────┼──────────────┤
│ Final Portfolio Value                   │ ₹1,18,500    │
│ Transaction Costs                       │    -₹840     │
│ Net Profit                              │  +₹18,500    │
└─────────────────────────────────────────┴──────────────┘

Trade Analysis:
  Best Trade: +8.2% (₹6,560 profit)
  Worst Trade: -3.1% (₹2,480 loss)
  Avg Holding Period: 6.2 days
  Longest Winning Streak: 7 trades
  Longest Losing Streak: 3 trades

Risk Metrics:
  Maximum Position Size: ₹42,000 (42% of capital)
  Avg Position Size: ₹35,000 (35% of capital)
  Portfolio at Risk (VaR 95%): ₹5,925
  Leverage Used: None

Strategy Rating: ⭐⭐⭐⭐⭐ (Excellent)

Reasoning:
This strategy demonstrates strong performance with:
• Sharpe ratio of 2.1 (excellent risk-adjusted returns)
• Win rate near 68% (robust signal quality)
• Profit factor of 2.3 (winners 2.3x larger than losers)
• Maximum drawdown of only 8.2% (good risk management)
• Consistent outperformance vs buy-and-hold (+6.2% alpha)

The backtest validates the current BUY recommendation with high confidence.

Monthly Breakdown:
  July 2023:    +3.2%  ✓
  August 2023:  +2.1%  ✓
  September 2023: -1.5%  ✗
  October 2023: +4.8%  ✓
  November 2023: +5.3%  ✓
  December 2023: +3.8%  ✓
```

---

## Understanding the Scores

### Technical Score (0-10)
- **8-10**: Very strong technical setup, multiple confirmations
- **6-8**: Good technical setup, some confirmations
- **4-6**: Neutral/mixed technical signals
- **0-4**: Weak technical setup, bearish indicators

### Fundamental Score (0-10)
- **8-10**: Excellent valuation and growth metrics
- **6-8**: Good fundamentals, fairly valued
- **4-6**: Average fundamentals
- **0-4**: Poor fundamentals or expensive valuation

### Sentiment Score (0-100)
- **70-100**: Very positive sentiment
- **50-70**: Positive sentiment
- **30-50**: Neutral sentiment
- **0-30**: Negative sentiment

### Confidence Level
- **90-100%**: Very high confidence (strong signals across all factors)
- **75-90%**: High confidence (most factors align)
- **60-75%**: Medium confidence (mixed signals)
- **Below 60%**: Low confidence (conflicting signals)

### Risk Rating
- **Low**: Volatility < 15%, stable price action
- **Medium**: Volatility 15-25%, normal for swing trading
- **High**: Volatility > 25%, increased risk

### Recommendation Actions
- **BUY**: Positive outlook, favorable entry point
- **HOLD**: Wait for better entry or take no action
- **SELL**: Exit positions, bearish outlook

---

## Notes

1. **Timeframe**: All recommendations are for swing trading (5-15 day holding periods)
2. **Stop Losses**: Always mandatory, typically 2-5% below entry
3. **Position Sizing**: Usually 3-5% of portfolio per trade
4. **Risk/Reward**: Minimum 1:2 ratio preferred
5. **Updates**: Recommendations valid for current market conditions only

For real-time analysis, run the command again to get updated recommendations.
