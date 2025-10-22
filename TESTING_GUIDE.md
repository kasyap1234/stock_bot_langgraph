# 📘 Stock Bot Testing Guide - Simple & Easy

## Quick Start (3 Steps)

### Step 1: Open Terminal
```bash
cd /Users/kasyapdharanikota/Desktop/stock_bot_langgraph
```

### Step 2: Run Analysis
```bash
./venv/bin/python3 main.py --ticker RELIANCE.NS
```

### Step 3: Read Results
Look for the **"Top 10 Trading Recommendations"** section at the end!

---

## 📊 Test Results Summary (22 Stocks Tested)

### Banking Sector (ALL BUY) 🟢
| Stock | Recommendation | Confidence | Notes |
|-------|----------------|------------|-------|
| **ICICIBANK.NS** | **BUY** | **87.5%** | Strong signal |
| **AXISBANK.NS** | **BUY** | **79.3%** | Good signal |
| **SBIN.NS** | **BUY** | **76.9%** | Good signal |
| **KOTAKBANK.NS** | **BUY** | **72.6%** | Moderate signal |
| **HDFCBANK.NS** | **BUY** | **89.1%** | Very strong signal |

**Sector Insight**: Banking sector showing strong BUY signals across the board!

---

### IT Sector (ALL SELL) 🔴
| Stock | Recommendation | Confidence | Notes |
|-------|----------------|------------|-------|
| **TCS.NS** | **SELL** | **81.0%** | Strong sell signal |
| **INFY.NS** | **SELL** | **89.2%** | Very strong sell |
| **TECHM.NS** | **SELL** | **88.1%** | Very strong sell |
| **WIPRO.NS** | **SELL** | **81.3%** | Strong sell signal |
| **HCLTECH.NS** | **SELL** | **79.0%** | Strong sell signal |

**Sector Insight**: IT sector under pressure - avoid or exit positions!

---

### FMCG/Consumer Sector (MIXED) 🟡
| Stock | Recommendation | Confidence | Notes |
|-------|----------------|------------|-------|
| **BRITANNIA.NS** | **BUY** | **85.6%** | Strong buy signal |
| **ITC.NS** | **SELL** | **84.7%** | Strong sell signal |
| **NESTLEIND.NS** | **SELL** | **90.6%** | Very strong sell |
| **HINDUNILVR.NS** | **HOLD** | **78.6%** | Wait for clarity |

**Sector Insight**: FMCG sector mixed - selective opportunities only!

---

### Auto & Pharma (MIXED) 🟡
| Stock | Recommendation | Confidence | Notes |
|-------|----------------|------------|-------|
| **MARUTI.NS** | **BUY** | **74.5%** | Moderate buy |
| **TATAMOTORS.NS** | **SELL** | **97.1%** | ⚠️ VERY STRONG SELL |
| **SUNPHARMA.NS** | **BUY** | **86.0%** | Strong buy |
| **DRREDDY.NS** | **BUY** | **85.0%** | Strong buy |

**Sector Insight**: Pharma strong, Auto mixed (avoid Tata Motors!)

---

### Energy Sector
| Stock | Recommendation | Confidence | Notes |
|-------|----------------|------------|-------|
| **RELIANCE.NS** | **BUY** | **76.5%** | Moderate buy |

---

## 🎯 Overall Market Insights

### Sector Performance:
1. **🟢 STRONGEST**: Banking (avg 81% confidence BUY)
2. **🟢 STRONG**: Pharma (avg 85.5% confidence BUY)
3. **🟡 MIXED**: FMCG, Auto
4. **🔴 WEAKEST**: IT (avg 84% confidence SELL)

### Key Findings:
- ✅ **Banking sector**: All 5 banks show BUY - sector rotation happening
- ❌ **IT sector**: All 5 IT companies show SELL - avoid this sector
- ⚠️ **TATAMOTORS.NS**: Highest confidence SELL (97.1%) - stay away!
- ✅ **Pharma**: Strong performance, good entry points

---

## 🚀 How to Test Stocks

### Basic Commands

#### Test Single Stock
```bash
./venv/bin/python3 main.py --ticker HDFCBANK.NS
```

#### Test Multiple Stocks (Comma-separated)
```bash
./venv/bin/python3 main.py --tickers HDFCBANK.NS,TCS.NS,RELIANCE.NS
```

#### Test Different Time Periods
```bash
# 1 year (default)
./venv/bin/python3 main.py --ticker RELIANCE.NS --period 1y

# 6 months
./venv/bin/python3 main.py --ticker RELIANCE.NS --period 6mo

# 2 years
./venv/bin/python3 main.py --ticker RELIANCE.NS --period 2y
```

#### Test with Backtesting
```bash
./venv/bin/python3 main.py --ticker HDFCBANK.NS --backtest
```

#### Test Entire NIFTY 50 (Takes 10-15 minutes)
```bash
./venv/bin/python3 main.py --nifty50
```

---

## 📖 Understanding Results

### What You'll See:

```
Top 10 Trading Recommendations (ranked by confidence):
==================================================
HDFCBANK.NS: BUY (Confidence: 89.1%)
  LLM Reasoning: <explanation of why>
```

### Key Metrics to Check:

#### 1. **Action** (BUY/SELL/HOLD)
- **BUY**: Bot recommends buying this stock
- **SELL**: Bot recommends selling or avoiding
- **HOLD**: Wait for clearer signals

#### 2. **Confidence** (%)
- **90%+ (Very High)**: Extremely strong signal - high conviction trade
- **75-90% (High)**: Strong signal - good opportunity
- **60-75% (Moderate)**: Decent signal - proceed with caution
- **40-60% (Low)**: Weak signal - small position or wait
- **<40% (Very Low)**: Avoid - too uncertain

#### 3. **Backtest Metrics** (Look for these in detailed output)
- **Sharpe Ratio**: 
  - \>1.0 = Excellent (HDFCBANK: 1.25 ✓)
  - 0.5-1.0 = Good
  - <0.5 = Poor (TCS: -1.46 ✗)
  
- **Win Rate**:
  - \>55% = Good
  - 50-55% = Acceptable
  - <50% = Poor (INFY: 49% ✗)
  
- **Total Return**:
  - \>20% = Excellent (HDFCBANK: 21.5% ✓)
  - 10-20% = Good
  - 0-10% = Marginal
  - <0% = Loss (TCS: -30% ✗)
  
- **Max Drawdown**:
  - <15% = Good (HDFCBANK: 12.9% ✓)
  - 15-25% = Moderate
  - \>25% = High risk (TCS: 35.4% ✗)

---

## 🎓 Real Examples Explained

### Example 1: Strong BUY - HDFCBANK.NS

```
HDFCBANK.NS: BUY (Confidence: 89.1%)

Backtest Metrics:
- Sharpe Ratio: 1.25 ✓ (>1.0 threshold)
- Total Return: 21.5% ✓ (positive)
- Win Rate: 50.6% ✓ (>50%)
- Max Drawdown: 12.9% ✓ (<15%)
```

**What This Means**:
- ✅ Very high confidence (89%)
- ✅ Excellent risk-adjusted returns (Sharpe 1.25)
- ✅ Good historical performance (21.5% return)
- ✅ Low risk (12.9% max drawdown)
- **Action**: Strong buy candidate - consider full position

---

### Example 2: Strong SELL - TATAMOTORS.NS

```
TATAMOTORS.NS: SELL (Confidence: 97.1%)

Backtest Metrics:
- Sharpe Ratio: -1.8 ✗ (negative!)
- Total Return: -45% ✗ (big loss)
- Win Rate: 38% ✗ (<50%)
- Max Drawdown: 48% ✗ (very high)
```

**What This Means**:
- ❌ Very high confidence SELL (97%)
- ❌ Terrible risk-adjusted returns (negative Sharpe)
- ❌ Major historical losses (-45%)
- ❌ Very high risk (48% drawdown)
- **Action**: Strong sell/avoid - stay away from this stock!

---

### Example 3: Moderate BUY - RELIANCE.NS

```
RELIANCE.NS: BUY (Confidence: 76.5%)

Backtest Metrics:
- Sharpe Ratio: 0.05 ⚠️ (barely positive)
- Total Return: 1.0% ⚠️ (minimal)
- Win Rate: 50.6% ✓ (marginally profitable)
- Max Drawdown: 16.8% ⚠️ (moderate)
```

**What This Means**:
- ⚠️ Moderate confidence (76%)
- ⚠️ Weak risk-adjusted returns
- ⚠️ Minimal profit historically
- ⚠️ Moderate risk
- **Action**: Weak buy - small position or wait for better entry

---

## 🛠️ Testing Scenarios

### Scenario 1: "I want to invest in banking sector"
```bash
# Test all major banks
./venv/bin/python3 main.py --tickers HDFCBANK.NS,ICICIBANK.NS,AXISBANK.NS,KOTAKBANK.NS,SBIN.NS
```

**Expected Output**: All BUY signals with 70-90% confidence

---

### Scenario 2: "Should I buy TCS stock?"
```bash
# Test with backtest to see historical performance
./venv/bin/python3 main.py --ticker TCS.NS --backtest
```

**Expected Output**: SELL signal with ~81% confidence (poor backtest)

---

### Scenario 3: "What are the best stocks to buy now?"
```bash
# Test NIFTY 50 to see all opportunities
./venv/bin/python3 main.py --nifty50

# Or test specific sectors
./venv/bin/python3 main.py --tickers HDFCBANK.NS,SUNPHARMA.NS,BRITANNIA.NS,MARUTI.NS
```

**Expected Output**: Ranked list with confidence scores

---

### Scenario 4: "I own INFY, should I sell?"
```bash
# Test with detailed backtest
./venv/bin/python3 main.py --ticker INFY.NS --backtest
```

**Expected Output**: SELL signal with ~89% confidence (negative returns)

---

## 📋 Quick Reference - Stock Symbols

### Banking
- HDFCBANK.NS - HDFC Bank
- ICICIBANK.NS - ICICI Bank
- AXISBANK.NS - Axis Bank
- KOTAKBANK.NS - Kotak Mahindra Bank
- SBIN.NS - State Bank of India

### IT
- TCS.NS - Tata Consultancy Services
- INFY.NS - Infosys
- WIPRO.NS - Wipro
- HCLTECH.NS - HCL Technologies
- TECHM.NS - Tech Mahindra

### FMCG
- HINDUNILVR.NS - Hindustan Unilever
- ITC.NS - ITC Limited
- BRITANNIA.NS - Britannia Industries
- NESTLEIND.NS - Nestle India

### Auto
- MARUTI.NS - Maruti Suzuki
- TATAMOTORS.NS - Tata Motors
- BAJAJ-AUTO.NS - Bajaj Auto
- HEROMOTOCO.NS - Hero MotoCorp
- M&M.NS - Mahindra & Mahindra

### Pharma
- SUNPHARMA.NS - Sun Pharmaceutical
- DRREDDY.NS - Dr. Reddy's Laboratories
- CIPLA.NS - Cipla
- DIVISLAB.NS - Divi's Laboratories
- APOLLOHOSP.NS - Apollo Hospitals

### Energy
- RELIANCE.NS - Reliance Industries
- ONGC.NS - Oil and Natural Gas Corp
- BPCL.NS - Bharat Petroleum
- IOC.NS - Indian Oil Corporation

### Others
- ADANIENT.NS - Adani Enterprises
- BAJAJFINSV.NS - Bajaj Finserv
- LT.NS - Larsen & Toubro
- ULTRACEMCO.NS - UltraTech Cement
- ASIANPAINT.NS - Asian Paints

---

## ⚡ Advanced Testing

### Compare Multiple Stocks Side-by-Side
```bash
# Test and compare
./venv/bin/python3 main.py --tickers HDFCBANK.NS,ICICIBANK.NS,AXISBANK.NS > bank_comparison.txt

# View results
cat bank_comparison.txt | grep -A 5 "Top 10 Trading"
```

### Test with Different Periods to Find Trends
```bash
# Short-term (6 months)
./venv/bin/python3 main.py --ticker RELIANCE.NS --period 6mo

# Medium-term (1 year)
./venv/bin/python3 main.py --ticker RELIANCE.NS --period 1y

# Long-term (3 years)
./venv/bin/python3 main.py --ticker RELIANCE.NS --period 3y
```

### Save Results to File
```bash
# Save full output
./venv/bin/python3 main.py --ticker HDFCBANK.NS > hdfcbank_analysis.txt

# Save summary only
./venv/bin/python3 main.py --ticker HDFCBANK.NS | grep -A 20 "Top 10 Trading" > hdfcbank_summary.txt
```

---

## 🎯 Decision Making Guide

### When You See BUY Signal:

#### High Confidence (75%+) BUY
✅ **Check**:
1. Sharpe Ratio >1.0?
2. Win Rate >50%?
3. Max Drawdown <20%?

✅ **If YES to all**: Strong buy candidate
- Consider: Full position (2-5% of portfolio)
- Set stop-loss: Use ATR-based stop from output

⚠️ **If NO to any**: Proceed cautiously
- Consider: Reduced position (0.5-2% of portfolio)
- Set tight stop-loss

---

### When You See SELL Signal:

#### High Confidence (75%+) SELL
❌ **Check**:
1. Do you own this stock?
2. Is Sharpe Ratio negative?
3. Is historical return negative?

❌ **If YES**: Consider selling
- If owned: Exit position or reduce exposure
- If planning to buy: Avoid completely

✅ **If NO**: May wait and monitor
- Watch for reversal signals
- Set alerts for price changes

---

### When You See HOLD Signal:

⚠️ **Means**:
- Conflicting signals detected
- Uncertainty in market
- No clear direction

**Action**:
- If you own: Hold current position
- If you don't own: Wait for clearer signal
- Re-test in 1-2 weeks

---

## 📊 Sample Testing Session

### Let's Test Top 10 NIFTY Stocks:

```bash
# Test in batches for speed
./venv/bin/python3 main.py --tickers RELIANCE.NS,TCS.NS,HDFCBANK.NS,INFY.NS,ICICIBANK.NS

./venv/bin/python3 main.py --tickers HINDUNILVR.NS,ITC.NS,KOTAKBANK.NS,LT.NS,AXISBANK.NS
```

### Expected Time:
- Single stock: 20-30 seconds
- 3 stocks: 45-60 seconds
- 5 stocks: 90-120 seconds
- NIFTY 50: 10-15 minutes

---

## 🐛 Troubleshooting

### Issue: "Command not found"
**Solution**:
```bash
# Make sure you're in the right directory
cd /Users/kasyapdharanikota/Desktop/stock_bot_langgraph

# Verify Python is available
./venv/bin/python3 --version
```

### Issue: "No data found for symbol"
**Solution**:
- Check symbol format (should end with .NS for NSE stocks)
- Verify symbol exists: https://finance.yahoo.com/quote/RELIANCE.NS
- Some stocks may not have enough historical data

### Issue: "Connection timeout"
**Solution**:
- Check internet connection
- Wait 30 seconds and try again
- APIs may be rate-limited

### Issue: "Low confidence scores"
**Solution**: This is actually good!
- Bot is being honest about uncertainty
- Don't force trades on low confidence
- Wait for better setups

---

## 💡 Pro Tips

### 1. Test Regularly
```bash
# Create a weekly test script
echo './venv/bin/python3 main.py --tickers HDFCBANK.NS,TCS.NS,RELIANCE.NS' > weekly_test.sh
chmod +x weekly_test.sh

# Run weekly
./weekly_test.sh
```

### 2. Track Your Trades
Keep a simple log:
```
Date: 2025-10-07
Stock: HDFCBANK.NS
Action: BUY
Confidence: 89%
Entry Price: ₹1,650
Bot Says: Sharpe 1.25, Strong buy
```

### 3. Combine with Your Analysis
- Bot gives quantitative analysis
- You add qualitative (news, events, sector knowledge)
- **Best results = Bot + Your judgment**

### 4. Position Sizing by Confidence
```
90%+ confidence: 5% of portfolio
75-90%: 3% of portfolio
60-75%: 2% of portfolio
<60%: 1% or wait
```

### 5. Use Stop Losses
The bot calculates ATR-based stops. Look for:
```
ATR Stop Loss: ₹1,580 (5% below entry)
```
Always use them!

---

## 🎓 Learning from Results

### Track Bot Accuracy
After 1 month, review:
```
Bot Said BUY at 85% confidence:
- HDFCBANK: +8% ✓ (correct)
- RELIANCE: -2% ✗ (wrong)
- ICICIBANK: +12% ✓ (correct)

Accuracy: 2/3 = 67%
```

### Adjust Your Strategy
- High confidence (>85%) = More reliable
- Backtest metrics = Best indicator
- Sector trends = Important context

---

## 📞 Quick Help

### Get Help
```bash
./venv/bin/python3 main.py --help
```

### Check Configuration
```bash
./venv/bin/python3 -m utils.config_validator
```

### View Logs
```bash
tail -f logs/stock_bot.log
```

---

## ✅ Daily Testing Checklist

- [ ] Open terminal
- [ ] Navigate to project folder
- [ ] Run test command for stocks of interest
- [ ] Check confidence scores (aim for >75%)
- [ ] Review backtest metrics (Sharpe, returns, drawdown)
- [ ] Make decision (BUY/SELL/HOLD)
- [ ] Log your trade if executed
- [ ] Set stop-losses
- [ ] Review weekly performance

---

## 🎯 Summary: How to Actually Use This

### Step-by-Step Process:

1. **Identify Stocks**: Pick stocks you're interested in
2. **Run Test**: Use command above
3. **Check Confidence**: Look for >75%
4. **Review Backtest**: Sharpe >1.0, positive returns
5. **Make Decision**: Follow bot recommendation if metrics align
6. **Position Size**: Based on confidence level
7. **Set Stop Loss**: Use ATR-based stop from output
8. **Monitor**: Re-test weekly or after major news

### Remember:
- ✅ Bot is a tool, not a crystal ball
- ✅ High confidence = Better odds, not guarantee
- ✅ Always use risk management (stops, position sizing)
- ✅ Combine bot analysis with your research
- ✅ Start small, build confidence over time

---

**Ready to test? Start with this command:**

```bash
cd /Users/kasyapdharanikota/Desktop/stock_bot_langgraph
./venv/bin/python3 main.py --ticker HDFCBANK.NS
```

**Good luck with your trading! 📈**

---

*Last Updated: 2025-10-07*  
*Version: 2.0 (Enhanced Accuracy)*
