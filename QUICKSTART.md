# Quick Start Guide

Get your stock swing trade recommender up and running in 5 minutes!

## Prerequisites

- Python 3.10 or higher
- Internet connection
- Terminal/Command prompt

## Installation (5 minutes)

### Step 1: Install uv (if not already installed)

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

### Step 2: Clone and Setup

```bash
# Navigate to the project directory
cd stock_bot_langgraph

# Install all dependencies (this will take a few minutes)
uv sync
```

**Note**: The first installation will download ~700MB of packages including TensorFlow. This is normal and only happens once.

### Step 3: Configure API Keys (Optional but Recommended)

The system works without API keys but with limited functionality. For best results, get free API keys:

```bash
# Create your .env file
cp .env.example .env
```

Edit `.env` and add your free API keys:

```env
# Get free key from https://console.groq.com/ (30 req/min free)
GROQ_API_KEY="your_groq_key_here"

# Get free key from https://newsapi.org/ (100 req/day free)
NEWS_API_KEY="your_news_key_here"

# Get free key from https://fred.stlouisfed.org/docs/api/api_key.html (free)
FRED_API_KEY="your_fred_key_here"
```

**No API keys?** The system still works with rule-based recommendations!

## Your First Analysis (1 minute)

### Analyze a Single Stock

```bash
uv run main.py --ticker RELIANCE.NS
```

That's it! You'll get:
- Technical analysis (40+ indicators)
- Fundamental metrics
- Sentiment score
- Risk assessment
- BUY/SELL/HOLD recommendation
- Entry/exit levels

### Try More Stocks

```bash
# Popular Indian stocks
uv run main.py --ticker TCS.NS          # Tata Consultancy Services
uv run main.py --ticker INFY.NS         # Infosys
uv run main.py --ticker HDFCBANK.NS     # HDFC Bank
uv run main.py --ticker SBIN.NS         # State Bank of India
```

### Compare Multiple Stocks

```bash
uv run main.py --tickers RELIANCE.NS,TCS.NS,INFY.NS
```

You'll get a ranked list of BUY opportunities!

### Analyze All NIFTY 50

```bash
uv run main.py --nifty50
```

### With Backtesting

```bash
uv run main.py --ticker RELIANCE.NS --backtest
```

See how the strategy would have performed over the last 6 months!

## Using the Quick Run Script

For convenience, use the provided script:

```bash
# Make it executable (first time only)
chmod +x run_analysis.sh

# Analyze a stock
./run_analysis.sh RELIANCE.NS

# With backtesting
./run_analysis.sh RELIANCE.NS --backtest

# All NIFTY 50
./run_analysis.sh --nifty50
```

## Understanding Your Results

### The Recommendation

```
ACTION: BUY
CONFIDENCE: 87.5%
ENTRY PRICE: ₹2,485.30
TARGET PRICE: ₹2,650.00 (+6.6%)
STOP LOSS: ₹2,420.00 (-2.6%)
TIME HORIZON: 7-10 days
```

- **ACTION**: What to do (BUY/SELL/HOLD)
- **CONFIDENCE**: How confident the system is (higher is better)
- **ENTRY**: Recommended entry price range
- **TARGET**: Profit target (sell at this price)
- **STOP LOSS**: Risk management (exit if price falls here)
- **TIME HORIZON**: Expected holding period

### The Scores

```
Technical Score: 8.2/10
Fundamental Score: 7.5/10
Sentiment Score: 72/100
Risk: Medium
```

- **Technical**: 8+ is very strong, 6-8 is good, 4-6 is neutral
- **Fundamental**: Based on P/E, earnings, growth
- **Sentiment**: News and market mood (70+ is very positive)
- **Risk**: Low/Medium/High volatility

### Composite Score

```
OVERALL COMPOSITE SCORE: 7.8/10
```

This is the weighted average of all factors:
- 7.5-10.0: Strong BUY
- 6.5-7.5: Moderate BUY
- 5.5-6.5: HOLD
- Below 5.5: SELL

## Common Workflows

### Daily Swing Trading Routine

```bash
# Morning: Check top NIFTY 50 opportunities
uv run main.py --nifty50 > daily_analysis.txt

# Find the top-ranked BUY candidates
# Enter positions based on recommendations

# Evening: Re-analyze your positions
uv run main.py --ticker YOUR_POSITION.NS
```

### Weekly Strategy Check

```bash
# Run with backtesting to validate strategy
uv run main.py --ticker RELIANCE.NS --backtest

# Check if Win Rate > 60% and Sharpe Ratio > 1.5
# Adjust positions accordingly
```

### Portfolio Monitoring

```bash
# Check all your positions at once
uv run main.py --tickers STOCK1.NS,STOCK2.NS,STOCK3.NS

# Compare and rebalance based on scores
```

## Troubleshooting

### "No module named 'X'"

```bash
# Reinstall dependencies
uv sync --force
```

### "Failed to fetch data for SYMBOL"

Check:
1. Symbol format is correct (e.g., `RELIANCE.NS` not `RELIANCE`)
2. Market is open or has recent data
3. Internet connection is working

### Slow Performance

```bash
# Use basic mode for faster results
uv run main.py --ticker RELIANCE.NS --basic

# Analyze fewer stocks
# Disable backtesting for speed
```

### "API key not valid"

1. Check `.env` file has correct keys
2. Verify no extra quotes or spaces
3. Get new keys from provider websites

## Next Steps

Once you're comfortable:

1. **Read the full README.md** for advanced features
2. **Check EXAMPLE_OUTPUT.md** to see detailed output samples
3. **Customize config/config.py** for your strategy
4. **Launch the web dashboard**: `uvicorn dashboard.app:app --port 8000`

## Important Reminders

⚠️ **This is NOT financial advice**
- Always do your own research
- Use proper risk management
- Never invest more than you can afford to lose
- Consider consulting a financial advisor

✅ **Best Practices**
- Always use stop losses
- Position size: 3-5% per trade
- Risk/Reward: Minimum 1:2 ratio
- Review positions daily
- Track your performance

## Getting Help

- Check the full documentation: `README.md`
- View example outputs: `EXAMPLE_OUTPUT.md`
- Architecture details: `ARCHITECTURE.md`
- Report issues on GitHub

---

**Happy Trading!** Remember: The best trade is the one you don't take if signals are unclear. Wait for high-confidence setups!
