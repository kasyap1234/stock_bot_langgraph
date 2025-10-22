# Stock Bot - Quick Start Guide

## What Was Fixed

### ✅ All Analysis is Now Real (No Mock/Placeholder Code)
- **Data Fetching**: Real API calls with 4-hour cache (down from 7 days)
- **Technical Analysis**: Real indicators calculated from actual price data
- **Fundamental Analysis**: Real metrics from Yahoo Finance, Alpha Vantage, web scraping
- **Sentiment Analysis**: Real VADER + keyword analysis on actual news/tweets
- **ML Models**: Real scikit-learn, XGBoost, LightGBM trained models
- **Risk Assessment**: Real Kelly criterion, portfolio optimization
- **Recommendation Engine**: Real multi-factor weighted decision making

### ✅ Better Error Handling
- **Macro Analysis**: Now properly reports when economic data is unavailable
- **Fundamental Analysis**: Tracks data quality and warns about missing metrics
- **Configuration Validator**: Checks all API keys and dependencies at startup

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install optional but recommended packages for better accuracy
pip install TA-Lib  # For advanced technical analysis
pip install tensorflow  # For neural networks
pip install xgboost lightgbm catboost  # For advanced ML models
pip install vaderSentiment  # For sentiment analysis
```

## Configuration

### 1. Set up API Keys (Optional but Recommended)

Create a `.env` file in the project root:

```bash
# Alpha Vantage (for fundamental data)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key

# News API (for sentiment analysis)
NEWS_API_KEY=your_news_api_key

# Twitter API (for social sentiment)
TWITTER_BEARER_TOKEN=your_twitter_bearer_token

# Groq (for LLM-based analysis)
GROQ_API_KEY=your_groq_api_key
```

**Note**: The system will work without these keys but will use fallback data sources. For best accuracy, configure at least Alpha Vantage and News API.

### 2. Validate Configuration

```bash
python -m utils.config_validator
```

This will show:
- ✅ What's working
- ⚠️ What's missing
- ❌ What needs to be fixed

## Usage

### Basic Analysis

```bash
# Analyze a single stock
python main.py --ticker RELIANCE.NS

# Analyze multiple stocks
python main.py --ticker RELIANCE.NS TCS.NS INFY.NS

# Or using comma-separated format
python main.py --tickers RELIANCE.NS,TCS.NS,INFY.NS

# Analyze NIFTY 50 stocks
python main.py --nifty50
```

### With Backtesting

```bash
# Run backtest simulation
python main.py --ticker RELIANCE.NS --backtest

# Run basic RSI strategy (baseline)
python main.py --ticker RELIANCE.NS --backtest --basic

# Custom period
python main.py --ticker TCS.NS --period 2y --backtest
```

### Skip Validation (if needed)

```bash
# Skip configuration validation
python main.py --ticker RELIANCE.NS --skip-validation
```

## Understanding the Output

### Recommendation Structure

```python
{
    'symbol': 'RELIANCE.NS',
    'action': 'BUY',  # or 'SELL', 'HOLD'
    'composite_score': 0.65,  # -1 to 1, higher is more bullish
    'confidence': 0.82,  # 0 to 1, higher is more confident
    'factor_contributions': {
        'technical': {'strength': 0.7, 'weight': 0.25, 'contribution': 0.175},
        'fundamental': {'strength': 0.5, 'weight': 0.15, 'contribution': 0.075},
        'sentiment': {'strength': 0.6, 'weight': 0.20, 'contribution': 0.120},
        # ... more factors
    }
}
```

### Confidence Levels
- **High (>0.7)**: Strong agreement between multiple analyses
- **Medium (0.4-0.7)**: Moderate agreement, some conflicting signals
- **Low (<0.4)**: Weak signals or conflicting analyses

### Data Quality Indicators
- **good**: 4+ fundamental metrics available
- **fair**: 2-3 fundamental metrics available  
- **poor**: < 2 fundamental metrics available
- **unavailable**: No data sources succeeded

## Interpreting Results

### Strong BUY Signal
```
Action: BUY
Composite Score: 0.65
Confidence: 0.85
Data Quality: good
```
- Multiple factors agree (high confidence)
- Strong positive score
- Good data quality
- **Consider**: Position size based on confidence

### Weak BUY Signal
```
Action: BUY  
Composite Score: 0.10
Confidence: 0.45
Data Quality: fair
```
- Marginal positive score
- Moderate confidence
- Fair data quality
- **Consider**: Smaller position or wait for confirmation

### HOLD with Warnings
```
Action: HOLD
Composite Score: 0.05
Confidence: 0.30
Data Quality: poor
Available Metrics: 1/6
```
- Insufficient data for confident decision
- Low confidence score
- **Recommendation**: Skip or gather more data

## Common Issues

### Issue: "Cache expired, fetching fresh data" messages
**Solution**: Normal behavior. Cache refreshes every 4 hours for accuracy.

### Issue: "Fundamental data unavailable" warnings
**Possible causes**:
1. API keys not configured
2. Rate limits exceeded
3. Symbol not found in data sources

**Solutions**:
1. Configure Alpha Vantage API key in `.env`
2. Wait a few minutes and retry
3. Verify symbol format (e.g., `RELIANCE.NS` for NSE stocks)

### Issue: "Macro analysis may be inaccurate" warnings
**Cause**: Economic data APIs (FRED, etc.) unavailable

**Impact**: Macro factor will be neutral (0.0) instead of bullish/bearish

**Solution**: This is acceptable - other factors will compensate

### Issue: Configuration validation fails
**Solution**: Run validator to see specific issues:
```bash
python -m utils.config_validator
```

Then fix reported errors/warnings.

## Tips for Best Accuracy

### 1. Use Fresh Data
- Don't rely on old cached data
- Run analysis during market hours for latest prices
- Clear cache if needed: `rm -rf data/cache/*`

### 2. Configure API Keys
- At minimum: Alpha Vantage (fundamental) + News API (sentiment)
- Bonus: Twitter API for social sentiment
- Optional: Groq for LLM-enhanced analysis

### 3. Analyze Multiple Stocks
- Compare recommendations across similar stocks
- Look for sector trends
- Use NIFTY 50 analysis for market-wide view

### 4. Run Backtests
- Validate strategy on historical data
- Compare basic vs. enhanced strategies
- Check walk-forward results for robustness

### 5. Monitor Data Quality
- Pay attention to `data_quality` indicators
- Low quality = low confidence
- Consider skipping stocks with poor data

### 6. Use Ensemble Results
- Don't rely on single factor
- High confidence = multiple factors agree
- Low confidence = conflicting signals, proceed with caution

## Testing

### Unit Tests
```bash
# Run all tests
pytest tests/

# Run specific tests
pytest tests/test_technical_analysis.py
pytest tests/test_fundamental_analysis.py
```

### Integration Test
```bash
# Test with real data
python test_reliance_fixed.py
```

### Accuracy Test
```bash
# Comprehensive test
python comprehensive_accuracy_test.py
```

## Logs

Check logs for detailed information:
```bash
# View real-time logs
tail -f logs/stock_bot.log

# Search for errors
grep ERROR logs/stock_bot.log

# Search for warnings
grep WARNING logs/stock_bot.log
```

## Performance

### Typical Analysis Time
- **Single stock**: 10-30 seconds
- **Multiple stocks (3-5)**: 30-60 seconds
- **NIFTY 50**: 5-10 minutes
- **With backtest**: +50% time

### Optimization Tips
1. Use parallel processing (automatic)
2. Enable caching (enabled by default)
3. Skip validation if running multiple times: `--skip-validation`
4. Reduce analysis period for faster results: `--period 1y`

## Support

### Documentation
- `FIXES_APPLIED.md` - Detailed fix documentation
- `README.md` - Project overview
- `ARCHITECTURE.md` - System architecture (if available)

### Troubleshooting
1. Check configuration: `python -m utils.config_validator`
2. Review logs: `tail -f logs/stock_bot.log`
3. Clear cache: `rm -rf data/cache/*`
4. Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

## Disclaimer

⚠️ **IMPORTANT**: This tool provides analysis and recommendations based on historical data and various indicators. It is NOT financial advice. Always:

1. Do your own research (DYOR)
2. Consult with a qualified financial advisor
3. Understand the risks before investing
4. Never invest more than you can afford to lose
5. Past performance does not guarantee future results

The developers and contributors are not responsible for any financial losses incurred from using this tool.

## Next Steps

1. ✅ Configure API keys in `.env`
2. ✅ Run configuration validator
3. ✅ Test with a single stock
4. ✅ Review and understand the output
5. ✅ Run backtests to validate strategy
6. ✅ Compare multiple stocks
7. ✅ Use responsibly with proper risk management

---

Happy trading! 📈
