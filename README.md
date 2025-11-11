# Stock Swing Trade Recommender

A sophisticated AI-powered stock analysis and recommendation system for short-term swing trading, specifically designed for Indian NSE stocks. This application uses advanced technical analysis, machine learning models, sentiment analysis, and macroeconomic factors to provide actionable BUY/SELL/HOLD recommendations.

## Features

### Core Capabilities
- **Multi-Agent Analysis System**: Parallel execution of specialized agents using LangGraph
- **40+ Technical Indicators**: RSI, MACD, Bollinger Bands, Ichimoku Cloud, Fibonacci, VWAP, and more
- **Machine Learning Models**: LSTM predictions, HMM regime detection, Random Forest signals
- **Fundamental Analysis**: P/E ratios, P/B ratios, earnings analysis
- **Sentiment Analysis**: VADER-based news sentiment scoring
- **Macroeconomic Factors**: GDP, inflation, interest rates via FRED API
- **Risk Assessment**: Volatility, Sharpe ratio, VaR, position sizing
- **Comprehensive Backtesting**: 5 trading strategies with realistic transaction costs
- **Multi-Stock Ranking**: Compare and rank multiple stocks for best opportunities
- **Web Dashboard**: Real-time monitoring with FastAPI and WebSocket support
- **LLM-Enhanced Reasoning**: Optional Groq API integration for intelligent recommendations

### Trading Strategies
1. **Trend Following**: Captures momentum in strong trends
2. **Mean Reversion**: Exploits price overshoots
3. **Breakout**: Identifies resistance/support breaks
4. **Sentiment-Driven**: Trades based on market sentiment
5. **Ensemble**: Combines multiple strategies for robust signals

## Quick Start

### Prerequisites
- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/stock_bot_langgraph.git
cd stock_bot_langgraph
```

2. **Install dependencies**

Using uv (recommended):
```bash
uv sync
```

Using pip:
```bash
pip install -r requirements.txt
```

3. **Configure API Keys**

Copy the example environment file and add your API keys:
```bash
cp .env.example .env
```

Edit `.env` and add your keys:
```env
# Required for LLM-enhanced recommendations (optional but recommended)
GROQ_API_KEY=your_groq_api_key_here

# Required for sentiment analysis
NEWS_API_KEY=your_newsapi_key_here

# Required for macroeconomic analysis
FRED_API_KEY=your_fred_api_key_here

# Optional - for additional data sources
ALPHA_VANTAGE_API_KEY=your_alphavantage_key_here
```

**Getting API Keys (Free)**:
- **Groq API**: https://console.groq.com/ (Free tier available)
- **News API**: https://newsapi.org/ (Free tier: 100 requests/day)
- **FRED API**: https://fred.stlouisfed.org/docs/api/api_key.html (Free)
- **Alpha Vantage**: https://www.alphavantage.co/support/#api-key (Free tier: 25 requests/day)

4. **Validate Setup**
```bash
uv run python validate_setup.py
```

### Basic Usage

#### Analyze a Single Stock
```bash
uv run main.py --ticker RELIANCE.NS
```

#### Analyze Multiple Stocks
```bash
uv run main.py --tickers RELIANCE.NS,TCS.NS,INFY.NS
```

#### Analyze All NIFTY 50 Stocks
```bash
uv run main.py --nifty50
```

#### Run with Backtesting
```bash
uv run main.py --ticker RELIANCE.NS --backtest
```

#### Use Basic RSI Strategy
```bash
uv run main.py --ticker RELIANCE.NS --basic
```

#### Quick Analysis Script
```bash
./run_analysis.sh RELIANCE.NS
```

### Supported Stock Symbols

The system works with NSE (National Stock Exchange) symbols. Common examples:
- **RELIANCE.NS** - Reliance Industries
- **TCS.NS** - Tata Consultancy Services
- **INFY.NS** - Infosys
- **HDFCBANK.NS** - HDFC Bank
- **ICICIBANK.NS** - ICICI Bank
- **SBIN.NS** - State Bank of India
- **WIPRO.NS** - Wipro
- **ITC.NS** - ITC Limited

For a complete list of NIFTY 50 stocks, see `config/config.py`.

## Understanding the Output

When you run an analysis, the system provides:

### 1. Technical Analysis
- **Trend Indicators**: Moving averages, ADX, Ichimoku Cloud
- **Momentum**: RSI, MACD, Stochastic, Williams %R
- **Volatility**: Bollinger Bands, ATR, Keltner Channels
- **Volume**: OBV, VWAP, Volume analysis
- **ML Signals**: LSTM predictions, HMM regime detection

### 2. Fundamental Analysis
- **Valuation Metrics**: P/E ratio, P/B ratio, Market Cap
- **Earnings**: EPS, earnings trends
- **Financial Health**: Revenue, profit margins

### 3. Sentiment Analysis
- **News Sentiment**: Positive/Negative/Neutral scores
- **Market Sentiment**: Overall market mood
- **Confidence Score**: Reliability of sentiment data

### 4. Risk Assessment
- **Volatility**: Historical and realized volatility
- **Sharpe Ratio**: Risk-adjusted returns
- **Value at Risk (VaR)**: Potential losses
- **Beta**: Correlation with market
- **Position Sizing**: Recommended position size based on risk

### 5. Final Recommendation
- **Action**: BUY / SELL / HOLD
- **Confidence Score**: 0-100% confidence in the recommendation
- **Entry/Exit Levels**: Suggested price points
- **Stop Loss**: Risk management level
- **Target Price**: Profit target
- **Time Horizon**: Recommended holding period (typically 5-15 days for swing trades)
- **Reasoning**: AI-generated explanation (if Groq API enabled)

### 6. Multi-Stock Ranking
When analyzing multiple stocks:
- **Buy Ranking**: Stocks ranked by attractiveness
- **Top Candidate**: Best stock to buy
- **Comparison Matrix**: Side-by-side comparison

### 7. Backtest Results (if enabled)
- **Total Returns**: Historical performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst loss period
- **Win Rate**: Percentage of profitable trades
- **Strategy Rating**: Overall strategy effectiveness

## Example Output

```
========================================
STOCK RECOMMENDATION FOR RELIANCE.NS
========================================

ACTION: BUY
CONFIDENCE: 87.5%
ENTRY PRICE: ₹2,485.30
TARGET PRICE: ₹2,650.00
STOP LOSS: ₹2,420.00
TIME HORIZON: 7-10 days

Technical Score: 8.2/10
  - RSI (45.3): Neutral with room to move up
  - MACD: Bullish crossover detected
  - Trend: Strong uptrend (ADX: 28.5)
  - Support at ₹2,450, Resistance at ₹2,550

Fundamental Score: 7.5/10
  - P/E Ratio: 24.3 (sector average: 26.1)
  - Strong earnings growth: +12% YoY
  - Revenue trend: Positive

Sentiment Score: 72%
  - Positive news coverage
  - Market sentiment: Bullish

Risk Assessment:
  - Volatility: Medium (18% annualized)
  - Sharpe Ratio: 1.8
  - Recommended Position Size: 4% of portfolio

LLM Reasoning:
Reliance shows strong technical momentum with a recent bullish MACD crossover
and price trading above key moving averages. The stock has found support at
₹2,450 and is likely to test ₹2,650 resistance. Fundamentals remain solid
with improving earnings. Entry at current levels offers favorable risk/reward
ratio for a swing trade targeting 5-7% gains over the next week.

Backtest Performance (Last 6 months):
  - Returns: +18.5%
  - Win Rate: 68%
  - Sharpe Ratio: 2.1
  - Max Drawdown: -8.2%
```

## Web Dashboard

Launch the interactive web dashboard for real-time monitoring:

```bash
uvicorn dashboard.app:app --host 0.0.0.0 --port 8000
```

Then open your browser to: http://localhost:8000

Features:
- Real-time stock analysis
- Portfolio tracking
- Risk metrics visualization
- WebSocket updates
- Session-based authentication

## Advanced Usage

### Custom Configuration

Edit `config/config.py` to customize:
- Technical indicator parameters
- Risk management rules
- ML model settings
- Position sizing strategies
- Backtest parameters

### Running Simulations

```python
from simulation.backtesting_engine import BacktestingEngine
from simulation.trading_strategies import TrendFollowingStrategy

engine = BacktestingEngine(initial_capital=100000)
strategy = TrendFollowingStrategy()
results = engine.run_backtest(strategy, 'RELIANCE.NS')
print(results)
```

### Using Individual Agents

```python
from agents.technical_analysis import technical_analysis_agent
from agents.sentiment_analysis import sentiment_analysis_agent

# Technical analysis only
state = {"stock_data": {"RELIANCE.NS": df}}
tech_results = technical_analysis_agent(state)
print(tech_results['technical_signals'])

# Sentiment analysis only
sentiment_results = sentiment_analysis_agent(state)
print(sentiment_results['sentiment_scores'])
```

### Multi-Timeframe Analysis

The system automatically analyzes:
- **Daily**: Short-term signals
- **Weekly**: Medium-term trends
- **Monthly**: Long-term context

### Real-Time Data Streaming

Enable real-time data in `config/config.py`:
```python
REAL_TIME_CONFIG = {
    "enabled": True,
    "update_interval": 60,  # seconds
    "streaming": True
}
```

## Architecture

The system uses a **LangGraph-based multi-agent architecture**:

```
User Input → Data Fetcher → [Technical, Fundamental, Sentiment, Macro] →
Risk Assessment → Final Recommendation → [Simulation] → Performance Analysis
```

Key components:
- **LangGraph Workflow**: Orchestrates agent execution
- **Parallel Processing**: Simultaneous data analysis
- **State Management**: Typed state object for data flow
- **Conditional Routing**: Smart workflow decisions
- **Error Handling**: Graceful degradation on failures

For detailed architecture, see `ARCHITECTURE.md`.

## Testing

Run the test suite:
```bash
pytest tests/
```

Run specific test modules:
```bash
pytest tests/test_workflow.py
pytest tests/test_technical_analysis.py
pytest tests/test_backtesting.py
```

## Troubleshooting

### "No module named 'talib'"
TA-Lib requires system libraries. The system will use fallback implementations if TA-Lib is not available.

To install TA-Lib (optional):
```bash
# Ubuntu/Debian
sudo apt-get install ta-lib

# macOS
brew install ta-lib

# Windows
# Download from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
```

### "API key not found"
Ensure your `.env` file is in the project root with valid API keys. The system will work with limited functionality without API keys:
- Without GROQ_API_KEY: Uses rule-based recommendations (no LLM reasoning)
- Without NEWS_API_KEY: Sentiment analysis will be limited
- Without FRED_API_KEY: Macroeconomic analysis will be skipped

### "Failed to fetch data for SYMBOL"
Check:
1. Symbol format is correct (e.g., RELIANCE.NS for NSE stocks)
2. Internet connection is active
3. Yahoo Finance is accessible
4. Market hours (some data updates during market hours only)

### Slow Performance
- Reduce number of stocks analyzed simultaneously
- Disable backtesting for faster analysis
- Use `--basic` flag for quick RSI-only analysis
- Check API rate limits

## Performance Optimization

### Caching
The system uses LRU caching for:
- Data fetching (reduces API calls)
- Technical indicator calculations
- Fundamental data

### Parallel Execution
- Multi-stock analysis runs in parallel
- Agent execution is parallelized by LangGraph
- Data fetching uses ThreadPoolExecutor

### API Rate Limits
- Yahoo Finance: No official limit (use responsibly)
- Alpha Vantage: 25 requests/day (free tier)
- News API: 100 requests/day (free tier)
- FRED API: No strict limit

## Important Notes

### Risk Disclaimer
**This is an analysis tool, NOT financial advice.**
- All recommendations are algorithmic and should be verified
- Past performance does not guarantee future results
- Always do your own research before trading
- Consider consulting a financial advisor
- The developers are not responsible for any trading losses

### Intended Use
- **Stock Analysis**: Identify potential swing trade opportunities
- **Backtesting**: Test trading strategies on historical data
- **Research**: Study market patterns and indicators
- **Education**: Learn about technical analysis and ML in trading

### NOT Intended For
- **Live Trading**: This tool does not place actual orders
- **High-Frequency Trading**: Not designed for millisecond-level trades
- **Guaranteed Profits**: No system can guarantee profits
- **Replacement for Due Diligence**: Always verify recommendations

### Swing Trading Timeframe
This system is optimized for **5-15 day holding periods**:
- Entry and exit signals for short-term momentum
- Risk management for swing trades
- Not suitable for day trading or long-term investing

## Contributing

Contributions are welcome! Areas for improvement:
- Additional technical indicators
- More sophisticated ML models
- Broker API integration for paper trading
- Enhanced dashboard features
- Additional data sources
- Performance optimizations

## License

MIT License - see LICENSE file for details

## Support

- **Documentation**: See `ARCHITECTURE.md` for technical details
- **Issues**: Report bugs or request features on GitHub
- **Examples**: See `examples/` directory for usage examples

## Acknowledgments

Built with:
- **LangGraph**: Agent orchestration
- **yfinance**: Stock data
- **TA-Lib**: Technical analysis
- **scikit-learn**: Machine learning
- **TensorFlow**: Deep learning
- **FastAPI**: Web dashboard
- **Groq**: LLM inference

## Roadmap

Planned features:
- [ ] Database integration for persistent storage
- [ ] Email/SMS alerts for trading signals
- [ ] Paper trading mode with broker integration
- [ ] Mobile app for monitoring
- [ ] Enhanced ML models (Transformer-based)
- [ ] Options analysis
- [ ] Sector rotation strategies
- [ ] Portfolio optimization
- [ ] Social sentiment analysis (Twitter, Reddit)
- [ ] Backtesting with more historical data

---

**Happy Trading!** Remember to always trade responsibly and never risk more than you can afford to lose.
