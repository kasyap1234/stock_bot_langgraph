# Automated Trading Strategies Framework

This document describes the comprehensive automated trading strategies framework implemented for the stock trading bot. The framework provides modular, extensible trading strategies with advanced backtesting capabilities.

## Overview

The trading strategies framework consists of:

- **Modular Strategy Classes**: Abstract base class with common interface
- **Multiple Strategy Types**: Trend Following, Mean Reversion, Breakout, Sentiment-Driven, Ensemble
- **Advanced Backtesting Engine**: Realistic market simulation with transaction costs
- **Risk Management**: Integrated position sizing and risk controls
- **Performance Analytics**: Comprehensive metrics and reporting

## Architecture

### Core Components

#### 1. BaseStrategy (Abstract Base Class)
```python
class BaseStrategy(ABC):
    def generate_signals(self, data: pd.DataFrame, state: Optional[State]) -> List[TradingSignal]
    def validate_signal(self, signal: TradingSignal, data: pd.DataFrame) -> bool
    def calculate_position_size(self, capital: float, price: float, risk_per_trade: float = 0.01) -> int
    def apply_risk_management(self, signal: TradingSignal, current_portfolio_value: float, current_positions: Dict[str, int]) -> TradingSignal
```

#### 2. StrategyConfig
Configuration class for strategy parameters:
```python
@dataclass
class StrategyConfig:
    name: str
    description: str
    parameters: Dict[str, Any]
    risk_management: Dict[str, Any]
    position_sizing: Dict[str, Any]
```

#### 3. TradingSignal
Represents individual trading signals:
```python
@dataclass
class TradingSignal:
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    price: float
    timestamp: datetime
    reason: str = ""
    metadata: Dict[str, Any] = None
```

## Available Strategies

### 1. Trend Following Strategy
**Description**: Follows market trends using moving averages and momentum indicators.

**Key Features**:
- Multi-timeframe trend analysis
- ML model confirmation
- Adaptive parameters based on volatility

**Parameters**:
- `trend_periods`: List of periods for trend calculation [20, 50, 200]
- `ml_confirmation`: Whether to use ML models for confirmation

### 2. Mean Reversion Strategy
**Description**: Trades on price deviations from statistical mean.

**Key Features**:
- RSI-based overbought/oversold detection
- Bollinger Bands analysis
- Statistical arbitrage approach

**Parameters**:
- `lookback_period`: Period for mean calculation (default: 20)
- `deviation_threshold`: Standard deviation threshold (default: 2.0)

### 3. Breakout Strategy
**Description**: Trades on price breakouts from consolidation patterns.

**Key Features**:
- Volume confirmation for breakouts
- Consolidation range detection
- False breakout filtering

**Parameters**:
- `consolidation_period`: Period to detect consolidation (default: 20)
- `breakout_threshold`: Minimum breakout percentage (default: 0.02)
- `volume_multiplier`: Volume confirmation multiplier (default: 1.5)

### 4. Sentiment-Driven Strategy
**Description**: Uses news and social media sentiment for trading decisions.

**Key Features**:
- VADER sentiment analysis
- Twitter sentiment integration
- Volume confirmation for sentiment signals

**Parameters**:
- `sentiment_threshold`: Minimum sentiment score for signals (default: 0.1)
- `volume_threshold`: Volume confirmation multiplier (default: 1.2)

### 5. Ensemble Strategy
**Description**: Combines multiple strategies with dynamic weighting.

**Key Features**:
- Dynamic weight adjustment based on market conditions
- Confidence-based signal aggregation
- Risk-adjusted ensemble decisions

**Parameters**:
- `strategies`: List of strategy instances to combine
- `weights`: Dictionary of strategy weights
- `confidence_threshold`: Minimum confidence for signals (default: 0.2)

## Usage Examples

### Basic Strategy Usage

```python
from simulation.trading_strategies import StrategyFactory
from simulation.backtesting_engine import BacktestingEngine

# Create a trend following strategy
config = StrategyFactory.get_default_configs()['trend_following']
strategy = StrategyFactory.create_strategy('trend_following', config)

# Create backtesting engine
engine = BacktestingEngine(initial_capital=1000000)

# Run backtest
results = engine.run_strategy_backtest(
    strategy=strategy,
    stock_data=your_stock_data,
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31)
)

print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
```

### Custom Strategy Configuration

```python
from simulation.trading_strategies import StrategyConfig, TrendFollowingStrategy

# Create custom configuration
custom_config = StrategyConfig(
    name="Custom Trend Strategy",
    description="Customized trend following",
    parameters={
        'trend_periods': [10, 30, 100],
        'ml_confirmation': True
    },
    risk_management={
        'max_drawdown': 0.15,
        'max_position_size': 0.15,
        'stop_loss_pct': 0.08
    },
    position_sizing={
        'method': 'percentage',
        'percentage': 0.08
    }
)

strategy = TrendFollowingStrategy(custom_config)
```

### Ensemble Strategy

```python
from simulation.trading_strategies import EnsembleStrategy

# Create individual strategies
trend_strategy = StrategyFactory.create_strategy('trend_following',
    StrategyFactory.get_default_configs()['trend_following'])
mean_rev_strategy = StrategyFactory.create_strategy('mean_reversion',
    StrategyFactory.get_default_configs()['mean_reversion'])

# Create ensemble
ensemble_config = StrategyConfig(
    name="My Ensemble",
    description="Combined strategies",
    parameters={
        'strategies': [trend_strategy, mean_rev_strategy],
        'weights': {'trend_following': 0.6, 'mean_reversion': 0.4}
    }
)

ensemble = EnsembleStrategy(ensemble_config)
```

## Backtesting Engine Integration

The framework integrates seamlessly with the existing backtesting engine:

### Enhanced Backtesting Features

1. **Realistic Market Conditions**:
   - Transaction costs (commissions, slippage)
   - Market impact modeling
   - Realistic trade execution

2. **Risk Management**:
   - Position size limits
   - Stop-loss orders
   - Maximum drawdown controls

3. **Performance Metrics**:
   - Sharpe ratio, Sortino ratio
   - Maximum drawdown
   - Win rate, profit factor
   - Calmar ratio

### Running Backtests

```python
# Initialize engine with custom parameters
engine = BacktestingEngine(
    initial_capital=1000000,
    commission_rate=0.001,  # 0.1%
    slippage_rate=0.0005,   # 0.05%
    max_position_size=0.1    # 10% per position
)

# Run strategy backtest
results = engine.run_strategy_backtest(
    strategy=your_strategy,
    stock_data=stock_data_dict,
    start_date=start_date,
    end_date=end_date,
    state=current_state  # Optional LangGraph state
)
```

## Integration with LangGraph Workflow

The strategies integrate with the existing LangGraph workflow:

### State Integration

```python
# Strategies can access LangGraph state
def generate_signals(self, data: pd.DataFrame, state: Optional[State]):
    # Access technical signals
    technical_signals = state.get('technical_signals', {})

    # Access sentiment data
    sentiment_scores = state.get('sentiment_scores', {})

    # Access ML predictions
    ml_predictions = state.get('ml_predictions', {})

    # Generate signals based on state data
    signals = self._generate_signals_from_state(data, state)
    return signals
```

### Workflow Integration

1. **Data Collection**: Strategies access processed market data
2. **Signal Generation**: Generate trading signals based on strategy logic
3. **Risk Assessment**: Apply risk management rules
4. **Order Execution**: Convert signals to orders via backtesting engine
5. **Performance Monitoring**: Track strategy performance

## Testing

Comprehensive test suite included:

```bash
# Run strategy tests
python -m pytest tests/test_trading_strategies.py -v

# Run with coverage
python -m pytest tests/test_trading_strategies.py --cov=simulation.trading_strategies
```

### Test Coverage

- **Unit Tests**: Individual strategy components
- **Integration Tests**: Strategy + backtesting engine
- **Performance Tests**: Large dataset handling
- **Edge Case Tests**: Error conditions and boundary cases

## Performance Considerations

### Optimization Features

1. **Efficient Data Processing**:
   - Vectorized pandas operations
   - Minimal data copying
   - Lazy evaluation where possible

2. **Memory Management**:
   - Streaming data processing for large datasets
   - Garbage collection optimization
   - Memory-efficient data structures

3. **Computational Efficiency**:
   - NumPy/SciPy for numerical computations
   - Parallel processing for multiple symbols
   - Caching for repeated calculations

### Scalability

- **Multi-Symbol Support**: Process multiple stocks simultaneously
- **Time Series Optimization**: Efficient historical data handling
- **Modular Architecture**: Easy to add new strategies without performance impact

## Risk Management

### Built-in Risk Controls

1. **Position Sizing**:
   - Fixed percentage of capital
   - Risk-based sizing (Kelly criterion)
   - Equal risk allocation

2. **Stop Loss Management**:
   - Fixed percentage stops
   - ATR-based stops
   - Trailing stops

3. **Portfolio-Level Controls**:
   - Maximum drawdown limits
   - Position concentration limits
   - Correlation-based diversification

### Custom Risk Rules

```python
# Custom risk management in strategy
def apply_risk_management(self, signal: TradingSignal, portfolio_value: float, positions: Dict[str, int]):
    # Check portfolio-level risk
    if portfolio_value < (1 - self.config.risk_management['max_drawdown']):
        signal.action = 'HOLD'
        signal.reason += " | Risk: Max drawdown exceeded"

    # Check position concentration
    current_position = positions.get(signal.symbol, 0)
    max_position = int(portfolio_value * self.config.risk_management['max_position_size'] / signal.price)

    if abs(current_position) >= max_position:
        signal.action = 'HOLD'
        signal.reason += f" | Risk: Position limit ({max_position}) exceeded"

    return signal
```

## Extension Guide

### Adding New Strategies

1. **Inherit from BaseStrategy**:
```python
class MyCustomStrategy(BaseStrategy):
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        # Custom initialization

    def generate_signals(self, data: pd.DataFrame, state: Optional[State]) -> List[TradingSignal]:
        # Implement signal generation logic
        signals = []
        # ... your logic here
        return signals

    def validate_signal(self, signal: TradingSignal, data: pd.DataFrame) -> bool:
        # Implement signal validation
        return True
```

2. **Add to StrategyFactory**:
```python
# In StrategyFactory class
strategies = {
    'my_custom': MyCustomStrategy,
    # ... existing strategies
}
```

3. **Create Default Configuration**:
```python
# In get_default_configs method
'my_custom': StrategyConfig(
    name="My Custom Strategy",
    description="Description of my strategy",
    parameters={'param1': 'value1'}
)
```

### Strategy Best Practices

1. **Data Validation**: Always validate input data
2. **Error Handling**: Implement robust error handling
3. **Logging**: Use appropriate logging levels
4. **Documentation**: Document parameters and behavior
5. **Testing**: Write comprehensive unit tests
6. **Performance**: Optimize for speed and memory usage

## Configuration

### Default Parameters

All strategies come with sensible default parameters optimized for NSE/BSE markets:

- **Commission Rates**: 0.1% (brokerage) + 0.05% (slippage)
- **Position Sizes**: 5-10% of portfolio per trade
- **Risk Limits**: 15% maximum drawdown
- **Time Horizons**: 20-200 day lookback periods

### Customization

Parameters can be customized for different market conditions:

```python
# High volatility markets
config.parameters.update({
    'trend_periods': [5, 15, 50],  # Shorter periods
    'deviation_threshold': 2.5     # Higher threshold
})

# Low volatility markets
config.parameters.update({
    'trend_periods': [30, 100, 300],  # Longer periods
    'deviation_threshold': 1.5       # Lower threshold
})
```

## Monitoring and Logging

### Performance Monitoring

The framework includes comprehensive logging and monitoring:

- **Signal Generation Logs**: Track signal creation and validation
- **Trade Execution Logs**: Monitor order execution and costs
- **Performance Metrics**: Real-time performance tracking
- **Error Handling**: Detailed error reporting and recovery

### Log Levels

- **DEBUG**: Detailed signal generation information
- **INFO**: Strategy execution and performance metrics
- **WARNING**: Non-critical issues and edge cases
- **ERROR**: Critical errors requiring attention

## Future Enhancements

### Planned Features

1. **Machine Learning Integration**:
   - Deep learning models for pattern recognition
   - Reinforcement learning for strategy optimization
   - AutoML for parameter tuning

2. **Advanced Risk Management**:
   - Portfolio optimization (Markowitz, Black-Litterman)
   - Dynamic risk parity
   - Stress testing and scenario analysis

3. **Real-time Adaptation**:
   - Market regime detection
   - Adaptive parameter adjustment
   - Real-time strategy switching

4. **Multi-Asset Support**:
   - Forex, commodities, crypto integration
   - Cross-market arbitrage
   - Multi-asset portfolio strategies

## Conclusion

The automated trading strategies framework provides a robust, extensible foundation for algorithmic trading. Its modular design allows for easy customization and extension while maintaining high performance and reliability.

For questions or contributions, please refer to the main project documentation or create an issue in the repository.