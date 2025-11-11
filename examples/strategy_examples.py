

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from simulation.trading_strategies import (
    StrategyFactory, StrategyConfig, EnsembleStrategy
)
from simulation.backtesting_engine import BacktestingEngine
from data.models import State

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data(symbol: str = "RELIANCE.NS", days: int = 252) -> pd.DataFrame:
    
    np.random.seed(42)

    # Generate realistic price data
    dates = pd.date_range('2023-01-01', periods=days, freq='D')

    # Base trend with some randomness
    trend = np.linspace(100, 120, days)
    noise = np.random.normal(0, 3, days)
    close_prices = trend + noise

    # Add some volatility clustering
    volatility = np.random.exponential(0.02, days)
    close_prices += np.random.normal(0, volatility * close_prices)

    # Ensure prices stay positive
    close_prices = np.maximum(close_prices, 50)

    # Generate OHLCV data
    high_prices = close_prices + np.abs(np.random.normal(0, 1, days))
    low_prices = close_prices - np.abs(np.random.normal(0, 1, days))
    open_prices = close_prices + np.random.normal(0, 1, days)
    volumes = np.random.randint(100000, 1000000, days)

    df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volumes
    }, index=dates)

    return df


def example_trend_following_strategy():
    
    print("\n" + "="*50)
    print("TREND FOLLOWING STRATEGY EXAMPLE")
    print("="*50)

    # Create strategy configuration
    config = StrategyFactory.get_default_configs()['trend_following']
    strategy = StrategyFactory.create_strategy('trend_following', config)

    # Create sample data
    stock_data = {'RELIANCE.NS': create_sample_data('RELIANCE.NS')}

    # Create backtesting engine
    engine = BacktestingEngine(initial_capital=1000000)

    # Run backtest
    print("Running trend following strategy backtest...")
    results = engine.run_strategy_backtest(
        strategy=strategy,
        stock_data=stock_data,
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 9, 1)
    )

    # Display results
    print("
Backtest Results:")
    print(".2%")
    print(".2%")
    print(".2%")
    print(".2f")
    print(f"Total Trades: {results.get('total_trades', 0)}")
    print(".2%")

    return results


def example_mean_reversion_strategy():
    
    print("\n" + "="*50)
    print("MEAN REVERSION STRATEGY EXAMPLE")
    print("="*50)

    # Create strategy configuration
    config = StrategyFactory.get_default_configs()['mean_reversion']
    strategy = StrategyFactory.create_strategy('mean_reversion', config)

    # Create oscillating data (more suitable for mean reversion)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    np.random.seed(42)

    # Create mean-reverting price series
    mean_price = 100
    oscillation = 15 * np.sin(np.linspace(0, 8*np.pi, 252))
    noise = np.random.normal(0, 2, 252)
    close_prices = mean_price + oscillation + noise

    stock_data = {
        'TCS.NS': pd.DataFrame({
            'Open': close_prices + np.random.normal(0, 1, 252),
            'High': close_prices + np.abs(np.random.normal(0, 1, 252)),
            'Low': close_prices - np.abs(np.random.normal(0, 1, 252)),
            'Close': close_prices,
            'Volume': np.random.randint(50000, 500000, 252)
        }, index=dates)
    }

    # Create backtesting engine
    engine = BacktestingEngine(initial_capital=1000000)

    # Run backtest
    print("Running mean reversion strategy backtest...")
    results = engine.run_strategy_backtest(
        strategy=strategy,
        stock_data=stock_data,
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 9, 1)
    )

    # Display results
    print("
Backtest Results:")
    print(".2%")
    print(".2%")
    print(".2%")
    print(".2f")
    print(f"Total Trades: {results.get('total_trades', 0)}")
    print(".2%")

    return results


def example_ensemble_strategy():
    
    print("\n" + "="*50)
    print("ENSEMBLE STRATEGY EXAMPLE")
    print("="*50)

    # Create individual strategies
    trend_config = StrategyFactory.get_default_configs()['trend_following']
    mean_rev_config = StrategyFactory.get_default_configs()['mean_reversion']
    breakout_config = StrategyFactory.get_default_configs()['breakout']

    trend_strategy = StrategyFactory.create_strategy('trend_following', trend_config)
    mean_rev_strategy = StrategyFactory.create_strategy('mean_reversion', mean_rev_config)
    breakout_strategy = StrategyFactory.create_strategy('breakout', breakout_config)

    # Create ensemble configuration
    ensemble_config = StrategyConfig(
        name="Advanced Ensemble",
        description="Combines trend following, mean reversion, and breakout strategies",
        parameters={
            'strategies': [trend_strategy, mean_rev_strategy, breakout_strategy],
            'weights': {'trend_following': 0.4, 'mean_reversion': 0.3, 'breakout': 0.3},
            'confidence_threshold': 0.6
        }
    )

    ensemble_strategy = EnsembleStrategy(ensemble_config)

    # Create sample data
    stock_data = {'INFY.NS': create_sample_data('INFY.NS')}

    # Create backtesting engine
    engine = BacktestingEngine(initial_capital=1000000)

    # Run backtest
    print("Running ensemble strategy backtest...")
    results = engine.run_strategy_backtest(
        strategy=ensemble_strategy,
        stock_data=stock_data,
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 9, 1)
    )

    # Display results
    print("
Backtest Results:")
    print(".2%")
    print(".2%")
    print(".2%")
    print(".2f")
    print(f"Total Trades: {results.get('total_trades', 0)}")
    print(".2%")

    return results


def example_sentiment_driven_strategy():
    
    print("\n" + "="*50)
    print("SENTIMENT-DRIVEN STRATEGY EXAMPLE")
    print("="*50)

    # Create strategy configuration
    config = StrategyFactory.get_default_configs()['sentiment_driven']
    strategy = StrategyFactory.create_strategy('sentiment_driven', config)

    # Create sample data
    stock_data = {'HDFCBANK.NS': create_sample_data('HDFCBANK.NS')}

    # Create mock state with sentiment data
    state = State()
    state.sentiment_scores = {
        'HDFCBANK.NS': {
            'compound': 0.7,  # Positive sentiment
            'positive': 0.7,
            'negative': 0.2,
            'articles_analyzed': 25
        }
    }

    # Create backtesting engine
    engine = BacktestingEngine(initial_capital=1000000)

    # Run backtest
    print("Running sentiment-driven strategy backtest...")
    results = engine.run_strategy_backtest(
        strategy=strategy,
        stock_data=stock_data,
        state=state,
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 9, 1)
    )

    # Display results
    print("
Backtest Results:")
    print(".2%")
    print(".2%")
    print(".2%")
    print(".2f")
    print(f"Total Trades: {results.get('total_trades', 0)}")
    print(".2%")

    return results


def example_custom_strategy_configuration():
    
    print("\n" + "="*50)
    print("CUSTOM STRATEGY CONFIGURATION EXAMPLE")
    print("="*50)

    # Create custom configuration
    custom_config = StrategyConfig(
        name="Custom Trend Strategy",
        description="Customized trend following with specific parameters",
        parameters={
            'trend_periods': [10, 30, 100],  # Shorter periods
            'ml_confirmation': True
        },
        risk_management={
            'max_drawdown': 0.15,  # Higher risk tolerance
            'max_position_size': 0.15,
            'stop_loss_pct': 0.08,
            'take_profit_pct': 0.12
        },
        position_sizing={
            'method': 'percentage',
            'percentage': 0.08  # 8% of portfolio per trade
        }
    )

    # Create strategy with custom config
    strategy = StrategyFactory.create_strategy('trend_following', custom_config)

    # Create sample data
    stock_data = {'BAJFINANCE.NS': create_sample_data('BAJFINANCE.NS')}

    # Create backtesting engine with custom settings
    engine = BacktestingEngine(
        initial_capital=2000000,  # Higher capital
        commission_rate=0.0005,   # Lower commission
        slippage_rate=0.0002      # Lower slippage
    )

    # Run backtest
    print("Running custom strategy backtest...")
    results = engine.run_strategy_backtest(
        strategy=strategy,
        stock_data=stock_data,
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 9, 1)
    )

    # Display results
    print("
Custom Strategy Results:")
    print(".2%")
    print(".2%")
    print(".2%")
    print(".2f")
    print(f"Total Trades: {results.get('total_trades', 0)}")
    print(".2%")

    return results


def compare_strategies():
    
    print("\n" + "="*50)
    print("STRATEGY COMPARISON EXAMPLE")
    print("="*50)

    strategies = {
        'Trend Following': StrategyFactory.create_strategy(
            'trend_following',
            StrategyFactory.get_default_configs()['trend_following']
        ),
        'Mean Reversion': StrategyFactory.create_strategy(
            'mean_reversion',
            StrategyFactory.get_default_configs()['mean_reversion']
        ),
        'Breakout': StrategyFactory.create_strategy(
            'breakout',
            StrategyFactory.get_default_configs()['breakout']
        )
    }

    # Create sample data
    stock_data = {'ITC.NS': create_sample_data('ITC.NS')}

    results_comparison = {}

    for name, strategy in strategies.items():
        print(f"\nRunning {name} strategy...")

        engine = BacktestingEngine(initial_capital=1000000)
        results = engine.run_strategy_backtest(
            strategy=strategy,
            stock_data=stock_data,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 9, 1)
        )

        results_comparison[name] = results

        print(".2%")

    # Summary comparison
    print("
STRATEGY COMPARISON SUMMARY:")
    print("-" * 60)
    print("<15")
    print("-" * 60)

    for name, results in results_comparison.items():
        print("<15")

    return results_comparison


def main():
    
    print("AUTOMATED TRADING STRATEGIES FRAMEWORK EXAMPLES")
    print("=" * 60)

    try:
        # Run individual strategy examples
        trend_results = example_trend_following_strategy()
        mean_rev_results = example_mean_reversion_strategy()
        ensemble_results = example_ensemble_strategy()
        sentiment_results = example_sentiment_driven_strategy()
        custom_results = example_custom_strategy_configuration()

        # Compare strategies
        comparison_results = compare_strategies()

        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("✓ Modular strategy framework")
        print("✓ Multiple trading strategies")
        print("✓ Customizable configurations")
        print("✓ Risk management integration")
        print("✓ Backtesting with realistic conditions")
        print("✓ Performance metrics calculation")
        print("✓ Strategy comparison capabilities")

    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()