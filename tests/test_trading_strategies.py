

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from simulation.trading_strategies import (
    BaseStrategy, StrategyConfig, TradingSignal,
    StrategyFactory, TrendFollowingStrategy,
    MeanReversionStrategy, BreakoutStrategy,
    SentimentDrivenStrategy, EnsembleStrategy
)
from data.models import State


class TestStrategyConfig:
    

    def test_strategy_config_creation(self):
        
        config = StrategyConfig(
            name="Test Strategy",
            description="Test description",
            parameters={"param1": "value1"},
            risk_management={"max_drawdown": 0.1}
        )

        assert config.name == "Test Strategy"
        assert config.description == "Test description"
        assert config.parameters["param1"] == "value1"
        assert config.risk_management["max_drawdown"] == 0.1


class TestTradingSignal:
    

    def test_trading_signal_creation(self):
        
        signal = TradingSignal(
            symbol="TEST.NS",
            action="BUY",
            confidence=0.8,
            price=100.0,
            timestamp=datetime.now(),
            reason="Test reason"
        )

        assert signal.symbol == "TEST.NS"
        assert signal.action == "BUY"
        assert signal.confidence == 0.8
        assert signal.price == 100.0
        assert signal.reason == "Test reason"


class TestStrategyFactory:
    

    def test_create_trend_following_strategy(self):
        
        config = StrategyFactory.get_default_configs()['trend_following']
        strategy = StrategyFactory.create_strategy('trend_following', config)

        assert isinstance(strategy, TrendFollowingStrategy)
        assert strategy.config.name == "Trend Following"

    def test_create_mean_reversion_strategy(self):
        
        config = StrategyFactory.get_default_configs()['mean_reversion']
        strategy = StrategyFactory.create_strategy('mean_reversion', config)

        assert isinstance(strategy, MeanReversionStrategy)
        assert strategy.config.name == "Mean Reversion"

    def test_create_breakout_strategy(self):
        
        config = StrategyFactory.get_default_configs()['breakout']
        strategy = StrategyFactory.create_strategy('breakout', config)

        assert isinstance(strategy, BreakoutStrategy)
        assert strategy.config.name == "Breakout Trading"

    def test_create_sentiment_strategy(self):
        
        config = StrategyFactory.get_default_configs()['sentiment_driven']
        strategy = StrategyFactory.create_strategy('sentiment_driven', config)

        assert isinstance(strategy, SentimentDrivenStrategy)
        assert strategy.config.name == "Sentiment Driven"

    def test_invalid_strategy_type(self):
        
        config = StrategyConfig("Test", "Test", {})

        with pytest.raises(ValueError, match="Unknown strategy type"):
            StrategyFactory.create_strategy('invalid_type', config)


class TestTrendFollowingStrategy:
    

    def setup_method(self):
        
        self.config = StrategyFactory.get_default_configs()['trend_following']
        self.strategy = TrendFollowingStrategy(self.config)

        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)

        # Create trending data
        trend = np.linspace(100, 150, 100)
        noise = np.random.normal(0, 2, 100)
        close_prices = trend + noise

        self.test_data = pd.DataFrame({
            'Close': close_prices,
            'High': close_prices + np.abs(np.random.normal(0, 1, 100)),
            'Low': close_prices - np.abs(np.random.normal(0, 1, 100)),
            'Open': close_prices + np.random.normal(0, 1, 100),
            'Volume': np.random.randint(1000, 10000, 100),
            'symbol': 'TEST.NS'
        }, index=dates)

    def test_generate_signals_trending_up(self):
        
        signals = self.strategy.generate_signals(self.test_data)

        assert len(signals) > 0
        # Should generate BUY signals for uptrend
        buy_signals = [s for s in signals if s.action == 'BUY']
        assert len(buy_signals) > 0

    def test_validate_signal(self):
        
        signal = TradingSignal(
            symbol="TEST.NS",
            action="BUY",
            confidence=0.8,
            price=120.0,
            timestamp=datetime.now()
        )

        is_valid = self.strategy.validate_signal(signal, self.test_data)
        assert isinstance(is_valid, bool)

    def test_calculate_position_size(self):
        
        quantity = self.strategy.calculate_position_size(100000, 100.0)
        assert quantity > 0
        assert isinstance(quantity, int)


class TestMeanReversionStrategy:
    

    def setup_method(self):
        
        self.config = StrategyFactory.get_default_configs()['mean_reversion']
        self.strategy = MeanReversionStrategy(self.config)

        # Create oscillating data around mean
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)

        mean_price = 100
        oscillation = 10 * np.sin(np.linspace(0, 4*np.pi, 100))
        noise = np.random.normal(0, 1, 100)
        close_prices = mean_price + oscillation + noise

        self.test_data = pd.DataFrame({
            'Close': close_prices,
            'High': close_prices + np.abs(np.random.normal(0, 1, 100)),
            'Low': close_prices - np.abs(np.random.normal(0, 1, 100)),
            'Open': close_prices + np.random.normal(0, 1, 100),
            'Volume': np.random.randint(1000, 10000, 100),
            'symbol': 'TEST.NS'
        }, index=dates)

    def test_generate_signals_oversold(self):
        
        # Make the last price very low (oversold)
        test_data = self.test_data.copy()
        test_data.loc[test_data.index[-1], 'Close'] = 85  # Below mean

        signals = self.strategy.generate_signals(test_data)

        # Should generate BUY signals for oversold condition
        buy_signals = [s for s in signals if s.action == 'BUY']
        assert len(buy_signals) > 0

    def test_generate_signals_overbought(self):
        
        # Make the last price very high (overbought)
        test_data = self.test_data.copy()
        test_data.loc[test_data.index[-1], 'Close'] = 115  # Above mean

        signals = self.strategy.generate_signals(test_data)

        # Should generate SELL signals for overbought condition
        sell_signals = [s for s in signals if s.action == 'SELL']
        assert len(sell_signals) > 0


class TestBreakoutStrategy:
    

    def setup_method(self):
        
        self.config = StrategyFactory.get_default_configs()['breakout']
        self.strategy = BreakoutStrategy(self.config)

        # Create consolidating data with breakout
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        np.random.seed(42)

        # Consolidating prices
        base_price = 100
        consolidation_noise = np.random.normal(0, 1, 40)
        breakout_move = np.linspace(100, 110, 10)  # Breakout up

        close_prices = np.concatenate([
            base_price + consolidation_noise,
            breakout_move
        ])

        self.test_data = pd.DataFrame({
            'Close': close_prices,
            'High': close_prices + np.abs(np.random.normal(0, 0.5, 50)),
            'Low': close_prices - np.abs(np.random.normal(0, 0.5, 50)),
            'Open': close_prices + np.random.normal(0, 0.5, 50),
            'Volume': np.random.randint(1000, 10000, 50),
            'symbol': 'TEST.NS'
        }, index=pd.date_range('2023-01-01', periods=50, freq='D'))

    def test_generate_signals_breakout(self):
        
        signals = self.strategy.generate_signals(self.test_data)

        # Should detect breakout and generate signals
        assert isinstance(signals, list)


class TestSentimentDrivenStrategy:
    

    def setup_method(self):
        
        self.config = StrategyFactory.get_default_configs()['sentiment_driven']
        self.strategy = SentimentDrivenStrategy(self.config)

        # Create test data
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        self.test_data = pd.DataFrame({
            'Close': np.random.uniform(95, 105, 50),
            'High': np.random.uniform(100, 110, 50),
            'Low': np.random.uniform(90, 100, 50),
            'Open': np.random.uniform(95, 105, 50),
            'Volume': np.random.randint(1000, 10000, 50),
            'symbol': 'TEST.NS'
        }, index=dates)

    def test_generate_signals_with_sentiment(self):
        
        # Mock state with sentiment data
        mock_state = State()
        mock_state.sentiment_scores = {
            'TEST.NS': {
                'compound': 0.8,  # Very positive sentiment
                'positive': 0.8,
                'negative': 0.1
            }
        }

        signals = self.strategy.generate_signals(self.test_data, mock_state)

        # Should generate BUY signals for positive sentiment
        buy_signals = [s for s in signals if s.action == 'BUY']
        assert len(buy_signals) > 0

    def test_generate_signals_without_sentiment(self):
        
        signals = self.strategy.generate_signals(self.test_data)

        # Should return empty list when no sentiment data
        assert signals == []


class TestEnsembleStrategy:
    

    def setup_method(self):
        
        # Create individual strategies
        trend_config = StrategyFactory.get_default_configs()['trend_following']
        mean_rev_config = StrategyFactory.get_default_configs()['mean_reversion']

        trend_strategy = TrendFollowingStrategy(trend_config)
        mean_rev_strategy = MeanReversionStrategy(mean_rev_config)

        # Create ensemble config
        ensemble_config = StrategyConfig(
            name="Test Ensemble",
            description="Test ensemble strategy",
            parameters={
                'strategies': [trend_strategy, mean_rev_strategy],
                'weights': {'trend_following': 0.6, 'mean_reversion': 0.4}
            }
        )

        self.strategy = EnsembleStrategy(ensemble_config)

        # Create test data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)

        # Create trending data
        trend = np.linspace(100, 120, 100)
        noise = np.random.normal(0, 2, 100)
        close_prices = trend + noise

        self.test_data = pd.DataFrame({
            'Close': close_prices,
            'High': close_prices + np.abs(np.random.normal(0, 1, 100)),
            'Low': close_prices - np.abs(np.random.normal(0, 1, 100)),
            'Open': close_prices + np.random.normal(0, 1, 100),
            'Volume': np.random.randint(1000, 10000, 100),
            'symbol': 'TEST.NS'
        }, index=dates)

    def test_generate_signals_ensemble(self):
        
        signals = self.strategy.generate_signals(self.test_data)

        # Should generate signals based on ensemble decision
        assert isinstance(signals, list)

        if signals:
            signal = signals[0]
            assert signal.action in ['BUY', 'SELL', 'HOLD']
            assert 0 <= signal.confidence <= 1

    def test_aggregate_signals(self):
        
        signals = [
            TradingSignal("TEST.NS", "BUY", 0.8, 100, datetime.now()),
            TradingSignal("TEST.NS", "BUY", 0.6, 100, datetime.now()),
            TradingSignal("TEST.NS", "SELL", 0.7, 100, datetime.now())
        ]

        weights = [0.5, 0.3, 0.2]
        result = self.strategy._aggregate_signals(signals, weights)

        assert 'action' in result
        assert 'confidence' in result
        assert result['action'] in ['BUY', 'SELL', 'HOLD']


class TestStrategyIntegration:
    

    def test_strategy_with_backtesting_engine(self):
        
        from simulation.backtesting_engine import BacktestingEngine

        # Create strategy
        config = StrategyFactory.get_default_configs()['trend_following']
        strategy = StrategyFactory.create_strategy('trend_following', config)

        # Create mock data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        close_prices = np.random.uniform(95, 105, 100)

        stock_data = {
            'TEST.NS': pd.DataFrame({
                'Close': close_prices,
                'High': close_prices + 1,
                'Low': close_prices - 1,
                'Open': close_prices,
                'Volume': np.random.randint(1000, 10000, 100)
            }, index=dates)
        }

        # Create backtesting engine
        engine = BacktestingEngine(initial_capital=100000)

        # Run backtest
        results = engine.run_strategy_backtest(
            strategy=strategy,
            stock_data=stock_data,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 2, 1)
        )

        # Should return results dictionary
        assert isinstance(results, dict)
        assert 'total_return' in results
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results


if __name__ == "__main__":
    pytest.main([__file__])