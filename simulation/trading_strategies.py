"""
Automated Trading Strategies Framework
Implements modular trading strategies with comprehensive backtesting capabilities.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from config.config import (
    RSI_OVERBOUGHT, RSI_OVERSOLD, CONFIRMATION_THRESHOLD,
    ENSEMBLE_THRESHOLD, TREND_STRENGTH_THRESHOLD,
    ADAPTIVE_THRESHOLDS, ENABLE_ADVANCED_TECH
)
from data.models import State
from agents.risk_management import RiskManager
from agents.market_risk_assessment import MarketRiskAssessor

# Configure logging
logger = logging.getLogger(__name__)

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    logger.warning("TA-Lib not available, using basic technical analysis")
    TALIB_AVAILABLE = False


class StrategyConfig:
    """Configuration class for trading strategies."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        risk_management: Dict[str, Any] = None,
        position_sizing: Dict[str, Any] = None
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.risk_management = risk_management or {
            'max_drawdown': 0.1,
            'max_position_size': 0.1,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.1
        }
        self.position_sizing = position_sizing or {
            'method': 'fixed_percentage',
            'percentage': 0.05
        }


class TradingSignal:
    """Represents a trading signal with metadata."""

    def __init__(
        self,
        symbol: str,
        action: str,  # 'BUY', 'SELL', 'HOLD'
        confidence: float,
        price: float,
        timestamp: datetime,
        reason: str = "",
        metadata: Dict[str, Any] = None
    ):
        self.symbol = symbol
        self.action = action.upper()
        self.confidence = confidence
        self.price = price
        self.timestamp = timestamp
        self.reason = reason
        self.metadata = metadata or {}


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def generate_signals(
        self,
        data: pd.DataFrame,
        state: Optional[State] = None
    ) -> List[TradingSignal]:
        """
        Generate trading signals for the given data.

        Args:
            data: Historical price data
            state: Current LangGraph state (optional)

        Returns:
            List of trading signals
        """
        pass

    @abstractmethod
    def validate_signal(self, signal: TradingSignal, data: pd.DataFrame) -> bool:
        """
        Validate a trading signal based on strategy rules.

        Args:
            signal: Trading signal to validate
            data: Historical price data

        Returns:
            True if signal is valid, False otherwise
        """
        pass

    def calculate_position_size(
        self,
        capital: float,
        price: float,
        risk_per_trade: float = 0.01
    ) -> int:
        """Calculate position size based on risk management."""
        if self.config.position_sizing['method'] == 'fixed_percentage':
            position_value = capital * self.config.position_sizing['percentage']
            return int(position_value / price)
        elif self.config.position_sizing['method'] == 'risk_based':
            risk_amount = capital * risk_per_trade
            stop_loss_distance = price * self.config.risk_management['stop_loss_pct']
            return int(risk_amount / stop_loss_distance)
        else:
            return 1  # Minimum position

    def apply_risk_management(
        self,
        signal: TradingSignal,
        current_portfolio_value: float,
        current_positions: Dict[str, int]
    ) -> TradingSignal:
        """Apply risk management rules to the signal."""
        # Check maximum drawdown
        if current_portfolio_value < (1 - self.config.risk_management['max_drawdown']):
            signal.action = 'HOLD'
            signal.reason += " | Risk: Max drawdown exceeded"

        # Check position size limits
        current_position = current_positions.get(signal.symbol, 0)
        max_position = int(current_portfolio_value * self.config.risk_management['max_position_size'] / signal.price)

        if abs(current_position) >= max_position:
            signal.action = 'HOLD'
            signal.reason += f" | Risk: Position limit ({max_position}) exceeded"

        return signal


class TrendFollowingStrategy(BaseStrategy):
    """Trend Following Strategy with ML confirmation."""

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.trend_periods = config.parameters.get('trend_periods', [20, 50, 200])
        self.ml_confirmation = config.parameters.get('ml_confirmation', True)

    def generate_signals(
        self,
        data: pd.DataFrame,
        state: Optional[State] = None
    ) -> List[TradingSignal]:
        """Generate trend-following signals."""
        signals = []

        if len(data) < max(self.trend_periods):
            return signals

        # Calculate trend indicators
        trend_signals = self._calculate_trend_indicators(data)

        # Get current price and timestamp
        current_price = data['Close'].iloc[-1]
        current_time = data.index[-1] if isinstance(data.index[-1], pd.Timestamp) else datetime.now()

        # Generate signals based on trend strength
        if trend_signals['strong_uptrend']:
            signal = TradingSignal(
                symbol=data.get('symbol', 'UNKNOWN'),
                action='BUY',
                confidence=trend_signals['trend_strength'],
                price=current_price,
                timestamp=current_time,
                reason=f"Strong uptrend detected (strength: {trend_signals['trend_strength']:.2f})"
            )
            signals.append(signal)

        elif trend_signals['strong_downtrend']:
            signal = TradingSignal(
                symbol=data.get('symbol', 'UNKNOWN'),
                action='SELL',
                confidence=trend_signals['trend_strength'],
                price=current_price,
                timestamp=current_time,
                reason=f"Strong downtrend detected (strength: {trend_signals['trend_strength']:.2f})"
            )
            signals.append(signal)

        # Apply ML confirmation if enabled
        if self.ml_confirmation and state and 'ml_predictions' in state:
            signals = self._apply_ml_confirmation(signals, state)

        return signals

    def _calculate_trend_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trend strength indicators."""
        result = {
            'strong_uptrend': False,
            'strong_downtrend': False,
            'trend_strength': 0.5
        }

        try:
            # Calculate moving averages
            sma_short = data['Close'].rolling(self.trend_periods[0]).mean()
            sma_medium = data['Close'].rolling(self.trend_periods[1]).mean()
            sma_long = data['Close'].rolling(self.trend_periods[2]).mean()

            # Calculate slope of short-term MA
            if len(sma_short) >= 5:
                slope = (sma_short.iloc[-1] - sma_short.iloc[-5]) / 5
                slope_pct = slope / sma_short.iloc[-1]

                # Trend strength based on MA alignment and slope
                ma_alignment = 0
                if sma_short.iloc[-1] > sma_medium.iloc[-1] > sma_long.iloc[-1]:
                    ma_alignment = 1  # Bullish alignment
                elif sma_short.iloc[-1] < sma_medium.iloc[-1] < sma_long.iloc[-1]:
                    ma_alignment = -1  # Bearish alignment

                # Combine slope and alignment for trend strength
                trend_strength = abs(slope_pct) * 2 + ma_alignment * 0.3
                trend_strength = min(max(trend_strength, 0), 1)

                result['trend_strength'] = trend_strength
                result['strong_uptrend'] = trend_strength > TREND_STRENGTH_THRESHOLD and ma_alignment > 0
                result['strong_downtrend'] = trend_strength > TREND_STRENGTH_THRESHOLD and ma_alignment < 0

        except Exception as e:
            self.logger.error(f"Error calculating trend indicators: {e}")

        return result

    def _apply_ml_confirmation(self, signals: List[TradingSignal], state: State) -> List[TradingSignal]:
        """Apply ML model confirmation to signals."""
        confirmed_signals = []

        for signal in signals:
            symbol = signal.symbol
            if symbol in state.get('ml_predictions', {}):
                ml_pred = state['ml_predictions'][symbol]
                latest_pred = ml_pred.get('latest_prediction', {})

                # Check if ML prediction aligns with signal
                if latest_pred.get('ensemble_prediction') == (1 if signal.action == 'BUY' else 0):
                    confidence_boost = latest_pred.get('confidence_score', 0) * 0.2
                    signal.confidence = min(signal.confidence + confidence_boost, 1.0)
                    signal.reason += f" | ML Confirmed (+{confidence_boost:.2f})"
                    confirmed_signals.append(signal)
                else:
                    # Reduce confidence if ML disagrees
                    signal.confidence *= 0.7
                    signal.reason += " | ML Disagrees"
                    if signal.confidence > 0.3:  # Still keep if confidence is reasonable
                        confirmed_signals.append(signal)
            else:
                confirmed_signals.append(signal)

        return confirmed_signals

    def validate_signal(self, signal: TradingSignal, data: pd.DataFrame) -> bool:
        """Validate trend-following signal."""
        if len(data) < max(self.trend_periods):
            return False

        # Check if signal aligns with current trend
        trend_indicators = self._calculate_trend_indicators(data)

        if signal.action == 'BUY':
            return trend_indicators['strong_uptrend']
        elif signal.action == 'SELL':
            return trend_indicators['strong_downtrend']

        return True


class MeanReversionStrategy(BaseStrategy):
    """Mean Reversion Strategy with pattern recognition."""

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.lookback_period = config.parameters.get('lookback_period', 20)
        self.deviation_threshold = config.parameters.get('deviation_threshold', 2.0)
        self.rsi_overbought = config.parameters.get('rsi_overbought', RSI_OVERBOUGHT)
        self.rsi_oversold = config.parameters.get('rsi_oversold', RSI_OVERSOLD)

    def generate_signals(
        self,
        data: pd.DataFrame,
        state: Optional[State] = None
    ) -> List[TradingSignal]:
        """Generate mean reversion signals."""
        signals = []

        if len(data) < self.lookback_period:
            return signals

        # Calculate mean reversion indicators
        mr_signals = self._calculate_mean_reversion_indicators(data)

        # Get current price and timestamp
        current_price = data['Close'].iloc[-1]
        current_time = data.index[-1] if isinstance(data.index[-1], pd.Timestamp) else datetime.now()

        # Generate signals based on deviation from mean
        if mr_signals['oversold_condition']:
            signal = TradingSignal(
                symbol=data.get('symbol', 'UNKNOWN'),
                action='BUY',
                confidence=mr_signals['reversion_probability'],
                price=current_price,
                timestamp=current_time,
                reason=f"Oversold condition (RSI: {mr_signals['rsi']:.1f}, Deviation: {mr_signals['z_score']:.2f})"
            )
            signals.append(signal)

        elif mr_signals['overbought_condition']:
            signal = TradingSignal(
                symbol=data.get('symbol', 'UNKNOWN'),
                action='SELL',
                confidence=mr_signals['reversion_probability'],
                price=current_price,
                timestamp=current_time,
                reason=f"Overbought condition (RSI: {mr_signals['rsi']:.1f}, Deviation: {mr_signals['z_score']:.2f})"
            )
            signals.append(signal)

        return signals

    def _calculate_mean_reversion_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate mean reversion indicators."""
        result = {
            'oversold_condition': False,
            'overbought_condition': False,
            'reversion_probability': 0.5,
            'rsi': 50.0,
            'z_score': 0.0
        }

        try:
            # Calculate RSI
            if TALIB_AVAILABLE:
                rsi = talib.RSI(data['Close'].values, timeperiod=14)
                result['rsi'] = rsi[-1] if not np.isnan(rsi[-1]) else 50.0
            else:
                # Basic RSI calculation
                price_changes = data['Close'].diff()
                gains = price_changes.where(price_changes > 0, 0)
                losses = -price_changes.where(price_changes < 0, 0)
                avg_gain = gains.rolling(14).mean()
                avg_loss = losses.rolling(14).mean()
                rs = avg_gain / avg_loss.replace(0, 1e-9)
                result['rsi'] = 100 - (100 / (1 + rs.iloc[-1])) if not pd.isna(rs.iloc[-1]) else 50.0

            # Calculate Bollinger Bands
            sma = data['Close'].rolling(self.lookback_period).mean()
            std = data['Close'].rolling(self.lookback_period).std()
            upper_band = sma + (std * self.deviation_threshold)
            lower_band = sma - (std * self.deviation_threshold)

            # Calculate Z-score
            current_price = data['Close'].iloc[-1]
            result['z_score'] = (current_price - sma.iloc[-1]) / std.iloc[-1] if std.iloc[-1] != 0 else 0

            # Determine conditions
            result['oversold_condition'] = (
                result['rsi'] < self.rsi_oversold or
                current_price < lower_band.iloc[-1]
            )
            result['overbought_condition'] = (
                result['rsi'] > self.rsi_overbought or
                current_price > upper_band.iloc[-1]
            )

            # Calculate reversion probability based on deviation
            deviation_pct = abs(result['z_score']) / self.deviation_threshold
            result['reversion_probability'] = min(deviation_pct, 1.0)

        except Exception as e:
            self.logger.error(f"Error calculating mean reversion indicators: {e}")

        return result

    def validate_signal(self, signal: TradingSignal, data: pd.DataFrame) -> bool:
        """Validate mean reversion signal."""
        if len(data) < self.lookback_period:
            return False

        mr_indicators = self._calculate_mean_reversion_indicators(data)

        if signal.action == 'BUY':
            return mr_indicators['oversold_condition']
        elif signal.action == 'SELL':
            return mr_indicators['overbought_condition']

        return True


class BreakoutStrategy(BaseStrategy):
    """Breakout Trading Strategy with volume analysis."""

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.consolidation_period = config.parameters.get('consolidation_period', 20)
        self.breakout_threshold = config.parameters.get('breakout_threshold', 0.02)
        self.volume_multiplier = config.parameters.get('volume_multiplier', 1.5)

    def generate_signals(
        self,
        data: pd.DataFrame,
        state: Optional[State] = None
    ) -> List[TradingSignal]:
        """Generate breakout signals."""
        signals = []

        if len(data) < self.consolidation_period + 5:
            return signals

        # Calculate breakout indicators
        breakout_signals = self._calculate_breakout_indicators(data)

        # Get current price and timestamp
        current_price = data['Close'].iloc[-1]
        current_time = data.index[-1] if isinstance(data.index[-1], pd.Timestamp) else datetime.now()

        # Generate signals based on breakout conditions
        if breakout_signals['bullish_breakout']:
            signal = TradingSignal(
                symbol=data.get('symbol', 'UNKNOWN'),
                action='BUY',
                confidence=breakout_signals['breakout_strength'],
                price=current_price,
                timestamp=current_time,
                reason=f"Bullish breakout detected (Strength: {breakout_signals['breakout_strength']:.2f})"
            )
            signals.append(signal)

        elif breakout_signals['bearish_breakout']:
            signal = TradingSignal(
                symbol=data.get('symbol', 'UNKNOWN'),
                action='SELL',
                confidence=breakout_signals['breakout_strength'],
                price=current_price,
                timestamp=current_time,
                reason=f"Bearish breakout detected (Strength: {breakout_signals['breakout_strength']:.2f})"
            )
            signals.append(signal)

        return signals

    def _calculate_breakout_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate breakout indicators."""
        result = {
            'bullish_breakout': False,
            'bearish_breakout': False,
            'breakout_strength': 0.5
        }

        try:
            # Calculate consolidation range
            high_max = data['High'].rolling(self.consolidation_period).max()
            low_min = data['Low'].rolling(self.consolidation_period).min()
            consolidation_range = high_max - low_min

            # Calculate volume confirmation
            avg_volume = data['Volume'].rolling(self.consolidation_period).mean()
            current_volume = data['Volume'].iloc[-1]

            # Check for breakout
            current_price = data['Close'].iloc[-1]
            prev_high = high_max.iloc[-2] if len(high_max) > 1 else high_max.iloc[-1]
            prev_low = low_min.iloc[-2] if len(low_min) > 1 else low_min.iloc[-1]

            # Bullish breakout: price breaks above recent high with volume confirmation
            if current_price > prev_high * (1 + self.breakout_threshold):
                volume_confirmed = current_volume > avg_volume.iloc[-1] * self.volume_multiplier
                if volume_confirmed:
                    result['bullish_breakout'] = True
                    # Strength based on how far above resistance and volume
                    breakout_pct = (current_price - prev_high) / prev_high
                    volume_ratio = current_volume / avg_volume.iloc[-1]
                    result['breakout_strength'] = min((breakout_pct * 10 + volume_ratio - 1) / 2, 1.0)

            # Bearish breakout: price breaks below recent low with volume confirmation
            elif current_price < prev_low * (1 - self.breakout_threshold):
                volume_confirmed = current_volume > avg_volume.iloc[-1] * self.volume_multiplier
                if volume_confirmed:
                    result['bearish_breakout'] = True
                    # Strength based on how far below support and volume
                    breakout_pct = (prev_low - current_price) / prev_low
                    volume_ratio = current_volume / avg_volume.iloc[-1]
                    result['breakout_strength'] = min((breakout_pct * 10 + volume_ratio - 1) / 2, 1.0)

        except Exception as e:
            self.logger.error(f"Error calculating breakout indicators: {e}")

        return result

    def validate_signal(self, signal: TradingSignal, data: pd.DataFrame) -> bool:
        """Validate breakout signal."""
        if len(data) < self.consolidation_period + 5:
            return False

        breakout_indicators = self._calculate_breakout_indicators(data)

        if signal.action == 'BUY':
            return breakout_indicators['bullish_breakout']
        elif signal.action == 'SELL':
            return breakout_indicators['bearish_breakout']

        return True


class SentimentDrivenStrategy(BaseStrategy):
    """Sentiment-Driven Strategy using news and social media data."""

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.sentiment_threshold = config.parameters.get('sentiment_threshold', 0.1)
        self.volume_threshold = config.parameters.get('volume_threshold', 1.2)
        self.news_lookback_days = config.parameters.get('news_lookback_days', 7)

    def generate_signals(
        self,
        data: pd.DataFrame,
        state: Optional[State] = None
    ) -> List[TradingSignal]:
        """Generate sentiment-driven signals."""
        signals = []

        if not state or 'sentiment_scores' not in state:
            return signals

        symbol = data.get('symbol', 'UNKNOWN')
        if symbol not in state['sentiment_scores']:
            return signals

        # Get sentiment data
        sentiment_data = state['sentiment_scores'][symbol]
        if 'error' in sentiment_data:
            return signals

        # Calculate sentiment indicators
        sentiment_signals = self._calculate_sentiment_indicators(sentiment_data, data)

        # Get current price and timestamp
        current_price = data['Close'].iloc[-1]
        current_time = data.index[-1] if isinstance(data.index[-1], pd.Timestamp) else datetime.now()

        # Generate signals based on sentiment
        if sentiment_signals['bullish_sentiment']:
            signal = TradingSignal(
                symbol=symbol,
                action='BUY',
                confidence=sentiment_signals['sentiment_strength'],
                price=current_price,
                timestamp=current_time,
                reason=f"Bullish sentiment detected (Score: {sentiment_signals['compound_score']:.2f})"
            )
            signals.append(signal)

        elif sentiment_signals['bearish_sentiment']:
            signal = TradingSignal(
                symbol=symbol,
                action='SELL',
                confidence=sentiment_signals['sentiment_strength'],
                price=current_price,
                timestamp=current_time,
                reason=f"Bearish sentiment detected (Score: {sentiment_signals['compound_score']:.2f})"
            )
            signals.append(signal)

        return signals

    def _calculate_sentiment_indicators(self, sentiment_data: Dict, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate sentiment-based indicators."""
        result = {
            'bullish_sentiment': False,
            'bearish_sentiment': False,
            'sentiment_strength': 0.5,
            'compound_score': 0.0
        }

        try:
            compound_score = sentiment_data.get('compound', 0.0)
            result['compound_score'] = compound_score

            # Check volume confirmation for sentiment signals
            if len(data) >= 5:
                avg_volume = data['Volume'].tail(5).mean()
                current_volume = data['Volume'].iloc[-1]
                volume_confirmed = current_volume > avg_volume * self.volume_threshold
            else:
                volume_confirmed = True

            # Determine sentiment direction
            if compound_score > self.sentiment_threshold and volume_confirmed:
                result['bullish_sentiment'] = True
                result['sentiment_strength'] = min(abs(compound_score), 1.0)
            elif compound_score < -self.sentiment_threshold and volume_confirmed:
                result['bearish_sentiment'] = True
                result['sentiment_strength'] = min(abs(compound_score), 1.0)

        except Exception as e:
            self.logger.error(f"Error calculating sentiment indicators: {e}")

        return result

    def validate_signal(self, signal: TradingSignal, data: pd.DataFrame) -> bool:
        """Validate sentiment-driven signal."""
        # Sentiment signals are primarily validated by the sentiment data itself
        return signal.confidence > 0.3


class EnsembleStrategy(BaseStrategy):
    """Ensemble Strategy combining multiple approaches with ML weighting."""

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.strategies = config.parameters.get('strategies', [])
        self.weights = config.parameters.get('weights', {})
        self.confidence_threshold = config.parameters.get('confidence_threshold', ENSEMBLE_THRESHOLD)

    def generate_signals(
        self,
        data: pd.DataFrame,
        state: Optional[State] = None
    ) -> List[TradingSignal]:
        """Generate ensemble signals by combining multiple strategies."""
        if not self.strategies:
            return []

        # Collect signals from all strategies
        all_signals = []
        strategy_weights = []

        for strategy in self.strategies:
            try:
                signals = strategy.generate_signals(data, state)
                if signals:
                    all_signals.extend(signals)
                    # Use strategy confidence as weight
                    weight = signals[0].confidence if signals else 0.5
                    strategy_weights.extend([weight] * len(signals))
            except Exception as e:
                self.logger.error(f"Error generating signals for {strategy.__class__.__name__}: {e}")

        if not all_signals:
            return []

        # Aggregate signals
        ensemble_signal = self._aggregate_signals(all_signals, strategy_weights)

        # Get current price and timestamp
        current_price = data['Close'].iloc[-1]
        current_time = data.index[-1] if isinstance(data.index[-1], pd.Timestamp) else datetime.now()

        # Generate final signal if confidence is high enough
        if ensemble_signal['confidence'] > self.confidence_threshold:
            signal = TradingSignal(
                symbol=data.get('symbol', 'UNKNOWN'),
                action=ensemble_signal['action'],
                confidence=ensemble_signal['confidence'],
                price=current_price,
                timestamp=current_time,
                reason=f"Ensemble signal: {ensemble_signal['action']} (Confidence: {ensemble_signal['confidence']:.2f})"
            )
            return [signal]

        return []

    def _aggregate_signals(self, signals: List[TradingSignal], weights: List[float]) -> Dict[str, Any]:
        """Aggregate signals from multiple strategies."""
        if not signals:
            return {'action': 'HOLD', 'confidence': 0.0}

        # Count votes for each action
        buy_votes = 0
        sell_votes = 0
        total_weight = 0

        for signal, weight in zip(signals, weights):
            if signal.action == 'BUY':
                buy_votes += weight
            elif signal.action == 'SELL':
                sell_votes += weight
            total_weight += weight

        # Determine ensemble action
        if buy_votes > sell_votes:
            action = 'BUY'
            confidence = buy_votes / total_weight if total_weight > 0 else 0.5
        elif sell_votes > buy_votes:
            action = 'SELL'
            confidence = sell_votes / total_weight if total_weight > 0 else 0.5
        else:
            action = 'HOLD'
            confidence = 0.5

        return {
            'action': action,
            'confidence': confidence,
            'buy_votes': buy_votes,
            'sell_votes': sell_votes
        }

    def validate_signal(self, signal: TradingSignal, data: pd.DataFrame) -> bool:
        """Validate ensemble signal."""
        return signal.confidence > self.confidence_threshold


# Strategy Factory
class StrategyFactory:
    """Factory class for creating trading strategies."""

    @staticmethod
    def create_strategy(strategy_type: str, config: StrategyConfig) -> BaseStrategy:
        """Create a strategy instance based on type."""
        strategies = {
            'trend_following': TrendFollowingStrategy,
            'mean_reversion': MeanReversionStrategy,
            'breakout': BreakoutStrategy,
            'sentiment_driven': SentimentDrivenStrategy,
            'ensemble': EnsembleStrategy
        }

        if strategy_type not in strategies:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

        return strategies[strategy_type](config)

    @staticmethod
    def get_default_configs() -> Dict[str, StrategyConfig]:
        """Get default configurations for all strategies."""
        return {
            'trend_following': StrategyConfig(
                name='Trend Following',
                description='Follows market trends with ML confirmation',
                parameters={
                    'trend_periods': [20, 50, 200],
                    'ml_confirmation': True
                }
            ),
            'mean_reversion': StrategyConfig(
                name='Mean Reversion',
                description='Trades on price deviations from mean',
                parameters={
                    'lookback_period': 20,
                    'deviation_threshold': 2.0
                }
            ),
            'breakout': StrategyConfig(
                name='Breakout Trading',
                description='Trades on price breakouts with volume confirmation',
                parameters={
                    'consolidation_period': 20,
                    'breakout_threshold': 0.02,
                    'volume_multiplier': 1.5
                }
            ),
            'sentiment_driven': StrategyConfig(
                name='Sentiment Driven',
                description='Trades based on news and social sentiment',
                parameters={
                    'sentiment_threshold': 0.1,
                    'volume_threshold': 1.2,
                    'news_lookback_days': 7
                }
            )
        }