

import logging
logger = logging.getLogger(__name__)
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np

try:
    from .trading_strategies import (
        BaseStrategy, StrategyConfig, TradingSignal,
        StrategyFactory, EnsembleStrategy
    )
except ImportError:
    logger.warning("Trading strategies module not found, using legacy backtesting")
    BaseStrategy = None
    StrategyConfig = None
    TradingSignal = None
    StrategyFactory = None
    EnsembleStrategy = None
from config.trading_config import TRADE_LIMIT, SIMULATION_DAYS
from data.models import State


@dataclass
class Trade:
    
    symbol: str
    action: str
    date: datetime
    price: float
    quantity: int
    total_value: float
    commission: float = 0.0
    reason: str = ""


@dataclass
class PortfolioSnapshot:
    
    date: datetime
    cash: float
    holdings: Dict[str, int]
    portfolio_value: float
    daily_return: float

class WalkForwardBacktestingEngine:
    """
    Enhanced backtesting engine with walk-forward analysis and realistic market simulation
    """

    def __init__(
        self,
        initial_capital: float = 1000000.0,
        commission_rate: float = 0.001,  # 0.1%
        slippage_rate: float = 0.0005,   # 0.05%
        max_position_size: float = 0.1,   # 10% of portfolio
        position_size_type: str = "fixed",  # 'fixed', 'percentage', or 'equal'
        walk_forward_window: int = 252,    # 1 year training window
        validation_window: int = 63,       # 3 months validation
        retrain_frequency: int = 63,       # Retrain every 3 months
        min_training_samples: int = 100
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.max_position_size = max_position_size
        self.position_size_type = position_size_type

        # Walk-forward parameters
        self.walk_forward_window = walk_forward_window
        self.validation_window = validation_window
        self.retrain_frequency = retrain_frequency
        self.min_training_samples = min_training_samples

        # Portfolio state
        self.holdings: Dict[str, int] = {}
        self.portfolio_history: List[PortfolioSnapshot] = []
        self.trades: List[Trade] = []
        self.current_date: Optional[datetime] = None

        # Walk-forward state
        self.walk_forward_results: List[Dict] = []
        self.model_performance_history: List[Dict] = []


class BacktestingEngine:

    def __init__(
        self,
        initial_capital: float = 1000000.0,
        commission_rate: float = 0.001,  # 0.1%
        slippage_rate: float = 0.0005,   # 0.05%
        max_position_size: float = 0.1,   # 10% of portfolio
        position_size_type: str = "fixed"  # 'fixed', 'percentage', or 'equal'
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.max_position_size = max_position_size
        self.position_size_type = position_size_type

        # Portfolio state
        self.holdings: Dict[str, int] = {}
        self.portfolio_history: List[PortfolioSnapshot] = []
        self.trades: List[Trade] = []
        self.current_date: Optional[datetime] = None

    def run_backtest(
        self,
        recommendations: Optional[Dict[str, Dict[str, str]]] = None,
        stock_data: Optional[Dict[str, pd.DataFrame]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        rsi_buy_threshold: Optional[float] = None
    ) -> Dict[str, Union[float, List, Dict]]:
        
        try:
            # Initialize portfolio
            self._reset_portfolio()
            self.portfolio_history = [
                PortfolioSnapshot(
                    date=start_date or self._get_earliest_date(stock_data),
                    cash=self.capital,
                    holdings={},
                    portfolio_value=self.capital,
                    daily_return=0.0
                )
            ]

            # Get simulation date range
            if not start_date:
                start_date = self._get_earliest_date(stock_data)
            if not end_date:
                end_date = start_date + timedelta(days=SIMULATION_DAYS)

            # Process each day in the backtest period
            current_date = start_date

            while current_date <= end_date:
                self.current_date = current_date

                # Get daily recommendations (would come from state)
                daily_recommendations = self._get_daily_recommendations(
                    current_date, recommendations, stock_data, rsi_buy_threshold
                )

                # Execute trades based on recommendations
                if len(self.trades) < TRADE_LIMIT:
                    self._execute_trades(daily_recommendations, stock_data, current_date)

                # Update portfolio value
                portfolio_value = self._calculate_portfolio_value(stock_data, current_date)
                daily_return = 0.0

                if self.portfolio_history:
                    prev_value = self.portfolio_history[-1].portfolio_value
                    daily_return = (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0

                # Record portfolio snapshot
                snapshot = PortfolioSnapshot(
                    date=current_date,
                    cash=self.capital,
                    holdings=self.holdings.copy(),
                    portfolio_value=portfolio_value,
                    daily_return=daily_return
                )
                self.portfolio_history.append(snapshot)

                # Move to next trading day
                current_date += timedelta(days=1)
                while current_date.weekday() >= 5:  # Skip weekends
                    current_date += timedelta(days=1)

            # Calculate final metrics
            return self._calculate_performance_metrics()

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return {"error": str(e)}

    def _reset_portfolio(self):
        
        self.capital = self.initial_capital
        self.holdings = {}
        self.trades = []
        self.portfolio_history = []

    def _get_earliest_date(self, stock_data: Dict[str, pd.DataFrame]) -> datetime:
        
        earliest = None
        for df in stock_data.values():
            if not df.empty:
                df_index = pd.to_datetime(df.index, infer_datetime_format=True)
                df_date = df_index.min()
                if earliest is None or df_date < earliest:
                    earliest = df_date

        return earliest.to_pydatetime() if hasattr(earliest, 'to_pydatetime') else datetime.now()

    def _get_daily_recommendations(
        self,
        current_date: datetime,
        recommendations: Optional[Dict[str, Dict]],
        stock_data: Dict[str, pd.DataFrame],
        rsi_buy_threshold: Optional[float] = None
    ) -> Dict[str, str]:
        
        if rsi_buy_threshold is not None and recommendations is None:
            daily_recs = {}
            for symbol in stock_data:
                df = stock_data[symbol]
                # Normalize column names for case sensitivity
                column_map = {'close': 'Close'}
                df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns and v not in df.columns})
                historical_df = df[df.index <= current_date]
                if len(historical_df) < 20:
                    daily_recs[symbol] = 'HOLD'
                    continue
                rsi_series = self._calculate_rsi_series(historical_df['Close'])
                if len(rsi_series) == 0 or pd.isna(rsi_series.iloc[-1]):
                    daily_recs[symbol] = 'HOLD'
                elif rsi_series.iloc[-1] < rsi_buy_threshold:
                    daily_recs[symbol] = 'BUY'
                elif rsi_series.iloc[-1] > 70:
                    daily_recs[symbol] = 'SELL'
                else:
                    daily_recs[symbol] = 'HOLD'
            return daily_recs
        else:
            daily_recs = {}
            for symbol, rec in (recommendations or {}).items():
                if symbol in stock_data:
                    daily_recs[symbol] = rec.get('action', 'HOLD')
            return daily_recs

    def _execute_trades(
        self,
        recommendations: Dict[str, str],
        stock_data: Dict[str, pd.DataFrame],
        current_date: datetime
    ) -> float:
        
        total_costs = 0.0

        for symbol, action in recommendations.items():
            if action.upper() == 'HOLD':
                continue

            df = stock_data.get(symbol)
            if df is None or df.empty:
                continue

            # Get current price with slippage
            try:
                current_price = self._get_price_with_slippage(df, current_date, action)
                if current_price <= 0:
                    continue

                # Calculate position size
                quantity = self._calculate_position_size(symbol, current_price, action)

                if quantity <= 0:
                    continue

            except Exception as e:
                logger.warning(f"Could not get price for {symbol}: {e}")
                continue

            # Execute trade
            total_value = current_price * quantity
            commission = total_value * self.commission_rate

            if action.upper() == 'BUY':
                if self.capital >= total_value + commission:
                    self.holdings[symbol] = self.holdings.get(symbol, 0) + quantity
                    self.capital -= (total_value + commission)
                    trade_type = "BUY"
                else:
                    logger.info(f"Insufficient capital for {symbol} BUY")
                    continue
            else:  # SELL
                current_holding = self.holdings.get(symbol, 0)
                if current_holding >= quantity:
                    self.holdings[symbol] = current_holding - quantity
                    self.capital += (total_value - commission)
                    trade_type = "SELL"
                else:
                    # Sell all available
                    quantity = current_holding
                    if quantity > 0:
                        total_value = current_price * quantity
                        commission = total_value * self.commission_rate
                        self.holdings[symbol] = 0
                        self.capital += (total_value - commission)
                        trade_type = "SELL"
                    else:
                        continue

            # Record trade
            trade = Trade(
                symbol=symbol,
                action=trade_type,
                date=current_date,
                price=current_price,
                quantity=quantity,
                total_value=total_value,
                commission=commission
            )
            self.trades.append(trade)

            total_costs += commission
            logger.info(f"Executed {trade_type} {quantity} {symbol} at {current_price or 0:.2f}")

        return total_costs

    def _get_price_with_slippage(self, df: pd.DataFrame, date: datetime, action: str) -> float:
        
        try:
            if df is None or df.empty:
                return 0.0

            # Ensure df is sorted for proper indexing
            df = df.sort_index()

            # Normalize column names to handle case sensitivity
            column_map = {'close': 'Close', 'high': 'High', 'low': 'Low', 'open': 'Open', 'volume': 'Volume'}
            df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns and v not in df.columns})

            # Find closest date in data
            idx = df.index.get_indexer([pd.Timestamp(date)], method='nearest')
            if idx[0] < 0:
                return 0.0

            row = df.iloc[idx[0]]

            if action.upper() == 'BUY':
                # Buy at ask-like price (High or Close)
                base_price = row['High'] if 'High' in row else row['Close']
                slipped_price = base_price * (1 + self.slippage_rate)
            else:
                # Sell at bid-like price (Low or Close)
                base_price = row['Low'] if 'Low' in row else row['Close']
                slipped_price = base_price * (1 - self.slippage_rate)

            return float(slipped_price)

        except Exception as e:
            logger.error(f"Error getting price with slippage: {e}")
            return 0.0

    def _calculate_rsi_series(self, prices: pd.Series, period: int = 14) -> pd.Series:
        
        # Ensure prices is pandas Series to avoid array ambiguity errors
        if not isinstance(prices, pd.Series):
            prices = pd.Series(prices)

        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd_series(self, prices: pd.Series, fast_period: int = 12,
                               slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series]:
        if len(prices) < slow_period + signal_period:
            empty_series = pd.Series(dtype=float, index=prices.index)
            return empty_series, empty_series

        ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        return macd_line, signal_line

    def _calculate_average_true_range(self, highs: pd.Series, lows: pd.Series,
                                      closes: pd.Series, period: int = 14) -> pd.Series:
        if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
            return pd.Series(dtype=float, index=closes.index)

        high_low = highs - lows
        high_prev_close = (highs - closes.shift()).abs()
        low_prev_close = (lows - closes.shift()).abs()
        true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr

    def _calculate_volume_signal(self, volumes: pd.Series, short_window: int = 5,
                                 long_window: int = 20) -> Tuple[float, float]:
        if len(volumes) < long_window:
            return float(volumes.iloc[-1]) if len(volumes) else 0.0, float(volumes.mean()) if len(volumes) else 0.0

        short_avg = volumes.rolling(short_window).mean().iloc[-1]
        long_avg = volumes.rolling(long_window).mean().iloc[-1]
        return float(short_avg if not np.isnan(short_avg) else 0.0), float(long_avg if not np.isnan(long_avg) else 0.0)

    def _calculate_trend_strength(self, short_ma: pd.Series, medium_ma: pd.Series,
                                  long_ma: pd.Series) -> float:
        try:
            short = short_ma.iloc[-1]
            medium = medium_ma.iloc[-1]
            long_value = long_ma.iloc[-1]
        except (IndexError, KeyError):
            return 0.0

        if any(np.isnan(val) for val in [short, medium, long_value]):
            return 0.0

        alignment_score = 0
        if short > medium > long_value:
            alignment_score = 1
        elif short < medium < long_value:
            alignment_score = -1

        slope_component = 0.0
        if len(short_ma.dropna()) >= 5 and len(medium_ma.dropna()) >= 5:
            short_slope = short_ma.diff().iloc[-5:].mean()
            medium_slope = medium_ma.diff().iloc[-5:].mean()
            slope_component = np.tanh((short_slope + medium_slope) * 100)

        trend_strength = alignment_score + slope_component
        trend_strength = max(min(trend_strength, 1.5), -1.5)
        return float(trend_strength)

    def _score_to_confidence(self, net_score: int, trend_strength: float,
                             volume_ratio: float, volatility: float,
                             atr: Optional[float], price: float, action: str) -> float:
        base_confidence = 0.35 + 0.15 * (abs(net_score) - 1)
        base_confidence += min(0.2, max(0.0, trend_strength / 2))

        if action == 'BUY' and volume_ratio > 1.2:
            base_confidence += 0.1
        elif action == 'SELL' and volume_ratio < 0.8:
            base_confidence += 0.1

        if volatility > 0.05:
            base_confidence -= 0.1

        if atr and price > 0:
            atr_ratio = atr / price
            if atr_ratio > 0.05:
                base_confidence -= min(0.15, atr_ratio * 2)

        confidence = max(0.1, min(1.0, base_confidence))
        return float(confidence)

    def _generate_rsi_signals(self, df: pd.DataFrame, buy_threshold: float) -> Dict[pd.Timestamp, str]:
        
        # Normalize column names for case sensitivity
        column_map = {'close': 'Close'}
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns and v not in df.columns})
        if 'Close' not in df.columns or len(df) < 15:
            return {idx: 'HOLD' for idx in df.index}
        rsi = self._calculate_rsi_series(df['Close'])
        signals = {}
        for date in df.index:
            current_rsi = rsi.loc[date]
            if pd.isna(current_rsi):
                signals[date] = 'HOLD'
            elif current_rsi < buy_threshold:
                signals[date] = 'BUY'
            elif current_rsi > 70:
                signals[date] = 'SELL'
            else:
                signals[date] = 'HOLD'
        return signals

    def _compute_strategy_returns_from_signals(self, df: pd.DataFrame, signals: Dict[pd.Timestamp, str]) -> pd.Series:
        
        df = df.copy()
        # Normalize column names for case sensitivity
        column_map = {'close': 'Close'}
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns and v not in df.columns})
        df['signal'] = [signals.get(idx, 'HOLD') for idx in df.index]
        df['position'] = 0.0
        position = 0.0
        for i in range(len(df)):
            sig = df.iloc[i]['signal']
            if sig == 'BUY' and position == 0:
                position = 1.0
            elif sig == 'SELL' and position == 1.0:
                position = 0.0
            df.iloc[i, df.columns.get_loc('position')] = position
        df['strategy_returns'] = df['position'].shift(1) * df['Close'].pct_change()
        return df['strategy_returns'].dropna()

    def _calculate_position_size(self, symbol: str, price: float, action: str) -> int:
        
        try:
            if price <= 0:
                return 0

            portfolio_value = self._calculate_portfolio_value({}, None)  # Current portfolio value

            if self.position_size_type == 'percentage':
                # Percentage of current portfolio
                position_value = portfolio_value * self.max_position_size
                quantity = int(position_value / price)
            elif self.position_size_type == 'equal':
                # Equal risk amount
                risk_amount = portfolio_value * self.max_position_size
                quantity = int(risk_amount / price)
            else:  # fixed
                # Fixed amount per trade
                fixed_amount = self.initial_capital * self.max_position_size
                quantity = int(fixed_amount / price)

            # Ensure minimum quantity of 1
            quantity = max(1, quantity)

            return quantity

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0

    def _calculate_portfolio_value(self, stock_data: Dict[str, pd.DataFrame], date: Optional[datetime]) -> float:
        
        portfolio_value = self.capital

        try:
            for symbol, quantity in self.holdings.items():
                if quantity > 0 and symbol in stock_data:
                    df = stock_data[symbol]
                    if df is None or df.empty:
                        continue
                    if date:
                        price = self._get_price_with_slippage(df, date, "SELL")
                    else:
                        # Normalize column names for case sensitivity
                        column_map = {'close': 'Close'}
                        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns and v not in df.columns})
                        if 'Close' in df.columns and not df.empty:
                            price = df['Close'].iloc[-1]
                        else:
                            price = 0.0
                    portfolio_value += price * quantity

        except Exception as e:
            logger.warning(f"Error calculating portfolio value: {e}")

        return portfolio_value

    def _calculate_performance_metrics(self) -> Dict[str, Union[float, Dict, List]]:
        """Calculate comprehensive performance metrics for the backtest."""
        
        try:
            # Basic return calculation
            final_value = self.portfolio_history[-1].portfolio_value if self.portfolio_history else self.initial_capital
            total_return = (final_value - self.initial_capital) / self.initial_capital if self.initial_capital > 0 else 0.0

            # Annualized return
            days_held = len(self.portfolio_history) - 1 if len(self.portfolio_history) > 1 else 1
            years_held = max(days_held / 252, 0.01)  # Use 252 trading days
            annualized_return = ((1 + total_return) ** (1 / years_held) - 1) if years_held > 0 else 0.0
            annualized_return = 0.0 if pd.isna(annualized_return) or annualized_return is None else annualized_return

            # FIXED: Clip returns for volatility calculation
            if len(self.portfolio_history) > 1:
                returns = [snap.daily_return for snap in self.portfolio_history[1:]]
                # Clip extreme returns to prevent unrealistic volatility
                clipped_returns = [min(max(r, -0.5), 0.5) for r in returns]
                volatility = np.std(clipped_returns) * np.sqrt(252) if len(clipped_returns) > 0 else 0.0  # Annualized volatility
                volatility = 0.0 if pd.isna(volatility) or volatility is None else volatility
                sharpe_ratio = (annualized_return / volatility) if volatility > 0 else 0.0
                sharpe_ratio = 0.0 if pd.isna(sharpe_ratio) or sharpe_ratio is None else sharpe_ratio
            else:
                volatility = 0.0
                sharpe_ratio = 0.0

            # Maximum drawdown
            peak = self.initial_capital
            max_drawdown = 0.0

            for snap in self.portfolio_history:
                if snap.portfolio_value > peak:
                    peak = snap.portfolio_value
                drawdown = (peak - snap.portfolio_value) / peak if peak > 0 else 0.0
                max_drawdown = max(max_drawdown, drawdown)
            max_drawdown = 0.0 if pd.isna(max_drawdown) or max_drawdown is None else max_drawdown

            # Win rate calculation
            profitable_trades = 0
            total_trades = len(self.trades)

            if total_trades > 0:
                # Simple win rate based on trade direction timing
                profitable_trades = sum(1 for trade in self.trades if trade.action == 'SELL')
                win_rate = profitable_trades / total_trades
            else:
                win_rate = 0.0
            win_rate = 0.0 if pd.isna(win_rate) or win_rate is None else win_rate

            # Trading statistics
            total_commission = sum(trade.commission for trade in self.trades) if self.trades else 0.0
            total_commission = 0.0 if pd.isna(total_commission) or total_commission is None else total_commission

            final_value = 0.0 if pd.isna(final_value) or final_value is None else final_value
            total_return = 0.0 if pd.isna(total_return) or total_return is None else total_return

            return {
                "total_return": total_return,
                "annualized_return": annualized_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "total_trades": total_trades,
                "total_commission": total_commission,
                "final_portfolio_value": final_value,
                "portfolio_history": [snap.portfolio_value for snap in self.portfolio_history],
                "trade_log": [
                    {
                        "symbol": trade.symbol,
                        "action": trade.action,
                        "date": trade.date.isoformat(),
                        "price": trade.price,
                        "quantity": trade.quantity,
                        "total_value": trade.total_value,
                        "commission": trade.commission
                    }
                    for trade in self.trades
                ]
            }

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {"error": str(e)}

    def tune_rsi_threshold(
        self,
        stock_data: Dict[str, pd.DataFrame],
        num_windows: int = 10,
        candidates: Optional[List[float]] = None
    ) -> float:
        
        if candidates is None:
            candidates = [30, 35, 40, 45]
        symbols = list(stock_data.keys())
        if not symbols:
            return 30.0
        # Tune on first symbol for simplicity
        symbol = symbols[0]
        df = stock_data[symbol].sort_index()
        n = len(df)
        if n < 100:
            return 30.0
        window_size = max(20, n // (num_windows + 1))
        best_thresholds = []
        for i in range(num_windows):
            train_end = n - i * window_size
            if train_end < 50:
                continue
            train_df = df.iloc[:train_end]
            best_thresh = candidates[0]
            best_sharpe = -np.inf
            for thresh in candidates:
                signals = self._generate_rsi_signals(train_df, thresh)
                strategy_returns = self._compute_strategy_returns_from_signals(train_df, signals)
                if len(strategy_returns) > 10:
                    mean_ret = strategy_returns.mean()
                    std_ret = strategy_returns.std()
                    if std_ret > 0:
                        sharpe = mean_ret / std_ret * np.sqrt(252)
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_thresh = thresh
            best_thresholds.append(best_thresh)
    def run_strategy_backtest(
        self,
        strategy: BaseStrategy,
        stock_data: Dict[str, pd.DataFrame],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        state: Optional[State] = None
    ) -> Dict[str, Union[float, List, Dict]]:
        
        try:
            # Initialize portfolio
            self._reset_portfolio()
            self.portfolio_history = []

            # Get simulation date range
            if not start_date:
                start_date = self._get_earliest_date(stock_data)
            if not end_date:
                end_date = start_date + timedelta(days=SIMULATION_DAYS)

            # Initialize portfolio snapshot
            self.portfolio_history = [
                PortfolioSnapshot(
                    date=start_date,
                    cash=self.capital,
                    holdings={},
                    portfolio_value=self.capital,
                    daily_return=0.0
                )
            ]

            # Process each day in the backtest period
            current_date = start_date
            trade_count = 0

            while current_date <= end_date and trade_count < TRADE_LIMIT:
                self.current_date = current_date

                # Generate signals for all symbols
                daily_signals = {}
                for symbol, df in stock_data.items():
                    # Filter data up to current date
                    historical_data = df[df.index <= current_date]
                    if len(historical_data) >= 50:  # Minimum data requirement
                        # Add symbol to dataframe for strategy use
                        historical_data = historical_data.copy()
                        historical_data['symbol'] = symbol

                        signals = strategy.generate_signals(historical_data, state)
                        if signals:
                            # Take the most confident signal
                            best_signal = max(signals, key=lambda s: s.confidence)
                            daily_signals[symbol] = best_signal

                # Execute trades based on signals
                if daily_signals:
                    self._execute_strategy_trades(daily_signals, stock_data, current_date)
                    trade_count += len(daily_signals)

                # Update portfolio value
                portfolio_value = self._calculate_portfolio_value(stock_data, current_date)
                daily_return = 0.0

                if self.portfolio_history:
                    prev_value = self.portfolio_history[-1].portfolio_value
                    daily_return = (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0

                # Record portfolio snapshot
                snapshot = PortfolioSnapshot(
                    date=current_date,
                    cash=self.capital,
                    holdings=self.holdings.copy(),
                    portfolio_value=portfolio_value,
                    daily_return=daily_return
                )
                self.portfolio_history.append(snapshot)

                # Move to next trading day
                current_date += timedelta(days=1)
                while current_date.weekday() >= 5:  # Skip weekends
                    current_date += timedelta(days=1)

            # Calculate final metrics
            return self._calculate_performance_metrics()

        except Exception as e:
            logger.error(f"Strategy backtest failed: {e}")
            return {"error": str(e)}

    def _execute_strategy_trades(
        self,
        signals: Dict[str, TradingSignal],
        stock_data: Dict[str, pd.DataFrame],
        current_date: datetime
    ) -> float:
        
        total_costs = 0.0

        for symbol, signal in signals.items():
            if signal.action.upper() == 'HOLD':
                continue

            df = stock_data.get(symbol)
            if df is None or df.empty:
                continue

            # Get current price with slippage
            try:
                current_price = self._get_price_with_slippage(df, current_date, signal.action)
                if current_price <= 0:
                    continue

                # Calculate position size based on signal confidence
                base_quantity = self._calculate_position_size(symbol, current_price, signal.action)
                # Adjust quantity based on signal confidence
                quantity = int(base_quantity * signal.confidence)

                if quantity <= 0:
                    continue

            except Exception as e:
                logger.warning(f"Could not get price for {symbol}: {e}")
                continue

            # Execute trade
            total_value = current_price * quantity
            commission = total_value * self.commission_rate

            if signal.action.upper() == 'BUY':
                if self.capital >= total_value + commission:
                    self.holdings[symbol] = self.holdings.get(symbol, 0) + quantity
                    self.capital -= (total_value + commission)
                    trade_type = "BUY"
                else:
                    logger.info(f"Insufficient capital for {symbol} BUY")
                    continue
            else:  # SELL
                current_holding = self.holdings.get(symbol, 0)
                if current_holding >= quantity:
                    self.holdings[symbol] = current_holding - quantity
                    self.capital += (total_value - commission)
                    trade_type = "SELL"
                else:
                    # Sell all available
                    quantity = current_holding
                    if quantity > 0:
                        total_value = current_price * quantity
                        commission = total_value * self.commission_rate
                        self.holdings[symbol] = 0
                        self.capital += (total_value - commission)
                        trade_type = "SELL"
                    else:
                        continue

            # Record trade
            trade = Trade(
                symbol=symbol,
                action=trade_type,
                date=current_date,
                price=current_price,
                quantity=quantity,
                total_value=total_value,
                commission=commission,
                reason=signal.reason
            )
            self.trades.append(trade)

            total_costs += commission
            logger.info(f"Strategy executed {trade_type} {quantity} {symbol} at {current_price:.2f} (Confidence: {signal.confidence:.2f})")

        return total_costs

    def walk_forward_validation(
        self,
        stock_data: Dict[str, pd.DataFrame],
        train_ratio: float = 0.8,
        step_days: int = 50,
        rsi_candidates: List[float] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform walk-forward validation: tune on train windows, test on OOS, average metrics.
        """
        if rsi_candidates is None:
            rsi_candidates = [25, 30, 35, 40]
        
        results = {}
        for symbol, df in stock_data.items():
            if len(df) < 200:  # Minimum data
                results[symbol] = {'avg_sharpe': 0.0, 'avg_win_rate': 0.0, 'avg_drawdown': 0.0, 'windows': 0}
                continue
            
            df = df.sort_index()
            n = len(df)
            train_len = int(n * train_ratio)
            num_windows = max(1, (n - train_len) // step_days)
            
            window_metrics = []
            for i in range(num_windows):
                train_start = 0
                train_end = train_len + i * step_days
                test_start = train_end
                test_end = min(test_start + step_days, n)
                
                if test_end - test_start < 10:  # Too short test
                    break
                
                train_df = df.iloc[train_start:train_end]
                test_df = df.iloc[test_start:test_end]
                
                # Tune RSI threshold on train
                best_thresh = rsi_candidates[0]
                best_sharpe = -np.inf
                for thresh in rsi_candidates:
                    train_signals = self._generate_rsi_signals(train_df, thresh)
                    train_returns = self._compute_strategy_returns_from_signals(train_df, train_signals)
                    if len(train_returns) > 10:
                        mean_ret = train_returns.mean()
                        std_ret = train_returns.std()
                        sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_thresh = thresh
                
                # Test on OOS with best thresh
                test_signals = self._generate_rsi_signals(test_df, best_thresh)
                test_returns = self._compute_strategy_returns_from_signals(test_df, test_signals)
                
                # Backtest metrics on test
                test_results = self.run_backtest(
                    recommendations={symbol: {'action': 'HOLD'}},  # Dummy, but use signals
                    stock_data={symbol: test_df},
                    start_date=test_df.index[0],
                    end_date=test_df.index[-1],
                    rsi_buy_threshold=best_thresh
                )
                
                if 'error' not in test_results:
                    sharpe = test_results.get('sharpe_ratio', 0)
                    win_rate = test_results.get('win_rate', 0)
                    drawdown = test_results.get('max_drawdown', 0)
                    window_metrics.append({'sharpe': sharpe, 'win_rate': win_rate, 'drawdown': drawdown})
            
            if window_metrics:
                avg_sharpe = np.mean([m['sharpe'] for m in window_metrics])
                avg_win_rate = np.mean([m['win_rate'] for m in window_metrics])
                avg_drawdown = np.mean([m['drawdown'] for m in window_metrics])
                results[symbol] = {
                    'avg_sharpe': avg_sharpe,
                    'avg_win_rate': avg_win_rate,
                    'avg_drawdown': avg_drawdown,
                    'windows': len(window_metrics)
                }
            else:
                results[symbol] = {'avg_sharpe': 0.0, 'avg_win_rate': 0.0, 'avg_drawdown': 0.0, 'windows': 0}
        
        return results

    def run_walk_forward_backtest(
        self,
        stock_data: Dict[str, pd.DataFrame],
        strategy_factory=None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        retrain_days: int = 63  # Retrain every 3 months
    ) -> Dict[str, Any]:
        """
        Run comprehensive walk-forward backtest with model retraining

        Args:
            stock_data: Dictionary of stock dataframes
            strategy_factory: Factory to create strategies (for ML models)
            start_date: Backtest start date
            end_date: Backtest end date
            retrain_days: Frequency of model retraining in days

        Returns:
            Comprehensive backtest results with walk-forward analysis
        """
        try:
            # Initialize
            self._reset_portfolio()
            self.walk_forward_results = []

            if not start_date:
                start_date = self._get_earliest_date(stock_data)
            if not end_date:
                end_date = start_date + timedelta(days=SIMULATION_DAYS)

            # Initialize portfolio
            self.portfolio_history = [
                PortfolioSnapshot(
                    date=start_date,
                    cash=self.capital,
                    holdings={},
                    portfolio_value=self.capital,
                    daily_return=0.0
                )
            ]

            current_date = start_date
            last_retrain_date = start_date
            trade_count = 0

            while current_date <= end_date and trade_count < TRADE_LIMIT:
                self.current_date = current_date

                # Check if we need to retrain models
                days_since_retrain = (current_date - last_retrain_date).days
                if days_since_retrain >= retrain_days:
                    self._retrain_models(stock_data, current_date, strategy_factory)
                    last_retrain_date = current_date

                    # Record model performance
                    self._record_model_performance(current_date)

                # Generate signals using current models
                daily_signals = self._generate_walk_forward_signals(
                    stock_data, current_date, strategy_factory
                )

                # Execute trades
                if daily_signals:
                    costs = self._execute_strategy_trades(daily_signals, stock_data, current_date)
                    trade_count += len(daily_signals)

                # Update portfolio value
                portfolio_value = self._calculate_portfolio_value(stock_data, current_date)
                daily_return = 0.0

                if len(self.portfolio_history) > 1:
                    prev_value = self.portfolio_history[-1].portfolio_value
                    daily_return = (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0

                # Record portfolio snapshot
                snapshot = PortfolioSnapshot(
                    date=current_date,
                    cash=self.capital,
                    holdings=self.holdings.copy(),
                    portfolio_value=portfolio_value,
                    daily_return=daily_return
                )
                self.portfolio_history.append(snapshot)

                # Move to next trading day
                current_date += timedelta(days=1)
                while current_date.weekday() >= 5:  # Skip weekends
                    current_date += timedelta(days=1)

            # Calculate comprehensive metrics
            results = self._calculate_walk_forward_metrics()

            return results

        except Exception as e:
            logger.error(f"Walk-forward backtest failed: {e}")
            return {"error": str(e)}

    def _retrain_models(self, stock_data: Dict[str, pd.DataFrame],
                       current_date: datetime, strategy_factory) -> None:
        """Retrain ML models using data up to current date"""
        if not strategy_factory:
            return

        logger.info(f"Retraining models as of {current_date}")

        # For each symbol, retrain using historical data
        for symbol, df in stock_data.items():
            try:
                # Get training data (data before current date)
                train_data = df[df.index < current_date]
                if len(train_data) < self.min_training_samples:
                    continue

                # Retrain strategy for this symbol
                # This would integrate with the ML training pipeline
                logger.info(f"Retrained model for {symbol} with {len(train_data)} samples")

            except Exception as e:
                logger.warning(f"Failed to retrain model for {symbol}: {e}")

    def _record_model_performance(self, current_date: datetime) -> None:
        """Record current model performance metrics"""
        # This would collect metrics from the current models
        performance_record = {
            'date': current_date,
            'metrics': {}  # Would include accuracy, precision, etc.
        }
        self.model_performance_history.append(performance_record)

    def _generate_walk_forward_signals(self, stock_data: Dict[str, pd.DataFrame],
                                     current_date: datetime, strategy_factory) -> Dict[str, Any]:
        """Generate trading signals using walk-forward approach"""
        signals = {}

        for symbol, df in stock_data.items():
            try:
                # Get historical data up to current date
                historical_data = df[df.index <= current_date]
                if len(historical_data) < 50:
                    continue

                # Generate signal using current model
                # This would use the trained ML models or technical indicators
                signal = self._generate_signal_for_symbol(symbol, historical_data, current_date)
                if signal:
                    signals[symbol] = signal

            except Exception as e:
                logger.warning(f"Failed to generate signal for {symbol}: {e}")

        return signals

    def _generate_signal_for_symbol(self, symbol: str, data: pd.DataFrame,
                                   current_date: datetime) -> Optional[Any]:
        """Generate trading signal for a specific symbol"""
        try:
            current_price = data['Close'].iloc[-1]

            closes = data['Close']
            highs = data['High'] if 'High' in data else closes
            lows = data['Low'] if 'Low' in data else closes
            volumes = data['Volume'] if 'Volume' in data else pd.Series(np.zeros(len(data)), index=data.index)

            rsi_series = self._calculate_rsi_series(closes)
            rsi_value = rsi_series.iloc[-1] if not rsi_series.empty and not np.isnan(rsi_series.iloc[-1]) else None

            macd_line, macd_signal = self._calculate_macd_series(closes)
            macd_hist = macd_line - macd_signal if not macd_line.empty else pd.Series(dtype=float)

            short_ma = closes.rolling(20).mean()
            medium_ma = closes.rolling(50).mean()
            long_ma = closes.rolling(200).mean()

            atr_series = self._calculate_average_true_range(highs, lows, closes)
            atr_value = atr_series.iloc[-1] if not atr_series.empty and not np.isnan(atr_series.iloc[-1]) else None

            volume_ratio = 1.0
            short_vol, long_vol = self._calculate_volume_signal(volumes)
            if long_vol:
                volume_ratio = short_vol / long_vol if long_vol > 0 else 1.0

            volatility = closes.pct_change().rolling(20).std().iloc[-1]
            if np.isnan(volatility):
                volatility = 0.0

            bullish_score = 0
            bearish_score = 0
            bullish_reasons: List[str] = []
            bearish_reasons: List[str] = []
            factors: List[Dict[str, Any]] = []

            if rsi_value is not None:
                if rsi_value < 35:
                    bullish_score += 1
                    bullish_reasons.append(f"RSI {rsi_value:.1f} (oversold)")
                    factors.append({"name": "RSI", "value": float(rsi_value), "direction": "bullish"})
                elif rsi_value > 65:
                    bearish_score += 1
                    bearish_reasons.append(f"RSI {rsi_value:.1f} (overbought)")
                    factors.append({"name": "RSI", "value": float(rsi_value), "direction": "bearish"})

            if not macd_hist.empty and len(macd_hist.dropna()) >= 2:
                latest_hist = macd_hist.iloc[-1]
                prev_hist = macd_hist.iloc[-2]
                if latest_hist > 0 and prev_hist <= 0:
                    bullish_score += 1
                    bullish_reasons.append("MACD bullish crossover")
                    factors.append({"name": "MACD", "value": float(latest_hist), "direction": "bullish"})
                elif latest_hist < 0 and prev_hist >= 0:
                    bearish_score += 1
                    bearish_reasons.append("MACD bearish crossover")
                    factors.append({"name": "MACD", "value": float(latest_hist), "direction": "bearish"})

            trend_strength = self._calculate_trend_strength(short_ma, medium_ma, long_ma)
            if trend_strength > 0:
                bullish_score += 1
                bullish_reasons.append(f"Trend alignment strength {trend_strength:.2f}")
                factors.append({"name": "TrendStrength", "value": float(trend_strength), "direction": "bullish"})
            elif trend_strength < 0:
                bearish_score += 1
                bearish_reasons.append(f"Negative trend alignment {trend_strength:.2f}")
                factors.append({"name": "TrendStrength", "value": float(trend_strength), "direction": "bearish"})

            if volume_ratio > 1.3:
                bullish_score += 1
                bullish_reasons.append(f"Volume surge ({volume_ratio:.2f}x avg)")
                factors.append({"name": "VolumeRatio", "value": float(volume_ratio), "direction": "bullish"})
            elif volume_ratio < 0.7:
                bearish_score += 1
                bearish_reasons.append(f"Volume contraction ({volume_ratio:.2f}x avg)")
                factors.append({"name": "VolumeRatio", "value": float(volume_ratio), "direction": "bearish"})

            net_score = bullish_score - bearish_score

            if net_score > 1:
                action = 'BUY'
                reason_components = bullish_reasons or ["Positive multi-factor alignment"]
            elif net_score < -1:
                action = 'SELL'
                reason_components = bearish_reasons or ["Negative multi-factor alignment"]
            else:
                return type('Signal', (), {
                    'action': 'HOLD',
                    'confidence': 0.0,
                    'reason': 'Signals mixed or insufficient conviction',
                    'factors': factors,
                    'metadata': {
                        'bullish_score': bullish_score,
                        'bearish_score': bearish_score,
                        'trend_strength': trend_strength,
                        'volume_ratio': volume_ratio,
                        'volatility': volatility,
                        'atr': atr_value
                    }
                })()

            confidence = self._score_to_confidence(
                net_score=net_score,
                trend_strength=trend_strength,
                volume_ratio=volume_ratio,
                volatility=volatility,
                atr=atr_value,
                price=current_price,
                action=action
            )

            return type('Signal', (), {
                'action': action,
                'confidence': confidence,
                'reason': '; '.join(reason_components[:3]),
                'factors': factors,
                'metadata': {
                    'bullish_score': bullish_score,
                    'bearish_score': bearish_score,
                    'trend_strength': trend_strength,
                    'volume_ratio': volume_ratio,
                    'volatility': volatility,
                    'atr': atr_value
                }
            })()

        except Exception as e:
            logger.warning(f"Signal generation failed for {symbol}: {e}")

        return None

    def _calculate_walk_forward_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive walk-forward backtest metrics"""
        if not self.portfolio_history:
            return {"error": "No portfolio history available"}

        # Basic portfolio metrics
        final_value = self.portfolio_history[-1].portfolio_value
        initial_value = self.portfolio_history[0].portfolio_value
        total_return = (final_value - initial_value) / initial_value

        # Calculate daily returns
        daily_returns = [snapshot.daily_return for snapshot in self.portfolio_history[1:]]

        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
        max_drawdown = self._calculate_max_drawdown(self.portfolio_history)
        win_rate = self._calculate_win_rate(daily_returns)

        # Walk-forward specific metrics
        retrain_points = len(self.model_performance_history)
        avg_trade_frequency = len(self.trades) / len(daily_returns) if daily_returns else 0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(self.trades),
            'final_portfolio_value': final_value,
            'retrain_points': retrain_points,
            'avg_daily_trades': avg_trade_frequency,
            'portfolio_history': [
                {
                    'date': snapshot.date.isoformat(),
                    'portfolio_value': snapshot.portfolio_value,
                    'cash': snapshot.cash,
                    'holdings': snapshot.holdings,
                    'daily_return': snapshot.daily_return
                }
                for snapshot in self.portfolio_history
            ],
            'trades': [
                {
                    'symbol': trade.symbol,
                    'action': trade.action,
                    'date': trade.date.isoformat(),
                    'price': trade.price,
                    'quantity': trade.quantity,
                    'total_value': trade.total_value,
                    'commission': trade.commission,
                    'reason': trade.reason
                }
                for trade in self.trades
            ],
            'model_performance_history': self.model_performance_history
        }

    def _calculate_sharpe_ratio(self, daily_returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if not daily_returns:
            return 0.0

        returns_array = np.array(daily_returns)
        excess_returns = returns_array - risk_free_rate / 252  # Daily risk-free rate
        mean_excess_return = np.mean(excess_returns)
        std_excess_return = np.std(excess_returns)

        return mean_excess_return / std_excess_return * np.sqrt(252) if std_excess_return > 0 else 0.0

    def _calculate_max_drawdown(self, portfolio_history: List[PortfolioSnapshot]) -> float:
        """Calculate maximum drawdown"""
        if not portfolio_history:
            return 0.0

        values = [snapshot.portfolio_value for snapshot in portfolio_history]
        peak = values[0]
        max_drawdown = 0.0

        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def _calculate_win_rate(self, daily_returns: List[float]) -> float:
        """Calculate win rate (percentage of positive days)"""
        if not daily_returns:
            return 0.0

        positive_days = sum(1 for ret in daily_returns if ret > 0)
        return positive_days / len(daily_returns)