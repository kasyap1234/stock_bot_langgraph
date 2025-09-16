

import logging
from typing import Dict, List, Optional, Union
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
from config.config import TRADE_LIMIT, SIMULATION_DAYS
from data.models import State

logger = logging.getLogger(__name__)


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
            logger.info(f"Executed {trade_type} {quantity} {symbol} at {current_price:.2f}")

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
        
        try:
            # Basic return calculation
            final_value = self.portfolio_history[-1].portfolio_value if self.portfolio_history else self.initial_capital
            total_return = (final_value - self.initial_capital) / self.initial_capital

            # Annualized return
            days_held = len(self.portfolio_history) - 1 if len(self.portfolio_history) > 1 else 1
            years_held = max(days_held / 252, 0.01)  # Use 252 trading days
            annualized_return = (1 + total_return) ** (1 / years_held) - 1

            # Volatility calculation
            if len(self.portfolio_history) > 1:
                returns = [snap.daily_return for snap in self.portfolio_history[1:]]
                volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
                sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            else:
                volatility = 0
                sharpe_ratio = 0

            # Maximum drawdown
            peak = self.initial_capital
            max_drawdown = 0.0

            for snap in self.portfolio_history:
                if snap.portfolio_value > peak:
                    peak = snap.portfolio_value
                drawdown = (peak - snap.portfolio_value) / peak
                max_drawdown = max(max_drawdown, drawdown)

            # Win rate calculation
            profitable_trades = 0
            total_trades = len(self.trades)

            if total_trades > 0:
                # Simple win rate based on trade direction timing
                profitable_trades = sum(1 for trade in self.trades if trade.action == 'SELL')
                win_rate = profitable_trades / total_trades
            else:
                win_rate = 0.0

            # Trading statistics
            total_commission = sum(trade.commission for trade in self.trades)

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
        return round(np.mean(best_thresholds)) if best_thresholds else 30.0