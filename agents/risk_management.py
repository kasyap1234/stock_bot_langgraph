

import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd

from config.config import (
    RISK_TOLERANCE, MAX_POSITIONS, MAX_PORTFOLIO_DRAWDOWN, MAX_DAILY_LOSS,
    MAX_POSITION_SIZE_PCT, MAX_SECTOR_EXPOSURE, KELLY_FRACTION, RISK_FREE_RATE,
    ATR_PERIOD, TRAILING_STOP_PCT, TIME_EXIT_DAYS, PROFIT_TARGET_LEVELS
)
from data.models import State

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class PositionStatus(Enum):
    
    OPEN = "open"
    CLOSED = "closed"
    STOPPED = "stopped"
    TARGET_HIT = "target_hit"
    TIME_EXIT = "time_exit"


@dataclass
class Position:
    
    symbol: str
    entry_price: float
    quantity: int
    entry_time: datetime
    position_type: str  # 'long' or 'short'
    stop_loss: Optional[float] = None
    trailing_stop: Optional[float] = None
    profit_targets: List[Tuple[float, float]] = field(default_factory=list)  # [(price, percentage)]
    max_holding_period: Optional[timedelta] = None
    status: PositionStatus = PositionStatus.OPEN
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None

    @property
    def current_value(self) -> float:
        
        return self.quantity * self.entry_price

    @property
    def unrealized_pnl(self) -> float:
        
        if self.exit_price:
            if self.position_type == 'long':
                return (self.exit_price - self.entry_price) * self.quantity
            else:
                return (self.entry_price - self.exit_price) * self.quantity
        return 0.0

    @property
    def holding_period(self) -> timedelta:
        
        end_time = self.exit_time if self.exit_time else datetime.now()
        return end_time - self.entry_time


@dataclass
class PortfolioRiskState:
    
    total_value: float
    cash: float
    positions: Dict[str, Position]
    daily_pnl: float
    max_drawdown: float
    daily_loss_limit: float
    portfolio_stop_triggered: bool = False
    last_update: datetime = field(default_factory=datetime.now)

    @property
    def total_exposure(self) -> float:
        
        position_value = sum(pos.current_value for pos in self.positions.values())
        return position_value / self.total_value if self.total_value > 0 else 0.0

    @property
    def concentration_risk(self) -> Dict[str, float]:
        
        concentrations = {}
        for symbol, position in self.positions.items():
            concentrations[symbol] = position.current_value / self.total_value
        return concentrations


class RiskManager:
    

    def __init__(self, config: Optional['RiskConfig'] = None):
        self.config = config or RiskConfig()
        self.portfolio_state = PortfolioRiskState(
            total_value=1000000.0,  # Default starting capital
            cash=1000000.0,
            positions={},
            daily_pnl=0.0,
            max_drawdown=0.0,
            daily_loss_limit=self.config.max_daily_loss
        )
        self.risk_alerts: List[Dict] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def open_position(
        self,
        symbol: str,
        entry_price: float,
        quantity: int,
        position_type: str,
        stop_loss: Optional[float] = None,
        trailing_stop_pct: Optional[float] = None,
        profit_targets: Optional[List[Tuple[float, float]]] = None
    ) -> bool:
        
        try:
            # Check portfolio risk limits
            if not self._check_portfolio_limits(symbol, entry_price, quantity):
                self.logger.warning(f"Portfolio limits exceeded for {symbol}")
                return False

            # Calculate stop-loss if not provided
            if stop_loss is None:
                stop_loss = self._calculate_dynamic_stop_loss(symbol, entry_price, position_type)

            # Set trailing stop
            trailing_stop = entry_price * (1 - trailing_stop_pct) if trailing_stop_pct else None

            # Set profit targets
            if profit_targets is None:
                profit_targets = self._calculate_profit_targets(entry_price, position_type)

            # Create position
            position = Position(
                symbol=symbol,
                entry_price=entry_price,
                quantity=quantity,
                entry_time=datetime.now(),
                position_type=position_type,
                stop_loss=stop_loss,
                trailing_stop=trailing_stop,
                profit_targets=profit_targets,
                max_holding_period=timedelta(days=self.config.time_exit_days)
            )

            # Update portfolio state
            self.portfolio_state.positions[symbol] = position
            self.portfolio_state.cash -= entry_price * quantity
            self.portfolio_state.total_value = self._calculate_total_value()

            self.logger.info(f"Opened {position_type} position in {symbol}: {quantity} @ {entry_price}")
            return True

        except Exception as e:
            self.logger.error(f"Error opening position for {symbol}: {e}")
            return False

    def update_positions(self, current_prices: Dict[str, float]) -> List[Dict]:
        
        updates = []

        for symbol, position in list(self.portfolio_state.positions.items()):
            if position.status != PositionStatus.OPEN:
                continue

            current_price = current_prices.get(symbol)
            if current_price is None:
                continue

            # Update trailing stop
            if position.trailing_stop:
                position.trailing_stop = self._update_trailing_stop(position, current_price)

            # Check for exits
            exit_signal = self._check_exit_conditions(position, current_price)
            if exit_signal:
                self._close_position(position, current_price, exit_signal['reason'])
                updates.append({
                    'symbol': symbol,
                    'action': 'close',
                    'price': current_price,
                    'reason': exit_signal['reason'],
                    'pnl': position.unrealized_pnl
                })

        # Update portfolio state
        self.portfolio_state.total_value = self._calculate_total_value()
        self._check_portfolio_risk_limits()

        return updates

    def _calculate_dynamic_stop_loss(self, symbol: str, entry_price: float, position_type: str) -> float:
        
        # This would need historical data - for now use percentage-based
        if position_type == 'long':
            return entry_price * (1 - self.config.atr_period * 0.01)  # Rough ATR approximation
        else:
            return entry_price * (1 + self.config.atr_period * 0.01)

    def _calculate_profit_targets(self, entry_price: float, position_type: str) -> List[Tuple[float, float]]:
        
        targets = []
        for target_pct in self.config.profit_target_levels:
            if position_type == 'long':
                target_price = entry_price * (1 + target_pct)
            else:
                target_price = entry_price * (1 - target_pct)
            targets.append((target_price, 0.25))  # Scale out 25% at each target
        return targets

    def _update_trailing_stop(self, position: Position, current_price: float) -> float:
        
        if position.position_type == 'long':
            # For long positions, trail below the highest price
            new_stop = current_price * (1 - self.config.trailing_stop_pct)
            return max(position.trailing_stop, new_stop) if position.trailing_stop else new_stop
        else:
            # For short positions, trail above the lowest price
            new_stop = current_price * (1 + self.config.trailing_stop_pct)
            return min(position.trailing_stop, new_stop) if position.trailing_stop else new_stop

    def _check_exit_conditions(self, position: Position, current_price: float) -> Optional[Dict]:
        
        # Check stop-loss
        if position.stop_loss:
            if position.position_type == 'long' and current_price <= position.stop_loss:
                return {'reason': 'stop_loss'}
            elif position.position_type == 'short' and current_price >= position.stop_loss:
                return {'reason': 'stop_loss'}

        # Check trailing stop
        if position.trailing_stop:
            if position.position_type == 'long' and current_price <= position.trailing_stop:
                return {'reason': 'trailing_stop'}
            elif position.position_type == 'short' and current_price >= position.trailing_stop:
                return {'reason': 'trailing_stop'}

        # Check profit targets
        for target_price, scale_pct in position.profit_targets:
            if position.position_type == 'long' and current_price >= target_price:
                return {'reason': 'profit_target', 'scale_pct': scale_pct}
            elif position.position_type == 'short' and current_price <= target_price:
                return {'reason': 'profit_target', 'scale_pct': scale_pct}

        # Check time-based exit
        if position.max_holding_period and position.holding_period >= position.max_holding_period:
            return {'reason': 'time_exit'}

        return None

    def _close_position(self, position: Position, exit_price: float, reason: str):
        
        position.status = PositionStatus(reason.split('_')[0].upper()) if '_' in reason else PositionStatus.CLOSED
        position.exit_price = exit_price
        position.exit_time = datetime.now()
        position.exit_reason = reason

        # Update portfolio cash
        self.portfolio_state.cash += exit_price * position.quantity

        self.logger.info(f"Closed {position.symbol} position: {reason} @ {exit_price}")

    def _check_portfolio_limits(self, symbol: str, price: float, quantity: int) -> bool:
        
        position_value = price * quantity
        new_total_value = self.portfolio_state.total_value - position_value

        # Check position size limit
        if position_value / self.portfolio_state.total_value > self.config.max_position_size:
            return False

        # Check concentration limit
        current_concentration = self.portfolio_state.concentration_risk.get(symbol, 0)
        new_concentration = (current_concentration * self.portfolio_state.total_value + position_value) / new_total_value
        if new_concentration > self.config.max_position_size:
            return False

        return True

    def _check_portfolio_risk_limits(self):
        
        # Check daily loss limit
        if self.portfolio_state.daily_pnl < -self.config.max_daily_loss:
            self.portfolio_state.portfolio_stop_triggered = True
            self._close_all_positions("daily_loss_limit")
            self._add_risk_alert("Daily loss limit exceeded", RiskLevel.HIGH)

        # Check max drawdown
        if self.portfolio_state.max_drawdown > self.config.max_portfolio_drawdown:
            self.portfolio_state.portfolio_stop_triggered = True
            self._close_all_positions("max_drawdown")
            self._add_risk_alert("Maximum drawdown exceeded", RiskLevel.EXTREME)

    def _close_all_positions(self, reason: str):
        
        for position in self.portfolio_state.positions.values():
            if position.status == PositionStatus.OPEN:
                # Assume market order at current price (would need actual prices in real implementation)
                self._close_position(position, position.entry_price, reason)

    def _calculate_total_value(self) -> float:
        
        position_value = sum(pos.current_value for pos in self.portfolio_state.positions.values())
        return self.portfolio_state.cash + position_value

    def _add_risk_alert(self, message: str, level: RiskLevel):
        
        alert = {
            'timestamp': datetime.now(),
            'message': message,
            'level': level.value,
            'portfolio_value': self.portfolio_state.total_value
        }
        self.risk_alerts.append(alert)
        self.logger.warning(f"Risk Alert [{level.value}]: {message}")

    def get_risk_metrics(self) -> Dict:
        
        return {
            'portfolio_value': self.portfolio_state.total_value,
            'cash': self.portfolio_state.cash,
            'total_exposure': self.portfolio_state.total_exposure,
            'concentration_risk': self.portfolio_state.concentration_risk,
            'daily_pnl': self.portfolio_state.daily_pnl,
            'max_drawdown': self.portfolio_state.max_drawdown,
            'portfolio_stop_triggered': self.portfolio_state.portfolio_stop_triggered,
            'open_positions': len([p for p in self.portfolio_state.positions.values() if p.status == PositionStatus.OPEN]),
            'risk_alerts': self.risk_alerts[-10:]  # Last 10 alerts
        }


from .risk_assessment import RiskConfig