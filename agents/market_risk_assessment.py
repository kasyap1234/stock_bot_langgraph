

import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

from config.trading_config import RISK_TOLERANCE
from data.models import State

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


class VolatilityRegime(Enum):
    
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class MarketRiskMetrics:
    
    vix_level: Optional[float] = None
    market_volatility: float = 0.0
    market_trend: str = "neutral"
    regime: MarketRegime = MarketRegime.SIDEWAYS
    volatility_regime: VolatilityRegime = VolatilityRegime.NORMAL
    gap_risk: float = 0.0
    liquidity_risk: float = 0.0
    correlation_risk: float = 0.0
    sector_rotation: Dict[str, float] = None
    last_update: datetime = None

    def __post_init__(self):
        if self.sector_rotation is None:
            self.sector_rotation = {}
        if self.last_update is None:
            self.last_update = datetime.now()


class MarketRiskAssessor:
    

    def __init__(self):
        self.market_data_cache: Dict[str, pd.DataFrame] = {}
        self.risk_metrics = MarketRiskMetrics()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def assess_market_risk(self, market_data: Dict[str, pd.DataFrame], state: Optional[State] = None) -> MarketRiskMetrics:
        
        try:
            # Update market data cache
            self.market_data_cache.update(market_data)

            # Detect market regime
            self.risk_metrics.regime = self._detect_market_regime(market_data)

            # Assess volatility regime
            self.risk_metrics.volatility_regime = self._assess_volatility_regime(market_data)

            # Calculate gap risk
            self.risk_metrics.gap_risk = self._calculate_gap_risk(market_data)

            # Assess liquidity risk
            self.risk_metrics.liquidity_risk = self._assess_liquidity_risk(market_data)

            # Calculate correlation risk
            self.risk_metrics.correlation_risk = self._calculate_correlation_risk(market_data)

            # Detect sector rotation
            self.risk_metrics.sector_rotation = self._detect_sector_rotation(market_data)

            # Calculate market volatility
            self.risk_metrics.market_volatility = self._calculate_market_volatility(market_data)

            # Determine market trend
            self.risk_metrics.market_trend = self._determine_market_trend(market_data)

            self.risk_metrics.last_update = datetime.now()

            return self.risk_metrics

        except Exception as e:
            self.logger.error(f"Error assessing market risk: {e}")
            return self.risk_metrics

    def get_volatility_adjustment(self, base_position_size: float, symbol: str) -> float:
        
        try:
            # Reduce position size in high volatility regimes
            vol_multiplier = {
                VolatilityRegime.LOW: 1.2,
                VolatilityRegime.NORMAL: 1.0,
                VolatilityRegime.HIGH: 0.7,
                VolatilityRegime.EXTREME: 0.3
            }.get(self.risk_metrics.volatility_regime, 1.0)

            # Reduce position size in bear markets
            regime_multiplier = {
                MarketRegime.BULL: 1.0,
                MarketRegime.BEAR: 0.6,
                MarketRegime.SIDEWAYS: 0.8,
                MarketRegime.HIGH_VOLATILITY: 0.5,
                MarketRegime.LOW_VOLATILITY: 1.1
            }.get(self.risk_metrics.regime, 1.0)

            # Apply gap risk adjustment
            gap_adjustment = max(0.5, 1.0 - self.risk_metrics.gap_risk)

            # Apply liquidity risk adjustment
            liquidity_adjustment = max(0.5, 1.0 - self.risk_metrics.liquidity_risk)

            total_multiplier = vol_multiplier * regime_multiplier * gap_adjustment * liquidity_adjustment

            return base_position_size * min(total_multiplier, 1.5)  # Cap at 1.5x

        except Exception as e:
            self.logger.error(f"Error calculating volatility adjustment: {e}")
            return base_position_size

    def _detect_market_regime(self, market_data: Dict[str, pd.DataFrame]) -> MarketRegime:
        
        try:
            if not market_data:
                return MarketRegime.SIDEWAYS

            # Use the first symbol as market proxy (could be NIFTY or broader index)
            market_symbol = list(market_data.keys())[0]
            df = market_data[market_symbol]

            if len(df) < 50:
                return MarketRegime.SIDEWAYS

            # Calculate trend indicators
            sma_50 = df['Close'].rolling(50).mean()
            sma_200 = df['Close'].rolling(200).mean()

            # Calculate volatility
            returns = df['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)

            # Determine regime
            current_price = df['Close'].iloc[-1]
            trend_strength = (current_price - sma_200.iloc[-1]) / sma_200.iloc[-1] if sma_200.iloc[-1] > 0 else 0

            if volatility > 0.30:  # 30% annualized volatility
                return MarketRegime.HIGH_VOLATILITY
            elif volatility < 0.15:  # 15% annualized volatility
                return MarketRegime.LOW_VOLATILITY
            elif trend_strength > 0.10:  # 10% above 200-day MA
                return MarketRegime.BULL
            elif trend_strength < -0.10:  # 10% below 200-day MA
                return MarketRegime.BEAR
            else:
                return MarketRegime.SIDEWAYS

        except Exception as e:
            self.logger.error(f"Error detecting market regime: {e}")
            return MarketRegime.SIDEWAYS

    def _assess_volatility_regime(self, market_data: Dict[str, pd.DataFrame]) -> VolatilityRegime:
        
        try:
            if not market_data:
                return VolatilityRegime.NORMAL

            # Calculate average volatility across symbols
            volatilities = []
            for df in market_data.values():
                if len(df) >= 30:
                    returns = df['Close'].pct_change().dropna()
                    vol = returns.std() * np.sqrt(252)
                    volatilities.append(vol)

            if not volatilities:
                return VolatilityRegime.NORMAL

            avg_volatility = np.mean(volatilities)

            if avg_volatility > 0.40:
                return VolatilityRegime.EXTREME
            elif avg_volatility > 0.25:
                return VolatilityRegime.HIGH
            elif avg_volatility < 0.15:
                return VolatilityRegime.LOW
            else:
                return VolatilityRegime.NORMAL

        except Exception as e:
            self.logger.error(f"Error assessing volatility regime: {e}")
            return VolatilityRegime.NORMAL

    def _calculate_gap_risk(self, market_data: Dict[str, pd.DataFrame]) -> float:
        
        try:
            if not market_data:
                return 0.0

            gap_risks = []
            for df in market_data.values():
                if len(df) >= 2 and 'Open' in df.columns and 'Close' in df.columns:
                    # Calculate overnight gaps
                    gaps = []
                    for i in range(1, min(len(df), 30)):  # Last 30 days
                        prev_close = df['Close'].iloc[-i-1]
                        current_open = df['Open'].iloc[-i]
                        if prev_close > 0:
                            gap_pct = abs(current_open - prev_close) / prev_close
                            gaps.append(gap_pct)

                    if gaps:
                        avg_gap = np.mean(gaps)
                        gap_risks.append(avg_gap)

            return np.mean(gap_risks) if gap_risks else 0.0

        except Exception as e:
            self.logger.error(f"Error calculating gap risk: {e}")
            return 0.0

    def _assess_liquidity_risk(self, market_data: Dict[str, pd.DataFrame]) -> float:
        
        try:
            if not market_data:
                return 0.0

            liquidity_risks = []
            for df in market_data.values():
                if 'Volume' in df.columns and len(df) >= 20:
                    # Calculate average volume
                    avg_volume = df['Volume'].tail(20).mean()
                    current_volume = df['Volume'].iloc[-1]

                    # Calculate volume ratio (lower ratio = higher liquidity risk)
                    if avg_volume > 0:
                        volume_ratio = current_volume / avg_volume
                        # Normalize to 0-1 scale where 1 = high liquidity risk
                        liquidity_risk = max(0, 1 - volume_ratio)
                        liquidity_risks.append(liquidity_risk)

            return np.mean(liquidity_risks) if liquidity_risks else 0.0

        except Exception as e:
            self.logger.error(f"Error assessing liquidity risk: {e}")
            return 0.0

    def _calculate_correlation_risk(self, market_data: Dict[str, pd.DataFrame]) -> float:
        
        try:
            if len(market_data) < 2:
                return 0.0

            # Calculate returns for correlation
            returns_list = []
            for df in market_data.values():
                if len(df) >= 30:
                    returns = df['Close'].pct_change().dropna().tail(30)
                    returns_list.append(returns.values)

            if len(returns_list) < 2:
                return 0.0

            # Calculate average correlation
            min_length = min(len(r) for r in returns_list)
            returns_matrix = np.array([r[-min_length:] for r in returns_list])
            corr_matrix = np.corrcoef(returns_matrix)

            # Average correlation (excluding diagonal)
            n = corr_matrix.shape[0]
            avg_corr = (np.sum(corr_matrix) - n) / (n * (n - 1)) if n > 1 else 0

            # Higher correlation = higher risk
            return max(0, avg_corr)

        except Exception as e:
            self.logger.error(f"Error calculating correlation risk: {e}")
            return 0.0

    def _detect_sector_rotation(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        
        # Simplified sector detection - in practice would need sector classifications
        try:
            sector_performance = {}

            # Group symbols by sector (simplified - would need actual sector data)
            sectors = {
                'IT': ['TCS.NS', 'INFY.NS', 'WIPRO.NS'],
                'BANKING': ['HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS'],
                'PHARMA': ['SUNPHARMA.NS', 'DRREDDY.NS'],
                'AUTO': ['MARUTI.NS', 'BAJAJ-AUTO.NS']
            }

            for sector, symbols in sectors.items():
                sector_returns = []
                for symbol in symbols:
                    if symbol in market_data:
                        df = market_data[symbol]
                        if len(df) >= 20:
                            returns = df['Close'].pct_change().tail(20).mean()
                            sector_returns.append(returns)

                if sector_returns:
                    sector_performance[sector] = np.mean(sector_returns)

            return sector_performance

        except Exception as e:
            self.logger.error(f"Error detecting sector rotation: {e}")
            return {}

    def _calculate_market_volatility(self, market_data: Dict[str, pd.DataFrame]) -> float:
        
        try:
            if not market_data:
                return 0.0

            volatilities = []
            for df in market_data.values():
                if len(df) >= 30:
                    returns = df['Close'].pct_change().dropna()
                    vol = returns.std() * np.sqrt(252)
                    volatilities.append(vol)

            return np.mean(volatilities) if volatilities else 0.0

        except Exception as e:
            self.logger.error(f"Error calculating market volatility: {e}")
            return 0.0

    def _determine_market_trend(self, market_data: Dict[str, pd.DataFrame]) -> str:
        
        try:
            if not market_data:
                return "neutral"

            # Use first symbol as market proxy
            market_symbol = list(market_data.keys())[0]
            df = market_data[market_symbol]

            if len(df) < 50:
                return "neutral"

            # Calculate trend using moving averages
            sma_20 = df['Close'].rolling(20).mean()
            sma_50 = df['Close'].rolling(50).mean()

            current_price = df['Close'].iloc[-1]

            if current_price > sma_20.iloc[-1] > sma_50.iloc[-1]:
                return "bullish"
            elif current_price < sma_20.iloc[-1] < sma_50.iloc[-1]:
                return "bearish"
            else:
                return "neutral"

        except Exception as e:
            self.logger.error(f"Error determining market trend: {e}")
            return "neutral"