"""
Risk assessment agent for stock portfolio analysis.
Calculates risk metrics including volatility, drawdown, Sharpe ratio, and portfolio optimization.
"""

import logging
from typing import Dict, Union, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

import numpy as np
import pandas as pd
try:
    from scipy import optimize
except Exception:
    optimize = None
    logging.warning("SciPy optimize not available, using equal weights for portfolio")

from config.config import (
    RISK_TOLERANCE, MAX_POSITIONS, MAX_PORTFOLIO_DRAWDOWN, MAX_DAILY_LOSS,
    MAX_POSITION_SIZE_PCT, MAX_SECTOR_EXPOSURE, KELLY_FRACTION, RISK_FREE_RATE,
    ATR_PERIOD, TRAILING_STOP_PCT, TIME_EXIT_DAYS, PROFIT_TARGET_LEVELS
)
from data.models import State

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """Configuration for risk management parameters."""
    max_portfolio_drawdown: float = MAX_PORTFOLIO_DRAWDOWN
    max_daily_loss: float = MAX_DAILY_LOSS
    max_position_size: float = MAX_POSITION_SIZE_PCT
    max_sector_exposure: float = MAX_SECTOR_EXPOSURE
    kelly_fraction: float = KELLY_FRACTION
    risk_free_rate: float = RISK_FREE_RATE
    atr_period: int = ATR_PERIOD
    trailing_stop_pct: float = TRAILING_STOP_PCT
    time_exit_days: int = TIME_EXIT_DAYS
    profit_target_levels: List[float] = None

    def __post_init__(self):
        if self.profit_target_levels is None:
            self.profit_target_levels = PROFIT_TARGET_LEVELS


def calculate_kelly_criterion(expected_return: float, volatility: float, risk_free_rate: float = 0.065) -> float:
    """
    Calculate Kelly Criterion for optimal position sizing.

    Args:
        expected_return: Expected annual return
        volatility: Annualized volatility
        risk_free_rate: Risk-free rate

    Returns:
        Kelly fraction (0-1)
    """
    if volatility <= 0:
        return 0.0

    # Kelly formula: f = (μ - r) / σ²
    kelly_fraction = (expected_return - risk_free_rate) / (volatility ** 2)

    # Apply half-Kelly for conservatism and ensure non-negative
    kelly_fraction = max(0.0, kelly_fraction)

    return kelly_fraction


def calculate_atr_stop_loss(df: pd.DataFrame, atr_period: int = 14, multiplier: float = 2.0) -> float:
    """
    Calculate dynamic stop-loss based on Average True Range (ATR).

    Args:
        df: DataFrame with OHLC data
        atr_period: Period for ATR calculation
        multiplier: ATR multiplier for stop distance

    Returns:
        Stop-loss price
    """
    try:
        if len(df) < atr_period + 1:
            return 0.0

        # Calculate True Range
        high = df['High'].iloc[-1]
        low = df['Low'].iloc[-1]
        close = df['Close'].iloc[-1]

        # True Range = max(high - low, |high - prev_close|, |low - prev_close|)
        prev_close = df['Close'].iloc[-2] if len(df) > 1 else close
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))

        # Calculate ATR (simple moving average of TR)
        if len(df) >= atr_period:
            atr_values = []
            for i in range(len(df) - atr_period + 1, len(df)):
                window_df = df.iloc[i-atr_period:i+1]
                window_tr = []
                for j in range(1, len(window_df)):
                    h = window_df['High'].iloc[j]
                    l = window_df['Low'].iloc[j]
                    c = window_df['Close'].iloc[j]
                    pc = window_df['Close'].iloc[j-1]
                    tr_val = max(h - l, abs(h - pc), abs(l - pc))
                    window_tr.append(tr_val)
                atr_values.append(np.mean(window_tr))
            atr = np.mean(atr_values) if atr_values else tr
        else:
            atr = tr

        # Stop-loss below current price for long positions
        stop_loss = close - (atr * multiplier)

        return max(0.0, stop_loss)

    except Exception as e:
        logger.error(f"Error calculating ATR stop-loss: {e}")
        return 0.0


def calculate_risk_parity_weights(volatilities: np.ndarray, correlations: np.ndarray) -> np.ndarray:
    """
    Calculate risk parity portfolio weights.

    Args:
        volatilities: Array of asset volatilities
        correlations: Correlation matrix

    Returns:
        Array of portfolio weights
    """
    try:
        n_assets = len(volatilities)
        if n_assets == 0:
            return np.array([])

        # Calculate covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * correlations

        # Risk parity: equal risk contribution from each asset
        inv_vol = 1.0 / volatilities
        weights = inv_vol / np.sum(inv_vol)

        # Scale weights to sum to 1
        weights = weights / np.sum(weights)

        return weights

    except Exception as e:
        logger.error(f"Error calculating risk parity weights: {e}")
        return np.ones(n_assets) / n_assets  # Equal weights fallback


def risk_assessment_agent(state: State) -> State:
    """
    Risk assessment agent for the LangGraph workflow.
    Calculates portfolio and individual asset risk metrics.

    Args:
        state: Current workflow state

    Returns:
        Updated state with risk metrics
    """
    logging.info("Starting risk assessment agent")

    stock_data = state.get("stock_data", {})
    risk_metrics = {}

    symbols = list(stock_data.keys())

    for symbol, df in stock_data.items():
        try:
            individual_risk = _calculate_individual_risk_metrics(symbol, df)
            risk_metrics[symbol] = individual_risk
            logger.info(f"Calculated risk metrics for {symbol}")

        except Exception as e:
            logger.error(f"Error in risk assessment for {symbol}: {e}")
            risk_metrics[symbol] = {"error": "Risk assessment failed"}

    # Calculate portfolio-level risk if multiple symbols
    if len(symbols) > 1:
        try:
            portfolio_risk = _calculate_portfolio_risk_metrics(stock_data, risk_metrics)
            risk_metrics.update(portfolio_risk)
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")

    state["risk_metrics"] = risk_metrics
    return state


def _calculate_individual_risk_metrics(symbol: str, df: pd.DataFrame) -> Dict[str, Union[float, bool]]:
    """
    Calculate individual stock risk metrics.
    
    Args:
        symbol: Stock symbol
        df: DataFrame with stock data
    
    Returns:
        Dictionary of risk metrics
    """
    try:
        logger.info(f"Individual risk input columns: {df.columns.tolist()}")
        if 'close' in df.columns:
            df = df.rename(columns={'close': 'Close'})
            logger.info(f"Renamed columns for individual risk: {df.columns.tolist()}")
        # Daily returns
        daily_returns = df['Close'].pct_change().dropna().values

        if len(daily_returns) < 30:
            return {"error": "Insufficient data for risk calculations"}

        # Annualized volatility (standard deviation)
        annual_vol = np.std(daily_returns) * np.sqrt(252)  # 252 trading days

        # Maximum drawdown
        cumulative = (df['Close'] / df['Close'].iloc[0]) - 1
        max_drawdown = (cumulative - cumulative.cummax()).min()

        # Sharpe ratio (assuming risk-free rate of 6.5% for Indian market)
        risk_free_rate = 0.065
        avg_return = np.mean(daily_returns) * 252
        sharpe = (avg_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0

        # Value at Risk (95% confidence, 1-day)
        var_95 = np.percentile(daily_returns, 5)  # 5th percentile

        # Risk rating
        risk_ok = annual_vol <= RISK_TOLERANCE

        # Kelly Criterion position sizing
        kelly_fraction = calculate_kelly_criterion(avg_return, annual_vol)
        kelly_weight = min(kelly_fraction * 0.5, MAX_POSITIONS)  # Half-Kelly with max limit

        # ATR-based dynamic stop-loss
        atr_stop_loss = calculate_atr_stop_loss(df)

        # Implied Volatility from options
        iv = _get_implied_volatility(symbol)
        implied_volatility = round(iv, 4) if iv is not None else 0.3  # default 30%

        # Additional risk metrics
        downside_deviation = np.std(daily_returns[daily_returns < 0]) * np.sqrt(252) if len(daily_returns[daily_returns < 0]) > 0 else 0
        sortino_ratio = (avg_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0

        # Maximum favorable excursion (MFE) - potential profit
        peak_return = np.max(np.cumsum(daily_returns))
        mfe = peak_return * np.sqrt(252)  # Annualized

        metrics = {
            "volatility": annual_vol,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino_ratio,
            "var_95": var_95,
            "avg_annual_return": avg_return,
            "weight": kelly_weight,
            "kelly_fraction": kelly_fraction,
            "atr_stop_loss": atr_stop_loss,
            "risk_ok": risk_ok,
            "implied_volatility": implied_volatility,
            "downside_volatility": downside_deviation,
            "max_favorable_excursion": mfe
        }

        return metrics

    except Exception as e:
        logger.error(f"Error calculating individual risk metrics: {e}")
        return {"error": "Risk calculation failed"}


def _calculate_portfolio_risk_metrics(stock_data: Dict, risk_metrics: Dict) -> Dict[str, Dict]:
    """
    Calculate portfolio-level risk metrics and optimal weights.

    Args:
        stock_data: Dictionary of stock dataframes
        risk_metrics: Dictionary of individual risk metrics

    Returns:
        Dictionary of portfolio-level metrics
    """
    try:
        symbols = list(stock_data.keys())

        # Extract daily returns for each stock
        returns_list = []

        for symbol in symbols:
            df = stock_data[symbol].copy()
            logger.info(f"Portfolio risk input columns for {symbol}: {df.columns.tolist()}")
            if 'close' in df.columns:
                df = df.rename(columns={'close': 'Close'})
                logger.info(f"Renamed columns for portfolio risk {symbol}: {df.columns.tolist()}")
            daily_returns = df['Close'].pct_change().dropna().values
            returns_list.append(daily_returns)

        # Use common period across all stocks
        min_length = min(len(r) for r in returns_list)
        returns_matrix = np.array([r[-min_length:] for r in returns_list])

        if returns_matrix.shape[0] < 2:
            return {}

        # Portfolio optimization with multiple approaches
        n_assets = len(symbols)

        # Extract volatilities and returns for optimization
        volatilities = np.array([risk_metrics[symbol]["volatility"] for symbol in symbols])
        returns = np.array([risk_metrics[symbol]["avg_annual_return"] for symbol in symbols])

        # Calculate correlation matrix
        correlations = np.corrcoef(returns_matrix)

        # Risk Parity weights
        risk_parity_weights = calculate_risk_parity_weights(volatilities, correlations)

        # Kelly-based weights (using individual Kelly fractions)
        kelly_weights = np.array([risk_metrics[symbol]["kelly_fraction"] * 0.5 for symbol in symbols])
        kelly_weights = kelly_weights / np.sum(kelly_weights) if np.sum(kelly_weights) > 0 else np.ones(n_assets) / n_assets

        # Sharpe ratio optimization
        def negative_sharpe_ratio(weights):
            portfolio_returns = np.dot(weights, returns_matrix)
            portfolio_vol = np.std(portfolio_returns) * np.sqrt(252)
            portfolio_return = np.mean(portfolio_returns) * 252
            risk_free_rate = 0.065

            sharpe = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
            return -sharpe

        # Constraints: weights sum to 1, each weight between 0 and max_positions
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
        ]
        bounds = [(0, MAX_POSITIONS) for _ in range(n_assets)]

        # Initial guess: risk parity weights
        init_weights = risk_parity_weights

        if optimize is None:
            logger.warning("Using risk parity weights due to missing optimizer")
            optimal_weights = risk_parity_weights
        else:
            try:
                result = optimize.minimize(
                    negative_sharpe_ratio,
                    init_weights,
                    bounds=bounds,
                    constraints=constraints,
                    method='SLSQP'
                )

                optimal_weights = result.x if result.success else risk_parity_weights
            except Exception as e:
                logger.warning(f"Portfolio optimization failed: {e}")
                optimal_weights = risk_parity_weights

        # Calculate optimized portfolio metrics
        portfolio_returns = np.dot(optimal_weights, returns_matrix)
        portfolio_vol = np.std(portfolio_returns) * np.sqrt(252)
        portfolio_return = np.mean(portfolio_returns) * 252
        risk_free_rate = 0.065
        sharpe = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0

        # Update individual weights with optimized weights
        for i, symbol in enumerate(symbols):
            risk_metrics[symbol]["weight"] = optimal_weights[i]
            risk_metrics[symbol]["risk_parity_weight"] = risk_parity_weights[i]
            risk_metrics[symbol]["kelly_weight"] = kelly_weights[i]

        # Portfolio-level metrics
        portfolio_vol_rp = np.std(np.dot(risk_parity_weights, returns_matrix)) * np.sqrt(252)
        portfolio_return_rp = np.mean(np.dot(risk_parity_weights, returns_matrix)) * 252
        sharpe_rp = (portfolio_return_rp - 0.065) / portfolio_vol_rp if portfolio_vol_rp > 0 else 0

        portfolio_metrics = {
            "portfolio_volatility": portfolio_vol,
            "portfolio_sharpe": sharpe,
            "portfolio_return": portfolio_return,
            "portfolio_volatility_rp": portfolio_vol_rp,
            "portfolio_sharpe_rp": sharpe_rp,
            "portfolio_return_rp": portfolio_return_rp,
            "diversification_benefit": _calculate_diversification_benefit(returns_matrix, optimal_weights),
            "risk_parity_weights": risk_parity_weights.tolist(),
            "kelly_weights": kelly_weights.tolist(),
            "optimal_weights": optimal_weights.tolist()
        }

        return portfolio_metrics

    except Exception as e:
        logger.error(f"Error calculating portfolio risk: {e}")
        return {"error": "Portfolio risk calculation failed"}


def _calculate_diversification_benefit(returns_matrix: np.ndarray, weights: np.ndarray) -> float:
    """
    Calculate the diversification benefit of the portfolio.
    
    Args:
        returns_matrix: Matrix of asset returns
        weights: Portfolio weights
    
    Returns:
        Diversification benefit ratio
    """
    try:
        # Weighted average of individual volatilities
        individual_vols = [np.std(returns_matrix[i]) for i in range(len(weights))]
        weighted_avg_vol = np.dot(weights, individual_vols)
    
        # Portfolio volatility
        portfolio_returns = np.dot(weights, returns_matrix)
        portfolio_vol = np.std(portfolio_returns)
    
        # Diversification benefit (reduction in volatility)
        diversification_ratio = portfolio_vol / weighted_avg_vol if weighted_avg_vol > 0 else 1.0
    
        return diversification_ratio
    
    except Exception as e:
        logger.error(f"Error calculating diversification benefit: {e}")
        return 1.0


def _get_implied_volatility(symbol: str) -> Optional[float]:
    """
    Fetch implied volatility from options data using yahooquery.

    Args:
        symbol: Stock symbol

    Returns:
        Average implied volatility or None if unavailable
    """
    try:
        from yahooquery import Ticker
        ticker = Ticker(symbol)

        # Try to get options chain
        try:
            options = ticker.option_chain()
        except Exception:
            logger.warning(f"Options chain not available for {symbol}")
            return None

        # Handle different response formats from yahooquery
        if isinstance(options, dict):
            calls = options.get('calls')
        elif hasattr(options, 'calls'):
            calls = options.calls
        else:
            calls = None

        if calls is not None:
            # Convert to DataFrame if it's not already
            if not isinstance(calls, pd.DataFrame):
                try:
                    calls = pd.DataFrame(calls)
                except Exception:
                    logger.warning(f"Could not convert calls data to DataFrame for {symbol}")
                    return None

            # Check if we have the impliedVolatility column
            if 'impliedVolatility' in calls.columns:
                iv_values = calls['impliedVolatility'].dropna()
                if not iv_values.empty:
                    iv_mean = float(iv_values.mean())
                    logger.info(f"Fetched IV for {symbol}: {iv_mean:.2%}")
                    return iv_mean

        logger.warning(f"No valid options data for {symbol}")
        return None
    except Exception as e:
        logger.error(f"Failed to fetch IV for {symbol}: {e}")
        return None