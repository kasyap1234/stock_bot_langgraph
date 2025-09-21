"""
Advanced Backtesting Framework with Walk-Forward Analysis, Monte Carlo Simulation,
Statistical Validation, and Comprehensive Performance Metrics.

This module provides sophisticated backtesting capabilities that go beyond simple
historical simulation to include robust validation methods and statistical analysis.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

from .trading_strategies import BaseStrategy, TradingSignal
from .backtesting_engine import BacktestingEngine, Trade, PortfolioSnapshot

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """Represents a single walk-forward analysis window."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    window_id: int
    
    
@dataclass
class BacktestResults:
    """Comprehensive backtesting results container."""
    strategy_name: str
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_return: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    portfolio_history: List[float] = field(default_factory=list)
    trade_log: List[Dict] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Statistical validation results."""
    is_statistically_significant: bool
    p_value: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    t_statistic: float
    degrees_of_freedom: int
    sample_size: int
    mean_return: float
    std_error: float
    validation_method: str
    additional_metrics: Dict[str, float] = field(default_factory=dict)


class WalkForwardAnalyzer:
    """
    Implements walk-forward analysis for robust strategy validation.
    
    Walk-forward analysis divides historical data into multiple training and testing
    periods to simulate real-world trading conditions and prevent overfitting.
    """
    
    def __init__(
        self,
        train_period_months: int = 12,
        test_period_months: int = 3,
        step_months: int = 1,
        min_trades_per_window: int = 5
    ):
        self.train_period_months = train_period_months
        self.test_period_months = test_period_months
        self.step_months = step_months
        self.min_trades_per_window = min_trades_per_window
        
    def create_windows(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[WalkForwardWindow]:
        """Create walk-forward analysis windows."""
        windows = []
        window_id = 0
        
        current_date = start_date
        
        while current_date < end_date:
            # Calculate training period
            train_start = current_date
            train_end = train_start + timedelta(days=self.train_period_months * 30)
            
            # Calculate testing period
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=self.test_period_months * 30)
            
            # Ensure we don't exceed the end date
            if test_end > end_date:
                test_end = end_date
                
            # Only create window if we have enough data
            if test_start < end_date:
                window = WalkForwardWindow(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    window_id=window_id
                )
                windows.append(window)
                window_id += 1
            
            # Move to next window
            current_date += timedelta(days=self.step_months * 30)
            
        return windows
    
    def run_walk_forward_analysis(
        self,
        strategy: BaseStrategy,
        stock_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 1000000.0
    ) -> Dict[str, Any]:
        """
        Run complete walk-forward analysis.
        
        Returns:
            Dictionary containing aggregated results and individual window results
        """
        try:
            windows = self.create_windows(start_date, end_date)
            
            if not windows:
                raise ValueError("No valid windows created for walk-forward analysis")
            
            window_results = []
            aggregated_returns = []
            
            for window in windows:
                logger.info(f"Running window {window.window_id}: "
                          f"Train {window.train_start.date()} to {window.train_end.date()}, "
                          f"Test {window.test_start.date()} to {window.test_end.date()}")
                
                # Run backtest for this window
                engine = BacktestingEngine(initial_capital=initial_capital)
                
                # Filter data for test period
                test_data = self._filter_data_by_date(
                    stock_data, window.test_start, window.test_end
                )
                
                if not test_data:
                    logger.warning(f"No data available for window {window.window_id}")
                    continue
                
                # Run strategy backtest on test period
                results = engine.run_strategy_backtest(
                    strategy=strategy,
                    stock_data=test_data,
                    start_date=window.test_start,
                    end_date=window.test_end
                )
                
                if 'error' not in results and results.get('total_trades', 0) >= self.min_trades_per_window:
                    window_result = {
                        'window_id': window.window_id,
                        'train_period': f"{window.train_start.date()} to {window.train_end.date()}",
                        'test_period': f"{window.test_start.date()} to {window.test_end.date()}",
                        'results': results
                    }
                    window_results.append(window_result)
                    aggregated_returns.append(results['total_return'])
                else:
                    logger.warning(f"Insufficient trades in window {window.window_id}: "
                                 f"{results.get('total_trades', 0)} trades")
            
            # Calculate aggregated statistics
            if aggregated_returns:
                aggregated_stats = self._calculate_aggregated_statistics(window_results)
                
                return {
                    'success': True,
                    'total_windows': len(windows),
                    'valid_windows': len(window_results),
                    'window_results': window_results,
                    'aggregated_statistics': aggregated_stats,
                    'out_of_sample_returns': aggregated_returns
                }
            else:
                return {
                    'success': False,
                    'error': 'No valid windows with sufficient trades',
                    'total_windows': len(windows),
                    'valid_windows': 0
                }
                
        except Exception as e:
            logger.error(f"Walk-forward analysis failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _filter_data_by_date(
        self,
        stock_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """Filter stock data by date range."""
        filtered_data = {}
        
        for symbol, df in stock_data.items():
            if df.empty:
                continue
                
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Filter by date range
            mask = (df.index >= start_date) & (df.index <= end_date)
            filtered_df = df[mask]
            
            if not filtered_df.empty:
                filtered_data[symbol] = filtered_df
                
        return filtered_data
    
    def _calculate_aggregated_statistics(self, window_results: List[Dict]) -> Dict[str, float]:
        """Calculate aggregated statistics across all windows."""
        if not window_results:
            return {}
        
        # Extract metrics from all windows
        returns = [w['results']['total_return'] for w in window_results]
        sharpe_ratios = [w['results']['sharpe_ratio'] for w in window_results]
        max_drawdowns = [w['results']['max_drawdown'] for w in window_results]
        win_rates = [w['results']['win_rate'] for w in window_results]
        
        return {
            'mean_return': np.mean(returns),
            'median_return': np.median(returns),
            'std_return': np.std(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'mean_sharpe': np.mean(sharpe_ratios),
            'median_sharpe': np.median(sharpe_ratios),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'worst_max_drawdown': np.max(max_drawdowns),
            'mean_win_rate': np.mean(win_rates),
            'consistency_ratio': len([r for r in returns if r > 0]) / len(returns),
            'profit_consistency': np.mean([1 if r > 0 else 0 for r in returns])
        }


@dataclass
class WalkForwardPeriod:
    """Represents a single walk-forward optimization period."""
    period_id: int
    is_start: int
    is_end: int
    oos_start: int
    oos_end: int
    is_dates: tuple[datetime, datetime]
    oos_dates: tuple[datetime, datetime]


class WalkForwardOptimizer:
    """
    Implements walk-forward optimization with fixed number of rolling periods.
    
    Creates exactly 5 periods with 80% IS for optimization and 20% OOS for testing,
    stepping by 20% of total data each time. Optimizes strategy parameters (RSI thresholds)
    on IS data and evaluates on OOS data.
    """
    
    def __init__(
        self,
        num_periods: int = 5,
        train_ratio: float = 0.8,
        step_ratio: float = 0.2,
        min_period_size: int = 100
    ):
        self.num_periods = num_periods
        self.train_ratio = train_ratio
        self.step_ratio = step_ratio
        self.min_period_size = min_period_size
    
    def create_periods(self, df: pd.DataFrame) -> List[WalkForwardPeriod]:
        """Create fixed number of rolling periods based on data length."""
        n = len(df)
        if n < self.min_period_size * self.num_periods:
            logger.warning(f"Insufficient data length {n} for {self.num_periods} periods")
            return []
        
        step_size = max(int(n * self.step_ratio), self.min_period_size)
        periods = []
        
        for i in range(self.num_periods):
            start = i * step_size
            end = min(start + n, n)  # Full data for first, rolling for others
            
            period_size = end - start
            if period_size < self.min_period_size:
                continue
            
            is_end = int(start + period_size * self.train_ratio)
            oos_start = is_end
            oos_end = end
            
            # Ensure OOS has minimum size
            if oos_end - oos_start < self.min_period_size * 0.2:
                continue
            
            period = WalkForwardPeriod(
                period_id=i,
                is_start=start,
                is_end=is_end,
                oos_start=oos_start,
                oos_end=oos_end,
                is_dates=(df.index[start], df.index[is_end - 1]),
                oos_dates=(df.index[oos_start], df.index[oos_end - 1])
            )
            periods.append(period)
            logger.info(f"Created period {i}: IS {period.is_dates} ({is_end - start} days), "
                       f"OOS {period.oos_dates} ({oos_end - oos_start} days)")
        
        return periods
    
    def optimize_and_test(
        self,
        df: pd.DataFrame,
        symbol: str,
        initial_capital: float = 100000.0
    ) -> Dict[str, Any]:
        """Optimize on IS periods and test on OOS, return per-period and aggregated metrics."""
        periods = self.create_periods(df)
        if not periods:
            return {'error': 'No valid periods created'}
        
        results = {
            'symbol': symbol,
            'periods': [],
            'aggregated_oos': {
                'sharpe': [],
                'drawdown': [],
                'win_rate': [],
                'returns': [],
                'equity_curves': []  # List of OOS equity curves
            },
            'num_periods': len(periods)
        }
        
        engine = BacktestingEngine(initial_capital=initial_capital)
        
        for period in periods:
            # IS data for optimization
            is_df = df.iloc[period.is_start:period.is_end].copy()
            if len(is_df) < 50:
                continue
            
            # Optimize: Tune RSI threshold on IS data
            tuned_threshold = engine.tune_rsi_threshold(
                {symbol: is_df},
                num_windows=3,  # Quick tuning
                candidates=[25, 30, 35, 40, 45, 50]  # From config.GRID_SEARCH_PARAMS rsi_period but for threshold
            )
            logger.info(f"Period {period.period_id}: Tuned RSI threshold = {tuned_threshold:.1f} on IS data")
            
            # Run IS backtest for IS metrics
            is_start_date = is_df.index[0]
            is_end_date = is_df.index[-1]
            is_results = engine.run_backtest(
                None,
                {symbol: is_df},
                is_start_date,
                is_end_date,
                rsi_buy_threshold=tuned_threshold
            )
            
            is_metrics = {
                'sharpe': is_results.get('sharpe_ratio', 0.0),
                'drawdown': is_results.get('max_drawdown', 0.0),
                'win_rate': is_results.get('win_rate', 0.0),
                'returns': is_results.get('total_return', 0.0),
                'equity_curve': is_results.get('portfolio_history', []),
                'total_trades': is_results.get('total_trades', 0)
            }
            
            # OOS data for testing
            oos_df = df.iloc[period.oos_start:period.oos_end].copy()
            if len(oos_df) < 20:
                continue
            
            oos_start_date = oos_df.index[0]
            oos_end_date = oos_df.index[-1]
            oos_results = engine.run_backtest(
                None,
                {symbol: oos_df},
                oos_start_date,
                oos_end_date,
                rsi_buy_threshold=tuned_threshold
            )
            
            oos_metrics = {
                'sharpe': oos_results.get('sharpe_ratio', 0.0),
                'drawdown': oos_results.get('max_drawdown', 0.0),
                'win_rate': oos_results.get('win_rate', 0.0),
                'returns': oos_results.get('total_return', 0.0),
                'equity_curve': oos_results.get('portfolio_history', []),
                'total_trades': oos_results.get('total_trades', 0)
            }
            
            period_result = {
                'period_id': period.period_id,
                'is_dates': period.is_dates,
                'oos_dates': period.oos_dates,
                'tuned_threshold': tuned_threshold,
                'is_metrics': is_metrics,
                'oos_metrics': oos_metrics
            }
            results['periods'].append(period_result)
            
            # Collect OOS for aggregation
            results['aggregated_oos']['sharpe'].append(oos_metrics['sharpe'])
            results['aggregated_oos']['drawdown'].append(oos_metrics['drawdown'])
            results['aggregated_oos']['win_rate'].append(oos_metrics['win_rate'])
            results['aggregated_oos']['returns'].append(oos_metrics['returns'])
            results['aggregated_oos']['equity_curves'].append(oos_metrics['equity_curve'])
            
            logger.info(f"Period {period.period_id} ({symbol}): "
                        f"IS Sharpe={is_metrics['sharpe'] or 0:.2f}, OOS Sharpe={oos_metrics['sharpe'] or 0:.2f}, "
                        f"OOS Win Rate={oos_metrics['win_rate'] or 0:.1%}, OOS Return={oos_metrics['returns'] or 0:.1%}")
        
        # Compute aggregated OOS metrics
        if results['aggregated_oos']['sharpe']:
            oos_data = results['aggregated_oos']
            results['aggregated_oos'].update({
                'avg_sharpe': np.mean([s or 0.0 for s in oos_data['sharpe']]),
                'std_sharpe': np.std([s or 0.0 for s in oos_data['sharpe']]),
                'avg_win_rate': np.mean([w or 0.0 for w in oos_data['win_rate']]),
                'avg_drawdown': np.mean([d or 0.0 for d in oos_data['drawdown']]),
                'avg_returns': np.mean([r or 0.0 for r in oos_data['returns']]),
                'total_oos_trades': sum(p['oos_metrics']['total_trades'] for p in results['periods']),
                'oos_win_rate_target_met': np.mean([w or 0.0 for w in oos_data['win_rate']]) > 0.5
            })
        else:
            results['aggregated_oos'] = {'error': 'No valid OOS periods', 'avg_sharpe': 0.0, 'avg_win_rate': 0.0, 'avg_drawdown': 0.0, 'avg_returns': 0.0, 'oos_win_rate_target_met': False}
        
        return results
    
    def run_optimization(
        self,
        stock_data: Dict[str, pd.DataFrame],
        symbols: List[str],
        initial_capital: float = 100000.0
    ) -> Dict[str, Dict[str, Any]]:
        """Run walk-forward optimization on multiple symbols."""
        results = {}
        for symbol in symbols:
            if symbol in stock_data and not stock_data[symbol].empty:
                df = stock_data[symbol].sort_index()
                results[symbol] = self.optimize_and_test(df, symbol, initial_capital)
            else:
                results[symbol] = {'error': f'No data for {symbol}'}
                logger.warning(f"No data available for {symbol}")
        return results


class MonteCarloSimulator:
    """
    Implements Monte Carlo simulation for strategy robustness testing.
    
    Uses bootstrap sampling and parameter perturbation to test strategy
    performance under various market conditions and parameter variations.
    """
    
    def __init__(
        self,
        num_simulations: int = 1000,
        confidence_level: float = 0.95,
        bootstrap_block_size: int = 252  # One year of trading days
    ):
        self.num_simulations = num_simulations
        self.confidence_level = confidence_level
        self.bootstrap_block_size = bootstrap_block_size
        
    def run_monte_carlo_simulation(
        self,
        strategy: BaseStrategy,
        stock_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 1000000.0,
        parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation with bootstrap sampling and parameter variation.
        
        Args:
            strategy: Trading strategy to test
            stock_data: Historical market data
            start_date: Simulation start date
            end_date: Simulation end date
            initial_capital: Starting capital
            parameter_ranges: Optional parameter ranges for perturbation
            
        Returns:
            Dictionary containing simulation results and statistics
        """
        try:
            simulation_results = []
            
            logger.info(f"Running {self.num_simulations} Monte Carlo simulations")
            
            for sim_id in range(self.num_simulations):
                if sim_id % 100 == 0:
                    logger.info(f"Completed {sim_id}/{self.num_simulations} simulations")
                
                # Create bootstrap sample of data
                bootstrap_data = self._create_bootstrap_sample(stock_data)
                
                # Perturb strategy parameters if ranges provided
                perturbed_strategy = self._perturb_strategy_parameters(
                    strategy, parameter_ranges
                ) if parameter_ranges else strategy
                
                # Run backtest with bootstrap data
                engine = BacktestingEngine(initial_capital=initial_capital)
                results = engine.run_strategy_backtest(
                    strategy=perturbed_strategy,
                    stock_data=bootstrap_data,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if 'error' not in results:
                    simulation_results.append({
                        'simulation_id': sim_id,
                        'total_return': results['total_return'],
                        'sharpe_ratio': results['sharpe_ratio'],
                        'max_drawdown': results['max_drawdown'],
                        'win_rate': results['win_rate'],
                        'total_trades': results['total_trades']
                    })
            
            if not simulation_results:
                return {'success': False, 'error': 'No successful simulations'}
            
            # Calculate Monte Carlo statistics
            mc_stats = self._calculate_monte_carlo_statistics(simulation_results)
            
            return {
                'success': True,
                'num_simulations': len(simulation_results),
                'simulation_results': simulation_results,
                'statistics': mc_stats
            }
            
        except Exception as e:
            logger.error(f"Monte Carlo simulation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _create_bootstrap_sample(
        self,
        stock_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Create bootstrap sample using block bootstrap method."""
        bootstrap_data = {}
        
        for symbol, df in stock_data.items():
            if len(df) < self.bootstrap_block_size:
                # If data is too short, use simple random sampling with replacement
                bootstrap_df = df.sample(n=len(df), replace=True).sort_index()
            else:
                # Block bootstrap to preserve time series structure
                bootstrap_df = self._block_bootstrap(df)
            
            bootstrap_data[symbol] = bootstrap_df
            
        return bootstrap_data
    
    def _block_bootstrap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform block bootstrap sampling."""
        n = len(df)
        num_blocks = n // self.bootstrap_block_size
        
        if num_blocks == 0:
            return df.sample(n=n, replace=True).sort_index()
        
        # Sample random blocks
        bootstrap_indices = []
        for _ in range(num_blocks):
            start_idx = np.random.randint(0, n - self.bootstrap_block_size + 1)
            block_indices = list(range(start_idx, start_idx + self.bootstrap_block_size))
            bootstrap_indices.extend(block_indices)
        
        # Handle remaining observations
        remaining = n - len(bootstrap_indices)
        if remaining > 0:
            start_idx = np.random.randint(0, n - remaining + 1)
            bootstrap_indices.extend(list(range(start_idx, start_idx + remaining)))
        
        # Create bootstrap sample
        bootstrap_df = df.iloc[bootstrap_indices[:n]].copy()
        
        # Reset index to maintain chronological order
        bootstrap_df.index = df.index
        
        return bootstrap_df
    
    def _perturb_strategy_parameters(
        self,
        strategy: BaseStrategy,
        parameter_ranges: Dict[str, Tuple[float, float]]
    ) -> BaseStrategy:
        """Create strategy copy with perturbed parameters."""
        # This is a simplified implementation
        # In practice, you'd need to implement parameter perturbation
        # based on the specific strategy type and its parameters
        return strategy
    
    def _calculate_monte_carlo_statistics(
        self,
        simulation_results: List[Dict]
    ) -> Dict[str, Any]:
        """Calculate comprehensive Monte Carlo statistics."""
        if not simulation_results:
            return {}
            
        returns = [r['total_return'] for r in simulation_results]
        sharpe_ratios = [r['sharpe_ratio'] for r in simulation_results]
        max_drawdowns = [r['max_drawdown'] for r in simulation_results]
        win_rates = [r['win_rate'] for r in simulation_results]
        
        # Calculate confidence intervals
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        return {
            'returns': {
                'mean': np.mean(returns),
                'median': np.median(returns),
                'std': np.std(returns),
                'min': np.min(returns),
                'max': np.max(returns),
                'confidence_interval': [
                    np.percentile(returns, lower_percentile),
                    np.percentile(returns, upper_percentile)
                ],
                'probability_positive': len([r for r in returns if r > 0]) / len(returns)
            },
            'sharpe_ratio': {
                'mean': np.mean(sharpe_ratios),
                'median': np.median(sharpe_ratios),
                'std': np.std(sharpe_ratios),
                'confidence_interval': [
                    np.percentile(sharpe_ratios, lower_percentile),
                    np.percentile(sharpe_ratios, upper_percentile)
                ]
            },
            'max_drawdown': {
                'mean': np.mean(max_drawdowns),
                'median': np.median(max_drawdowns),
                'worst_case': np.max(max_drawdowns),
                'confidence_interval': [
                    np.percentile(max_drawdowns, lower_percentile),
                    np.percentile(max_drawdowns, upper_percentile)
                ]
            },
            'win_rate': {
                'mean': np.mean(win_rates),
                'median': np.median(win_rates),
                'std': np.std(win_rates),
                'confidence_interval': [
                    np.percentile(win_rates, lower_percentile),
                    np.percentile(win_rates, upper_percentile)
                ]
            }
        }


class StatisticalValidator:
    """
    Implements statistical significance testing for strategy performance.
    
    Provides various statistical tests to validate that strategy performance
    is statistically significant and not due to random chance.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    def validate_strategy_performance(
        self,
        strategy_returns: List[float],
        benchmark_returns: Optional[List[float]] = None,
        risk_free_rate: float = 0.02
    ) -> ValidationReport:
        """
        Perform comprehensive statistical validation of strategy performance.
        
        Args:
            strategy_returns: List of strategy returns
            benchmark_returns: Optional benchmark returns for comparison
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            
        Returns:
            ValidationReport with statistical test results
        """
        try:
            if not strategy_returns:
                raise ValueError("Strategy returns cannot be empty")
            
            returns_array = np.array(strategy_returns)
            
            # Remove any NaN or infinite values
            returns_array = returns_array[np.isfinite(returns_array)]
            
            if len(returns_array) == 0:
                raise ValueError("No valid returns after filtering")
            
            # Calculate basic statistics
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array, ddof=1)
            n = len(returns_array)
            
            # Perform t-test against zero (testing if mean return > 0)
            if std_return > 1e-10:  # Use a small threshold to handle floating point precision
                t_statistic = mean_return / (std_return / np.sqrt(n))
                degrees_of_freedom = n - 1
                p_value = 1 - stats.t.cdf(t_statistic, degrees_of_freedom)
            else:
                # Handle zero or near-zero volatility
                t_statistic = 0.0
                degrees_of_freedom = n - 1
                if mean_return > 0:
                    p_value = 0.0  # Perfectly significant if mean > 0 with no volatility
                elif mean_return < 0:
                    p_value = 1.0  # Not significant if mean < 0 with no volatility
                else:
                    p_value = 0.5  # Neutral if mean = 0 with no volatility
            
            # Calculate confidence interval for mean return
            std_error = std_return / np.sqrt(n) if n > 0 else 0
            t_critical = stats.t.ppf(1 - self.alpha/2, degrees_of_freedom)
            
            ci_lower = mean_return - t_critical * std_error
            ci_upper = mean_return + t_critical * std_error
            
            # Determine statistical significance
            is_significant = bool(p_value < self.alpha)
            
            # Additional tests
            additional_metrics = self._perform_additional_tests(
                returns_array, benchmark_returns, risk_free_rate
            )
            
            return ValidationReport(
                is_statistically_significant=is_significant,
                p_value=p_value,
                confidence_interval_lower=ci_lower,
                confidence_interval_upper=ci_upper,
                t_statistic=t_statistic,
                degrees_of_freedom=degrees_of_freedom,
                sample_size=n,
                mean_return=mean_return,
                std_error=std_error,
                validation_method="One-sample t-test",
                additional_metrics=additional_metrics
            )
            
        except Exception as e:
            logger.error(f"Statistical validation failed: {e}")
            return ValidationReport(
                is_statistically_significant=False,
                p_value=1.0,
                confidence_interval_lower=0.0,
                confidence_interval_upper=0.0,
                t_statistic=0.0,
                degrees_of_freedom=0,
                sample_size=0,
                mean_return=0.0,
                std_error=0.0,
                validation_method="Failed",
                additional_metrics={'error': str(e)}
            )
    
    def _perform_additional_tests(
        self,
        returns: np.ndarray,
        benchmark_returns: Optional[List[float]],
        risk_free_rate: float
    ) -> Dict[str, float]:
        """Perform additional statistical tests."""
        additional_metrics = {}
        
        try:
            # Sharpe ratio test
            excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
            if np.std(excess_returns) > 0:
                sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
                additional_metrics['sharpe_ratio'] = sharpe_ratio
                
                # Sharpe ratio confidence interval (using Jobson-Korkie method)
                n = len(returns)
                if n > 3:
                    sharpe_std_error = np.sqrt((1 + 0.5 * sharpe_ratio**2) / n)
                    sharpe_ci_lower = sharpe_ratio - 1.96 * sharpe_std_error
                    sharpe_ci_upper = sharpe_ratio + 1.96 * sharpe_std_error
                    additional_metrics['sharpe_ci_lower'] = sharpe_ci_lower
                    additional_metrics['sharpe_ci_upper'] = sharpe_ci_upper
            
            # Normality test (Jarque-Bera)
            if len(returns) > 7:
                jb_statistic, jb_p_value = stats.jarque_bera(returns)
                additional_metrics['jarque_bera_statistic'] = jb_statistic
                additional_metrics['jarque_bera_p_value'] = jb_p_value
                additional_metrics['returns_normally_distributed'] = bool(jb_p_value > 0.05)
            
            # Autocorrelation test (Ljung-Box)
            if len(returns) > 10 and STATSMODELS_AVAILABLE:
                lb_result = acorr_ljungbox(returns, lags=min(10, len(returns)//4), return_df=True)
                additional_metrics['ljung_box_p_value'] = lb_result['lb_pvalue'].iloc[-1]
                additional_metrics['returns_independent'] = bool(lb_result['lb_pvalue'].iloc[-1] > 0.05)
            
            # Benchmark comparison if provided
            if benchmark_returns and len(benchmark_returns) == len(returns):
                benchmark_array = np.array(benchmark_returns)
                benchmark_array = benchmark_array[np.isfinite(benchmark_array)]
                
                if len(benchmark_array) == len(returns):
                    # Paired t-test
                    diff_returns = returns - benchmark_array
                    if np.std(diff_returns) > 0:
                        t_stat, p_val = stats.ttest_1samp(diff_returns, 0)
                        additional_metrics['vs_benchmark_t_statistic'] = t_stat
                        additional_metrics['vs_benchmark_p_value'] = p_val
                        additional_metrics['outperforms_benchmark'] = bool(p_val < 0.05 and t_stat > 0)
                    
                    # Information ratio
                    tracking_error = np.std(diff_returns)
                    if tracking_error > 0:
                        information_ratio = np.mean(diff_returns) / tracking_error * np.sqrt(252)
                        additional_metrics['information_ratio'] = information_ratio
            
        except Exception as e:
            logger.warning(f"Error in additional statistical tests: {e}")
            additional_metrics['additional_tests_error'] = str(e)
        
        return additional_metrics


class PerformanceMetricsCalculator:
    """
    Calculates comprehensive performance metrics for trading strategies.
    
    Provides a wide range of risk-adjusted return metrics and performance
    statistics commonly used in quantitative finance.
    """
    
    @staticmethod
    def calculate_comprehensive_metrics(
        portfolio_values: List[float],
        trades: List[Trade],
        benchmark_returns: Optional[List[float]] = None,
        risk_free_rate: float = 0.02
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            portfolio_values: Time series of portfolio values
            trades: List of executed trades
            benchmark_returns: Optional benchmark returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Dictionary of performance metrics
        """
        try:
            if len(portfolio_values) < 2:
                return {'error': 'Insufficient data for metrics calculation'}
            
            # Calculate returns
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            returns = returns[np.isfinite(returns)]
            
            if len(returns) == 0:
                return {'error': 'No valid returns calculated'}
            
            # Basic metrics
            total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
            
            # Annualized metrics
            trading_days = len(returns)
            years = trading_days / 252
            annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
            
            # Volatility
            volatility = np.std(returns) * np.sqrt(252)
            
            # Risk-adjusted returns
            daily_rf_rate = risk_free_rate / 252
            excess_returns = returns - daily_rf_rate
            
            # Sharpe ratio
            returns_std = np.std(returns)
            if returns_std > 1e-10:  # Use threshold to handle floating point precision
                sharpe_ratio = np.mean(excess_returns) / returns_std * np.sqrt(252)
            else:
                # Handle zero volatility case
                if np.mean(excess_returns) > 0:
                    sharpe_ratio = float('inf')  # Infinite Sharpe ratio for positive returns with no risk
                elif np.mean(excess_returns) < 0:
                    sharpe_ratio = float('-inf')  # Negative infinite for negative returns with no risk
                else:
                    sharpe_ratio = 0.0  # Zero for zero returns with no risk
            
            # Sortino ratio (using downside deviation)
            downside_returns = returns[returns < 0]
            downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = np.mean(excess_returns) * np.sqrt(252) / downside_deviation if downside_deviation > 0 else 0
            
            # Maximum drawdown
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (peak - portfolio_values) / peak
            max_drawdown = np.max(drawdown)
            
            # Calmar ratio
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
            
            # Trade-based metrics
            trade_metrics = PerformanceMetricsCalculator._calculate_trade_metrics(trades)
            
            # Additional risk metrics
            additional_metrics = PerformanceMetricsCalculator._calculate_additional_risk_metrics(
                returns, portfolio_values
            )
            
            # Combine all metrics
            metrics = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio,
                **trade_metrics,
                **additional_metrics
            }
            
            # Benchmark comparison if provided
            if benchmark_returns:
                benchmark_metrics = PerformanceMetricsCalculator._calculate_benchmark_metrics(
                    returns, benchmark_returns
                )
                metrics.update(benchmark_metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def _calculate_trade_metrics(trades: List[Trade]) -> Dict[str, float]:
        """Calculate trade-based performance metrics."""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_trade_return': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'consecutive_wins': 0,
                'consecutive_losses': 0
            }
        
        # Group trades by symbol to calculate P&L
        trade_pnl = []
        positions = {}
        
        for trade in trades:
            symbol = trade.symbol
            
            if trade.action == 'BUY':
                if symbol not in positions:
                    positions[symbol] = []
                positions[symbol].append({
                    'quantity': trade.quantity,
                    'price': trade.price,
                    'date': trade.date
                })
            elif trade.action == 'SELL' and symbol in positions:
                # Calculate P&L for sold shares (FIFO)
                remaining_quantity = trade.quantity
                
                while remaining_quantity > 0 and positions[symbol]:
                    position = positions[symbol][0]
                    
                    if position['quantity'] <= remaining_quantity:
                        # Sell entire position
                        pnl = (trade.price - position['price']) * position['quantity']
                        trade_pnl.append(pnl)
                        remaining_quantity -= position['quantity']
                        positions[symbol].pop(0)
                    else:
                        # Partial sell
                        pnl = (trade.price - position['price']) * remaining_quantity
                        trade_pnl.append(pnl)
                        position['quantity'] -= remaining_quantity
                        remaining_quantity = 0
        
        if not trade_pnl:
            return {
                'total_trades': len(trades),
                'win_rate': 0,
                'profit_factor': 0,
                'avg_trade_return': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'consecutive_wins': 0,
                'consecutive_losses': 0
            }
        
        # Calculate metrics
        winning_trades = [pnl for pnl in trade_pnl if pnl > 0]
        losing_trades = [pnl for pnl in trade_pnl if pnl < 0]
        
        win_rate = len(winning_trades) / len(trade_pnl) if trade_pnl else 0
        
        total_wins = sum(winning_trades) if winning_trades else 0
        total_losses = abs(sum(losing_trades)) if losing_trades else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0
        
        avg_trade_return = np.mean(trade_pnl)
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        largest_win = max(winning_trades) if winning_trades else 0
        largest_loss = min(losing_trades) if losing_trades else 0
        
        # Calculate consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for pnl in trade_pnl:
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                consecutive_wins = max(consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                consecutive_losses = max(consecutive_losses, current_losses)
        
        return {
            'total_trades': len(trade_pnl),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade_return': avg_trade_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'consecutive_wins': consecutive_wins,
            'consecutive_losses': consecutive_losses
        }
    
    @staticmethod
    def _calculate_additional_risk_metrics(
        returns: np.ndarray,
        portfolio_values: List[float]
    ) -> Dict[str, float]:
        """Calculate additional risk and performance metrics."""
        try:
            # Value at Risk (VaR) - 95% confidence
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            # Conditional Value at Risk (CVaR)
            cvar_95 = np.mean(returns[returns <= var_95]) if np.any(returns <= var_95) else 0
            cvar_99 = np.mean(returns[returns <= var_99]) if np.any(returns <= var_99) else 0
            
            # Skewness and Kurtosis
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
            
            # Ulcer Index (alternative to max drawdown)
            peak = np.maximum.accumulate(portfolio_values)
            drawdown_pct = (peak - portfolio_values) / peak * 100
            ulcer_index = np.sqrt(np.mean(drawdown_pct ** 2))
            
            # Recovery factor
            total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
            max_dd = np.max((peak - portfolio_values) / peak)
            recovery_factor = total_return / max_dd if max_dd > 0 else 0
            
            # Tail ratio
            tail_ratio = abs(np.percentile(returns, 95)) / abs(np.percentile(returns, 5)) if np.percentile(returns, 5) != 0 else 0
            
            return {
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'ulcer_index': ulcer_index,
                'recovery_factor': recovery_factor,
                'tail_ratio': tail_ratio
            }
            
        except Exception as e:
            logger.warning(f"Error calculating additional risk metrics: {e}")
            return {}
    
    @staticmethod
    def _calculate_benchmark_metrics(
        strategy_returns: np.ndarray,
        benchmark_returns: List[float]
    ) -> Dict[str, float]:
        """Calculate benchmark comparison metrics."""
        try:
            benchmark_array = np.array(benchmark_returns)
            
            # Ensure same length
            min_length = min(len(strategy_returns), len(benchmark_array))
            strat_ret = strategy_returns[:min_length]
            bench_ret = benchmark_array[:min_length]
            
            # Active returns
            active_returns = strat_ret - bench_ret
            
            # Tracking error
            tracking_error = np.std(active_returns) * np.sqrt(252)
            
            # Information ratio
            information_ratio = np.mean(active_returns) / np.std(active_returns) * np.sqrt(252) if np.std(active_returns) > 0 else 0
            
            # Beta
            covariance = np.cov(strat_ret, bench_ret)[0, 1]
            benchmark_variance = np.var(bench_ret)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            # Alpha (Jensen's alpha)
            risk_free_rate = 0.02 / 252  # Daily risk-free rate
            alpha = np.mean(strat_ret - risk_free_rate) - beta * np.mean(bench_ret - risk_free_rate)
            alpha_annualized = alpha * 252
            
            # Correlation
            correlation = np.corrcoef(strat_ret, bench_ret)[0, 1] if len(strat_ret) > 1 else 0
            
            return {
                'tracking_error': tracking_error,
                'information_ratio': information_ratio,
                'beta': beta,
                'alpha': alpha_annualized,
                'correlation_with_benchmark': correlation
            }
            
        except Exception as e:
            logger.warning(f"Error calculating benchmark metrics: {e}")
            return {}


class AdvancedBacktestingEngine:
    """
    Main class that orchestrates advanced backtesting capabilities.
    
    Combines walk-forward analysis, Monte Carlo simulation, statistical validation,
    and comprehensive performance metrics into a unified backtesting framework.
    """
    
    def __init__(
        self,
        initial_capital: float = 1000000.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        
        # Initialize components
        self.walk_forward_analyzer = WalkForwardAnalyzer()
        self.monte_carlo_simulator = MonteCarloSimulator()
        self.statistical_validator = StatisticalValidator()
        self.performance_calculator = PerformanceMetricsCalculator()
        
    def run_comprehensive_backtest(
        self,
        strategy: BaseStrategy,
        stock_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
        benchmark_data: Optional[pd.DataFrame] = None,
        enable_walk_forward: bool = True,
        enable_monte_carlo: bool = True,
        enable_statistical_validation: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive backtesting with all advanced features.
        
        Args:
            strategy: Trading strategy to test
            stock_data: Historical market data
            start_date: Backtest start date
            end_date: Backtest end date
            benchmark_data: Optional benchmark data for comparison
            enable_walk_forward: Whether to run walk-forward analysis
            enable_monte_carlo: Whether to run Monte Carlo simulation
            enable_statistical_validation: Whether to perform statistical validation
            
        Returns:
            Comprehensive backtest results
        """
        try:
            results = {
                'strategy_name': strategy.__class__.__name__,
                'backtest_period': f"{start_date.date()} to {end_date.date()}",
                'success': True
            }
            
            # 1. Standard backtest
            logger.info("Running standard backtest...")
            standard_engine = BacktestingEngine(
                initial_capital=self.initial_capital,
                commission_rate=self.commission_rate,
                slippage_rate=self.slippage_rate
            )
            
            standard_results = standard_engine.run_strategy_backtest(
                strategy=strategy,
                stock_data=stock_data,
                start_date=start_date,
                end_date=end_date
            )
            
            if 'error' in standard_results:
                return {'success': False, 'error': f"Standard backtest failed: {standard_results['error']}"}
            
            results['standard_backtest'] = standard_results
            
            # 2. Walk-forward analysis
            if enable_walk_forward:
                logger.info("Running walk-forward analysis...")
                wf_results = self.walk_forward_analyzer.run_walk_forward_analysis(
                    strategy=strategy,
                    stock_data=stock_data,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=self.initial_capital
                )
                results['walk_forward_analysis'] = wf_results
            
            # 3. Monte Carlo simulation
            if enable_monte_carlo:
                logger.info("Running Monte Carlo simulation...")
                mc_results = self.monte_carlo_simulator.run_monte_carlo_simulation(
                    strategy=strategy,
                    stock_data=stock_data,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=self.initial_capital
                )
                results['monte_carlo_simulation'] = mc_results
            
            # 4. Statistical validation
            if enable_statistical_validation and 'portfolio_history' in standard_results:
                logger.info("Performing statistical validation...")
                
                # Calculate returns from portfolio history
                portfolio_values = standard_results['portfolio_history']
                if len(portfolio_values) > 1:
                    returns = np.diff(portfolio_values) / portfolio_values[:-1]
                    returns = returns[np.isfinite(returns)].tolist()
                    
                    # Prepare benchmark returns if available
                    benchmark_returns = None
                    if benchmark_data is not None and not benchmark_data.empty:
                        benchmark_returns = benchmark_data['Close'].pct_change().dropna().tolist()
                    
                    validation_report = self.statistical_validator.validate_strategy_performance(
                        strategy_returns=returns,
                        benchmark_returns=benchmark_returns
                    )
                    results['statistical_validation'] = validation_report
            
            # 5. Comprehensive performance metrics
            logger.info("Calculating comprehensive performance metrics...")
            if 'portfolio_history' in standard_results and 'trade_log' in standard_results:
                # Convert trade log to Trade objects
                trades = []
                for trade_dict in standard_results['trade_log']:
                    trade = Trade(
                        symbol=trade_dict['symbol'],
                        action=trade_dict['action'],
                        date=datetime.fromisoformat(trade_dict['date']),
                        price=trade_dict['price'],
                        quantity=trade_dict['quantity'],
                        total_value=trade_dict['total_value'],
                        commission=trade_dict['commission']
                    )
                    trades.append(trade)
                
                # Prepare benchmark returns
                benchmark_returns = None
                if benchmark_data is not None and not benchmark_data.empty:
                    benchmark_returns = benchmark_data['Close'].pct_change().dropna().tolist()
                
                comprehensive_metrics = self.performance_calculator.calculate_comprehensive_metrics(
                    portfolio_values=standard_results['portfolio_history'],
                    trades=trades,
                    benchmark_returns=benchmark_returns
                )
                results['comprehensive_metrics'] = comprehensive_metrics
            
            logger.info("Comprehensive backtest completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive backtest failed: {e}")
            return {'success': False, 'error': str(e)}