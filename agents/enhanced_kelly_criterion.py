"""
Enhanced Kelly Criterion Calculator

This module implements an enhanced Kelly Criterion calculator with volatility adjustments
and portfolio correlation considerations for optimal position sizing in the advanced
risk assessment system.

Requirements addressed:
- 2.1 - Kelly Criterion with volatility adjustments for optimal capital allocation
- 2.3 - Portfolio correlation considerations for position sizing
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime
import math

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm

logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class KellyParameters:
    """Kelly Criterion calculation parameters"""
    expected_return: float
    volatility: float
    risk_free_rate: float
    win_probability: Optional[float] = None
    avg_win: Optional[float] = None
    avg_loss: Optional[float] = None
    
    def __post_init__(self):
        """Validate parameters"""
        if self.volatility <= 0:
            raise ValueError("Volatility must be positive")
        if self.expected_return <= self.risk_free_rate:
            logger.warning("Expected return is not greater than risk-free rate")


@dataclass
class KellyResult:
    """Kelly Criterion calculation result"""
    kelly_fraction: float
    adjusted_kelly: float
    confidence_interval: Tuple[float, float]
    risk_adjusted_kelly: float
    correlation_adjusted_kelly: float
    recommended_position: float
    calculation_method: str
    parameters_used: KellyParameters
    warnings: List[str]


@dataclass
class PortfolioKellyResult:
    """Portfolio-level Kelly optimization result"""
    individual_kellys: Dict[str, float]
    portfolio_kelly: float
    optimal_weights: Dict[str, float]
    correlation_adjustment: float
    diversification_benefit: float
    total_leverage: float
    risk_budget_allocation: Dict[str, float]


class EnhancedKellyCriterion:
    """
    Enhanced Kelly Criterion calculator with multiple improvements:
    
    1. Volatility adjustments for different market regimes
    2. Portfolio correlation considerations
    3. Multiple calculation methods (continuous, discrete, fractional)
    4. Risk management overlays (maximum position limits, drawdown protection)
    5. Dynamic parameter estimation with confidence intervals
    """
    
    def __init__(self, 
                 max_kelly_fraction: float = 0.25,
                 min_kelly_fraction: float = 0.01,
                 volatility_adjustment_factor: float = 1.0,
                 correlation_penalty_factor: float = 0.5,
                 confidence_level: float = 0.95):
        """
        Initialize enhanced Kelly calculator.
        
        Args:
            max_kelly_fraction: Maximum allowed Kelly fraction (risk management)
            min_kelly_fraction: Minimum Kelly fraction (avoid zero positions)
            volatility_adjustment_factor: Factor to adjust for volatility regime
            correlation_penalty_factor: Penalty factor for high correlations
            confidence_level: Confidence level for parameter estimation
        """
        self.max_kelly_fraction = max_kelly_fraction
        self.min_kelly_fraction = min_kelly_fraction
        self.volatility_adjustment_factor = volatility_adjustment_factor
        self.correlation_penalty_factor = correlation_penalty_factor
        self.confidence_level = confidence_level
        
    def calculate_kelly_fraction(self, 
                                parameters: KellyParameters,
                                method: str = 'continuous',
                                apply_adjustments: bool = True) -> KellyResult:
        """
        Calculate Kelly fraction with various methods and adjustments.
        
        Args:
            parameters: Kelly calculation parameters
            method: Calculation method ('continuous', 'discrete', 'fractional')
            apply_adjustments: Whether to apply volatility and other adjustments
            
        Returns:
            Kelly calculation result with adjustments
        """
        warnings_list = []
        
        # Basic Kelly calculation
        if method == 'continuous':
            kelly_fraction = self._calculate_continuous_kelly(parameters)
        elif method == 'discrete':
            kelly_fraction = self._calculate_discrete_kelly(parameters)
        elif method == 'fractional':
            kelly_fraction = self._calculate_fractional_kelly(parameters)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Apply risk management bounds
        original_kelly = kelly_fraction
        kelly_fraction = max(self.min_kelly_fraction, 
                           min(self.max_kelly_fraction, kelly_fraction))
        
        if kelly_fraction != original_kelly:
            warnings_list.append(f"Kelly fraction capped from {original_kelly:.4f} to {kelly_fraction:.4f}")
        
        # Calculate adjustments
        adjusted_kelly = kelly_fraction
        risk_adjusted_kelly = kelly_fraction
        
        if apply_adjustments:
            # Volatility adjustment
            volatility_factor = self._calculate_volatility_adjustment(parameters.volatility)
            risk_adjusted_kelly = kelly_fraction * volatility_factor
            
            # Half-Kelly for conservatism
            adjusted_kelly = risk_adjusted_kelly * 0.5
            
            if volatility_factor != 1.0:
                warnings_list.append(f"Applied volatility adjustment factor: {volatility_factor:.3f}")
        
        # Calculate confidence intervals
        confidence_interval = self._calculate_kelly_confidence_interval(
            parameters, kelly_fraction
        )
        
        # Correlation adjustment (will be applied at portfolio level)
        correlation_adjusted_kelly = adjusted_kelly
        
        # Final recommended position
        recommended_position = max(0.0, min(self.max_kelly_fraction, adjusted_kelly))
        
        return KellyResult(
            kelly_fraction=kelly_fraction,
            adjusted_kelly=adjusted_kelly,
            confidence_interval=confidence_interval,
            risk_adjusted_kelly=risk_adjusted_kelly,
            correlation_adjusted_kelly=correlation_adjusted_kelly,
            recommended_position=recommended_position,
            calculation_method=method,
            parameters_used=parameters,
            warnings=warnings_list
        )
    
    def calculate_portfolio_kelly(self, 
                                asset_parameters: Dict[str, KellyParameters],
                                correlation_matrix: np.ndarray,
                                asset_names: List[str],
                                method: str = 'mean_variance') -> PortfolioKellyResult:
        """
        Calculate optimal Kelly fractions for a portfolio considering correlations.
        
        Args:
            asset_parameters: Dictionary of Kelly parameters for each asset
            correlation_matrix: Correlation matrix between assets
            asset_names: List of asset names
            method: Portfolio optimization method ('mean_variance', 'risk_parity', 'equal_risk')
            
        Returns:
            Portfolio Kelly optimization result
        """
        # Calculate individual Kelly fractions
        individual_kellys = {}
        individual_results = {}
        
        for asset_name in asset_names:
            if asset_name in asset_parameters:
                result = self.calculate_kelly_fraction(
                    asset_parameters[asset_name], 
                    apply_adjustments=False  # Apply adjustments at portfolio level
                )
                individual_kellys[asset_name] = result.kelly_fraction
                individual_results[asset_name] = result
            else:
                individual_kellys[asset_name] = 0.0
        
        # Portfolio optimization considering correlations
        if method == 'mean_variance':
            optimal_weights = self._optimize_portfolio_kelly_mean_variance(
                asset_parameters, correlation_matrix, asset_names
            )
        elif method == 'risk_parity':
            optimal_weights = self._optimize_portfolio_kelly_risk_parity(
                asset_parameters, correlation_matrix, asset_names
            )
        elif method == 'equal_risk':
            optimal_weights = self._optimize_portfolio_kelly_equal_risk(
                asset_parameters, correlation_matrix, asset_names
            )
        else:
            # Fallback to individual Kelly fractions
            total_kelly = sum(individual_kellys.values())
            if total_kelly > 0:
                optimal_weights = {name: kelly / total_kelly for name, kelly in individual_kellys.items()}
            else:
                optimal_weights = {name: 1.0 / len(asset_names) for name in asset_names}
        
        # Calculate correlation adjustment
        correlation_adjustment = self._calculate_correlation_adjustment(
            correlation_matrix, list(optimal_weights.values())
        )
        
        # Apply correlation adjustment to weights
        adjusted_weights = {}
        for asset_name, weight in optimal_weights.items():
            adjusted_weights[asset_name] = weight * correlation_adjustment
        
        # Calculate portfolio-level metrics
        portfolio_kelly = sum(adjusted_weights.values())
        total_leverage = portfolio_kelly
        
        # Calculate diversification benefit
        diversification_benefit = self._calculate_diversification_benefit(
            correlation_matrix, list(optimal_weights.values())
        )
        
        # Risk budget allocation
        risk_budget_allocation = self._calculate_risk_budget_allocation(
            asset_parameters, adjusted_weights, asset_names
        )
        
        return PortfolioKellyResult(
            individual_kellys=individual_kellys,
            portfolio_kelly=portfolio_kelly,
            optimal_weights=adjusted_weights,
            correlation_adjustment=correlation_adjustment,
            diversification_benefit=diversification_benefit,
            total_leverage=total_leverage,
            risk_budget_allocation=risk_budget_allocation
        )
    
    def estimate_parameters_from_returns(self, 
                                       returns: pd.Series,
                                       risk_free_rate: float = 0.02,
                                       estimation_method: str = 'historical') -> KellyParameters:
        """
        Estimate Kelly parameters from historical returns.
        
        Args:
            returns: Historical return series
            risk_free_rate: Risk-free rate (annualized)
            estimation_method: Parameter estimation method
            
        Returns:
            Estimated Kelly parameters
        """
        if len(returns) < 30:
            raise ValueError("Need at least 30 observations for parameter estimation")
        
        # Remove NaN values
        clean_returns = returns.dropna()
        
        if estimation_method == 'historical':
            # Simple historical estimation
            expected_return = clean_returns.mean() * 252  # Annualized
            volatility = clean_returns.std() * np.sqrt(252)  # Annualized
            
        elif estimation_method == 'robust':
            # Robust estimation (trimmed mean, winsorized std)
            trimmed_returns = self._trim_outliers(clean_returns, trim_pct=0.05)
            expected_return = trimmed_returns.mean() * 252
            volatility = trimmed_returns.std() * np.sqrt(252)
            
        elif estimation_method == 'bayesian':
            # Bayesian estimation with priors
            expected_return, volatility = self._bayesian_parameter_estimation(clean_returns)
            
        else:
            raise ValueError(f"Unknown estimation method: {estimation_method}")
        
        # Calculate win/loss statistics
        positive_returns = clean_returns[clean_returns > 0]
        negative_returns = clean_returns[clean_returns < 0]
        
        win_probability = len(positive_returns) / len(clean_returns) if len(clean_returns) > 0 else 0.5
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0.01
        avg_loss = abs(negative_returns.mean()) if len(negative_returns) > 0 else 0.01
        
        return KellyParameters(
            expected_return=expected_return,
            volatility=volatility,
            risk_free_rate=risk_free_rate,
            win_probability=win_probability,
            avg_win=avg_win,
            avg_loss=avg_loss
        )
    
    def _calculate_continuous_kelly(self, parameters: KellyParameters) -> float:
        """Calculate Kelly fraction using continuous formula."""
        excess_return = parameters.expected_return - parameters.risk_free_rate
        
        if parameters.volatility <= 0:
            return 0.0
        
        kelly_fraction = excess_return / (parameters.volatility ** 2)
        return max(0.0, kelly_fraction)
    
    def _calculate_discrete_kelly(self, parameters: KellyParameters) -> float:
        """Calculate Kelly fraction using discrete win/loss formula."""
        if (parameters.win_probability is None or 
            parameters.avg_win is None or 
            parameters.avg_loss is None):
            # Fallback to continuous formula
            return self._calculate_continuous_kelly(parameters)
        
        p = parameters.win_probability
        b = parameters.avg_win / parameters.avg_loss  # Win/loss ratio
        
        if p <= 0 or b <= 0:
            return 0.0
        
        kelly_fraction = (p * b - (1 - p)) / b
        return max(0.0, kelly_fraction)
    
    def _calculate_fractional_kelly(self, parameters: KellyParameters) -> float:
        """Calculate fractional Kelly (conservative approach)."""
        full_kelly = self._calculate_continuous_kelly(parameters)
        
        # Use fractional Kelly based on confidence in parameters
        # Higher volatility = lower confidence = smaller fraction
        confidence_factor = 1.0 / (1.0 + parameters.volatility)
        fractional_kelly = full_kelly * confidence_factor * 0.5  # Half Kelly with confidence adjustment
        
        return max(0.0, fractional_kelly)
    
    def _calculate_volatility_adjustment(self, volatility: float) -> float:
        """Calculate volatility-based adjustment factor."""
        # Reduce Kelly fraction for high volatility assets
        # Normal volatility around 0.2 (20%), high volatility > 0.4 (40%)
        
        if volatility <= 0.2:
            # Low volatility - no penalty
            adjustment = 1.0
        elif volatility <= 0.4:
            # Medium volatility - linear penalty
            adjustment = 1.0 - (volatility - 0.2) * self.volatility_adjustment_factor
        else:
            # High volatility - significant penalty
            adjustment = 1.0 - 0.2 * self.volatility_adjustment_factor - (volatility - 0.4) * 2.0 * self.volatility_adjustment_factor
        
        return max(0.1, min(1.0, adjustment))  # Keep between 10% and 100%
    
    def _calculate_kelly_confidence_interval(self, 
                                           parameters: KellyParameters, 
                                           kelly_fraction: float) -> Tuple[float, float]:
        """Calculate confidence interval for Kelly fraction."""
        # Simplified confidence interval based on parameter uncertainty
        # In practice, this would use bootstrap or analytical methods
        
        # Assume 20% uncertainty in parameters
        uncertainty = 0.2
        
        lower_bound = kelly_fraction * (1 - uncertainty)
        upper_bound = kelly_fraction * (1 + uncertainty)
        
        return (max(0.0, lower_bound), min(self.max_kelly_fraction, upper_bound))
    
    def _optimize_portfolio_kelly_mean_variance(self, 
                                              asset_parameters: Dict[str, KellyParameters],
                                              correlation_matrix: np.ndarray,
                                              asset_names: List[str]) -> Dict[str, float]:
        """Optimize portfolio Kelly using mean-variance approach."""
        n_assets = len(asset_names)
        
        # Extract expected returns and volatilities
        expected_returns = np.array([
            asset_parameters.get(name, KellyParameters(0, 0.2, 0.02)).expected_return 
            for name in asset_names
        ])
        
        volatilities = np.array([
            asset_parameters.get(name, KellyParameters(0, 0.2, 0.02)).volatility 
            for name in asset_names
        ])
        
        # Create covariance matrix
        # Ensure correlation matrix matches the number of assets
        if correlation_matrix.shape[0] != n_assets or correlation_matrix.shape[1] != n_assets:
            logger.warning(f"Correlation matrix size {correlation_matrix.shape} doesn't match number of assets {n_assets}")
            # Use identity matrix as fallback
            correlation_matrix = np.eye(n_assets)
        
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        
        # Objective function: maximize Kelly criterion
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            
            if portfolio_variance <= 0:
                return -1e6  # Penalty for invalid portfolio
            
            # Kelly fraction for portfolio
            kelly = portfolio_return / portfolio_variance
            return -kelly  # Negative because we minimize
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
        ]
        
        # Bounds (0 to max_kelly_fraction for each asset)
        bounds = [(0, self.max_kelly_fraction) for _ in range(n_assets)]
        
        # Initial guess
        initial_weights = np.ones(n_assets) / n_assets
        
        try:
            result = optimize.minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                optimal_weights = result.x
            else:
                # Fallback to equal weights
                optimal_weights = np.ones(n_assets) / n_assets
                
        except Exception as e:
            logger.warning(f"Portfolio optimization failed: {e}")
            optimal_weights = np.ones(n_assets) / n_assets
        
        return dict(zip(asset_names, optimal_weights))
    
    def _optimize_portfolio_kelly_risk_parity(self, 
                                            asset_parameters: Dict[str, KellyParameters],
                                            correlation_matrix: np.ndarray,
                                            asset_names: List[str]) -> Dict[str, float]:
        """Optimize portfolio Kelly using risk parity approach."""
        # Risk parity: equal risk contribution from each asset
        volatilities = np.array([
            asset_parameters.get(name, KellyParameters(0, 0.2, 0.02)).volatility 
            for name in asset_names
        ])
        
        # Inverse volatility weighting
        inv_vol_weights = 1.0 / volatilities
        inv_vol_weights = inv_vol_weights / np.sum(inv_vol_weights)
        
        return dict(zip(asset_names, inv_vol_weights))
    
    def _optimize_portfolio_kelly_equal_risk(self, 
                                           asset_parameters: Dict[str, KellyParameters],
                                           correlation_matrix: np.ndarray,
                                           asset_names: List[str]) -> Dict[str, float]:
        """Optimize portfolio Kelly using equal risk contribution."""
        # Equal risk contribution approach
        n_assets = len(asset_names)
        equal_weights = np.ones(n_assets) / n_assets
        
        return dict(zip(asset_names, equal_weights))
    
    def _calculate_correlation_adjustment(self, 
                                        correlation_matrix: np.ndarray, 
                                        weights: List[float]) -> float:
        """Calculate correlation-based adjustment factor."""
        weights_array = np.array(weights)
        
        # Calculate weighted average correlation
        weighted_corr = 0.0
        total_weight = 0.0
        
        n = len(weights)
        for i in range(n):
            for j in range(i + 1, n):
                pair_weight = weights_array[i] * weights_array[j]
                weighted_corr += pair_weight * abs(correlation_matrix[i, j])
                total_weight += pair_weight
        
        if total_weight > 0:
            weighted_corr /= total_weight
        
        # Adjustment factor: reduce allocation for high correlations
        adjustment = 1.0 - weighted_corr * self.correlation_penalty_factor
        
        return max(0.1, min(1.0, adjustment))
    
    def _calculate_diversification_benefit(self, 
                                         correlation_matrix: np.ndarray, 
                                         weights: List[float]) -> float:
        """Calculate diversification benefit ratio."""
        weights_array = np.array(weights)
        n_weights = len(weights_array)
        
        # Ensure correlation matrix matches weights size
        if correlation_matrix.shape[0] != n_weights or correlation_matrix.shape[1] != n_weights:
            logger.warning(f"Correlation matrix size {correlation_matrix.shape} doesn't match weights size {n_weights}")
            # Use identity matrix as fallback
            correlation_matrix = np.eye(n_weights)
        
        # Portfolio variance
        portfolio_var = np.dot(weights_array, np.dot(correlation_matrix, weights_array))
        
        # Weighted average individual variance (assuming unit variances)
        weighted_avg_var = np.sum(weights_array ** 2)
        
        # Diversification ratio
        if weighted_avg_var > 0:
            diversification_ratio = portfolio_var / weighted_avg_var
        else:
            diversification_ratio = 1.0
        
        return 1.0 - diversification_ratio  # Higher is better
    
    def _calculate_risk_budget_allocation(self, 
                                        asset_parameters: Dict[str, KellyParameters],
                                        weights: Dict[str, float],
                                        asset_names: List[str]) -> Dict[str, float]:
        """Calculate risk budget allocation for each asset."""
        risk_contributions = {}
        total_risk = 0.0
        
        for name in asset_names:
            if name in asset_parameters and name in weights:
                # Risk contribution = weight * volatility
                risk_contrib = (weights[name] * 
                              asset_parameters[name].volatility)
                risk_contributions[name] = risk_contrib
                total_risk += risk_contrib
        
        # Normalize to percentages
        if total_risk > 0:
            risk_budget = {name: contrib / total_risk 
                          for name, contrib in risk_contributions.items()}
        else:
            risk_budget = {name: 1.0 / len(asset_names) for name in asset_names}
        
        return risk_budget
    
    def _trim_outliers(self, returns: pd.Series, trim_pct: float = 0.05) -> pd.Series:
        """Trim outliers from returns series."""
        lower_bound = returns.quantile(trim_pct)
        upper_bound = returns.quantile(1 - trim_pct)
        
        return returns[(returns >= lower_bound) & (returns <= upper_bound)]
    
    def _bayesian_parameter_estimation(self, returns: pd.Series) -> Tuple[float, float]:
        """Bayesian parameter estimation with priors."""
        # Simple Bayesian estimation with normal priors
        # Prior: expected return ~ N(0.08, 0.1), volatility ~ InvGamma(2, 0.2)
        
        sample_mean = returns.mean() * 252
        sample_std = returns.std() * np.sqrt(252)
        n = len(returns)
        
        # Bayesian update (simplified)
        prior_mean = 0.08
        prior_precision = 1.0 / (0.1 ** 2)
        
        posterior_precision = prior_precision + n / (sample_std ** 2)
        posterior_mean = (prior_precision * prior_mean + n * sample_mean / (sample_std ** 2)) / posterior_precision
        
        # For volatility, use sample estimate (more complex Bayesian update would be needed)
        posterior_std = sample_std
        
        return posterior_mean, posterior_std


def calculate_enhanced_kelly(returns_data: pd.DataFrame,
                           correlation_matrix: Optional[np.ndarray] = None,
                           risk_free_rate: float = 0.02) -> Dict[str, Any]:
    """
    Convenience function for enhanced Kelly calculation.
    
    Args:
        returns_data: DataFrame with returns for assets
        correlation_matrix: Optional correlation matrix
        risk_free_rate: Risk-free rate
        
    Returns:
        Dictionary with Kelly calculation results
    """
    calculator = EnhancedKellyCriterion()
    
    try:
        # Calculate individual Kelly fractions
        individual_results = {}
        asset_parameters = {}
        
        for column in returns_data.columns:
            returns_series = returns_data[column].dropna()
            
            if len(returns_series) >= 30:
                params = calculator.estimate_parameters_from_returns(
                    returns_series, risk_free_rate
                )
                result = calculator.calculate_kelly_fraction(params)
                
                individual_results[column] = {
                    'kelly_fraction': result.kelly_fraction,
                    'adjusted_kelly': result.adjusted_kelly,
                    'recommended_position': result.recommended_position,
                    'expected_return': params.expected_return,
                    'volatility': params.volatility
                }
                asset_parameters[column] = params
        
        # Portfolio-level calculation if correlation matrix provided
        portfolio_result = None
        if correlation_matrix is not None and len(asset_parameters) > 1:
            asset_names = list(asset_parameters.keys())
            portfolio_result = calculator.calculate_portfolio_kelly(
                asset_parameters, correlation_matrix, asset_names
            )
        
        return {
            'individual_results': individual_results,
            'portfolio_result': portfolio_result.__dict__ if portfolio_result else None,
            'total_kelly': sum(result['recommended_position'] for result in individual_results.values())
        }
        
    except Exception as e:
        logger.error(f"Enhanced Kelly calculation failed: {e}")
        return {
            'error': f'Kelly calculation failed: {str(e)}',
            'individual_results': {},
            'portfolio_result': None
        }