"""
GARCH Volatility Estimation Module

This module implements GARCH(1,1) model for volatility forecasting as part of the
advanced risk assessment system. It provides volatility prediction capabilities
with proper parameter estimation and forecasting interfaces.

Requirements addressed: 2.2 - Dynamic risk parameter adjustment based on volatility regime
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm

logger = logging.getLogger(__name__)

# Suppress convergence warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class GARCHParameters:
    """GARCH(1,1) model parameters"""
    omega: float  # Constant term
    alpha: float  # ARCH coefficient (lagged squared residual)
    beta: float   # GARCH coefficient (lagged conditional variance)
    
    def __post_init__(self):
        """Validate GARCH parameters"""
        if self.omega <= 0:
            raise ValueError("Omega must be positive")
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("Alpha and beta must be non-negative")
        if self.alpha + self.beta >= 1:
            raise ValueError("Alpha + beta must be less than 1 for stationarity")


@dataclass
class VolatilityForecast:
    """Volatility forecast results"""
    forecast_variance: float
    forecast_volatility: float
    confidence_interval: Tuple[float, float]
    forecast_date: datetime
    model_params: GARCHParameters
    log_likelihood: float


class GARCHVolatilityEstimator:
    """
    GARCH(1,1) volatility estimator for dynamic risk assessment.
    
    Implements the GARCH(1,1) model:
    σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}
    
    Where:
    - σ²_t is the conditional variance at time t
    - ω is the constant term (omega)
    - α is the ARCH coefficient (alpha)
    - β is the GARCH coefficient (beta)
    - ε²_{t-1} is the squared residual from previous period
    """
    
    def __init__(self, 
                 max_iter: int = 1000,
                 tolerance: float = 1e-6,
                 initial_variance_method: str = 'sample'):
        """
        Initialize GARCH volatility estimator.
        
        Args:
            max_iter: Maximum iterations for optimization
            tolerance: Convergence tolerance
            initial_variance_method: Method for initial variance ('sample' or 'ewma')
        """
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.initial_variance_method = initial_variance_method
        self.fitted_params: Optional[GARCHParameters] = None
        self.log_likelihood: Optional[float] = None
        self.residuals: Optional[np.ndarray] = None
        self.conditional_variances: Optional[np.ndarray] = None
        
    def fit(self, returns: Union[pd.Series, np.ndarray]) -> GARCHParameters:
        """
        Fit GARCH(1,1) model to return series.
        
        Args:
            returns: Time series of returns
            
        Returns:
            Fitted GARCH parameters
            
        Raises:
            ValueError: If returns series is too short or contains invalid data
        """
        # Convert to numpy array and validate
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        returns = np.asarray(returns, dtype=float)
        
        if len(returns) < 50:
            raise ValueError("Need at least 50 observations for GARCH estimation")
        
        if np.any(np.isnan(returns)) or np.any(np.isinf(returns)):
            raise ValueError("Returns contain NaN or infinite values")
        
        # Remove mean (assume zero mean for simplicity)
        residuals = returns - np.mean(returns)
        self.residuals = residuals
        
        # Initial parameter estimates
        initial_params = self._get_initial_parameters(residuals)
        
        # Optimize log-likelihood
        try:
            result = optimize.minimize(
                fun=self._negative_log_likelihood,
                x0=initial_params,
                args=(residuals,),
                method='L-BFGS-B',
                bounds=[(1e-6, None), (1e-6, 0.99), (1e-6, 0.99)],
                options={'maxiter': self.max_iter, 'ftol': self.tolerance}
            )
            
            if not result.success:
                logger.warning(f"GARCH optimization did not converge: {result.message}")
                # Fall back to initial parameters if optimization fails
                omega, alpha, beta = initial_params
            else:
                omega, alpha, beta = result.x
                
        except Exception as e:
            logger.error(f"GARCH optimization failed: {e}")
            omega, alpha, beta = initial_params
        
        # Ensure parameter constraints
        alpha = max(1e-6, min(alpha, 0.99))
        beta = max(1e-6, min(beta, 0.99))
        
        # Ensure stationarity
        if alpha + beta >= 1:
            total = alpha + beta
            alpha = alpha / total * 0.99
            beta = beta / total * 0.99
        
        omega = max(1e-6, omega)
        
        # Store fitted parameters
        self.fitted_params = GARCHParameters(omega=omega, alpha=alpha, beta=beta)
        
        # Calculate conditional variances for fitted model
        self.conditional_variances = self._calculate_conditional_variances(
            residuals, self.fitted_params
        )
        
        # Calculate log-likelihood
        self.log_likelihood = -self._negative_log_likelihood(
            [omega, alpha, beta], residuals
        )
        
        logger.info(f"GARCH(1,1) fitted: ω={omega:.6f}, α={alpha:.6f}, β={beta:.6f}")
        
        return self.fitted_params
    
    def forecast(self, 
                 horizon: int = 1,
                 confidence_level: float = 0.95) -> VolatilityForecast:
        """
        Forecast volatility using fitted GARCH model.
        
        Args:
            horizon: Forecast horizon (days ahead)
            confidence_level: Confidence level for intervals
            
        Returns:
            Volatility forecast with confidence intervals
            
        Raises:
            RuntimeError: If model hasn't been fitted
        """
        if self.fitted_params is None:
            raise RuntimeError("Model must be fitted before forecasting")
        
        # Get last conditional variance
        last_variance = self.conditional_variances[-1]
        
        # Calculate unconditional variance
        unconditional_var = (self.fitted_params.omega / 
                           (1 - self.fitted_params.alpha - self.fitted_params.beta))
        
        # Multi-step ahead forecast
        if horizon == 1:
            # One-step ahead forecast
            forecast_var = (self.fitted_params.omega + 
                          self.fitted_params.alpha * self.residuals[-1]**2 + 
                          self.fitted_params.beta * last_variance)
        else:
            # Multi-step ahead forecast (converges to unconditional variance)
            persistence = self.fitted_params.alpha + self.fitted_params.beta
            forecast_var = (unconditional_var + 
                          (last_variance - unconditional_var) * (persistence ** horizon))
        
        forecast_vol = np.sqrt(forecast_var)
        
        # Calculate confidence intervals (approximate)
        # Using normal approximation for volatility forecast uncertainty
        z_score = norm.ppf((1 + confidence_level) / 2)
        
        # Approximate standard error (simplified)
        std_error = forecast_vol * 0.1  # Rough approximation
        
        ci_lower = max(0, forecast_vol - z_score * std_error)
        ci_upper = forecast_vol + z_score * std_error
        
        return VolatilityForecast(
            forecast_variance=forecast_var,
            forecast_volatility=forecast_vol,
            confidence_interval=(ci_lower, ci_upper),
            forecast_date=datetime.now() + timedelta(days=horizon),
            model_params=self.fitted_params,
            log_likelihood=self.log_likelihood or 0.0
        )
    
    def get_conditional_volatilities(self) -> Optional[np.ndarray]:
        """
        Get conditional volatilities from fitted model.
        
        Returns:
            Array of conditional volatilities, or None if not fitted
        """
        if self.conditional_variances is None:
            return None
        return np.sqrt(self.conditional_variances)
    
    def _get_initial_parameters(self, residuals: np.ndarray) -> List[float]:
        """Get initial parameter estimates for optimization."""
        # Sample variance for omega initialization
        sample_var = np.var(residuals)
        
        # Simple initial estimates
        omega = sample_var * 0.1  # 10% of sample variance
        alpha = 0.1  # Typical starting value
        beta = 0.8   # Typical starting value
        
        return [omega, alpha, beta]
    
    def _negative_log_likelihood(self, params: List[float], residuals: np.ndarray) -> float:
        """
        Calculate negative log-likelihood for GARCH(1,1) model.
        
        Args:
            params: [omega, alpha, beta]
            residuals: Residual series
            
        Returns:
            Negative log-likelihood value
        """
        omega, alpha, beta = params
        
        # Parameter constraints
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e8  # Large penalty for invalid parameters
        
        try:
            # Calculate conditional variances
            conditional_vars = self._calculate_conditional_variances(
                residuals, GARCHParameters(omega, alpha, beta)
            )
            
            # Avoid numerical issues
            conditional_vars = np.maximum(conditional_vars, 1e-8)
            
            # Log-likelihood calculation
            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * conditional_vars) + 
                (residuals**2) / conditional_vars
            )
            
            return -log_likelihood
            
        except Exception:
            return 1e8  # Return large value if calculation fails
    
    def _calculate_conditional_variances(self, 
                                       residuals: np.ndarray, 
                                       params: GARCHParameters) -> np.ndarray:
        """
        Calculate conditional variances using GARCH(1,1) recursion.
        
        Args:
            residuals: Residual series
            params: GARCH parameters
            
        Returns:
            Array of conditional variances
        """
        n = len(residuals)
        conditional_vars = np.zeros(n)
        
        # Initial variance
        if self.initial_variance_method == 'sample':
            conditional_vars[0] = np.var(residuals)
        else:  # EWMA
            conditional_vars[0] = residuals[0]**2
        
        # GARCH recursion
        for t in range(1, n):
            conditional_vars[t] = (params.omega + 
                                 params.alpha * residuals[t-1]**2 + 
                                 params.beta * conditional_vars[t-1])
        
        return conditional_vars


class VolatilityPredictor:
    """
    High-level interface for volatility prediction using GARCH models.
    
    This class provides a simplified interface for volatility forecasting
    that integrates with the existing risk assessment system.
    """
    
    def __init__(self, lookback_window: int = 252):
        """
        Initialize volatility predictor.
        
        Args:
            lookback_window: Number of days to use for model fitting
        """
        self.lookback_window = lookback_window
        self.estimator = GARCHVolatilityEstimator()
        self.last_fit_date: Optional[datetime] = None
        
    def predict_volatility(self, 
                          price_data: pd.DataFrame,
                          horizon: int = 1,
                          refit_threshold_days: int = 30) -> Dict[str, Union[float, Tuple[float, float]]]:
        """
        Predict volatility for given price data.
        
        Args:
            price_data: DataFrame with 'Close' prices
            horizon: Forecast horizon in days
            refit_threshold_days: Refit model if last fit is older than this
            
        Returns:
            Dictionary with volatility forecast and confidence intervals
        """
        try:
            # Calculate returns
            if 'Close' not in price_data.columns:
                raise ValueError("Price data must contain 'Close' column")
            
            returns = price_data['Close'].pct_change().dropna()
            
            if len(returns) < self.lookback_window:
                # Use all available data if less than lookback window
                model_returns = returns
            else:
                # Use most recent lookback_window observations
                model_returns = returns.tail(self.lookback_window)
            
            # Check if we need to refit the model
            should_refit = (
                self.last_fit_date is None or 
                (datetime.now() - self.last_fit_date).days > refit_threshold_days or
                self.estimator.fitted_params is None
            )
            
            if should_refit:
                logger.info("Fitting GARCH model for volatility prediction")
                self.estimator.fit(model_returns)
                self.last_fit_date = datetime.now()
            
            # Generate forecast
            forecast = self.estimator.forecast(horizon=horizon)
            
            # Convert to annualized volatility (assuming daily returns)
            annual_vol = forecast.forecast_volatility * np.sqrt(252)
            annual_ci = (
                forecast.confidence_interval[0] * np.sqrt(252),
                forecast.confidence_interval[1] * np.sqrt(252)
            )
            
            return {
                'volatility_forecast': annual_vol,
                'confidence_interval': annual_ci,
                'forecast_date': forecast.forecast_date,
                'model_params': {
                    'omega': forecast.model_params.omega,
                    'alpha': forecast.model_params.alpha,
                    'beta': forecast.model_params.beta
                },
                'log_likelihood': forecast.log_likelihood
            }
            
        except Exception as e:
            logger.error(f"Error in volatility prediction: {e}")
            # Fallback to simple historical volatility
            try:
                if 'Close' in price_data.columns:
                    returns = price_data['Close'].pct_change().dropna()
                    if len(returns) > 0:
                        hist_vol = returns.std() * np.sqrt(252)
                        if hist_vol > 0:
                            return {
                                'volatility_forecast': hist_vol,
                                'confidence_interval': (hist_vol * 0.8, hist_vol * 1.2),
                                'forecast_date': datetime.now() + timedelta(days=horizon),
                                'model_params': None,
                                'log_likelihood': None,
                                'fallback': True
                            }
            except Exception:
                pass
            
            # Ultimate fallback
            return {
                'volatility_forecast': 0.3,  # Default 30% volatility
                'confidence_interval': (0.2, 0.4),
                'forecast_date': datetime.now() + timedelta(days=horizon),
                'model_params': None,
                'log_likelihood': None,
                'fallback': True
            }


def estimate_garch_volatility(price_data: pd.DataFrame, 
                            horizon: int = 1) -> Dict[str, Union[float, Tuple[float, float]]]:
    """
    Convenience function for GARCH volatility estimation.
    
    Args:
        price_data: DataFrame with 'Close' prices
        horizon: Forecast horizon in days
        
    Returns:
        Dictionary with volatility forecast results
    """
    predictor = VolatilityPredictor()
    return predictor.predict_volatility(price_data, horizon=horizon)