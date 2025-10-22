"""
Component Health Check System

This module provides health checks for various system components
to ensure graceful degradation when APIs or services are unavailable.
"""

import logging
from typing import Dict, Any, Callable
from functools import wraps
import time

logger = logging.getLogger(__name__)

class ComponentHealthChecker:
    """Tracks the health status of system components."""

    def __init__(self):
        self.component_status = {}
        self.last_check = {}

    def check_component(self, component_name: str, check_func: Callable, timeout: int = 10) -> bool:
        """
        Check if a component is healthy.

        Args:
            component_name: Name of the component
            check_func: Function that returns True if healthy, False otherwise
            timeout: Timeout in seconds for the check

        Returns:
            True if component is healthy, False otherwise
        """
        try:
            start_time = time.time()
            result = check_func()
            check_time = time.time() - start_time

            self.component_status[component_name] = {
                'healthy': result,
                'last_check': time.time(),
                'check_time': check_time,
                'error': None
            }

            if result:
                logger.info(f"Component '{component_name}' is healthy")
            else:
                logger.warning(f"Component '{component_name}' check failed")

            return result

        except Exception as e:
            self.component_status[component_name] = {
                'healthy': False,
                'last_check': time.time(),
                'check_time': 0,
                'error': str(e)
            }
            logger.warning(f"Component '{component_name}' check error: {e}")
            return False

    def is_component_healthy(self, component_name: str, max_age: int = 300) -> bool:
        """
        Check if a component is currently considered healthy.

        Args:
            max_age: Maximum age in seconds for cached health checks

        Returns:
            True if component is healthy and check is recent
        """
        if component_name not in self.component_status:
            return False

        status = self.component_status[component_name]
        if time.time() - status['last_check'] > max_age:
            # Health check is too old
            return False

        return status['healthy']

    def get_component_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all components."""
        return self.component_status.copy()

    def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of component health."""
        total_components = len(self.component_status)
        healthy_components = sum(1 for status in self.component_status.values() if status['healthy'])

        return {
            'total_components': total_components,
            'healthy_components': healthy_components,
            'unhealthy_components': total_components - healthy_components,
            'overall_health': healthy_components / total_components if total_components > 0 else 0.0,
            'component_details': self.get_component_status()
        }

# Global health checker instance
health_checker = ComponentHealthChecker()

def with_health_check(component_name: str, fallback_return=None):
    """
    Decorator to add health checking to functions.

    Args:
        component_name: Name of the component
        fallback_return: Value to return if component is unhealthy
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not health_checker.is_component_healthy(component_name):
                logger.warning(f"Component '{component_name}' is unhealthy, using fallback")
                return fallback_return
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in component '{component_name}': {e}")
                # Mark component as unhealthy
                health_checker.component_status[component_name] = {
                    'healthy': False,
                    'last_check': time.time(),
                    'check_time': 0,
                    'error': str(e)
                }
                return fallback_return
        return wrapper
    return decorator

def check_data_apis():
    """Check health of data API components."""
    from config.api_config import ALPHA_VANTAGE_API_KEY, NEWS_API_KEY, FRED_API_KEY

    # Check Yahoo Finance (always available)
    def check_yahoo():
        try:
            import yfinance as yf
            ticker = yf.Ticker("AAPL")
            data = ticker.history(period="1d")
            return not data.empty
        except:
            return False

    health_checker.check_component("yahoo_finance", check_yahoo)

    # Check Alpha Vantage
    def check_alpha_vantage():
        return bool(ALPHA_VANTAGE_API_KEY and ALPHA_VANTAGE_API_KEY not in ["", "demo", "9YM9MF6IN0GJMMCO"])

    health_checker.check_component("alpha_vantage", check_alpha_vantage)

    # Check NewsAPI
    def check_newsapi():
        return bool(NEWS_API_KEY and NEWS_API_KEY not in ["", "YOUR_NEWS_API_KEY_HERE"])

    health_checker.check_component("newsapi", check_newsapi)

    # Check FRED
    def check_fred():
        return bool(FRED_API_KEY and FRED_API_KEY not in ["", "YOUR_FRED_API_KEY_HERE"])

    health_checker.check_component("fred", check_fred)

def check_ml_components():
    """Check health of ML components."""
    # Check if required ML libraries are available
    try:
        import sklearn
        sklearn_available = True
    except ImportError:
        sklearn_available = False

    health_checker.check_component("scikit_learn", lambda: sklearn_available)

    try:
        import tensorflow
        tf_available = True
    except ImportError:
        tf_available = False

    health_checker.check_component("tensorflow", lambda: tf_available)

    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False

    health_checker.check_component("pytorch", lambda: torch_available)

def perform_health_checks():
    """Perform all health checks."""
    logger.info("Performing component health checks...")

    check_data_apis()
    check_ml_components()

    summary = health_checker.get_health_summary()
    logger.info(f"Health check summary: {summary['healthy_components']}/{summary['total_components']} components healthy")

    return summary</content>
</xai:function_call">The file `utils/component_health.py` has been created successfully.