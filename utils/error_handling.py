

import logging
import time
from typing import Any, Callable, Optional, Type, Union
from functools import wraps
import requests

logger = logging.getLogger(__name__)


class StockBotError(Exception):
    
    pass


class DataFetchError(StockBotError):
    
    pass


class APILimitError(StockBotError):
    
    pass


class ValidationError(StockBotError):
    
    pass


def retry_with_exponential_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
                        raise StockBotError(f"Function {func.__name__} failed after retries: {e}") from e

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)

                    # Add jitter to prevent thundering herd
                    import random
                    jitter = delay * 0.1 * random.uniform(-1, 1)
                    delay = max(0.1, delay + jitter)

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )

                    time.sleep(delay)

            # This should never be reached
            return None

        return wrapper
    return decorator


def handle_api_errors(exceptions_to_handle: tuple = (requests.RequestException,), custom_message: str = ""):
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except exceptions_to_handle as e:
                error_msg = f"{custom_message} {str(e)}".strip()
                logger.error(f"API error in {func.__name__}: {error_msg}")
                raise DataFetchError(error_msg) from e
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {e}")
                raise StockBotError(f"Unexpected error in {func.__name__}: {e}") from e

        return wrapper
    return decorator


def validate_data(func_data: Any, validator_func: Callable[[Any], bool], error_message: str = "Data validation failed"):
    
    if not validator_func(func_data):
        logger.error(error_message)
        raise ValidationError(error_message)


def safe_get(data: dict, keys: list, default: Any = None) -> Any:
    
    try:
        for key in keys:
            if isinstance(data, dict) and key in data:
                data = data[key]
            else:
                return default
        return data
    except (KeyError, TypeError, IndexError):
        return default


def robust_float_conversion(value: Any, default: float = 0.0) -> float:
    
    if value is None or value == "":
        return default

    try:
        # Handle string representations with currency symbols
        if isinstance(value, str):
            import re
            # Remove common currency symbols and separators
            clean_value = re.sub(r'[₹$€£,]', '', value.strip())
            return float(clean_value)
        else:
            return float(value)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert {value} to float, using default {default}")
        return default


def robust_int_conversion(value: Any, default: int = 0) -> int:
    
    try:
        # Handle float values (e.g., 100.0 to 100)
        if isinstance(value, float):
            # Check if it's a whole number
            if value.is_integer():
                return int(value)
            else:
                # Round to nearest integer
                return int(round(value))

        # Handle string values
        if isinstance(value, str):
            import re
            # Remove decimal part if present
            clean_value = re.sub(r'\..*$', '', re.sub(r'[₹$€£,]', '', value.strip()))
            return int(clean_value)

        return int(value)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert {value} to int, using default {default}")
        return default


def check_api_limits(api_name: str, rate_limits: dict) -> bool:
    
    # This is a placeholder for rate limiting logic
    # In production, you'd implement actual rate tracking

    current_time = time.time()
    request_count = rate_limits.get('count', 0)
    window_start = rate_limits.get('window_start', current_time)
    requests_per_minute = rate_limits.get('rpm', 5)

    # Reset counter if window has passed (60 seconds)
    if current_time - window_start > 60:
        rate_limits['count'] = 0
        rate_limits['window_start'] = current_time

    # Check if limit exceeded
    if request_count >= requests_per_minute:
        logger.warning(f"API rate limit exceeded for {api_name}")
        return False

    # Increment counter
    rate_limits['count'] = request_count + 1

    return True


def graceful_shutdown(signals: list = None):
    
    import signal
    import sys

    if signals is None:
        signals = [signal.SIGINT, signal.SIGTERM]

    def signal_handler(signum, frame):
        
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        # Perform cleanup here if needed
        sys.exit(0)

    for sig in signals:
        signal.signal(sig, signal_handler)

    logger.info("Graceful shutdown handler configured")


def retry_indicator_calculation(
    max_retries: int = 3,
    fallback_method: Callable = None,
    exceptions: tuple = (Exception,)
):
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            symbol = kwargs.get('symbol', 'unknown')

            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"Indicator {func.__name__} succeeded on attempt {attempt + 1} for {symbol}")
                    return result
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        # Try fallback method if available
                        if fallback_method:
                            try:
                                logger.warning(f"Using fallback method for {func.__name__} after {max_retries + 1} failed attempts for {symbol}")
                                return fallback_method(*args, **kwargs)
                            except Exception as fallback_e:
                                logger.error(f"Fallback method also failed for {func.__name__} on {symbol}: {fallback_e}")

                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__} on {symbol}: {e}")
                        # Return neutral signal instead of raising exception
                        return {"error": f"Indicator calculation failed after retries: {str(e)}"}

                    # Log retry attempt
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__} on {symbol}: {e}. "
                        f"Retrying..."
                    )

                    # Brief pause before retry
                    time.sleep(0.1 * (attempt + 1))

            # This should never be reached
            return {"error": "Unexpected error in retry logic"}

        return wrapper
    return decorator


def log_error_context(func: Callable) -> Callable:
    
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            import inspect

            # Get function arguments safely
            try:
                # Only log first few args for security
                safe_args = []
                for arg in args[:3]:  # Limit to first 3 args
                    if isinstance(arg, dict):
                        # Don't log entire dicts for security
                        safe_arg = f"dict({len(arg)} keys)"
                    elif isinstance(arg, (list, tuple)):
                        safe_arg = f"{type(arg).__name__}({len(arg)} items)"
                    else:
                        safe_arg = str(arg)[:50] + "..." if len(str(arg)) > 50 else str(arg)
                    safe_args.append(safe_arg)

                args_str = ", ".join(safe_args)
                if len(args) > 3:
                    args_str += ", ..."

                logger.error(
                    f"Error in {func.__module__}.{func.__name__}({args_str}): {type(e).__name__}: {e}"
                )
            except Exception:
                # If logging fails, just log the basics
                logger.error(f"Error in {func.__module__}.{func.__name__}: {type(e).__name__}: {e}")

            raise

    return wrapper