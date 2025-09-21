import logging
import time
import functools
from typing import Any, Dict, Optional, Callable

logger = logging.getLogger(__name__)


class StockBotError(Exception):
    """Base exception for stock bot errors."""
    pass


class DataFetchError(StockBotError):
    """Exception for data fetching errors."""
    pass


class APILimitError(StockBotError):
    """Exception for API limit errors."""
    pass


class ValidationError(StockBotError):
    """Exception for data validation errors."""
    pass


class DataFetchingError(Exception):
    """Custom exception for data fetching failures."""
    def __init__(self, message, source=None):
        self.message = message
        self.source = source
        super().__init__(f"[{self.source}] {self.message}" if self.source else self.message)


class APIKeyMissingError(Exception):
    """Exception raised when an API key is missing."""
    pass


def retry_with_exponential_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for retrying functions with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator


def handle_api_errors(func: Callable) -> Callable:
    """Decorator for handling API errors."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"API error in {func.__name__}: {e}")
            raise APILimitError(f"API error: {e}")
    return wrapper


def validate_data(data: Any, required_fields: list = None) -> bool:
    """Validate data structure."""
    if data is None:
        return False
    if required_fields:
        if isinstance(data, dict):
            return all(field in data for field in required_fields)
    return True


def safe_get(data: Dict, key: str, default: Any = None) -> Any:
    """Safely get value from dictionary."""
    try:
        return data.get(key, default)
    except (AttributeError, TypeError):
        return default


def robust_float_conversion(value: Any, default: float = 0.0) -> float:
    """Robustly convert value to float."""
    try:
        if value is None or value == '':
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def robust_int_conversion(value: Any, default: int = 0) -> int:
    """Robustly convert value to int."""
    try:
        if value is None or value == '':
            return default
        return int(float(value))
    except (ValueError, TypeError):
        return default


def check_api_limits() -> bool:
    """Check if API limits are exceeded."""
    # Placeholder implementation
    return True


def graceful_shutdown():
    """Handle graceful shutdown."""
    logger.info("Initiating graceful shutdown...")


def log_error_context(error: Exception, context: Dict[str, Any] = None):
    """Log error with context."""
    context_str = f" Context: {context}" if context else ""
    logger.error(f"Error: {error}{context_str}")


def retry_indicator_calculation(max_retries: int = 3, fallback_method: Optional[Callable] = None):
    """Decorator for retrying indicator calculations with optional fallback."""
    logger.info(f"retry_indicator_calculation called with max_retries={max_retries}, fallback_method={fallback_method is not None}")

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"retry_indicator_calculation wrapper called for function: {func.__name__}")
            logger.info(f"Arguments: {args}, kwargs: {kwargs}")

            for attempt in range(max_retries):
                try:
                    logger.info(f"Attempt {attempt + 1}/{max_retries} for {func.__name__}")
                    result = func(*args, **kwargs)
                    logger.info(f"Attempt {attempt + 1} succeeded for {func.__name__}")
                    return result
                except Exception as e:
                    logger.warning(f"Indicator calculation attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        logger.error(f"Indicator calculation failed after {max_retries} attempts: {e}")
                        if fallback_method is not None:
                            logger.info(f"Using fallback method: {fallback_method.__name__}")
                            try:
                                return fallback_method(*args, **kwargs)
                            except Exception as fallback_e:
                                logger.error(f"Fallback method also failed: {fallback_e}")
                                return None
                        return None
                    logger.warning(f"Retrying {func.__name__} in next attempt...")
            return None
        return wrapper
    return decorator