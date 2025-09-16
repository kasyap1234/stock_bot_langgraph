

import logging
import logging.handlers
import sys
from typing import Optional, Dict, Any
from pathlib import Path

from config.config import LOG_LEVEL, LOG_FORMAT


def setup_logging(
    log_level: str = LOG_LEVEL,
    log_format: str = LOG_FORMAT,
    log_file: Optional[str] = "logs/stock_bot.log",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Formatter
    formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Suppress noisy third-party library logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)

    logger.info("Logging configuration initialized")
    logger.info(f"Log level: {log_level}")
    if log_file:
        logger.info(f"Log file: {log_file}")


def get_logger(name: str) -> logging.Logger:
    
    return logging.getLogger(name)


class LogContextManager:
    

    def __init__(self, logger: logging.Logger, context: Dict[str, Any]):
        self.logger = logger
        self.context = context
        self.old_extra = getattr(logger, 'extra', {})

    def __enter__(self):
        # Add context to logger
        setattr(self.logger, 'extra', {**self.old_extra, **self.context})
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original context
        setattr(self.logger, 'extra', self.old_extra)


def log_function_call(logger: Optional[logging.Logger] = None):
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)

            logger.debug(f"Entering {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Exiting {func.__name__} with result type: {type(result).__name__}")
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                raise
        return wrapper
    return decorator


def log_performance(logger: logging.Logger, threshold_ms: float = 1000):
    
    import time

    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000

                if execution_time > threshold_ms:
                    logger.warning(
                        f"{func.__name__} execution time: {execution_time:.2f}ms "
                        f"(threshold: {threshold_ms}ms)"
                    )
                else:
                    logger.debug(f"{func.__name__} execution time: {execution_time:.2f}ms")

                return result
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                logger.error(
                    f"{func.__name__} failed after {execution_time:.2f}ms: {e}"
                )
                raise
        return wrapper
    return decorator