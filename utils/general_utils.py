

import hashlib
from typing import List, Dict, Any, Union
import pandas as pd


def generate_cache_key(*args, **kwargs) -> str:
    
    # Create a deterministic string from all arguments
    key_parts = []

    for arg in args:
        key_parts.append(str(type(arg).__name__))
        if isinstance(arg, (list, tuple)):
            key_parts.append(str(len(arg)))
        elif isinstance(arg, dict):
            key_parts.append(str(sorted(arg.keys()) if arg else ""))
        else:
            key_parts.append(str(arg))

    for key, value in sorted(kwargs.items()):
        key_parts.extend([key, str(value)])

    key_string = "|".join(key_parts)
    return hashlib.sha256(key_string.encode()).hexdigest()


def safe_divide(numerator: Union[float, int], denominator: Union[float, int], default: float = 0.0) -> float:
    
    try:
        if denominator == 0:
            return default
        return float(numerator) / float(denominator)
    except (TypeError, ValueError):
        return default


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    
    if not text or len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


def flatten_dict(data: Dict[str, Any], prefix: str = "", separator: str = ".") -> Dict[str, Any]:
    
    flattened = {}

    for key, value in data.items():
        full_key = f"{prefix}{separator}{key}" if prefix else key

        if isinstance(value, dict):
            flattened.update(flatten_dict(value, full_key, separator))
        else:
            flattened[full_key] = value

    return flattened


def round_to_significant_digits(value: float, sig_digits: int = 3) -> float:
    
    if value == 0:
        return 0.0

    import math
    magnitude = math.floor(math.log10(abs(value)))
    scale = 10 ** (magnitude - sig_digits + 1)

    return round(value / scale) * scale


def format_currency(amount: Union[float, int], currency: str = "INR", precision: int = 2) -> str:
    
    try:
        if currency.upper() == "INR":
            symbol = "₹"
        elif currency.upper() == "USD":
            symbol = "$"
        elif currency.upper() == "EUR":
            symbol = "€"
        else:
            symbol = f"{currency} "

        return f"{symbol}{amount:,.{precision}f}"
    except (ValueError, TypeError):
        return f"{amount}"


def format_percentage(value: float, precision: int = 2, signed: bool = True) -> str:
    
    try:
        formatted = f"{value * 100:.{precision}f}%"
        if signed and value > 0 and not formatted.startswith('+'):
            formatted = "+" + formatted
        return formatted
    except (ValueError, TypeError):
        return f"{value}"


def validate_date_string(date_str: str, formats: List[str] = None) -> bool:
    
    from datetime import datetime

    if formats is None:
        formats = ["%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d"]

    for fmt in formats:
        try:
            datetime.strptime(date_str, fmt)
            return True
        except ValueError:
            continue

    return False


def get_recent_business_days(n_days: int = 30) -> List[str]:
    
    from datetime import datetime, timedelta

    dates = []
    current_date = datetime.now().date()

    while len(dates) < n_days:
        # Check if weekday (0=Monday, 6=Sunday)
        if current_date.weekday() < 5:  # Monday to Friday
            dates.append(current_date.strftime("%Y-%m-%d"))

        current_date -= timedelta(days=1)

    return dates


def calculate_returns(prices: List[float]) -> List[float]:
    
    if len(prices) < 2:
        return []

    returns = []
    for i in range(1, len(prices)):
        if prices[i-1] != 0:
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)

    return returns


def get_summary_stats(data: List[float]) -> Dict[str, float]:
    
    import statistics

    if not data:
        return {"count": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    try:
        return {
            "count": len(data),
            "mean": statistics.mean(data),
            "median": statistics.median(data),
            "std": statistics.stdev(data) if len(data) > 1 else 0,
            "min": min(data),
            "max": max(data)
        }
    except (statistics.StatisticsError, TypeError):
        return {"count": len(data), "error": "Could not calculate statistics"}


def merge_dataframes_by_date(dfs: List[pd.DataFrame], date_column: str = "date") -> pd.DataFrame:
    
    if not dfs:
        return pd.DataFrame()

    if len(dfs) == 1:
        return dfs[0]

    try:
        # Start with first dataframe
        result = dfs[0]

        # Merge each subsequent dataframe
        for df in dfs[1:]:
            merge_cols = [date_column] if date_column in df.columns else None
            result = result.merge(df, on=date_column, how="outer") if merge_cols else result

        # Sort by date
        if date_column in result.columns:
            result = result.sort_values(date_column)

        return result

    except Exception as e:
        # Return first dataframe if merge fails
        print(f"Warning: DataFrame merge failed: {e}")
        return dfs[0]