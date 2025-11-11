#!/usr/bin/env python3
"""
Setup Validation Script for Stock Swing Trade Recommender
Checks configuration, dependencies, and API connectivity
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def print_header(text: str):
    """Print a formatted section header"""
    print(f"\n{'=' * 80}")
    print(f"  {text}")
    print(f"{'=' * 80}\n")


def print_result(check: str, passed: bool, message: str = ""):
    """Print a check result with color coding"""
    status = "✓ PASS" if passed else "✗ FAIL"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    print(f"{color}{status}{reset} - {check}")
    if message:
        print(f"       {message}")


def check_python_version() -> bool:
    """Check if Python version is 3.10 or higher"""
    version = sys.version_info
    required = (3, 10)
    passed = version >= required
    message = f"Python {version.major}.{version.minor}.{version.micro}"
    if not passed:
        message += f" (Required: {required[0]}.{required[1]}+)"
    print_result("Python Version", passed, message)
    return passed


def check_dependencies() -> bool:
    """Check if all required dependencies are installed"""
    required_packages = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("yfinance", "yfinance"),
        ("yahooquery", "yahooquery"),
        ("scikit-learn", "sklearn"),
        ("tensorflow", "tensorflow"),
        ("fastapi", "fastapi"),
        ("langchain", "langchain"),
        ("langgraph", "langgraph"),
        ("groq", "groq"),
    ]

    all_installed = True
    missing = []

    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            all_installed = False
            missing.append(package_name)

    message = "All core dependencies installed" if all_installed else f"Missing: {', '.join(missing)}"
    print_result("Core Dependencies", all_installed, message)
    return all_installed


def check_optional_dependencies() -> Dict[str, bool]:
    """Check optional dependencies"""
    optional_packages = [
        ("TA-Lib", "talib", "Technical analysis library (fallback available)"),
        ("redis", "redis", "For caching (optional)"),
    ]

    results = {}
    for package_name, import_name, description in optional_packages:
        try:
            __import__(import_name)
            installed = True
            message = f"{description} - Installed"
        except ImportError:
            installed = False
            message = f"{description} - Not installed (optional)"

        print_result(f"Optional: {package_name}", installed, message)
        results[package_name] = installed

    return results


def check_env_file() -> Tuple[bool, Dict[str, bool]]:
    """Check if .env file exists and has required keys"""
    env_path = Path(".env")

    if not env_path.exists():
        print_result(".env File", False, "File not found. Copy from .env.example")
        return False, {}

    print_result(".env File", True, "File exists")

    # Check for API keys
    from dotenv import load_dotenv
    load_dotenv()

    api_keys = {
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY", ""),
        "NEWS_API_KEY": os.getenv("NEWS_API_KEY", ""),
        "FRED_API_KEY": os.getenv("FRED_API_KEY", ""),
        "ALPHA_VANTAGE_API_KEY": os.getenv("ALPHA_VANTAGE_API_KEY", ""),
    }

    key_status = {}
    for key_name, key_value in api_keys.items():
        is_set = bool(key_value and key_value != "your_" in key_value.lower())
        key_status[key_name] = is_set

        if key_name == "ALPHA_VANTAGE_API_KEY":
            status_msg = "Configured" if is_set else "Not set (optional)"
            print_result(f"  {key_name}", is_set or True, status_msg)
        else:
            status_msg = "Configured" if is_set else "Not set (limited functionality)"
            print_result(f"  {key_name}", is_set, status_msg)

    return True, key_status


def check_data_connectivity() -> bool:
    """Check if we can fetch sample stock data"""
    try:
        import yfinance as yf

        # Try to fetch a sample stock
        ticker = yf.Ticker("RELIANCE.NS")
        hist = ticker.history(period="5d")

        if not hist.empty:
            print_result("Yahoo Finance Connectivity", True, "Successfully fetched sample data")
            return True
        else:
            print_result("Yahoo Finance Connectivity", False, "No data returned")
            return False
    except Exception as e:
        print_result("Yahoo Finance Connectivity", False, f"Error: {str(e)[:50]}")
        return False


def check_config_file() -> bool:
    """Check if config file is accessible"""
    try:
        from config import config
        print_result("Configuration File", True, "config/config.py loaded successfully")
        return True
    except Exception as e:
        print_result("Configuration File", False, f"Error: {str(e)[:50]}")
        return False


def check_api_connectivity(key_status: Dict[str, bool]) -> Dict[str, bool]:
    """Check API connectivity for configured keys"""
    connectivity = {}

    # Check Groq API
    if key_status.get("GROQ_API_KEY"):
        try:
            from groq import Groq
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            # Simple test - list models
            models = client.models.list()
            print_result("Groq API Connectivity", True, "API key valid")
            connectivity["GROQ"] = True
        except Exception as e:
            error_msg = str(e)[:50]
            print_result("Groq API Connectivity", False, f"Error: {error_msg}")
            connectivity["GROQ"] = False
    else:
        print_result("Groq API Connectivity", False, "API key not configured (skipping)")
        connectivity["GROQ"] = False

    # Check News API
    if key_status.get("NEWS_API_KEY"):
        try:
            import requests
            api_key = os.getenv("NEWS_API_KEY")
            url = f"https://newsapi.org/v2/top-headlines?country=in&apiKey={api_key}&pageSize=1"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print_result("News API Connectivity", True, "API key valid")
                connectivity["NEWS"] = True
            else:
                print_result("News API Connectivity", False, f"HTTP {response.status_code}")
                connectivity["NEWS"] = False
        except Exception as e:
            print_result("News API Connectivity", False, f"Error: {str(e)[:50]}")
            connectivity["NEWS"] = False
    else:
        print_result("News API Connectivity", False, "API key not configured (skipping)")
        connectivity["NEWS"] = False

    # Check FRED API
    if key_status.get("FRED_API_KEY"):
        try:
            import requests
            api_key = os.getenv("FRED_API_KEY")
            url = f"https://api.stlouisfed.org/fred/series?series_id=GDP&api_key={api_key}&file_type=json"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print_result("FRED API Connectivity", True, "API key valid")
                connectivity["FRED"] = True
            else:
                print_result("FRED API Connectivity", False, f"HTTP {response.status_code}")
                connectivity["FRED"] = False
        except Exception as e:
            print_result("FRED API Connectivity", False, f"Error: {str(e)[:50]}")
            connectivity["FRED"] = False
    else:
        print_result("FRED API Connectivity", False, "API key not configured (skipping)")
        connectivity["FRED"] = False

    return connectivity


def print_summary(results: Dict[str, bool]):
    """Print validation summary"""
    print_header("VALIDATION SUMMARY")

    critical_checks = ["python", "dependencies", "env_file", "config", "data_connectivity"]
    critical_passed = sum(1 for k in critical_checks if results.get(k, False))
    critical_total = len(critical_checks)

    optional_passed = sum(1 for k, v in results.items() if k not in critical_checks and v)
    optional_total = len([k for k in results.keys() if k not in critical_checks])

    print(f"Critical Checks: {critical_passed}/{critical_total} passed")
    print(f"Optional Checks: {optional_passed}/{optional_total} passed")

    if critical_passed == critical_total:
        print("\n✓ System is ready to use!")
        print("\nYou can now run:")
        print("  uv run main.py --ticker RELIANCE.NS")
        return True
    else:
        print("\n✗ Some critical checks failed. Please review the issues above.")
        print("\nCommon fixes:")
        print("  1. Install dependencies: uv sync")
        print("  2. Create .env file: cp .env.example .env")
        print("  3. Add API keys to .env file")
        return False


def main():
    """Run all validation checks"""
    print_header("STOCK SWING TRADE RECOMMENDER - SETUP VALIDATION")

    results = {}

    # Critical checks
    print_header("CRITICAL CHECKS")
    results["python"] = check_python_version()
    results["dependencies"] = check_dependencies()
    results["config"] = check_config_file()

    # Environment configuration
    print_header("ENVIRONMENT CONFIGURATION")
    env_exists, key_status = check_env_file()
    results["env_file"] = env_exists

    # Optional dependencies
    print_header("OPTIONAL DEPENDENCIES")
    optional_deps = check_optional_dependencies()
    results.update(optional_deps)

    # Data connectivity
    print_header("DATA CONNECTIVITY")
    results["data_connectivity"] = check_data_connectivity()

    # API connectivity (only if keys are configured)
    if env_exists and any(key_status.values()):
        print_header("API CONNECTIVITY")
        api_connectivity = check_api_connectivity(key_status)
        results.update(api_connectivity)

    # Print summary
    system_ready = print_summary(results)

    # Exit with appropriate code
    sys.exit(0 if system_ready else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error during validation: {e}")
        sys.exit(1)
