"""
Test Suite Module

This module contains the comprehensive test suite for the automated
trading system. It includes unit tests, integration tests, and
performance tests for all major components.

Test Categories:
- Data fetching and API integration tests
- Technical analysis and signal generation tests
- Risk management and portfolio optimization tests
- Machine learning model validation tests
- Backtesting engine and simulation tests
- Dashboard and web interface tests
- Configuration and utility function tests

The test suite ensures system reliability, validates algorithmic
accuracy, and maintains code quality through continuous testing.

Usage:
    # Run all tests
    python -m pytest tests/

    # Run specific test category
    python -m pytest tests/test_data_apis.py

    # Run with coverage
    python -m pytest tests/ --cov=. --cov-report=html
"""

# Test module initialization
# All test files are imported and organized by the pytest framework