#!/usr/bin/env python3
"""
Main entry point for the StockBot application.
Run with: uv run python main.py
"""

from utils import setup_logging
from cli import run_cli

setup_logging()

if __name__ == "__main__":
    run_cli()