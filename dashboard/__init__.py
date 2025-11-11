"""
Trading Dashboard Module

This module provides a web-based dashboard for monitoring and controlling
the automated trading system. It includes real-time data visualization,
portfolio monitoring, strategy management, and risk assessment tools.

Main Components:
- FastAPI web application with WebSocket support
- Real-time data streaming and updates
- Authentication and authorization
- Portfolio and risk metrics visualization
- Strategy backtesting interface
- Alert system for risk management

Usage:
    from dashboard import app
    # Run the dashboard
    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
"""

from .app import (
    app,
    ConnectionManager,
    dashboard_state,
    User,
    UserInDB,
    Token,
    TokenData,
    manager,
    update_dashboard_data,
    update_dashboard_from_analysis,
    check_alerts,
    verify_password,
    get_password_hash,
    get_user,
    authenticate_user,
    load_user,
)

__all__ = [
    # FastAPI application
    "app",

    # WebSocket management
    "ConnectionManager",
    "manager",

    # Authentication models
    "User",
    "UserInDB",
    "Token",
    "TokenData",

    # Authentication functions
    "verify_password",
    "get_password_hash",
    "get_user",
    "authenticate_user",
    "load_user",

    # Dashboard state and utilities
    "dashboard_state",
    "update_dashboard_data",
    "update_dashboard_from_analysis",
    "check_alerts",
]