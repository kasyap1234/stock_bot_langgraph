"""
Static Files Module

This module contains static assets for the trading dashboard including
CSS stylesheets, JavaScript files, and images.

The static files are organized into subdirectories:
- css/: Stylesheets for the dashboard interface
- js/: JavaScript files for client-side functionality
- images/: Static images and icons

These files are served by the FastAPI application via the StaticFiles
middleware and are accessible at the /static route.
"""

# This module is primarily for static file organization
# No imports needed as static files are served directly by FastAPI