# Project Context

## Purpose
The Stock Bot LangGraph is an advanced stock recommendation system specifically designed for Indian markets. It leverages LangGraph for orchestrating complex analysis workflows to provide intelligent buy/sell recommendations. The system fetches stock data from free APIs, performs multi-dimensional analysis including technical, fundamental, sentiment, and macroeconomic factors, and validates recommendations through backtesting and simulation. The primary goal is to assist Indian investors in making informed trading decisions by combining various analytical approaches into a cohesive recommendation engine.

## Tech Stack
- Python 3.12 (managed via uv)
- LangGraph (for workflow orchestration and state management)
- Flask/Dash (for the web dashboard interface)
- NumPy, pandas (for data processing and analysis)
- Various ML libraries (scikit-learn, TensorFlow/PyTorch for advanced models)
- Free APIs (Yahoo Finance, NSE data sources for Indian stocks)

## Project Conventions

### Code Style
- No comments or docstrings in the codebase
- Clean, modular, and maintainable code
- Small, single-purpose, composable functions
- Consistent naming conventions across files, functions, and variables
- No emojis or decorative formatting

### Architecture Patterns
- Agent-based architecture with specialized analysis agents
- StateGraph pattern using LangGraph for workflow management
- Parallel processing for multiple analysis types
- Circuit breaker pattern for error handling in analysis pipelines
- Modular component design with clear separation of concerns

### Testing Strategy
- Comprehensive test suite covering all major components
- Unit tests for individual agents and utilities
- Integration tests for workflow orchestration
- Backtesting and simulation validation tests
- Performance and accuracy verification tests

### Git Workflow
- Standard Git workflow with feature branches
- Commit conventions focused on clear, descriptive messages
- Regular testing and validation before merges

## Domain Context
The system operates in the Indian stock market context, focusing on NSE-listed stocks. It handles Indian market specifics such as corporate actions, sector analysis, and macroeconomic indicators relevant to Indian economy. The analysis incorporates Indian market volatility patterns, regulatory considerations, and investor sentiment from Indian financial news sources.

## Important Constraints
- Must use only free APIs to avoid subscription costs
- Optimized for Indian stocks and market conditions
- Deterministic and reproducible outputs
- Real-time performance requirements for analysis workflows
- Memory-efficient processing for large datasets

## External Dependencies
- Yahoo Finance API (for historical stock data)
- NSE/BSE data sources (for Indian market data)
- Financial news APIs (for sentiment analysis)
- Macroeconomic data providers (for economic indicators)
- No paid API dependencies - all integrations use free tiers