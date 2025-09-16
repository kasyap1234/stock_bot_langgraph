# Trading Dashboard

A comprehensive web-based interface for monitoring and controlling automated trading systems.

## Features

### Dashboard Framework
- **Modern Web Interface**: Built with FastAPI backend and HTML/CSS/JavaScript frontend
- **Real-time Updates**: WebSocket connections for live data streaming
- **Responsive Design**: Optimized for desktop and mobile access
- **User Authentication**: Secure session management with login/logout

### Core Components

#### Portfolio Overview
- Real-time P&L tracking and portfolio composition
- Live position monitoring with market values
- Performance metrics and returns visualization

#### Strategy Performance
- Individual strategy returns and win rates
- Sharpe ratios and risk-adjusted performance
- Comparative strategy analysis

#### Risk Monitoring
- Drawdown charts and VaR metrics
- Risk alerts with customizable thresholds
- Position size and sector exposure monitoring

#### Market Data
- Live price feeds from multiple sources
- Technical indicators and market sentiment
- Real-time news and macroeconomic data

### Customization Features
- **Widget Layouts**: Configurable dashboard arrangements
- **Alert Preferences**: Customizable notification thresholds
- **Strategy Parameters**: Real-time parameter adjustment interface
- **Chart Customization**: Timeframes, indicators, and themes

### Interactive Features
- **Real-time Charts**: Interactive charts with technical overlays
- **Backtesting Interface**: Strategy optimization with parameter tuning
- **Risk Controls**: Position adjustments and risk management
- **Historical Analysis**: Performance analysis tools

## Installation

### Prerequisites
- Python 3.12+
- uv package manager
- bun runtime (for frontend dependencies)

### Setup

1. **Install Python dependencies:**
   ```bash
   uv sync
   ```

2. **Install frontend dependencies:**
   ```bash
   cd dashboard
   bun install
   ```

3. **Run the dashboard:**
   ```bash
   uv run uvicorn dashboard.app:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Access the dashboard:**
   - Open http://localhost:8000 in your browser
   - Login with default credentials: `admin` / `admin123`

## API Endpoints

### Authentication
- `POST /auth/token` - Login and get access token
- `POST /auth/logout` - Logout and clear session

### Dashboard Data
- `GET /api/portfolio` - Get portfolio data
- `GET /api/strategies` - Get strategy performance
- `GET /api/risk-metrics` - Get risk metrics
- `GET /api/market-data` - Get market data
- `GET /api/alerts` - Get active alerts

### Actions
- `POST /api/run-analysis` - Run trading analysis
- `POST /api/run-backtest` - Execute backtesting simulation

### Real-time
- `WebSocket /ws/{client_id}` - Real-time data streaming

## Configuration

### Environment Variables
- `GROQ_API_KEY` - API key for LLM services
- `ALPHA_VANTAGE_API_KEY` - Alpha Vantage API key
- `NEWS_API_KEY` - News API key
- `REAL_TIME_ENABLED` - Enable/disable real-time features

### Dashboard Settings
- Edit `config/config.py` for trading parameters
- Modify `dashboard/static/css/dashboard.css` for styling
- Update `dashboard/static/js/dashboard.js` for functionality

## Architecture

### Backend (FastAPI)
- **Authentication**: Session-based with fastapi-login
- **WebSocket**: Real-time data broadcasting
- **API**: RESTful endpoints for data operations
- **Integration**: Connects with existing LangGraph workflow

### Frontend
- **Framework**: Vanilla JavaScript with Bootstrap 5
- **Charts**: Chart.js for data visualization
- **Real-time**: WebSocket client for live updates
- **Responsive**: Mobile-first design approach

### Data Flow
1. **Real-time Data**: Collected from Yahoo Finance, Alpha Vantage, etc.
2. **Analysis Engine**: LangGraph workflow processes market data
3. **Dashboard State**: Centralized state management
4. **WebSocket Broadcast**: Live updates to connected clients

## Security

- **Authentication**: Secure login with bcrypt password hashing
- **Session Management**: HTTP-only cookies for session handling
- **CORS**: Configured for secure cross-origin requests
- **Input Validation**: Pydantic models for API validation

## Development

### Project Structure
```
dashboard/
├── app.py                 # FastAPI application
├── templates/            # Jinja2 HTML templates
│   ├── login.html
│   └── dashboard.html
├── static/               # Static assets
│   ├── css/
│   ├── js/
│   └── images/
├── package.json          # Frontend dependencies
└── README.md            # This file
```

### Adding New Features

1. **Backend**: Add routes in `app.py`
2. **Frontend**: Update templates and JavaScript
3. **Styling**: Modify CSS for new components
4. **Testing**: Add tests for new functionality

### Customization

- **Themes**: Modify CSS variables for branding
- **Widgets**: Extend dashboard sections
- **Charts**: Add new Chart.js visualizations
- **Alerts**: Configure notification thresholds

## Troubleshooting

### Common Issues

1. **WebSocket Connection Failed**
   - Check firewall settings
   - Verify WebSocket URL configuration

2. **Data Not Loading**
   - Check API keys in environment variables
   - Verify network connectivity

3. **Authentication Issues**
   - Clear browser cookies
   - Check password hashing configuration

### Logs
- Application logs: `logs/stock_bot.log`
- Console logs: Browser developer tools
- Server logs: Terminal output

## Contributing

1. Follow the existing code style and patterns
2. Add tests for new functionality
3. Update documentation for changes
4. Ensure responsive design for mobile devices

## License

This project is part of the Stock Trading Bot system. See main project license for details.