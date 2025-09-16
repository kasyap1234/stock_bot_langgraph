"""
Trading Dashboard - FastAPI Backend
Real-time monitoring and control interface for automated trading system.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi_login import LoginManager
from fastapi_login.exceptions import InvalidCredentialsException
from passlib.context import CryptContext
from pydantic import BaseModel

from config.config import GROQ_API_KEY, MODEL_NAME, DEFAULT_STOCKS
from data.models import State
from main import build_workflow_graph, create_initial_state
from agents import data_fetcher_agent
from data import real_time_data
from simulation import run_trading_simulation
from analysis import PerformanceAnalyzer
from utils import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Login manager
SECRET_KEY = "your-secret-key-change-in-production"  # TODO: Move to config
login_manager = LoginManager(SECRET_KEY, token_url="/auth/token", use_cookie=True)
login_manager.cookie_name = "trading_dashboard_auth"

# In-memory user store (replace with database in production)
fake_users_db = {
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash("admin123"),
        "disabled": False,
    }
}

class User(BaseModel):
    username: str
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscriptions: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client {client_id} connected")

    def disconnect(self, websocket: WebSocket, client_id: str):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client {client_id} disconnected")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Failed to send message: {e}")

    async def broadcast_to_subscribers(self, topic: str, message: str):
        if topic in self.subscriptions:
            for websocket in self.subscriptions[topic]:
                try:
                    await websocket.send_text(message)
                except Exception as e:
                    logger.error(f"Failed to send message to subscriber: {e}")

manager = ConnectionManager()

# Global state for dashboard data
dashboard_state = {
    "portfolio": {},
    "strategies": {},
    "risk_metrics": {},
    "market_data": {},
    "alerts": [],
    "last_update": None
}

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Trading Dashboard")
    # Initialize real-time data feeds
    asyncio.create_task(update_dashboard_data())
    yield
    # Shutdown
    logger.info("Shutting down Trading Dashboard")

# Create FastAPI app
app = FastAPI(
    title="Trading Dashboard",
    description="Real-time monitoring and control interface for automated trading system",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="dashboard/static"), name="static")

# Templates
templates = Jinja2Templates(directory="dashboard/templates")

# Auth functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

@login_manager.user_loader
def load_user(username: str):
    user = get_user(fake_users_db, username)
    return user

# Routes
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, user=Depends(login_manager.optional)):
    if not user:
        return templates.TemplateResponse("login.html", {"request": request})
    return templates.TemplateResponse("dashboard.html", {"request": request, "user": user})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/auth/token")
async def login_for_access_token(request: Request):
    form_data = await request.form()
    username = form_data.get("username")
    password = form_data.get("password")

    user = authenticate_user(fake_users_db, username, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = login_manager.create_access_token(data={"sub": username})
    response = JSONResponse({"access_token": access_token, "token_type": "bearer"})
    login_manager.set_cookie(response, access_token)
    return response

@app.post("/auth/logout")
async def logout():
    response = JSONResponse({"message": "Logged out"})
    login_manager.set_cookie(response, "")
    return response

# API Routes
@app.get("/api/portfolio")
async def get_portfolio(user=Depends(login_manager)):
    return dashboard_state["portfolio"]

@app.get("/api/strategies")
async def get_strategies(user=Depends(login_manager)):
    return dashboard_state["strategies"]

@app.get("/api/risk-metrics")
async def get_risk_metrics(user=Depends(login_manager)):
    return dashboard_state["risk_metrics"]

@app.get("/api/market-data")
async def get_market_data(user=Depends(login_manager)):
    return dashboard_state["market_data"]

@app.get("/api/alerts")
async def get_alerts(user=Depends(login_manager)):
    return dashboard_state["alerts"]

@app.post("/api/run-analysis")
async def run_analysis(request: Request, user=Depends(login_manager)):
    try:
        data = await request.json()
        stocks = data.get("stocks", DEFAULT_STOCKS)

        # Build and run analysis workflow
        graph = build_workflow_graph(stocks)
        initial_state = create_initial_state()
        final_state = graph.invoke(initial_state)

        # Update dashboard state
        update_dashboard_from_analysis(final_state)

        return {"status": "success", "message": "Analysis completed"}
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/run-backtest")
async def run_backtest(request: Request, user=Depends(login_manager)):
    try:
        data = await request.json()
        stocks = data.get("stocks", DEFAULT_STOCKS)
        strategy_params = data.get("params", {})

        # Run backtest simulation
        simulation_results = run_trading_simulation({}, **strategy_params)

        # Analyze performance
        analyzer = PerformanceAnalyzer()
        performance_analysis = analyzer.analyze_strategy_performance(simulation_results)

        # Update dashboard state
        dashboard_state["simulation_results"] = simulation_results
        dashboard_state["performance_analysis"] = performance_analysis

        return {"status": "success", "results": simulation_results, "analysis": performance_analysis}
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages if needed
            await manager.send_personal_message(f"Echo: {data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket, client_id)

# Background tasks
async def update_dashboard_data():
    """Background task to update dashboard data periodically."""
    while True:
        try:
            # Update market data
            market_data = await real_time_data.get_real_time_data(DEFAULT_STOCKS)
            dashboard_state["market_data"] = market_data

            # Check for alerts
            alerts = check_alerts()
            dashboard_state["alerts"] = alerts

            # Broadcast updates
            update_message = {
                "type": "dashboard_update",
                "data": {
                    "market_data": market_data,
                    "alerts": alerts,
                    "timestamp": datetime.now().isoformat()
                }
            }
            await manager.broadcast(json.dumps(update_message))

            dashboard_state["last_update"] = datetime.now()

        except Exception as e:
            logger.error(f"Dashboard update failed: {e}")

        await asyncio.sleep(60)  # Update every minute

def update_dashboard_from_analysis(final_state: State):
    """Update dashboard state from analysis results."""
    # Update portfolio
    dashboard_state["portfolio"] = final_state.get("simulation_results", {})

    # Update strategies
    dashboard_state["strategies"] = final_state.get("final_recommendation", {})

    # Update risk metrics
    dashboard_state["risk_metrics"] = final_state.get("risk_metrics", {})

def check_alerts() -> List[Dict]:
    """Check for trading alerts based on current data."""
    alerts = []

    # Check portfolio drawdown
    portfolio = dashboard_state.get("portfolio", {})
    max_drawdown = portfolio.get("max_drawdown", 0)
    if max_drawdown > 0.15:  # 15% threshold
        alerts.append({
            "type": "warning",
            "message": f"Portfolio drawdown exceeded 15%: {max_drawdown:.1%}",
            "timestamp": datetime.now().isoformat()
        })

    # Check risk metrics
    risk_metrics = dashboard_state.get("risk_metrics", {})
    for symbol, metrics in risk_metrics.items():
        volatility = metrics.get("volatility", 0)
        if volatility > 0.25:  # 25% volatility threshold
            alerts.append({
                "type": "warning",
                "message": f"High volatility detected for {symbol}: {volatility:.1%}",
                "timestamp": datetime.now().isoformat()
            })

    return alerts

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)