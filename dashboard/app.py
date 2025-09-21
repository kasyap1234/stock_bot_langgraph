import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Depends, HTTPException, status, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi_csrf_protect import CsrfProtect
from fastapi_csrf_protect.exceptions import CsrfProtectError
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from jose import JWTError, jwt
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from config.config import settings
from data.models import State
from main import build_workflow_graph, create_initial_state
from agents import data_fetcher_agent
from data import real_time_data
from simulation import run_trading_simulation
from analysis import PerformanceAnalyzer
from utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# SQLAlchemy setup for in-memory SQLite
SQLALCHEMY_DATABASE_URL = "sqlite:///./dashboard.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    disabled = Column(Boolean, default=False)

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)

def get_user(db: Session, username: str):
    """Get user from database."""
    return db.query(User).filter(User.username == username).first()

def authenticate_user(db: Session, username: str, password: str):
    """Authenticate user."""
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Get current user from JWT token."""
    credentials = await credentials()
    token = credentials.credentials
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    user = get_user(db, username=username)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

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

dashboard_state = {
    "portfolio": {},
    "strategies": {},
    "risk_metrics": {},
    "market_data": {},
    "alerts": [],
    "last_update": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Trading Dashboard")
    # Initialize real-time data feeds
    asyncio.create_task(update_dashboard_data())
    yield
    # Shutdown
    logger.info("Shutting down Trading Dashboard")

app = FastAPI(
    title="Trading Dashboard",
    description="Real-time monitoring and control interface for automated trading system",
    version="1.0.0",
    lifespan=lifespan
)

# CSRF Protection
csrf = CsrfProtect(app, config={
    'secret_key': settings.secret_key,
    'cookie_name': 'csrf_token',
})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="dashboard/static"), name="static")

templates = Jinja2Templates(directory="dashboard/templates")

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, current_user: Optional[User] = Depends(get_current_user)):
    if current_user is None:
        return templates.TemplateResponse("login.html", {"request": request})
    return templates.TemplateResponse("dashboard.html", {"request": request, "user": current_user})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/auth/token")
async def login_for_access_token(
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    user = authenticate_user(db, username, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/auth/logout", response_class=RedirectResponse)
async def logout(current_user: User = Depends(get_current_user)):
    return RedirectResponse(url="/login", status_code=303)

@app.get("/api/portfolio")
async def get_portfolio(current_user: User = Depends(get_current_user)):
    return dashboard_state["portfolio"]

@app.get("/api/strategies")
async def get_strategies(current_user: User = Depends(get_current_user)):
    return dashboard_state["strategies"]

@app.get("/api/risk-metrics")
async def get_risk_metrics(current_user: User = Depends(get_current_user)):
    return dashboard_state["risk_metrics"]

@app.get("/api/market-data")
async def get_market_data(current_user: User = Depends(get_current_user)):
    return dashboard_state["market_data"]

@app.get("/api/alerts")
async def get_alerts(current_user: User = Depends(get_current_user)):
    return dashboard_state["alerts"]

@app.post("/api/run-analysis")
async def run_analysis(request: Request, current_user: User = Depends(get_current_user)):
    try:
        data = await request.json()
        stocks = data.get("stocks", settings.default_stocks)

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
async def run_backtest(request: Request, current_user: User = Depends(get_current_user)):
    try:
        data = await request.json()
        stocks = data.get("stocks", settings.default_stocks)
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

async def update_dashboard_data():
    
    while True:
        try:
            # Update market data
            market_data = await real_time_data.get_real_time_data(settings.default_stocks)
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
    
    # Update portfolio
    dashboard_state["portfolio"] = final_state.get("simulation_results", {})

    # Update strategies
    dashboard_state["strategies"] = final_state.get("final_recommendation", {})

    # Update risk metrics
    dashboard_state["risk_metrics"] = final_state.get("risk_metrics", {})

def check_alerts() -> List[Dict]:
    
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