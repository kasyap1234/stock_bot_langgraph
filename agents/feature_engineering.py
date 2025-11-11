import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import yfinance as yf

from data.models import State
from .improved_features import create_enhanced_features, get_important_features

logger = logging.getLogger(__name__)

class FeatureEngineer:
    
    def __init__(self):
        self.sector_mapping = self._load_sector_mapping()
        self.market_cap_cache = {}

    def _load_sector_mapping(self) -> Dict[str, str]:
        # This can be expanded or loaded from a file
        return {
            'RELIANCE.NS': 'Energy', 'TCS.NS': 'Technology', 'HDFCBANK.NS': 'Financial Services',
            'ICICIBANK.NS': 'Financial Services', 'INFY.NS': 'Technology', 'HINDUNILVR.NS': 'Consumer Goods',
            'ITC.NS': 'Consumer Goods', 'KOTAKBANK.NS': 'Financial Services', 'LT.NS': 'Industrials',
            'AXISBANK.NS': 'Financial Services', 'MARUTI.NS': 'Consumer Goods', 'BAJFINANCE.NS': 'Financial Services',
            'BHARTIARTL.NS': 'Telecommunications', 'HCLTECH.NS': 'Technology', 'WIPRO.NS': 'Technology',
            'ULTRACEMCO.NS': 'Materials', 'NESTLEIND.NS': 'Consumer Goods', 'POWERGRID.NS': 'Utilities',
            'NTPC.NS': 'Utilities', 'ONGC.NS': 'Energy', 'COALINDIA.NS': 'Energy', 'GRASIM.NS': 'Materials',
            'JSWSTEEL.NS': 'Materials', 'TATASTEEL.NS': 'Materials', 'ADANIPORTS.NS': 'Industrials',
            'SHREECEM.NS': 'Materials', 'BAJAJ-AUTO.NS': 'Consumer Goods', 'TITAN.NS': 'Consumer Goods',
            'HEROMOTOCO.NS': 'Consumer Goods', 'DRREDDY.NS': 'Healthcare', 'SUNPHARMA.NS': 'Healthcare',
            'CIPLA.NS': 'Healthcare', 'DIVISLAB.NS': 'Healthcare', 'APOLLOHOSP.NS': 'Healthcare',
            'INDUSINDBK.NS': 'Financial Services', 'HDFCLIFE.NS': 'Financial Services', 'SBILIFE.NS': 'Financial Services',
            'BRITANNIA.NS': 'Consumer Goods', 'TECHM.NS': 'Technology', 'EICHERMOT.NS': 'Consumer Goods',
            'BPCL.NS': 'Energy', 'UPL.NS': 'Materials', 'M&M.NS': 'Consumer Goods',
            'TATACONSUM.NS': 'Consumer Goods', 'ASIANPAINT.NS': 'Materials', 'PIDILITIND.NS': 'Materials',
            'NMDC.NS': 'Materials', 'GAIL.NS': 'Utilities', 'VEDL.NS': 'Materials'
        }

    def get_market_cap(self, symbol: str) -> float:
        if symbol in self.market_cap_cache:
            return self.market_cap_cache[symbol]
        try:
            ticker = yf.Ticker(symbol)
            market_cap = ticker.info.get('marketCap', 0)
            if market_cap:
                self.market_cap_cache[symbol] = market_cap
                return market_cap
        except Exception as e:
            logger.warning(f"Could not fetch market cap for {symbol}: {e}")
        return 0

    def create_all_features(self, state: State, symbol: str) -> pd.DataFrame:
        stock_data = state.get('stock_data', {})
        if symbol not in stock_data:
            logger.warning(f"No stock data found for {symbol}")
            return pd.DataFrame()

        df = stock_data[symbol]
        if df.empty:
            logger.warning(f"Empty stock data for {symbol}")
            return pd.DataFrame()

        # Centralized feature creation
        features = create_enhanced_features(df)
        
        # Add cross-sectional features
        sector = self.sector_mapping.get(symbol, 'Unknown')
        market_cap = self.get_market_cap(symbol)
        
        all_sectors = set(self.sector_mapping.values())
        for sec in all_sectors:
            features[f'sector_{sec.lower().replace(" ", "_")}'] = 1 if sector == sec else 0
            
        features['market_cap_log'] = np.log(market_cap + 1)

        logger.info(f"Created {len(features.columns)} features for {symbol}")
        return features

    def prepare_training_data(self, features: pd.DataFrame, target_horizon: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
        if 'Close' not in features.columns:
            raise ValueError("DataFrame must contain 'Close' column for target creation.")

        if len(features) < target_horizon + 10:
            logger.warning("Insufficient data for training")
            return pd.DataFrame(), pd.Series()

        # Create target: future price movement
        future_close = features['Close'].shift(-target_horizon)
        target = (future_close > features['Close']).astype(int)

        # Align features and target
        data = features.copy()
        data['target'] = target
        data = data.dropna(subset=['target'])
        
        y = data['target']
        X = data.drop(columns=['target'])

        return X, y

def feature_engineering_agent(state: State) -> State:
    logging.info("Starting feature engineering agent")
    stock_data = state.get("stock_data", {})

    if not stock_data:
        logger.warning("No stock data available for feature engineering")
        return {**state, "engineered_features": {}}

    engineer = FeatureEngineer()
    engineered_features = {}

    for symbol in stock_data.keys():
        try:
            features = engineer.create_all_features(state, symbol)
            if not features.empty:
                # Prepare training data with target column
                X, y = engineer.prepare_training_data(features)
                if not X.empty:
                    # Combine features and target
                    features_with_target = X.copy()
                    features_with_target['target'] = y
                    engineered_features[symbol] = features_with_target
                    logger.info(f"Engineered {len(features_with_target.columns)} features (including target) for {symbol}")
                else:
                    logger.warning(f"Insufficient data after target creation for {symbol}")
        except Exception as e:
            logger.error(f"Error in feature engineering for {symbol}: {e}", exc_info=True)
            continue

    return {**state, "engineered_features": engineered_features}