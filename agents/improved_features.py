"""
Improved Feature Engineering for Better ML Predictions
Adds predictive features that improve model accuracy
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum-based features that are highly predictive"""
    df = df.copy()
    
    # Rate of Change (ROC) - multiple periods
    for period in [5, 10, 20]:
        df[f'ROC_{period}'] = df['Close'].pct_change(period) * 100
    
    # Momentum indicator
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    df['Momentum_20'] = df['Close'] - df['Close'].shift(20)
    
    # Price acceleration (second derivative)
    df['Price_Accel'] = df['Close'].diff().diff()
    
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume-based features for confirmation"""
    df = df.copy()
    
    # Volume ratios
    df['Volume_Ratio_5'] = df['Volume'] / df['Volume'].rolling(5).mean()
    df['Volume_Ratio_20'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_EMA'] = df['OBV'].ewm(span=20).mean()
    
    # Volume-Price Trend
    df['VPT'] = (df['Volume'] * df['Close'].pct_change()).fillna(0).cumsum()
    
    # Money Flow Index (MFI) - volume-weighted RSI
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    
    money_ratio = positive_flow.rolling(14).sum() / negative_flow.rolling(14).sum()
    df['MFI'] = 100 - (100 / (1 + money_ratio))
    
    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility features for risk assessment"""
    df = df.copy()
    
    # Historical Volatility (different periods)
    for period in [10, 20, 30]:
        df[f'HV_{period}'] = df['Close'].pct_change().rolling(period).std() * np.sqrt(252)
    
    # Average True Range (ATR) - properly calculated
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    df['ATR_Percent'] = (df['ATR'] / df['Close']) * 100
    
    # Volatility ratio
    df['Volatility_Ratio'] = df['HV_10'] / df['HV_30']
    
    return df


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add trend strength and direction features"""
    df = df.copy()
    
    # Multiple moving averages
    for period in [5, 10, 20, 50, 200]:
        df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
        df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
    
    # Price position relative to MAs
    for period in [20, 50, 200]:
        df[f'Price_Above_SMA_{period}'] = (df['Close'] > df[f'SMA_{period}']).astype(int)
        df[f'Price_Distance_SMA_{period}'] = ((df['Close'] - df[f'SMA_{period}']) / df[f'SMA_{period}']) * 100
    
    # MA crossovers
    df['SMA_5_20_Cross'] = ((df['SMA_5'] > df['SMA_20']).astype(int) - 
                            (df['SMA_5'] > df['SMA_20']).shift().astype(int))
    df['SMA_20_50_Cross'] = ((df['SMA_20'] > df['SMA_50']).astype(int) - 
                             (df['SMA_20'] > df['SMA_50']).shift().astype(int))
    
    # ADX (Average Directional Index) for trend strength
    # Simplified ADX calculation
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff() * -1
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr = true_range = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low'] - df['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
    
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['ADX'] = dx.rolling(14).mean()
    
    return df


def add_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add candlestick pattern features"""
    df = df.copy()
    
    # Basic candlestick patterns
    df['Body'] = df['Close'] - df['Open']
    df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    
    # Doji (small body relative to range)
    df['Is_Doji'] = (abs(df['Body']) / (df['High'] - df['Low']) < 0.1).astype(int)
    
    # Hammer/Hanging Man
    df['Is_Hammer'] = ((df['Lower_Shadow'] > 2 * abs(df['Body'])) & 
                       (df['Upper_Shadow'] < abs(df['Body']))).astype(int)
    
    # Engulfing patterns
    df['Bullish_Engulfing'] = ((df['Body'] > 0) & 
                                (df['Body'].shift() < 0) &
                                (df['Close'] > df['Open'].shift()) &
                                (df['Open'] < df['Close'].shift())).astype(int)
    
    df['Bearish_Engulfing'] = ((df['Body'] < 0) & 
                                (df['Body'].shift() > 0) &
                                (df['Close'] < df['Open'].shift()) &
                                (df['Open'] > df['Close'].shift())).astype(int)
    
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features between important indicators"""
    df = df.copy()

    # Price and volume interaction
    df['Price_Volume_Interaction'] = df['Close'].pct_change() * df['Volume']

    # Momentum and volatility interaction
    if 'ROC_10' in df.columns and 'HV_10' in df.columns:
        df['Momentum_Volatility_Interaction'] = df['ROC_10'] * df['HV_10']

    # Trend and momentum interaction
    if 'ADX' in df.columns and 'Momentum_10' in df.columns:
        df['Trend_Momentum_Interaction'] = df['ADX'] * df['Momentum_10']
        
    return df


def add_fourier_transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Fourier transform features to capture cyclical patterns"""
    df = df.copy()
    
    close_prices = df['Close'].values
    fft_result = np.fft.fft(close_prices)
    
    # Get dominant frequencies
    fft_freq = np.fft.fftfreq(len(close_prices))
    dominant_freq_idx = np.argsort(np.abs(fft_result))[::-1][1:4] # Top 3 dominant frequencies
    
    for i, idx in enumerate(dominant_freq_idx):
        freq = fft_freq[idx]
        if freq != 0:
            period = 1 / freq
            df[f'Dominant_Cycle_{i+1}'] = np.sin(2 * np.pi * freq * np.arange(len(df)))
            
    return df


def add_wavelet_transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Wavelet transform features for multi-resolution analysis"""
    df = df.copy()
    
    try:
        import pywt
        
        # Decompose price series using Daubechies wavelet
        coeffs = pywt.wavedec(df['Close'], 'db4', level=4)
        
        # Add approximation and detail coefficients as features
        df['Wavelet_A4'] = np.repeat(coeffs[0], len(df) // len(coeffs[0]) + 1)[:len(df)]
        for i, d in enumerate(coeffs[1:]):
            df[f'Wavelet_D{4-i}'] = np.repeat(d, len(df) // len(d) + 1)[:len(df)]
            
    except ImportError:
        logger.warning("PyWavelets not installed, skipping wavelet features.")
        
    return df


def create_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all enhanced features for improved ML predictions
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with additional predictive features
    """
    try:
        logger.info("Creating enhanced features for ML models...")
        
        if 'close' in df.columns:
            df = df.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low',
                'close': 'Close', 'volume': 'Volume'
            })
        
        # Base features
        df = add_momentum_features(df)
        df = add_volume_features(df)
        df = add_volatility_features(df)
        df = add_trend_features(df)
        df = add_pattern_features(df)
        
        # Advanced signal processing features
        df = add_fourier_transform_features(df)
        df = add_wavelet_transform_features(df)
        
        # Interaction features (should be added after base features)
        df = add_interaction_features(df)
        
        # Remove infs and NaNs
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(0)
        
        logger.info(f"Enhanced features created. Total features: {len(df.columns)}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error creating enhanced features: {e}", exc_info=True)
        return df


def get_important_features() -> List[str]:
    """
    Return list of most important features for ML models
    Based on feature importance analysis and new additions
    """
    return [
        # Momentum
        'ROC_10', 'Momentum_10', 'Price_Accel',
        
        # Volume
        'Volume_Ratio_5', 'MFI', 'OBV_EMA',
        
        # Volatility
        'ATR_Percent', 'HV_20', 'Volatility_Ratio',
        
        # Trend
        'ADX', 'Price_Distance_SMA_50', 'SMA_20_50_Cross',
        
        # Patterns
        'Bullish_Engulfing', 'Bearish_Engulfing',
        
        # Signal Processing
        'Dominant_Cycle_1', 'Wavelet_D4',
        
        # Interaction
        'Momentum_Volatility_Interaction', 'Trend_Momentum_Interaction',
        
        # Standard ML
        'Return_Lag_1', 'Return_Mean_10', 'Price_Percentile_50',
        'Distance_From_52W_High'
    ]
