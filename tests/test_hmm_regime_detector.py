import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from yahooquery import Ticker

from agents.technical_analysis import HMMRegimeDetector
import logging

# Enable logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_hmm_regime_detector():
    """Test HMMRegimeDetector with RELIANCE.NS data."""
    symbol = "RELIANCE.NS"
    logger.info(f"Loading data for {symbol}")
    
    try:
        ticker = Ticker(symbol)
        df = ticker.history(period="1y")
        
        if df.empty:
            raise ValueError("No data fetched")
        
        # Handle MultiIndex if present
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
            df.set_index('date', inplace=True)
        
        # Standardize column names (yahooquery uses lowercase)
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        # Ensure datetime index
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        df = df.sort_index()
        
        logger.info(f"Loaded data shape: {df.shape}")
        logger.info(f"Data columns: {df.columns.tolist()}")
        
        detector = HMMRegimeDetector()
        
        # Prepare features
        features = detector.prepare_features(df)
        logger.info(f"Features shape: {features.shape}")
        
        # Debug volume
        if 'Volume' in df.columns:
            print("Raw Volume describe:")
            print(df['Volume'].describe())
            inf_indices = np.where(np.isinf(df['Volume']))[0]
            if len(inf_indices) > 0:
                print("Raw Volume inf indices:", inf_indices)
                print("Raw Volume at inf indices:", df['Volume'].iloc[inf_indices].tolist())
        
        assert not features.empty, "Features should not be empty"
        assert len(features) > 0, f"Expected >0 rows, got {len(features)}"
        
        # Debug if not finite
        if not np.all(np.isfinite(features)):
            logger.error("Non-finite values found!")
            for col in features.columns:
                col_inf = np.isinf(features[col].values).sum()
                col_nan = features[col].isna().sum()
                if col_inf > 0 or col_nan > 0:
                    logger.error(f"Column '{col}': inf={col_inf}, nan={col_nan}")
                    logger.error(f"Min: {features[col].min()}, Max: {features[col].max()}")
                    logger.error(f"Sample values: {features[col].tail(5).tolist()}")
                    # Print the rows with inf
                    inf_mask = np.isinf(features[col].values)
                    inf_rows = features[inf_mask]
                    logger.error(f"Inf rows for {col}: {inf_rows.index.tolist()}")
                    if len(inf_rows) > 0:
                        logger.error(f"Raw volume at inf rows: {df['Volume'].iloc[inf_rows.index].tolist()}")
            raise AssertionError("All features must be finite")
        
        # Assert no inf/nan
        assert np.all(np.isfinite(features)), "All features must be finite"
        assert features.isna().sum().sum() == 0, "No NaN values allowed"
        
        logger.info("Features validation passed: all finite, no NaN")
        
        # Fit model
        success = detector.fit_hmm_model(features, df)
        assert success, "Fit should succeed (HMM or fallback)"
        
        logger.info(f"Fit success: {success}")
        
        # Get signal
        signal = detector.get_hmm_signal(df)
        logger.info(f"HMM signal: {signal}")
        
        assert signal is not None, "Signal should be generated"
        
        logger.info("Test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    test_hmm_regime_detector()