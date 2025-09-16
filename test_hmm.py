import yfinance as yf
import pandas as pd
import numpy as np
import logging
from agents.technical_analysis import HMMRegimeDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ticker = "RELIANCE.NS"
df = yf.download(ticker, period="1y", progress=False)
df = df.reset_index()
logger.info(f"Downloaded data for {ticker}: shape {df.shape}")

if df.empty:
    logger.error("No data downloaded")
    exit(1)

df.columns = [col.capitalize() for col in df.columns]
df = df.rename(columns={'Adj Close': 'Close'})

hmm = HMMRegimeDetector()

features = hmm.prepare_features(df)
if features.empty:
    logger.warning("Features empty, cannot fit")
    exit(1)

logger.info("Feature statistics:")
for col in features.columns:
    feat = features[col]
    logger.info(f"{col}: shape={len(feat)}, inf={np.isinf(feat).any()}, nan={feat.isna().any()}, min={feat.min()}, max={feat.max()}")

regime, confidence = hmm.fit_hmm_model(features, df)
logger.info(f"HMM fit result: regime={regime}, confidence={confidence}")
if hmm.model is not None:
    logger.info("HMM model fitted successfully without errors")
else:
    logger.warning("HMM model not fitted")