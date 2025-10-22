import logging
import os
import random
from datetime import datetime, timedelta
from data.apis import get_stock_history
from data.ingest import clean_stock_data
from data.models import HistoricalData

# Simple RSI calculation (basic implementation)
def calculate_rsi(data: HistoricalData, period: int = 14) -> float:
    if len(data) < period + 1:
        return 50.0  # Neutral if insufficient data
    
    closes = [r['close'] for r in data[-period-1:]]  # Last period+1 closes
    deltas = [closes[i+1] - closes[i] for i in range(len(closes)-1)]
    
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Simple MACD calculation
def calculate_macd(data: HistoricalData, fast: int = 12, slow: int = 26, signal: int = 9) -> float:
    if len(data) < slow:
        return 0.0
    
    closes = [r['close'] for r in data]
    
    def ema(prices, period):
        alpha = 2 / (period + 1)
        ema_val = prices[0]
        for price in prices[1:]:
            ema_val = alpha * price + (1 - alpha) * ema_val
        return ema_val
    
    ema_fast = ema(closes[-fast:], fast)
    ema_slow = ema(closes[-slow:], slow)
    macd_line = ema_fast - ema_slow
    return macd_line

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def run_end_to_end(symbol: str):
    logger.info(f"Running end-to-end test for {symbol}...")
    try:
        # Step 1: Fetch
        raw_data: HistoricalData = get_stock_history(symbol, period="1y", interval="1d")
        logger.info(f"Fetched {len(raw_data)} raw records for {symbol}")
        
        if not raw_data:
            raise ValueError("No data fetched")
        
        # Log sample raw
        logger.info("Raw sample:")
        for r in raw_data[:3]:
            logger.info(f"  Date={r['date']}, Close={r['close']}, Volume={r['volume']}")
        
        # Step 2: Ingest/Clean
        cleaned_data = clean_stock_data(raw_data)
        logger.info(f"Cleaned to {len(cleaned_data)} records")
        
        if not cleaned_data:
            raise ValueError("Cleaning returned empty data")
        
        # Log sample cleaned
        logger.info("Cleaned sample:")
        for r in cleaned_data[:3]:
            logger.info(f"  Date={r['date']}, Close={r['close']}, Volume={r['volume']}")
        
        # Confirm non-zero
        zero_close = sum(1 for r in cleaned_data if r['close'] == 0)
        zero_volume = sum(1 for r in cleaned_data if r['volume'] == 0)
        logger.info(f"Zero Close: {zero_close}/{len(cleaned_data)}, Zero Volume: {zero_volume}/{len(cleaned_data)}")
        
        if zero_close > 0 or zero_volume > 0:
            logger.warning("Some zero values present, but proceeding if not excessive")
        
        # Step 3: Basic TA
        rsi = calculate_rsi(cleaned_data)
        macd = calculate_macd(cleaned_data)
        logger.info(f"RSI: {rsi:.2f}, MACD: {macd:.4f}")
        
        # Confirm valid signals
        assert 0 < rsi < 100, f"Invalid RSI: {rsi}"
        assert macd != 0, f"Zero MACD: {macd}"
        logger.info(f"Valid signals confirmed: RSI={rsi:.2f} (expected ~50), MACD={macd:.4f} (non-zero)")
        
        logger.info(f"End-to-end test PASSED for {symbol}")
        
    except Exception as e:
        logger.error(f"End-to-end test FAILED for {symbol}: {str(e)}")
        raise

if __name__ == "__main__":
    symbols = ["RELIANCE.NS", "TATAMOTORS.NS"]
    for symbol in symbols:
        run_end_to_end(symbol)
        import time
        time.sleep(2)  # Delay between tests
    logger.info("All end-to-end tests complete")