import logging
from data.apis import get_stock_history
from data.quality_validator import validate_data_quality
from data.models import HistoricalData

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def diagnose_symbol(symbol: str):
    logger.info(f"Diagnosing {symbol}...")
    try:
        # Fetch historical data
        data: HistoricalData = get_stock_history(symbol, period="1y", interval="1d")
        logger.info(f"Fetched {len(data)} records for {symbol}")
        
        if not data:
            logger.error(f"No data fetched for {symbol}")
            return
        
        # Sample first 5 records
        logger.info(f"Sample data for {symbol} (first 5):")
        for i, record in enumerate(data[:5]):
            logger.info(f"  {i+1}: Date={record['date']}, Open={record['open']}, High={record['high']}, Low={record['low']}, Close={record['close']}, Volume={record['volume']}")
        
        # Check for zero values
        zero_close = sum(1 for r in data if r['close'] == 0)
        zero_volume = sum(1 for r in data if r['volume'] == 0)
        logger.info(f"Zero Close counts: {zero_close}/{len(data)}")
        logger.info(f"Zero Volume counts: {zero_volume}/{len(data)}")
        
        if zero_close > 0 or zero_volume > 0:
            logger.warning(f"Zero values detected in {symbol}: Close={zero_close}, Volume={zero_volume}")
        
        # Run quality validation
        from data.quality_validator import DataQualityValidator
        validator = DataQualityValidator()
        report = validate_data_quality(data, symbol, validator)
        logger.info(f"Quality score for {symbol}: {report.overall_quality_score}")
        logger.info(f"Issues found: {len(report.issues)}")
        for issue in report.issues:
            logger.info(f"  - {issue.issue_type.value}: {issue.description} (severity: {issue.severity})")
        
        # Basic TA check - compute simple RSI (placeholder, but log if data valid)
        if len(data) >= 14:
            logger.info(f"Data sufficient for TA indicators (min 14 periods)")
        else:
            logger.warning(f"Insufficient data for TA: {len(data)} < 14")
            
    except Exception as e:
        logger.error(f"Error diagnosing {symbol}: {str(e)}", exc_info=True)

if __name__ == "__main__":
    symbols = ["RELIANCE.NS", "TATAMOTORS.NS"]
    for symbol in symbols:
        diagnose_symbol(symbol)
        # Add delay to avoid throttling
        import time
        time.sleep(5)
    logger.info("Diagnosis complete")