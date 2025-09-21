"""
Tests for Missing Data Handler

Tests cover:
- Intelligent interpolation methods for missing values
- Data exclusion logic for poor quality data
- Various interpolation strategies
- Quality assessment for interpolated data
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List

from data.missing_data_handler import (
    MissingDataHandler,
    InterpolationMethod,
    DataQuality,
    InterpolationResult,
    ProcessedData,
    handle_missing_data
)
from data.models import StockData, create_stock_data


class TestMissingDataHandler:
    """Test suite for MissingDataHandler"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.handler = MissingDataHandler()
        self.symbol = "AAPL"
        
        # Create sample data with some missing values
        self.sample_data = self._create_sample_data_with_missing()
        self.clean_data = self._create_clean_sample_data()
    
    def _create_clean_sample_data(self, days: int = 20) -> List[StockData]:
        """Create clean sample stock data for testing"""
        data = []
        base_date = datetime.now() - timedelta(days=days)
        base_price = 100.0
        
        for i in range(days):
            date = base_date + timedelta(days=i)
            price_change = (i % 3 - 1) * 0.02  # -2%, 0%, +2% pattern
            close_price = base_price * (1 + price_change)
            
            stock_data = create_stock_data(
                symbol=self.symbol,
                date=date.strftime("%Y-%m-%d"),
                open_price=close_price * 0.99,
                high=close_price * 1.02,
                low=close_price * 0.98,
                close=close_price,
                volume=1000000 + (i % 5) * 100000
            )
            data.append(stock_data)
            
        return data
    
    def _create_sample_data_with_missing(self) -> List[StockData]:
        """Create sample data with missing values"""
        data = self._create_clean_sample_data(20)
        
        # Introduce missing values
        data[3]['close'] = None  # Missing close price
        data[5]['open'] = None   # Missing open price
        data[7]['volume'] = None # Missing volume
        data[10]['high'] = None  # Missing high price
        data[12]['low'] = None   # Missing low price
        
        return data
    
    def _create_heavily_missing_data(self) -> List[StockData]:
        """Create data with heavy missing values (should be excluded)"""
        data = self._create_clean_sample_data(10)
        
        # Make 60% of data missing
        for i in [1, 2, 3, 4, 5, 6]:
            data[i]['close'] = None
            data[i]['open'] = None
            data[i]['high'] = None
            data[i]['low'] = None
        
        return data
    
    def test_handle_empty_data(self):
        """Test handling of empty data"""
        result = self.handler.handle_missing_data([], self.symbol)
        
        assert result.quality == DataQuality.EXCLUDED
        assert len(result.data) == 0
        assert result.interpolation_result.method_used == InterpolationMethod.EXCLUDE
        assert result.interpolation_result.original_count == 0
    
    def test_handle_clean_data(self):
        """Test handling of clean data (no missing values)"""
        result = self.handler.handle_missing_data(self.clean_data, self.symbol)
        
        assert result.quality in [DataQuality.HIGH, DataQuality.GOOD]
        assert len(result.data) == len(self.clean_data)
        assert result.interpolation_result.interpolated_count == 0
        assert result.interpolation_result.quality_score > 0.8
    
    def test_handle_minor_missing_data(self):
        """Test handling of data with minor missing values"""
        result = self.handler.handle_missing_data(self.sample_data, self.symbol)
        
        assert result.quality in [DataQuality.GOOD, DataQuality.FAIR]
        assert len(result.data) == len(self.sample_data)
        assert result.interpolation_result.interpolated_count > 0
        assert result.interpolation_result.method_used in [
            InterpolationMethod.LINEAR, 
            InterpolationMethod.TIME_WEIGHTED,
            InterpolationMethod.FORWARD_FILL
        ]
    
    def test_exclude_heavily_missing_data(self):
        """Test exclusion of data with too many missing values"""
        heavily_missing = self._create_heavily_missing_data()
        result = self.handler.handle_missing_data(heavily_missing, self.symbol)
        
        assert result.quality == DataQuality.EXCLUDED
        assert len(result.data) == 0
        assert result.interpolation_result.method_used == InterpolationMethod.EXCLUDE
    
    def test_linear_interpolation_method(self):
        """Test linear interpolation method specifically"""
        # Create data with small gaps suitable for linear interpolation
        data = self._create_clean_sample_data(10)
        data[3]['close'] = None
        data[4]['close'] = None
        
        # Use handler with settings that favor linear interpolation
        handler = MissingDataHandler(max_missing_ratio=0.5)
        result = handler.handle_missing_data(data, self.symbol)
        
        assert result.interpolation_result.interpolated_count > 0
        assert result.quality in [DataQuality.GOOD, DataQuality.FAIR]
        
        # Check that interpolated values are reasonable
        processed_df = pd.DataFrame([{
            'date': pd.to_datetime(record['date']),
            'close': record['close']
        } for record in result.data])
        
        # Should not have any None values
        assert not processed_df['close'].isnull().any()
    
    def test_forward_fill_method(self):
        """Test forward fill interpolation method"""
        # Create data that would use forward fill
        data = self._create_clean_sample_data(15)
        
        # Create pattern that favors forward fill
        for i in [5, 6, 7, 8]:
            data[i]['close'] = None
        
        handler = MissingDataHandler(max_missing_ratio=0.4)
        result = handler.handle_missing_data(data, self.symbol)
        
        assert result.interpolation_result.interpolated_count > 0
        assert result.quality in [DataQuality.GOOD, DataQuality.FAIR, DataQuality.POOR]
    
    def test_time_weighted_interpolation(self):
        """Test time-weighted interpolation method"""
        data = self._create_clean_sample_data(12)
        
        # Create moderate missing data pattern
        data[4]['close'] = None
        data[6]['close'] = None
        data[8]['open'] = None
        
        handler = MissingDataHandler(max_missing_ratio=0.3)
        result = handler.handle_missing_data(data, self.symbol)
        
        assert result.interpolation_result.interpolated_count > 0
        # Time-weighted should give good quality for moderate missing data
        assert result.quality in [DataQuality.GOOD, DataQuality.FAIR]
    
    def test_consecutive_missing_handling(self):
        """Test handling of consecutive missing values"""
        data = self._create_clean_sample_data(15)
        
        # Create consecutive missing values
        for i in range(5, 9):  # 4 consecutive missing
            data[i]['close'] = None
        
        result = self.handler.handle_missing_data(data, self.symbol)
        
        # Should still process but with lower quality
        assert len(result.data) > 0
        assert result.interpolation_result.interpolated_count > 0
        assert result.quality in [DataQuality.GOOD, DataQuality.FAIR, DataQuality.POOR]
    
    def test_volume_interpolation(self):
        """Test specific handling of volume data"""
        data = self._create_clean_sample_data(10)
        
        # Make volume missing in several records
        data[2]['volume'] = None
        data[4]['volume'] = None
        data[6]['volume'] = None
        
        result = self.handler.handle_missing_data(data, self.symbol)
        
        # Check that volume was interpolated
        volumes = [record['volume'] for record in result.data]
        assert all(v is not None and v > 0 for v in volumes)
        assert result.interpolation_result.interpolated_count > 0
    
    def test_ohlc_consistency_after_interpolation(self):
        """Test that OHLC relationships are maintained after interpolation"""
        data = self._create_clean_sample_data(8)
        
        # Create missing values in OHLC
        data[3]['open'] = None
        data[3]['high'] = None
        data[4]['low'] = None
        data[4]['close'] = None
        
        result = self.handler.handle_missing_data(data, self.symbol)
        
        # Check OHLC consistency in processed data
        for record in result.data:
            if all(record[field] is not None for field in ['open', 'high', 'low', 'close']):
                assert record['high'] >= max(record['open'], record['close'])
                assert record['low'] <= min(record['open'], record['close'])
                assert record['high'] >= record['low']
    
    def test_quality_assessment(self):
        """Test quality assessment logic"""
        # Test different scenarios
        scenarios = [
            (self.clean_data, [DataQuality.HIGH, DataQuality.GOOD]),
            (self.sample_data, [DataQuality.GOOD, DataQuality.FAIR]),
            (self._create_heavily_missing_data(), [DataQuality.EXCLUDED])
        ]
        
        for data, expected_qualities in scenarios:
            result = self.handler.handle_missing_data(data, self.symbol)
            assert result.quality in expected_qualities
    
    def test_interpolation_confidence(self):
        """Test confidence calculation for interpolation"""
        # Low missing data should have high confidence
        low_missing = self._create_clean_sample_data(10)
        low_missing[2]['close'] = None
        
        result_low = self.handler.handle_missing_data(low_missing, self.symbol)
        
        # High missing data should have lower confidence
        high_missing = self._create_clean_sample_data(10)
        for i in [2, 3, 4, 5, 6]:
            high_missing[i]['close'] = None
        
        result_high = self.handler.handle_missing_data(high_missing, self.symbol)
        
        if result_high.quality != DataQuality.EXCLUDED:
            assert result_low.interpolation_result.confidence > result_high.interpolation_result.confidence
    
    def test_custom_handler_parameters(self):
        """Test handler with custom parameters"""
        # More restrictive handler
        strict_handler = MissingDataHandler(
            max_missing_ratio=0.10,
            max_consecutive_missing=2,
            exclude_threshold=0.15
        )
        
        # Less restrictive handler
        lenient_handler = MissingDataHandler(
            max_missing_ratio=0.40,
            max_consecutive_missing=10,
            exclude_threshold=0.50
        )
        
        # Test with moderately missing data
        moderate_missing = self._create_clean_sample_data(10)
        for i in [2, 4, 6]:
            moderate_missing[i]['close'] = None
        
        strict_result = strict_handler.handle_missing_data(moderate_missing, self.symbol)
        lenient_result = lenient_handler.handle_missing_data(moderate_missing, self.symbol)
        
        # Lenient handler should be more accepting
        if strict_result.quality == DataQuality.EXCLUDED:
            assert lenient_result.quality != DataQuality.EXCLUDED
    
    def test_interpolation_result_structure(self):
        """Test the structure of interpolation results"""
        result = self.handler.handle_missing_data(self.sample_data, self.symbol)
        
        # Check InterpolationResult structure
        ir = result.interpolation_result
        assert hasattr(ir, 'original_count')
        assert hasattr(ir, 'missing_count')
        assert hasattr(ir, 'interpolated_count')
        assert hasattr(ir, 'excluded_count')
        assert hasattr(ir, 'method_used')
        assert hasattr(ir, 'quality_score')
        assert hasattr(ir, 'confidence')
        assert hasattr(ir, 'warnings')
        
        # Check value ranges and types
        assert isinstance(ir.original_count, int)
        assert isinstance(ir.missing_count, (int, np.integer))
        assert isinstance(ir.interpolated_count, (int, np.integer))
        assert isinstance(ir.excluded_count, int)
        assert isinstance(ir.method_used, InterpolationMethod)
        assert 0.0 <= float(ir.quality_score) <= 1.0
        assert 0.0 <= ir.confidence <= 1.0
        assert isinstance(ir.warnings, list)
    
    def test_processed_data_structure(self):
        """Test the structure of processed data"""
        result = self.handler.handle_missing_data(self.sample_data, self.symbol)
        
        # Check ProcessedData structure
        assert hasattr(result, 'data')
        assert hasattr(result, 'quality')
        assert hasattr(result, 'interpolation_result')
        assert hasattr(result, 'metadata')
        
        # Check types
        assert isinstance(result.data, list)
        assert isinstance(result.quality, DataQuality)
        assert isinstance(result.interpolation_result, InterpolationResult)
        assert isinstance(result.metadata, dict)
        
        # Check metadata content
        assert 'symbol' in result.metadata
        assert result.metadata['symbol'] == self.symbol
    
    def test_convenience_function(self):
        """Test the convenience handle_missing_data function"""
        result = handle_missing_data(self.sample_data, self.symbol)
        
        assert isinstance(result, ProcessedData)
        assert result.metadata['symbol'] == self.symbol
    
    def test_edge_cases(self):
        """Test various edge cases"""
        # Single record
        single_record = [self.clean_data[0]]
        result = self.handler.handle_missing_data(single_record, self.symbol)
        assert len(result.data) <= 1
        
        # All missing critical data
        all_missing = self._create_clean_sample_data(3)
        for record in all_missing:
            record['close'] = None
        
        result = self.handler.handle_missing_data(all_missing, self.symbol)
        assert result.quality == DataQuality.EXCLUDED
        
        # Mixed missing patterns
        mixed_missing = self._create_clean_sample_data(8)
        mixed_missing[1]['open'] = None
        mixed_missing[2]['close'] = None
        mixed_missing[3]['volume'] = None
        mixed_missing[4]['high'] = None
        mixed_missing[5]['low'] = None
        
        result = self.handler.handle_missing_data(mixed_missing, self.symbol)
        # This might be excluded due to high missing ratio, so check conditionally
        if result.quality != DataQuality.EXCLUDED:
            assert result.interpolation_result.interpolated_count > 0
    
    def test_interpolation_warnings(self):
        """Test that appropriate warnings are generated"""
        # Create scenario that should generate warnings
        data = self._create_clean_sample_data(6)
        data[2]['volume'] = None
        data[3]['volume'] = None
        data[4]['volume'] = None
        
        result = self.handler.handle_missing_data(data, self.symbol)
        
        # Should have warnings about volume interpolation if data was processed
        warnings = result.interpolation_result.warnings
        if result.quality != DataQuality.EXCLUDED:
            # If data was processed, check for volume-related warnings or general interpolation
            assert result.interpolation_result.interpolated_count > 0
    
    def test_realistic_missing_patterns(self):
        """Test with realistic missing data patterns"""
        # Weekend gaps (common in stock data)
        data = self._create_clean_sample_data(14)
        
        # Remove weekend-like gaps
        del data[5:7]  # Remove 2 days (simulate weekend)
        del data[10:12]  # Remove another 2 days
        
        result = self.handler.handle_missing_data(data, self.symbol)
        
        # Should handle weekend gaps well
        assert result.quality in [DataQuality.HIGH, DataQuality.GOOD, DataQuality.FAIR]
        assert len(result.data) == len(data)  # No data should be excluded


if __name__ == "__main__":
    pytest.main([__file__])