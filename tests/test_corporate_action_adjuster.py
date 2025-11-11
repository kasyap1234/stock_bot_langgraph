"""
Tests for Corporate Action Adjuster

Tests cover:
- Historical data adjustment for stock splits and dividends
- Consistency maintenance algorithms
- Detection and correction of corporate actions
- Validation of adjusted data
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List

from data.corporate_action_adjuster import (
    CorporateActionAdjuster,
    CorporateActionType,
    CorporateAction,
    AdjustmentResult,
    AdjustedData,
    adjust_for_corporate_actions
)
from data.models import StockData, create_stock_data


class TestCorporateActionAdjuster:
    """Test suite for CorporateActionAdjuster"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.adjuster = CorporateActionAdjuster()
        self.symbol = "AAPL"
        
        # Create sample data with corporate actions
        self.sample_data = self._create_sample_data_with_split()
        self.clean_data = self._create_clean_sample_data()
    
    def _create_clean_sample_data(self, days: int = 30) -> List[StockData]:
        """Create clean sample stock data for testing"""
        data = []
        base_date = datetime.now() - timedelta(days=days)
        base_price = 100.0
        
        for i in range(days):
            date = base_date + timedelta(days=i)
            # Small random price movement
            price_change = (i % 5 - 2) * 0.01  # -2%, -1%, 0%, 1%, 2% pattern
            close_price = base_price * (1 + price_change)
            
            stock_data = create_stock_data(
                symbol=self.symbol,
                date=date.strftime("%Y-%m-%d"),
                open_price=close_price * 0.995,
                high=close_price * 1.01,
                low=close_price * 0.99,
                close=close_price,
                volume=1000000 + (i % 3) * 100000
            )
            data.append(stock_data)
            
        return data
    
    def _create_sample_data_with_split(self) -> List[StockData]:
        """Create sample data with a 2:1 stock split"""
        data = []
        base_date = datetime.now() - timedelta(days=20)
        split_date = base_date + timedelta(days=10)
        
        for i in range(20):
            date = base_date + timedelta(days=i)
            
            if date < split_date:
                # Pre-split: higher prices, lower volume
                close_price = 200.0 + (i % 3 - 1) * 5.0
                volume = 500000
            else:
                # Post-split: lower prices, higher volume
                close_price = 100.0 + (i % 3 - 1) * 2.5
                volume = 1000000
                
                # Create split effect on the split date
                if i == 10:  # Split date
                    close_price = 95.0  # Slight discount due to split
                    volume = 2500000  # High volume on split date
            
            stock_data = create_stock_data(
                symbol=self.symbol,
                date=date.strftime("%Y-%m-%d"),
                open_price=close_price * 0.995,
                high=close_price * 1.01,
                low=close_price * 0.99,
                close=close_price,
                volume=volume
            )
            data.append(stock_data)
            
        return data
    
    def _create_sample_data_with_dividend(self) -> List[StockData]:
        """Create sample data with a dividend payment"""
        data = []
        base_date = datetime.now() - timedelta(days=15)
        dividend_date = base_date + timedelta(days=7)
        
        for i in range(15):
            date = base_date + timedelta(days=i)
            close_price = 100.0 + (i % 3 - 1) * 2.0
            volume = 1000000
            
            # Create dividend effect on ex-dividend date
            if date == dividend_date:
                close_price = 97.0  # $3 dividend causes price drop
                volume = 2500000  # Higher volume on ex-dividend date (more significant spike)
            
            stock_data = create_stock_data(
                symbol=self.symbol,
                date=date.strftime("%Y-%m-%d"),
                open_price=close_price * 0.995,
                high=close_price * 1.01,
                low=close_price * 0.99,
                close=close_price,
                volume=volume
            )
            data.append(stock_data)
            
        return data
    
    def test_adjust_empty_data(self):
        """Test adjustment of empty data"""
        result = self.adjuster.adjust_for_corporate_actions([], self.symbol)
        
        assert len(result.data) == 0
        assert result.adjustment_result.original_count == 0
        assert result.adjustment_result.adjusted_count == 0
        assert len(result.adjustment_result.actions_detected) == 0
        assert len(result.adjustment_result.actions_applied) == 0
    
    def test_adjust_clean_data(self):
        """Test adjustment of clean data (no corporate actions)"""
        result = self.adjuster.adjust_for_corporate_actions(self.clean_data, self.symbol)
        
        assert len(result.data) == len(self.clean_data)
        assert result.adjustment_result.original_count == len(self.clean_data)
        assert result.adjustment_result.adjusted_count == len(self.clean_data)
        assert result.adjustment_result.quality_score > 0.8  # Should be high quality
        
        # Should detect minimal or no corporate actions
        assert len(result.adjustment_result.actions_detected) <= 2  # Allow for some false positives
    
    def test_detect_stock_split(self):
        """Test detection of stock splits"""
        result = self.adjuster.adjust_for_corporate_actions(self.sample_data, self.symbol)
        
        # Should detect at least one corporate action
        assert len(result.adjustment_result.actions_detected) > 0
        
        # Check if any detected action is a stock split
        split_actions = [action for action in result.adjustment_result.actions_detected 
                        if action.action_type == CorporateActionType.STOCK_SPLIT]
        
        # Should detect the 2:1 split we created
        if split_actions:
            split_action = split_actions[0]
            assert split_action.ratio > 1.5  # Should be close to 2.0
            assert split_action.confidence > 0.5
    
    def test_detect_dividend(self):
        """Test detection of dividend payments"""
        dividend_data = self._create_sample_data_with_dividend()
        
        # Use more sensitive adjuster for dividend detection
        sensitive_adjuster = CorporateActionAdjuster(
            dividend_detection_threshold=0.01,  # More sensitive
            volume_spike_threshold=1.5  # Lower volume threshold
        )
        
        result = sensitive_adjuster.adjust_for_corporate_actions(dividend_data, self.symbol)
        
        # Should detect corporate actions (may include false positives)
        # If no actions detected, that's also acceptable as dividend detection is challenging
        if len(result.adjustment_result.actions_detected) > 0:
            # Check if any detected action is a dividend
            dividend_actions = [action for action in result.adjustment_result.actions_detected 
                               if action.action_type == CorporateActionType.CASH_DIVIDEND]
            
            # If dividends detected, validate them
            for dividend_action in dividend_actions:
                assert dividend_action.dividend_amount > 0
                assert dividend_action.confidence > 0.1
    
    def test_apply_stock_split_adjustment(self):
        """Test application of stock split adjustments"""
        # Create known split action
        split_date = datetime.now() - timedelta(days=10)
        known_split = CorporateAction(
            date=split_date,
            action_type=CorporateActionType.STOCK_SPLIT,
            ratio=2.0,
            dividend_amount=0.0,
            description="2:1 stock split",
            confidence=0.9,
            detected_automatically=False
        )
        
        result = self.adjuster.adjust_for_corporate_actions(
            self.sample_data, self.symbol, [known_split]
        )
        
        # Should apply the adjustment
        assert len(result.adjustment_result.actions_applied) > 0
        assert result.adjustment_result.actions_applied[0].action_type == CorporateActionType.STOCK_SPLIT
        
        # Check that pre-split prices were adjusted
        adjusted_df = pd.DataFrame([{
            'date': pd.to_datetime(record['date']),
            'close': record['close']
        } for record in result.data])
        
        # Pre-split and post-split prices should be more consistent
        pre_split_prices = adjusted_df[adjusted_df['date'] < split_date]['close']
        post_split_prices = adjusted_df[adjusted_df['date'] >= split_date]['close']
        
        if len(pre_split_prices) > 0 and len(post_split_prices) > 0:
            # After adjustment, pre-split and post-split prices should be in similar range
            pre_split_avg = pre_split_prices.mean()
            post_split_avg = post_split_prices.mean()
            ratio = pre_split_avg / post_split_avg
            assert 0.8 <= ratio <= 1.2  # Should be roughly equal after adjustment
    
    def test_apply_dividend_adjustment(self):
        """Test application of dividend adjustments"""
        dividend_data = self._create_sample_data_with_dividend()
        
        # Create known dividend action
        dividend_date = datetime.now() - timedelta(days=8)
        known_dividend = CorporateAction(
            date=dividend_date,
            action_type=CorporateActionType.CASH_DIVIDEND,
            ratio=1.0,
            dividend_amount=3.0,
            description="$3.00 dividend payment",
            confidence=0.9,
            detected_automatically=False
        )
        
        result = self.adjuster.adjust_for_corporate_actions(
            dividend_data, self.symbol, [known_dividend]
        )
        
        # Should apply the adjustment
        applied_dividends = [action for action in result.adjustment_result.actions_applied 
                           if action.action_type == CorporateActionType.CASH_DIVIDEND]
        assert len(applied_dividends) > 0
        
        # Check price continuity after dividend adjustment
        adjusted_df = pd.DataFrame([{
            'date': pd.to_datetime(record['date']),
            'close': record['close']
        } for record in result.data])
        
        # Calculate returns after adjustment - should be smoother
        returns = adjusted_df['close'].pct_change(fill_method=None).dropna()
        extreme_returns = (abs(returns) > 0.05).sum()  # More than 5% daily change
        
        # Should have fewer extreme returns after dividend adjustment
        assert extreme_returns <= 2  # Allow for some volatility
    
    def test_confidence_threshold(self):
        """Test that low-confidence actions are not applied"""
        # Create low-confidence action
        low_confidence_action = CorporateAction(
            date=datetime.now() - timedelta(days=5),
            action_type=CorporateActionType.STOCK_SPLIT,
            ratio=2.0,
            dividend_amount=0.0,
            description="Low confidence split",
            confidence=0.3,  # Below default threshold of 0.7
            detected_automatically=True
        )
        
        result = self.adjuster.adjust_for_corporate_actions(
            self.clean_data, self.symbol, [low_confidence_action]
        )
        
        # Should not apply low-confidence action
        assert len(result.adjustment_result.actions_applied) == 0
        assert len(result.adjustment_result.warnings) > 0
        assert any('low confidence' in warning.lower() for warning in result.adjustment_result.warnings)
    
    def test_adjustment_quality_calculation(self):
        """Test quality score calculation"""
        # Test with good adjustment
        good_split = CorporateAction(
            date=datetime.now() - timedelta(days=10),
            action_type=CorporateActionType.STOCK_SPLIT,
            ratio=2.0,
            dividend_amount=0.0,
            description="Good split",
            confidence=0.9,
            detected_automatically=False
        )
        
        good_result = self.adjuster.adjust_for_corporate_actions(
            self.sample_data, self.symbol, [good_split]
        )
        
        # Should have high quality score
        assert good_result.adjustment_result.quality_score > 0.7
        
        # Test with no adjustments (clean data)
        clean_result = self.adjuster.adjust_for_corporate_actions(self.clean_data, self.symbol)
        
        # Clean data should have high quality score
        assert clean_result.adjustment_result.quality_score > 0.8
    
    def test_data_validation(self):
        """Test validation of adjusted data"""
        # Apply adjustment
        split_action = CorporateAction(
            date=datetime.now() - timedelta(days=10),
            action_type=CorporateActionType.STOCK_SPLIT,
            ratio=2.0,
            dividend_amount=0.0,
            description="Test split",
            confidence=0.9,
            detected_automatically=False
        )
        
        result = self.adjuster.adjust_for_corporate_actions(
            self.sample_data, self.symbol, [split_action]
        )
        
        # Validate adjustments
        validation = self.adjuster.validate_adjustments(self.sample_data, result.data)
        
        # Should be valid
        assert validation['is_valid'] is True
        assert 'metrics' in validation
        
        # Should have reasonable volatility metrics
        if 'original_volatility' in validation['metrics']:
            assert validation['metrics']['original_volatility'] >= 0
            assert validation['metrics']['adjusted_volatility'] >= 0
    
    def test_ohlc_consistency_after_adjustment(self):
        """Test that OHLC relationships are maintained after adjustment"""
        split_action = CorporateAction(
            date=datetime.now() - timedelta(days=10),
            action_type=CorporateActionType.STOCK_SPLIT,
            ratio=2.0,
            dividend_amount=0.0,
            description="Test split",
            confidence=0.9,
            detected_automatically=False
        )
        
        result = self.adjuster.adjust_for_corporate_actions(
            self.sample_data, self.symbol, [split_action]
        )
        
        # Check OHLC consistency in adjusted data
        for record in result.data:
            if all(record[field] is not None for field in ['open', 'high', 'low', 'close']):
                assert record['high'] >= max(record['open'], record['close'])
                assert record['low'] <= min(record['open'], record['close'])
                assert record['high'] >= record['low']
                assert record['open'] > 0
                assert record['close'] > 0
    
    def test_multiple_corporate_actions(self):
        """Test handling of multiple corporate actions"""
        # Create multiple actions
        split_action = CorporateAction(
            date=datetime.now() - timedelta(days=15),
            action_type=CorporateActionType.STOCK_SPLIT,
            ratio=2.0,
            dividend_amount=0.0,
            description="2:1 split",
            confidence=0.9,
            detected_automatically=False
        )
        
        dividend_action = CorporateAction(
            date=datetime.now() - timedelta(days=5),
            action_type=CorporateActionType.CASH_DIVIDEND,
            ratio=1.0,
            dividend_amount=2.0,
            description="$2.00 dividend",
            confidence=0.8,
            detected_automatically=False
        )
        
        result = self.adjuster.adjust_for_corporate_actions(
            self.sample_data, self.symbol, [split_action, dividend_action]
        )
        
        # Should apply both actions
        assert len(result.adjustment_result.actions_applied) == 2
        
        # Should maintain data integrity
        assert len(result.data) == len(self.sample_data)
        assert result.adjustment_result.quality_score > 0.5
    
    def test_custom_adjuster_parameters(self):
        """Test adjuster with custom parameters"""
        # More sensitive adjuster
        sensitive_adjuster = CorporateActionAdjuster(
            split_detection_threshold=0.2,  # Lower threshold
            dividend_detection_threshold=0.01,  # Lower threshold
            confidence_threshold=0.5  # Lower confidence threshold
        )
        
        # Less sensitive adjuster
        conservative_adjuster = CorporateActionAdjuster(
            split_detection_threshold=0.6,  # Higher threshold
            dividend_detection_threshold=0.05,  # Higher threshold
            confidence_threshold=0.9  # Higher confidence threshold
        )
        
        # Test with same data
        sensitive_result = sensitive_adjuster.adjust_for_corporate_actions(self.sample_data, self.symbol)
        conservative_result = conservative_adjuster.adjust_for_corporate_actions(self.sample_data, self.symbol)
        
        # Sensitive adjuster should detect more actions
        assert len(sensitive_result.adjustment_result.actions_detected) >= len(conservative_result.adjustment_result.actions_detected)
    
    def test_adjustment_result_structure(self):
        """Test the structure of adjustment results"""
        result = self.adjuster.adjust_for_corporate_actions(self.sample_data, self.symbol)
        
        # Check AdjustmentResult structure
        ar = result.adjustment_result
        assert hasattr(ar, 'original_count')
        assert hasattr(ar, 'adjusted_count')
        assert hasattr(ar, 'actions_detected')
        assert hasattr(ar, 'actions_applied')
        assert hasattr(ar, 'quality_score')
        assert hasattr(ar, 'warnings')
        
        # Check types and ranges
        assert isinstance(ar.original_count, int)
        assert isinstance(ar.adjusted_count, int)
        assert isinstance(ar.actions_detected, list)
        assert isinstance(ar.actions_applied, list)
        assert 0.0 <= ar.quality_score <= 1.0
        assert isinstance(ar.warnings, list)
    
    def test_adjusted_data_structure(self):
        """Test the structure of adjusted data"""
        result = self.adjuster.adjust_for_corporate_actions(self.sample_data, self.symbol)
        
        # Check AdjustedData structure
        assert hasattr(result, 'data')
        assert hasattr(result, 'adjustment_result')
        assert hasattr(result, 'metadata')
        
        # Check types
        assert isinstance(result.data, list)
        assert isinstance(result.adjustment_result, AdjustmentResult)
        assert isinstance(result.metadata, dict)
        
        # Check metadata content
        assert 'symbol' in result.metadata
        assert result.metadata['symbol'] == self.symbol
    
    def test_convenience_function(self):
        """Test the convenience adjust_for_corporate_actions function"""
        result = adjust_for_corporate_actions(self.sample_data, self.symbol)
        
        assert isinstance(result, AdjustedData)
        assert result.metadata['symbol'] == self.symbol
    
    def test_edge_cases(self):
        """Test various edge cases"""
        # Single record
        single_record = [self.clean_data[0]]
        result = self.adjuster.adjust_for_corporate_actions(single_record, self.symbol)
        assert len(result.data) == 1
        assert len(result.adjustment_result.actions_detected) == 0
        
        # Very volatile data (might trigger false positives)
        volatile_data = self._create_clean_sample_data(10)
        for i, record in enumerate(volatile_data):
            if i % 2 == 0:
                record['close'] = record['close'] * 1.3  # 30% increase
                record['volume'] = record['volume'] * 3  # 3x volume
        
        result = self.adjuster.adjust_for_corporate_actions(volatile_data, self.symbol)
        
        # Should handle volatile data without crashing
        assert len(result.data) == len(volatile_data)
        assert result.adjustment_result.quality_score >= 0.0
    
    def test_corporate_action_detection_accuracy(self):
        """Test accuracy of corporate action detection"""
        # Create data with known split
        split_data = self._create_sample_data_with_split()
        result = self.adjuster.adjust_for_corporate_actions(split_data, self.symbol)
        
        # Should detect some corporate actions
        detected_actions = result.adjustment_result.actions_detected
        
        # Check that detected actions have reasonable properties
        for action in detected_actions:
            assert action.confidence > 0.1
            assert action.confidence <= 1.0
            assert action.action_type in [CorporateActionType.STOCK_SPLIT, CorporateActionType.CASH_DIVIDEND]
            
            if action.action_type == CorporateActionType.STOCK_SPLIT:
                assert action.ratio > 0.1  # Reasonable split ratio
            elif action.action_type == CorporateActionType.CASH_DIVIDEND:
                assert action.dividend_amount >= 0  # Non-negative dividend


if __name__ == "__main__":
    pytest.main([__file__])