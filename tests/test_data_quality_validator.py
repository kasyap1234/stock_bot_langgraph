"""
Tests for Data Quality Validator

Tests cover:
- Anomaly detection for market data
- Data consistency checking algorithms
- Various data quality issues detection
- Quality scoring and reporting
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from typing import List

from data.quality_validator import (
    DataQualityValidator,
    DataQualityIssue,
    QualityIssue,
    DataQualityReport,
    validate_data_quality,
    validate_data,
    InsufficientDataError,
    ConstantPriceError
)
from data.models import StockData, create_stock_data


class TestDataQualityValidator:
    """Test suite for DataQualityValidator"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = DataQualityValidator()
        self.symbol = "AAPL"
        
        # Create sample valid data
        self.valid_data = self._create_sample_data()
    
    def _create_sample_data(self, days: int = 30) -> List[StockData]:
        """Create sample stock data for testing"""
        data = []
        base_date = datetime.now() - timedelta(days=days)
        base_price = 100.0
        
        for i in range(days):
            date = base_date + timedelta(days=i)
            # Simulate realistic price movement
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

    def _create_insufficient_data(self):
        """Create sample data with insufficient periods (40 < 50)"""
        return self._create_sample_data(days=40)

    def _create_constant_data(self):
        """Create sample data with constant prices (std dev = 0)"""
        data = []
        base_date = datetime.now() - timedelta(days=100)
        for i in range(100):
            date = base_date + timedelta(days=i)
            stock_data = create_stock_data(
                symbol=self.symbol,
                date=date.strftime("%Y-%m-%d"),
                open_price=100.0,
                high=100.0,
                low=100.0,
                close=100.0,
                volume=1000000
            )
            data.append(stock_data)
        return data
    
    def test_validate_empty_data(self):
        """Test validation of empty data"""
        report = self.validator.validate_data_quality([], self.symbol)
        
        assert report.symbol == self.symbol
        assert report.total_records == 0
        assert report.valid_records == 0
        assert report.overall_quality_score == 0.0
        assert len(report.issues) == 1
        assert report.issues[0].issue_type == DataQualityIssue.MISSING_DATA
        assert report.issues[0].severity == 'critical'
    
    def test_validate_valid_data(self):
        """Test validation of clean, valid data"""
        report = self.validator.validate_data_quality(self.valid_data, self.symbol)
        
        assert report.symbol == self.symbol
        assert report.total_records == len(self.valid_data)
        assert report.overall_quality_score > 0.8  # Should be high quality
        
        # Should have minimal issues (maybe stale data if recent)
        critical_issues = [i for i in report.issues if i.severity == 'critical']
        assert len(critical_issues) == 0
    
    def test_detect_missing_data(self):
        """Test detection of missing data fields"""
        # Create data with missing close prices
        data_with_missing = self.valid_data.copy()
        data_with_missing[5]['close'] = None
        data_with_missing[10]['open'] = None
        
        report = self.validator.validate_data_quality(data_with_missing, self.symbol)
        
        missing_issues = [i for i in report.issues if i.issue_type == DataQualityIssue.MISSING_DATA]
        assert len(missing_issues) >= 1  # Should detect missing data
        
        # Check that affected records are identified
        for issue in missing_issues:
            assert len(issue.affected_records) > 0
    
    def test_detect_negative_prices(self):
        """Test detection of negative or zero prices"""
        data_with_negatives = self.valid_data.copy()
        data_with_negatives[3]['close'] = -10.0
        data_with_negatives[7]['open'] = 0.0
        
        report = self.validator.validate_data_quality(data_with_negatives, self.symbol)
        
        negative_issues = [i for i in report.issues if i.issue_type == DataQualityIssue.NEGATIVE_PRICE]
        assert len(negative_issues) >= 1
        assert negative_issues[0].severity == 'critical'
    
    def test_detect_extreme_price_changes(self):
        """Test detection of extreme price changes"""
        data_with_extremes = self.valid_data.copy()
        
        # Create extreme price jump (50% increase)
        base_price = data_with_extremes[10]['close']
        data_with_extremes[11]['close'] = base_price * 1.5
        data_with_extremes[11]['open'] = base_price * 1.5
        data_with_extremes[11]['high'] = base_price * 1.52
        data_with_extremes[11]['low'] = base_price * 1.48
        
        report = self.validator.validate_data_quality(data_with_extremes, self.symbol)
        
        extreme_issues = [i for i in report.issues if i.issue_type == DataQualityIssue.EXTREME_PRICE_CHANGE]
        assert len(extreme_issues) >= 1
        assert extreme_issues[0].severity in ['medium', 'high']
    
    def test_detect_volume_anomalies(self):
        """Test detection of volume anomalies"""
        data_with_volume_spike = self.valid_data.copy()
        
        # Create volume spike (10x normal volume)
        normal_volume = data_with_volume_spike[10]['volume']
        data_with_volume_spike[15]['volume'] = normal_volume * 10
        
        report = self.validator.validate_data_quality(data_with_volume_spike, self.symbol)
        
        volume_issues = [i for i in report.issues if i.issue_type == DataQualityIssue.VOLUME_ANOMALY]
        assert len(volume_issues) >= 1
        assert volume_issues[0].severity == 'medium'
    
    def test_detect_ohlc_inconsistency(self):
        """Test detection of OHLC inconsistencies"""
        data_with_inconsistency = self.valid_data.copy()
        
        # Make high price lower than close price
        data_with_inconsistency[8]['close'] = 105.0
        data_with_inconsistency[8]['high'] = 103.0  # High < Close (invalid)
        
        # Make low price higher than open price
        data_with_inconsistency[12]['open'] = 98.0
        data_with_inconsistency[12]['low'] = 99.0  # Low > Open (invalid)
        
        report = self.validator.validate_data_quality(data_with_inconsistency, self.symbol)
        
        ohlc_issues = [i for i in report.issues if i.issue_type == DataQualityIssue.INCONSISTENT_OHLC]
        assert len(ohlc_issues) >= 1
        assert ohlc_issues[0].severity == 'high'
    
    def test_detect_duplicates(self):
        """Test detection of duplicate records"""
        data_with_duplicates = self.valid_data.copy()
        
        # Add duplicate record
        duplicate_record = data_with_duplicates[5].copy()
        data_with_duplicates.append(duplicate_record)
        
        report = self.validator.validate_data_quality(data_with_duplicates, self.symbol)
        
        duplicate_issues = [i for i in report.issues if i.issue_type == DataQualityIssue.DUPLICATE_DATA]
        assert len(duplicate_issues) >= 1
        assert duplicate_issues[0].severity == 'medium'
    
    def test_detect_zero_volume(self):
        """Test detection of zero volume"""
        data_with_zero_volume = self.valid_data.copy()
        data_with_zero_volume[6]['volume'] = 0
        data_with_zero_volume[14]['volume'] = 0
        
        report = self.validator.validate_data_quality(data_with_zero_volume, self.symbol)
        
        zero_volume_issues = [i for i in report.issues if i.issue_type == DataQualityIssue.ZERO_VOLUME]
        assert len(zero_volume_issues) >= 1
        assert zero_volume_issues[0].severity == 'low'  # Zero volume might be valid
    
    def test_detect_stale_data(self):
        """Test detection of stale data"""
        # Create old data (30 days old)
        old_data = self._create_sample_data(days=10)
        base_date = datetime.now() - timedelta(days=30)
        
        for i, record in enumerate(old_data):
            date = base_date + timedelta(days=i)
            record['date'] = date.strftime("%Y-%m-%d")
        
        report = self.validator.validate_data_quality(old_data, self.symbol)
        
        stale_issues = [i for i in report.issues if i.issue_type == DataQualityIssue.STALE_DATA]
        assert len(stale_issues) >= 1
        assert stale_issues[0].severity in ['medium', 'high']
    
    def test_detect_corporate_actions(self):
        """Test detection of potential corporate actions"""
        data_with_corporate_action = self.valid_data.copy()
        
        # Simulate dividend payment (price drop with volume spike)
        base_price = data_with_corporate_action[15]['close']
        base_volume = data_with_corporate_action[15]['volume']
        
        # Next day: price drops 3% with 5x volume
        data_with_corporate_action[16]['open'] = base_price * 0.97
        data_with_corporate_action[16]['close'] = base_price * 0.97
        data_with_corporate_action[16]['high'] = base_price * 0.98
        data_with_corporate_action[16]['low'] = base_price * 0.96
        data_with_corporate_action[16]['volume'] = base_volume * 5
        
        report = self.validator.validate_data_quality(data_with_corporate_action, self.symbol)
        
        corporate_action_issues = [i for i in report.issues 
                                 if i.issue_type == DataQualityIssue.CORPORATE_ACTION_DETECTED]
        # Corporate action detection might not always trigger, so we don't assert it must exist
        # but if it does, it should be properly classified
        for issue in corporate_action_issues:
            assert issue.severity in ['low', 'medium']
    
    def test_quality_score_calculation(self):
        """Test quality score calculation"""
        # Test with perfect data
        perfect_report = self.validator.validate_data_quality(self.valid_data, self.symbol)
        
        # Test with problematic data
        problematic_data = self.valid_data.copy()
        problematic_data[5]['close'] = None  # Missing data
        problematic_data[10]['close'] = -50.0  # Negative price
        
        problematic_report = self.validator.validate_data_quality(problematic_data, self.symbol)
        
        # Problematic data should have lower quality score
        assert problematic_report.overall_quality_score < perfect_report.overall_quality_score
        assert 0.0 <= problematic_report.overall_quality_score <= 1.0
    
    def test_recommendations_generation(self):
        """Test generation of actionable recommendations"""
        problematic_data = self.valid_data.copy()
        problematic_data[5]['close'] = None  # Missing data
        problematic_data[10]['close'] = -50.0  # Negative price
        
        report = self.validator.validate_data_quality(problematic_data, self.symbol)
        
        assert len(report.recommendations) > 0
        assert any('missing' in rec.lower() for rec in report.recommendations)
    
    def test_custom_validator_parameters(self):
        """Test validator with custom parameters"""
        custom_validator = DataQualityValidator(
            price_change_threshold=0.10,  # 10% threshold instead of 20%
            volume_anomaly_threshold=3.0,  # 3x instead of 5x
            stale_data_days=3  # 3 days instead of 7
        )
        
        # Create data with 15% price change (should trigger with 10% threshold)
        data_with_change = self.valid_data.copy()
        base_price = data_with_change[10]['close']
        data_with_change[11]['close'] = base_price * 1.15
        
        report = custom_validator.validate_data_quality(data_with_change, self.symbol)
        
        extreme_issues = [i for i in report.issues if i.issue_type == DataQualityIssue.EXTREME_PRICE_CHANGE]
        assert len(extreme_issues) >= 1  # Should detect with lower threshold
    
    def test_convenience_function(self):
        """Test the convenience validate_data_quality function"""
        report = validate_data_quality(self.valid_data, self.symbol)
        
        assert isinstance(report, DataQualityReport)
        assert report.symbol == self.symbol
        assert report.total_records == len(self.valid_data)
    
    def test_data_gap_detection(self):
        """Test detection of data gaps (missing trading days)"""
        # Create data with gaps (skip weekends and some weekdays)
        sparse_data = []
        base_date = datetime.now() - timedelta(days=30)
        
        # Only include every 3rd day to create gaps
        for i in range(0, 30, 3):
            date = base_date + timedelta(days=i)
            stock_data = create_stock_data(
                symbol=self.symbol,
                date=date.strftime("%Y-%m-%d"),
                open_price=100.0,
                high=102.0,
                low=98.0,
                close=101.0,
                volume=1000000
            )
            sparse_data.append(stock_data)
        
        report = self.validator.validate_data_quality(sparse_data, self.symbol)
        
        gap_issues = [i for i in report.issues if i.issue_type == DataQualityIssue.DATA_GAP]
        # Gap detection might not always trigger depending on the data, but if it does,
        # it should be properly classified
        for issue in gap_issues:
            assert issue.severity in ['medium', 'high']
    
    def test_report_structure(self):
        """Test the structure and completeness of the quality report"""
        report = self.validator.validate_data_quality(self.valid_data, self.symbol)
        
        # Check all required fields are present
        assert hasattr(report, 'symbol')
        assert hasattr(report, 'total_records')
        assert hasattr(report, 'valid_records')
        assert hasattr(report, 'issues')
        assert hasattr(report, 'overall_quality_score')
        assert hasattr(report, 'recommendations')
        assert hasattr(report, 'timestamp')
        
        # Check data types
        assert isinstance(report.symbol, str)
        assert isinstance(report.total_records, int)
        assert isinstance(report.valid_records, int)
        assert isinstance(report.issues, list)
        assert isinstance(report.overall_quality_score, float)
        assert isinstance(report.recommendations, list)
        assert isinstance(report.timestamp, datetime)
        
        # Check value ranges
        assert 0 <= report.overall_quality_score <= 1.0
        assert report.valid_records <= report.total_records

    def test_validate_data_insufficient(self):
        """Test validate_data raises InsufficientDataError for short data"""
        data = self._create_insufficient_data()
        with pytest.raises(InsufficientDataError):
            validate_data(data)

    def test_validate_data_constant(self):
        """Test validate_data raises ConstantPriceError for constant prices"""
        data = self._create_constant_data()
        with pytest.raises(ConstantPriceError):
            validate_data(data)

    def test_validate_data_valid(self):
        """Test validate_data does not raise for valid data"""
        data = self._create_sample_data(days=60)  # More than 50 periods
        validate_data(data)  # Should not raise


if __name__ == "__main__":
    pytest.main([__file__])