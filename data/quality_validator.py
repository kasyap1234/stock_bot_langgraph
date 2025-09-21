"""
Data Quality Validation System for Stock Bot

This module implements comprehensive data quality validation including:
- Anomaly detection for market data
- Data consistency checking algorithms
- Missing data detection and handling
- Corporate action adjustment validation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import statistics
import re

from .models import StockData, HistoricalData, validate_stock_data

logger = logging.getLogger(__name__)

class InsufficientDataError(Exception):
    """Raised when there is insufficient historical data for analysis."""
    pass

class ConstantPriceError(Exception):
    """Raised when prices are constant (zero standard deviation)."""
    pass


class DataQualityIssue(Enum):
    """Types of data quality issues that can be detected"""
    MISSING_DATA = "missing_data"
    PRICE_ANOMALY = "price_anomaly"
    VOLUME_ANOMALY = "volume_anomaly"
    INCONSISTENT_OHLC = "inconsistent_ohlc"
    DUPLICATE_DATA = "duplicate_data"
    INVALID_DATE = "invalid_date"
    NEGATIVE_PRICE = "negative_price"
    ZERO_VOLUME = "zero_volume"
    EXTREME_PRICE_CHANGE = "extreme_price_change"
    CORPORATE_ACTION_DETECTED = "corporate_action_detected"
    DATA_GAP = "data_gap"
    STALE_DATA = "stale_data"


@dataclass
class QualityIssue:
    """Represents a data quality issue"""
    issue_type: DataQualityIssue
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    affected_records: List[int]  # indices of affected records
    suggested_action: str
    confidence: float  # 0.0 to 1.0


@dataclass
class DataQualityReport:
    """Comprehensive data quality assessment report"""
    symbol: str
    total_records: int
    valid_records: int
    issues: List[QualityIssue]
    overall_quality_score: float  # 0.0 to 1.0
    recommendations: List[str]
    timestamp: datetime


class DataQualityValidator:
    """
    Comprehensive data quality validator for stock market data
    """
    
    def __init__(self, 
                 price_change_threshold: float = 0.20,  # 20% daily change threshold
                 volume_anomaly_threshold: float = 5.0,  # 5x average volume
                 missing_data_threshold: float = 0.05,   # 5% missing data tolerance
                 stale_data_days: int = 7):              # Data older than 7 days is stale
        
        self.price_change_threshold = price_change_threshold
        self.volume_anomaly_threshold = volume_anomaly_threshold
        self.missing_data_threshold = missing_data_threshold
        self.stale_data_days = stale_data_days
        
    def validate_data_quality(self, data: HistoricalData, symbol: str) -> DataQualityReport:
        """
        Perform comprehensive data quality validation
        
        Args:
            data: Historical stock data to validate
            symbol: Stock symbol being validated
            
        Returns:
            DataQualityReport with detailed quality assessment
        """
        if not data:
            return DataQualityReport(
                symbol=symbol,
                total_records=0,
                valid_records=0,
                issues=[QualityIssue(
                    issue_type=DataQualityIssue.MISSING_DATA,
                    severity='critical',
                    description='No data available for validation',
                    affected_records=[],
                    suggested_action='Check data source and fetch process',
                    confidence=1.0
                )],
                overall_quality_score=0.0,
                recommendations=['Verify data source connectivity', 'Check symbol validity'],
                timestamp=datetime.now()
            )
        
        issues = []
        
        # Convert to DataFrame for easier analysis
        df = self._convert_to_dataframe(data)
        
        # Run all validation checks
        issues.extend(self._check_basic_data_integrity(df))
        issues.extend(self._check_price_anomalies(df))
        issues.extend(self._check_volume_anomalies(df))
        issues.extend(self._check_ohlc_consistency(df))
        issues.extend(self._check_duplicates(df))
        issues.extend(self._check_data_gaps(df))
        issues.extend(self._check_stale_data(df))
        issues.extend(self._detect_corporate_actions(df))
        
        # Calculate quality metrics
        valid_records = len(df) - sum(len(issue.affected_records) for issue in issues 
                                    if issue.severity in ['high', 'critical'])
        quality_score = self._calculate_quality_score(df, issues)
        recommendations = self._generate_recommendations(issues)
        
        return DataQualityReport(
            symbol=symbol,
            total_records=len(data),
            valid_records=max(0, valid_records),
            issues=issues,
            overall_quality_score=quality_score,
            recommendations=recommendations,
            timestamp=datetime.now()
        )
    
    def _convert_to_dataframe(self, data: HistoricalData) -> pd.DataFrame:
        """Convert HistoricalData to pandas DataFrame for analysis"""
        df_data = []
        for i, record in enumerate(data):
            df_data.append({
                'index': i,
                'symbol': record.get('symbol'),
                'date': pd.to_datetime(record.get('date')),
                'open': record.get('open'),
                'high': record.get('high'),
                'low': record.get('low'),
                'close': record.get('close'),
                'volume': record.get('volume')
            })
        
        df = pd.DataFrame(df_data)
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        return df
    
    def _check_basic_data_integrity(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Check for basic data integrity issues"""
        issues = []
        
        # Check for missing required fields
        required_fields = ['open', 'high', 'low', 'close', 'volume']
        for field in required_fields:
            missing_count = df[field].isnull().sum()
            if missing_count > 0:
                missing_indices = df[df[field].isnull()]['index'].tolist()
                severity = 'critical' if missing_count > len(df) * 0.1 else 'high'
                
                issues.append(QualityIssue(
                    issue_type=DataQualityIssue.MISSING_DATA,
                    severity=severity,
                    description=f'Missing {field} values in {missing_count} records',
                    affected_records=missing_indices,
                    suggested_action=f'Interpolate or exclude records with missing {field}',
                    confidence=1.0
                ))
        
        # Check for negative prices
        price_fields = ['open', 'high', 'low', 'close']
        for field in price_fields:
            negative_mask = df[field] <= 0
            if negative_mask.any():
                negative_indices = df[negative_mask]['index'].tolist()
                issues.append(QualityIssue(
                    issue_type=DataQualityIssue.NEGATIVE_PRICE,
                    severity='critical',
                    description=f'Negative or zero {field} prices detected',
                    affected_records=negative_indices,
                    suggested_action=f'Remove or correct negative {field} values',
                    confidence=1.0
                ))
        
        # Check for zero volume (might be valid for some stocks)
        zero_volume_mask = df['volume'] == 0
        if zero_volume_mask.any():
            zero_volume_indices = df[zero_volume_mask]['index'].tolist()
            issues.append(QualityIssue(
                issue_type=DataQualityIssue.ZERO_VOLUME,
                severity='low',
                description=f'Zero volume detected in {zero_volume_mask.sum()} records',
                affected_records=zero_volume_indices,
                suggested_action='Verify if zero volume is expected for these dates',
                confidence=0.7
            ))
        
        return issues
    
    def _check_price_anomalies(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Detect price anomalies using statistical methods"""
        issues = []
        
        if len(df) < 2:
            return issues
        
        # Calculate daily returns
        df['daily_return'] = df['close'].pct_change(fill_method=None)
        
        # Detect extreme price changes
        extreme_changes = abs(df['daily_return']) > self.price_change_threshold
        if extreme_changes.any():
            extreme_indices = df[extreme_changes]['index'].tolist()
            max_change = abs(df['daily_return']).max()
            
            issues.append(QualityIssue(
                issue_type=DataQualityIssue.EXTREME_PRICE_CHANGE,
                severity='high' if max_change > 0.5 else 'medium',
                description=f'Extreme price changes detected (max: {max_change:.2%})',
                affected_records=extreme_indices,
                suggested_action='Verify if extreme changes are due to corporate actions',
                confidence=0.8
            ))
        
        # Statistical outlier detection using IQR method
        if len(df) >= 10:  # Need sufficient data for statistical analysis
            Q1 = df['close'].quantile(0.25)
            Q3 = df['close'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (df['close'] < lower_bound) | (df['close'] > upper_bound)
            if outliers.any():
                outlier_indices = df[outliers]['index'].tolist()
                issues.append(QualityIssue(
                    issue_type=DataQualityIssue.PRICE_ANOMALY,
                    severity='medium',
                    description=f'Statistical price outliers detected in {outliers.sum()} records',
                    affected_records=outlier_indices,
                    suggested_action='Review outlier prices for data entry errors',
                    confidence=0.7
                ))
        
        return issues
    
    def _check_volume_anomalies(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Detect volume anomalies"""
        issues = []
        
        if len(df) < 5:  # Need sufficient data for volume analysis
            return issues
        
        # Calculate rolling average volume
        df['volume_ma'] = df['volume'].rolling(window=20, min_periods=5).mean()
        
        # Detect volume spikes
        volume_ratio = df['volume'] / df['volume_ma']
        volume_spikes = volume_ratio > self.volume_anomaly_threshold
        
        if volume_spikes.any():
            spike_indices = df[volume_spikes]['index'].tolist()
            max_ratio = volume_ratio.max()
            
            issues.append(QualityIssue(
                issue_type=DataQualityIssue.VOLUME_ANOMALY,
                severity='medium',
                description=f'Volume spikes detected (max: {max_ratio:.1f}x average)',
                affected_records=spike_indices,
                suggested_action='Verify volume spikes with news or corporate actions',
                confidence=0.6
            ))
        
        return issues
    
    def _check_ohlc_consistency(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Check OHLC price consistency"""
        issues = []
        
        # Check if high >= max(open, close) and low <= min(open, close)
        inconsistent_high = df['high'] < df[['open', 'close']].max(axis=1)
        inconsistent_low = df['low'] > df[['open', 'close']].min(axis=1)
        
        if inconsistent_high.any():
            high_indices = df[inconsistent_high]['index'].tolist()
            issues.append(QualityIssue(
                issue_type=DataQualityIssue.INCONSISTENT_OHLC,
                severity='high',
                description=f'Inconsistent high prices in {inconsistent_high.sum()} records',
                affected_records=high_indices,
                suggested_action='Correct high prices to be >= max(open, close)',
                confidence=1.0
            ))
        
        if inconsistent_low.any():
            low_indices = df[inconsistent_low]['index'].tolist()
            issues.append(QualityIssue(
                issue_type=DataQualityIssue.INCONSISTENT_OHLC,
                severity='high',
                description=f'Inconsistent low prices in {inconsistent_low.sum()} records',
                affected_records=low_indices,
                suggested_action='Correct low prices to be <= min(open, close)',
                confidence=1.0
            ))
        
        return issues
    
    def _check_duplicates(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Check for duplicate records"""
        issues = []
        
        # Check for duplicate dates
        duplicate_dates = df.index.duplicated()
        if duplicate_dates.any():
            duplicate_indices = df[duplicate_dates]['index'].tolist()
            issues.append(QualityIssue(
                issue_type=DataQualityIssue.DUPLICATE_DATA,
                severity='medium',
                description=f'Duplicate dates found in {duplicate_dates.sum()} records',
                affected_records=duplicate_indices,
                suggested_action='Remove or merge duplicate records',
                confidence=1.0
            ))
        
        return issues
    
    def _check_data_gaps(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Check for gaps in data (missing trading days)"""
        issues = []
        
        if len(df) < 2:
            return issues
        
        # Calculate expected business days vs actual data points
        date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
        expected_days = len(date_range)
        actual_days = len(df)
        
        gap_ratio = (expected_days - actual_days) / expected_days
        
        if gap_ratio > 0.1:  # More than 10% missing days
            issues.append(QualityIssue(
                issue_type=DataQualityIssue.DATA_GAP,
                severity='medium' if gap_ratio < 0.3 else 'high',
                description=f'Data gaps detected: {gap_ratio:.1%} of expected trading days missing',
                affected_records=[],
                suggested_action='Fill gaps with appropriate interpolation or mark as holidays',
                confidence=0.8
            ))
        
        return issues
    
    def _check_stale_data(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Check if data is stale (too old)"""
        issues = []
        
        if df.empty:
            return issues
        
        latest_date = df.index.max()
        days_old = (datetime.now() - latest_date).days
        
        if days_old > self.stale_data_days:
            issues.append(QualityIssue(
                issue_type=DataQualityIssue.STALE_DATA,
                severity='medium' if days_old < 30 else 'high',
                description=f'Data is {days_old} days old (latest: {latest_date.date()})',
                affected_records=[],
                suggested_action='Update data source to get recent market data',
                confidence=1.0
            ))
        
        return issues
    
    def _detect_corporate_actions(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Detect potential corporate actions (splits, dividends)"""
        issues = []
        
        if len(df) < 2:
            return issues
        
        # Look for sudden price drops with volume spikes (potential dividend)
        df['price_drop'] = -df['close'].pct_change(fill_method=None)
        df['volume_spike'] = df['volume'] / df['volume'].rolling(10, min_periods=3).mean()
        
        # Potential dividend: price drop > 2% with volume spike > 2x
        dividend_candidates = (df['price_drop'] > 0.02) & (df['volume_spike'] > 2.0)
        
        if dividend_candidates.any():
            dividend_indices = df[dividend_candidates]['index'].tolist()
            issues.append(QualityIssue(
                issue_type=DataQualityIssue.CORPORATE_ACTION_DETECTED,
                severity='low',
                description=f'Potential dividend payments detected on {dividend_candidates.sum()} dates',
                affected_records=dividend_indices,
                suggested_action='Verify and adjust for dividend payments',
                confidence=0.6
            ))
        
        # Look for stock splits (sudden price halving/doubling with volume spike)
        price_changes = df['close'].pct_change(fill_method=None)
        potential_splits = (abs(price_changes) > 0.4) & (df['volume_spike'] > 3.0)
        
        if potential_splits.any():
            split_indices = df[potential_splits]['index'].tolist()
            issues.append(QualityIssue(
                issue_type=DataQualityIssue.CORPORATE_ACTION_DETECTED,
                severity='medium',
                description=f'Potential stock splits detected on {potential_splits.sum()} dates',
                affected_records=split_indices,
                suggested_action='Verify and adjust historical prices for stock splits',
                confidence=0.7
            ))
        
        return issues
    
    def _calculate_quality_score(self, df: pd.DataFrame, issues: List[QualityIssue]) -> float:
        """Calculate overall data quality score (0.0 to 1.0)"""
        if df.empty:
            return 0.0
        
        # Base score starts at 1.0
        score = 1.0
        
        # Deduct points based on issue severity
        severity_weights = {
            'low': 0.01,
            'medium': 0.05,
            'high': 0.15,
            'critical': 0.30
        }
        
        for issue in issues:
            weight = severity_weights.get(issue.severity, 0.1)
            affected_ratio = len(issue.affected_records) / len(df) if issue.affected_records else 0.1
            deduction = weight * affected_ratio * issue.confidence
            score -= deduction
        
        return max(0.0, score)
    
    def _generate_recommendations(self, issues: List[QualityIssue]) -> List[str]:
        """Generate actionable recommendations based on detected issues"""
        recommendations = []
        
        # Group issues by type for consolidated recommendations
        issue_types = {}
        for issue in issues:
            if issue.issue_type not in issue_types:
                issue_types[issue.issue_type] = []
            issue_types[issue.issue_type].append(issue)
        
        # Generate specific recommendations
        if DataQualityIssue.MISSING_DATA in issue_types:
            recommendations.append("Implement data interpolation or exclusion logic for missing values")
        
        if DataQualityIssue.PRICE_ANOMALY in issue_types:
            recommendations.append("Review and validate extreme price movements")
        
        if DataQualityIssue.CORPORATE_ACTION_DETECTED in issue_types:
            recommendations.append("Implement corporate action adjustment system")
        
        if DataQualityIssue.STALE_DATA in issue_types:
            recommendations.append("Set up automated data refresh processes")
        
        if DataQualityIssue.DATA_GAP in issue_types:
            recommendations.append("Implement gap detection and filling mechanisms")
        
        # Add general recommendations if multiple issues exist
        if len(issues) > 5:
            recommendations.append("Consider implementing automated data quality monitoring")
        
        return recommendations


def validate_data_quality(data: HistoricalData, symbol: str, 
                         validator: Optional[DataQualityValidator] = None) -> DataQualityReport:
    """
    Convenience function to validate data quality
    
    Args:
        data: Historical stock data to validate
        symbol: Stock symbol
        validator: Optional custom validator instance
        
    Returns:
        DataQualityReport with validation results
    """
    if validator is None:
        validator = DataQualityValidator()
    
    return validator.validate_data_quality(data, symbol)


def validate_data(data: HistoricalData, min_periods: int = 50, epsilon: float = 1e-6) -> None:
    """
    Validate data for sufficient length and price variation.
    
    Args:
        data: Historical stock data
        min_periods: Minimum number of periods required
        epsilon: Minimum standard deviation threshold for price variation
    
    Raises:
        InsufficientDataError: If data has fewer than min_periods
        ConstantPriceError: If close prices have std dev < epsilon
    """
    if data is None or data.empty:
        raise InsufficientDataError("No data provided")
    
    if len(data) < min_periods:
        raise InsufficientDataError(
            f"Insufficient data: {len(data)} periods, minimum {min_periods} required"
        )
    
    # Convert to DataFrame using existing method
    validator = DataQualityValidator()
    df = validator._convert_to_dataframe(data)
    
    close_prices = df['close'].dropna()
    
    if len(close_prices) < 2:
        raise InsufficientDataError("Insufficient valid close prices for variation check")
    
    std_dev = close_prices.std()
    
    if pd.isna(std_dev) or std_dev < epsilon:
        raise ConstantPriceError(
            f"Constant prices detected: std dev = {std_dev:.6f} < {epsilon}"
        )