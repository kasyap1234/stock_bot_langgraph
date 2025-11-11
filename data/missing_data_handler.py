
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings

from .models import StockData, HistoricalData, create_stock_data

logger = logging.getLogger(__name__)


class InterpolationMethod(Enum):
    LINEAR = "linear"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    SPLINE = "spline"
    POLYNOMIAL = "polynomial"
    TIME_WEIGHTED = "time_weighted"
    EXCLUDE = "exclude"


class DataQuality(Enum):
    HIGH = "high"          # Original data, no interpolation
    GOOD = "good"          # Minor interpolation with high confidence
    FAIR = "fair"          # Moderate interpolation with medium confidence
    POOR = "poor"          # Heavy interpolation with low confidence
    EXCLUDED = "excluded"  # Data excluded due to poor quality


@dataclass
class InterpolationResult:
    original_count: int
    missing_count: int
    interpolated_count: int
    excluded_count: int
    method_used: InterpolationMethod
    quality_score: float  # 0.0 to 1.0
    confidence: float     # 0.0 to 1.0
    warnings: List[str]


@dataclass
class ProcessedData:
    data: HistoricalData
    quality: DataQuality
    interpolation_result: InterpolationResult
    metadata: Dict[str, Any]


class MissingDataHandler:
    def __init__(self,
                 max_missing_ratio: float = 0.30,      # Max 30% missing data
                 max_consecutive_missing: int = 7,      # Max 7 consecutive missing days
                 interpolation_confidence_threshold: float = 0.7,  # Min confidence for interpolation
                 exclude_threshold: float = 0.50):     # Exclude if >50% missing
        
        self.max_missing_ratio = max_missing_ratio
        self.max_consecutive_missing = max_consecutive_missing
        self.interpolation_confidence_threshold = interpolation_confidence_threshold
        self.exclude_threshold = exclude_threshold
    
    def handle_missing_data(self, data: HistoricalData, symbol: str) -> ProcessedData:
        if not data:
            return ProcessedData(
                data=[],
                quality=DataQuality.EXCLUDED,
                interpolation_result=InterpolationResult(
                    original_count=0,
                    missing_count=0,
                    interpolated_count=0,
                    excluded_count=0,
                    method_used=InterpolationMethod.EXCLUDE,
                    quality_score=0.0,
                    confidence=0.0,
                    warnings=["No data provided"]
                ),
                metadata={"symbol": symbol, "reason": "empty_data"}
            )
        
        # Convert to DataFrame for easier processing
        df = self._convert_to_dataframe(data)
        original_count = len(df)
        
        # Analyze missing data patterns
        missing_analysis = self._analyze_missing_patterns(df)
        
        # Determine if data should be excluded entirely
        if self._should_exclude_data(missing_analysis):
            return ProcessedData(
                data=[],
                quality=DataQuality.EXCLUDED,
                interpolation_result=InterpolationResult(
                    original_count=original_count,
                    missing_count=missing_analysis['total_missing'],
                    interpolated_count=0,
                    excluded_count=original_count,
                    method_used=InterpolationMethod.EXCLUDE,
                    quality_score=0.0,
                    confidence=1.0,
                    warnings=[f"Data excluded: {missing_analysis['missing_ratio']:.1%} missing"]
                ),
                metadata={"symbol": symbol, "reason": "too_much_missing_data"}
            )
        
        # Choose interpolation method based on data characteristics
        method = self._choose_interpolation_method(df, missing_analysis)
        
        # Apply interpolation
        processed_df, interpolation_result = self._apply_interpolation(df, method, missing_analysis)
        
        # Convert back to HistoricalData format
        processed_data = self._convert_to_historical_data(processed_df, symbol)
        
        # Determine overall quality
        quality = self._assess_data_quality(interpolation_result)
        
        return ProcessedData(
            data=processed_data,
            quality=quality,
            interpolation_result=interpolation_result,
            metadata={
                "symbol": symbol,
                "method": method.value,
                "original_length": original_count,
                "processed_length": len(processed_data)
            }
        )
    
    def _convert_to_dataframe(self, data: HistoricalData) -> pd.DataFrame:
        df_data = []
        for record in data:
            df_data.append({
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
    
    def _analyze_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        price_fields = ['open', 'high', 'low', 'close']
        
        analysis = {
            'total_records': len(df),
            'missing_by_field': {},
            'total_missing': 0,
            'missing_ratio': 0.0,
            'consecutive_missing': {},
            'missing_patterns': []
        }
        
        # Analyze missing data by field
        for field in price_fields + ['volume']:
            missing_mask = df[field].isnull()
            missing_count = missing_mask.sum()
            
            analysis['missing_by_field'][field] = {
                'count': int(missing_count),
                'ratio': float(missing_count) / len(df) if len(df) > 0 else 0.0,
                'consecutive_max': self._max_consecutive_missing(missing_mask)
            }
            
            analysis['total_missing'] += int(missing_count)
        
        # Calculate overall missing ratio (any field missing)
        any_missing = df[price_fields + ['volume']].isnull().any(axis=1)
        analysis['missing_ratio'] = float(any_missing.sum()) / len(df) if len(df) > 0 else 0.0
        
        # Identify patterns
        if analysis['missing_ratio'] > 0.5:
            analysis['missing_patterns'].append('high_missing_ratio')
        
        for field, info in analysis['missing_by_field'].items():
            if info['consecutive_max'] > self.max_consecutive_missing:
                analysis['missing_patterns'].append(f'long_consecutive_{field}')
        
        return analysis
    
    def _max_consecutive_missing(self, missing_mask: pd.Series) -> int:
        if not missing_mask.any():
            return 0
        
        # Find consecutive groups of True values
        groups = missing_mask.ne(missing_mask.shift()).cumsum()
        consecutive_counts = missing_mask.groupby(groups).sum()
        
        return consecutive_counts.max() if len(consecutive_counts) > 0 else 0
    
    def _should_exclude_data(self, missing_analysis: Dict[str, Any]) -> bool:
        # Exclude if too much data is missing overall
        if missing_analysis['missing_ratio'] > self.exclude_threshold:
            return True
        
        # Exclude if critical fields have too much missing data
        critical_fields = ['close']  # Close price is most critical
        for field in critical_fields:
            if missing_analysis['missing_by_field'][field]['ratio'] > 0.6:  # More lenient
                return True
        
        # Exclude if too many consecutive missing values (more lenient)
        for field, info in missing_analysis['missing_by_field'].items():
            if info['consecutive_max'] > self.max_consecutive_missing * 3:  # More lenient
                return True
        
        return False
    
    def _choose_interpolation_method(self, df: pd.DataFrame,
                                   missing_analysis: Dict[str, Any]) -> InterpolationMethod:
        missing_ratio = missing_analysis['missing_ratio']
        
        # If very little missing data, use linear interpolation
        if missing_ratio < 0.10:  # More lenient
            return InterpolationMethod.LINEAR
        
        # If moderate missing data, use time-weighted interpolation
        if missing_ratio < 0.25:  # More lenient
            return InterpolationMethod.TIME_WEIGHTED
        
        # For higher missing ratios, use forward fill (more conservative)
        if missing_ratio < 0.40:  # More lenient
            return InterpolationMethod.FORWARD_FILL
        
        # For very high missing ratios, exclude
        return InterpolationMethod.EXCLUDE
    
    def _apply_interpolation(self, df: pd.DataFrame, method: InterpolationMethod,
                           missing_analysis: Dict[str, Any]) -> Tuple[pd.DataFrame, InterpolationResult]:
        original_df = df.copy()
        warnings_list = []
        
        if method == InterpolationMethod.EXCLUDE:
            return df, InterpolationResult(
                original_count=len(df),
                missing_count=missing_analysis['total_missing'],
                interpolated_count=0,
                excluded_count=len(df),
                method_used=method,
                quality_score=0.0,
                confidence=1.0,
                warnings=["Data excluded due to poor quality"]
            )
        
        interpolated_count = 0
        
        if method == InterpolationMethod.LINEAR:
            interpolated_count = self._apply_linear_interpolation(df, warnings_list)
        
        elif method == InterpolationMethod.FORWARD_FILL:
            interpolated_count = self._apply_forward_fill(df, warnings_list)
        
        elif method == InterpolationMethod.BACKWARD_FILL:
            interpolated_count = self._apply_backward_fill(df, warnings_list)
        
        elif method == InterpolationMethod.TIME_WEIGHTED:
            interpolated_count = self._apply_time_weighted_interpolation(df, warnings_list)
        
        elif method == InterpolationMethod.SPLINE:
            interpolated_count = self._apply_spline_interpolation(df, warnings_list)
        
        # Calculate quality metrics
        quality_score = self._calculate_interpolation_quality(original_df, df, interpolated_count)
        confidence = self._calculate_interpolation_confidence(missing_analysis, method)
        
        return df, InterpolationResult(
            original_count=len(original_df),
            missing_count=missing_analysis['total_missing'],
            interpolated_count=interpolated_count,
            excluded_count=0,
            method_used=method,
            quality_score=quality_score,
            confidence=confidence,
            warnings=warnings_list
        )
    
    def _apply_linear_interpolation(self, df: pd.DataFrame, warnings: List[str]) -> int:
        interpolated_count = 0
        price_fields = ['open', 'high', 'low', 'close']
        
        for field in price_fields:
            before_count = df[field].isnull().sum()
            df[field] = df[field].interpolate(method='linear')
            after_count = df[field].isnull().sum()
            interpolated_count += (before_count - after_count)
        
        # Handle volume separately (use forward fill for volume)
        if df['volume'].isnull().any():
            before_count = df['volume'].isnull().sum()
            df['volume'] = df['volume'].ffill().bfill()
            after_count = df['volume'].isnull().sum()
            interpolated_count += (before_count - after_count)
            
            if before_count > 0:
                warnings.append(f"Volume interpolated using forward/backward fill for {before_count} records")
        
        return interpolated_count
    
    def _apply_forward_fill(self, df: pd.DataFrame, warnings: List[str]) -> int:
        interpolated_count = 0
        fields = ['open', 'high', 'low', 'close', 'volume']
        
        for field in fields:
            before_count = df[field].isnull().sum()
            df[field] = df[field].ffill()
            after_count = df[field].isnull().sum()
            interpolated_count += (before_count - after_count)
        
        # Handle any remaining NaN values at the beginning with backward fill
        for field in fields:
            if df[field].isnull().any():
                before_count = df[field].isnull().sum()
                df[field] = df[field].bfill()
                after_count = df[field].isnull().sum()
                interpolated_count += (before_count - after_count)
        
        return interpolated_count
    
    def _apply_backward_fill(self, df: pd.DataFrame, warnings: List[str]) -> int:
        interpolated_count = 0
        fields = ['open', 'high', 'low', 'close', 'volume']
        
        for field in fields:
            before_count = df[field].isnull().sum()
            df[field] = df[field].bfill()
            after_count = df[field].isnull().sum()
            interpolated_count += (before_count - after_count)
        
        # Handle any remaining NaN values at the end with forward fill
        for field in fields:
            if df[field].isnull().any():
                before_count = df[field].isnull().sum()
                df[field] = df[field].ffill()
                after_count = df[field].isnull().sum()
                interpolated_count += (before_count - after_count)
        
        return interpolated_count
    
    def _apply_time_weighted_interpolation(self, df: pd.DataFrame, warnings: List[str]) -> int:
        interpolated_count = 0
        price_fields = ['open', 'high', 'low', 'close']
        
        # For price fields, use time-based interpolation
        for field in price_fields:
            before_count = df[field].isnull().sum()
            df[field] = df[field].interpolate(method='time')
            after_count = df[field].isnull().sum()
            interpolated_count += (before_count - after_count)
        
        # For volume, use forward fill as it's not continuous
        if df['volume'].isnull().any():
            before_count = df['volume'].isnull().sum()
            df['volume'] = df['volume'].ffill().bfill()
            after_count = df['volume'].isnull().sum()
            interpolated_count += (before_count - after_count)
        
        return interpolated_count
    
    def _apply_spline_interpolation(self, df: pd.DataFrame, warnings: List[str]) -> int:
        interpolated_count = 0
        price_fields = ['open', 'high', 'low', 'close']
        
        for field in price_fields:
            before_count = df[field].isnull().sum()
            try:
                # Use cubic spline if we have enough data points
                if len(df.dropna(subset=[field])) >= 4:
                    df[field] = df[field].interpolate(method='spline', order=3)
                else:
                    # Fall back to linear if not enough points
                    df[field] = df[field].interpolate(method='linear')
                    warnings.append(f"Fell back to linear interpolation for {field} due to insufficient data")
            except Exception as e:
                # Fall back to linear interpolation on error
                df[field] = df[field].interpolate(method='linear')
                warnings.append(f"Spline interpolation failed for {field}, used linear: {str(e)}")
            
            after_count = df[field].isnull().sum()
            interpolated_count += (before_count - after_count)
        
        # Handle volume with forward fill
        if df['volume'].isnull().any():
            before_count = df['volume'].isnull().sum()
            df['volume'] = df['volume'].ffill().bfill()
            after_count = df['volume'].isnull().sum()
            interpolated_count += (before_count - after_count)
        
        return interpolated_count
    
    def _calculate_interpolation_quality(self, original_df: pd.DataFrame,
                                       interpolated_df: pd.DataFrame,
                                       interpolated_count: int) -> float:
        if len(original_df) == 0:
            return 0.0
        
        # Base quality starts high
        quality = 1.0
        
        # Reduce quality based on interpolation ratio
        interpolation_ratio = interpolated_count / (len(original_df) * 5)  # 5 fields
        quality -= interpolation_ratio * 0.5  # Max 50% reduction for full interpolation
        
        # Additional quality checks
        price_fields = ['open', 'high', 'low', 'close']
        
        # Check for unrealistic interpolated values
        for field in price_fields:
            if field in interpolated_df.columns:
                # Check for negative values (shouldn't happen with good interpolation)
                if (interpolated_df[field] <= 0).any():
                    quality -= 0.2
                
                # Check for extreme volatility in interpolated sections
                daily_changes = interpolated_df[field].pct_change(fill_method=None).abs()
                if daily_changes.max() > 0.5:  # 50% daily change
                    quality -= 0.1
        
        return max(0.0, quality)
    
    def _calculate_interpolation_confidence(self, missing_analysis: Dict[str, Any],
                                          method: InterpolationMethod) -> float:
        base_confidence = {
            InterpolationMethod.LINEAR: 0.8,
            InterpolationMethod.TIME_WEIGHTED: 0.85,
            InterpolationMethod.FORWARD_FILL: 0.7,
            InterpolationMethod.BACKWARD_FILL: 0.7,
            InterpolationMethod.SPLINE: 0.9,
            InterpolationMethod.EXCLUDE: 1.0
        }.get(method, 0.5)
        
        # Reduce confidence based on missing data ratio
        missing_ratio = missing_analysis['missing_ratio']
        confidence_reduction = missing_ratio * 0.5  # Max 50% reduction
        
        # Additional reduction for consecutive missing values
        max_consecutive = max(
            info['consecutive_max'] for info in missing_analysis['missing_by_field'].values()
        )
        if max_consecutive > 3:
            confidence_reduction += 0.1
        
        return max(0.1, base_confidence - confidence_reduction)
    
    def _assess_data_quality(self, interpolation_result: InterpolationResult) -> DataQuality:
        if interpolation_result.method_used == InterpolationMethod.EXCLUDE:
            return DataQuality.EXCLUDED
        
        quality_score = interpolation_result.quality_score
        confidence = interpolation_result.confidence
        
        # Combine quality score and confidence
        overall_score = (quality_score + confidence) / 2
        
        if overall_score >= 0.9:
            return DataQuality.HIGH
        elif overall_score >= 0.7:
            return DataQuality.GOOD
        elif overall_score >= 0.5:
            return DataQuality.FAIR
        else:
            return DataQuality.POOR
    
    def _convert_to_historical_data(self, df: pd.DataFrame, symbol: str) -> HistoricalData:
        """Convert DataFrame back to HistoricalData format"""
        data = []
        for date, row in df.iterrows():
            # Handle NaN values properly
            open_price = row['open'] if pd.notna(row['open']) else None
            high = row['high'] if pd.notna(row['high']) else None
            low = row['low'] if pd.notna(row['low']) else None
            close = row['close'] if pd.notna(row['close']) else None
            volume = int(row['volume']) if pd.notna(row['volume']) else 0
            
            stock_data = create_stock_data(
                symbol=symbol,
                date=date.strftime("%Y-%m-%d"),
                open_price=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume
            )
            data.append(stock_data)
        
        return data


def handle_missing_data(data: HistoricalData, symbol: str, 
                       handler: Optional[MissingDataHandler] = None) -> ProcessedData:
    """
    Convenience function to handle missing data
    
    Args:
        data: Historical stock data with potential missing values
        symbol: Stock symbol
        handler: Optional custom handler instance
        
    Returns:
        ProcessedData with handled missing values
    """
    if handler is None:
        handler = MissingDataHandler()
    
    return handler.handle_missing_data(data, symbol)