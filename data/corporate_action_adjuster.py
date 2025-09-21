"""
Corporate Action Adjustment System for Stock Bot

This module implements corporate action adjustments including:
- Historical data adjustment for stock splits and dividends
- Consistency maintenance algorithms
- Detection and correction of corporate actions
- Validation of adjusted data
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .models import StockData, HistoricalData, create_stock_data

logger = logging.getLogger(__name__)


class CorporateActionType(Enum):
    """Types of corporate actions"""
    STOCK_SPLIT = "stock_split"
    STOCK_DIVIDEND = "stock_dividend"
    CASH_DIVIDEND = "cash_dividend"
    SPIN_OFF = "spin_off"
    MERGER = "merger"
    RIGHTS_OFFERING = "rights_offering"
    UNKNOWN = "unknown"


@dataclass
class CorporateAction:
    """Represents a corporate action"""
    date: datetime
    action_type: CorporateActionType
    ratio: float  # Split ratio (e.g., 2.0 for 2:1 split)
    dividend_amount: float  # Dividend amount per share
    description: str
    confidence: float  # 0.0 to 1.0
    detected_automatically: bool


@dataclass
class AdjustmentResult:
    """Result of corporate action adjustment"""
    original_count: int
    adjusted_count: int
    actions_detected: List[CorporateAction]
    actions_applied: List[CorporateAction]
    quality_score: float  # 0.0 to 1.0
    warnings: List[str]


@dataclass
class AdjustedData:
    """Data after corporate action adjustments"""
    data: HistoricalData
    adjustment_result: AdjustmentResult
    metadata: Dict[str, Any]


class CorporateActionAdjuster:
    """
    Corporate action adjustment system for stock market data
    """
    
    def __init__(self,
                 split_detection_threshold: float = 0.4,    # 40% price change for split detection
                 dividend_detection_threshold: float = 0.02, # 2% price drop for dividend detection
                 volume_spike_threshold: float = 2.0,       # 2x volume spike
                 confidence_threshold: float = 0.7):        # Min confidence for auto-adjustment
        
        self.split_detection_threshold = split_detection_threshold
        self.dividend_detection_threshold = dividend_detection_threshold
        self.volume_spike_threshold = volume_spike_threshold
        self.confidence_threshold = confidence_threshold
    
    def adjust_for_corporate_actions(self, data: HistoricalData, symbol: str,
                                   known_actions: Optional[List[CorporateAction]] = None) -> AdjustedData:
        """
        Adjust historical data for corporate actions
        
        Args:
            data: Historical stock data to adjust
            symbol: Stock symbol
            known_actions: Optional list of known corporate actions
            
        Returns:
            AdjustedData with corporate action adjustments applied
        """
        # FIXED: Added input validation
        if not data:
            return AdjustedData(
                data=[],
                adjustment_result=AdjustmentResult(
                    original_count=0,
                    adjusted_count=0,
                    actions_detected=[],
                    actions_applied=[],
                    quality_score=0.0,
                    warnings=["No data provided"]
                ),
                metadata={"symbol": symbol, "reason": "empty_data"}
            )
        
        # Convert to DataFrame for easier processing
        df = self._convert_to_dataframe(data)
        original_count = len(df)
        
        # Detect corporate actions if not provided
        detected_actions = []
        if known_actions is None:
            detected_actions = self._detect_corporate_actions(df, symbol)
        else:
            detected_actions = known_actions
        
        # Apply adjustments
        adjusted_df, applied_actions, warnings = self._apply_adjustments(df, detected_actions)
        
        # Convert back to HistoricalData format
        adjusted_data = self._convert_to_historical_data(adjusted_df, symbol)
        
        # Calculate quality score
        quality_score = self._calculate_adjustment_quality(df, adjusted_df, applied_actions)
        
        return AdjustedData(
            data=adjusted_data,
            adjustment_result=AdjustmentResult(
                original_count=original_count,
                adjusted_count=len(adjusted_data),
                actions_detected=detected_actions,
                actions_applied=applied_actions,
                quality_score=quality_score,
                warnings=warnings
            ),
            metadata={
                "symbol": symbol,
                "adjustments_applied": len(applied_actions),
                "original_length": original_count,
                "adjusted_length": len(adjusted_data)
            }
        )
    
    def _convert_to_dataframe(self, data: HistoricalData) -> pd.DataFrame:
        """Convert HistoricalData to pandas DataFrame"""
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
    
    def _detect_corporate_actions(self, df: pd.DataFrame, symbol: str) -> List[CorporateAction]:
        """Detect corporate actions from price and volume patterns"""
        actions = []
        
        if len(df) < 2:
            return actions
        
        # Calculate daily returns and volume ratios
        df['daily_return'] = df['close'].pct_change(fill_method=None)
        df['volume_ma'] = df['volume'].rolling(window=20, min_periods=5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Detect stock splits
        actions.extend(self._detect_stock_splits(df, symbol))
        
        # Detect dividends
        actions.extend(self._detect_dividends(df, symbol))
        
        return actions
    
    def _detect_stock_splits(self, df: pd.DataFrame, symbol: str) -> List[CorporateAction]:
        """Detect stock splits from price patterns"""
        # FIXED: Added input validation
        if df.empty:
            return []
        
        splits = []
        
        # Look for sudden large price drops with volume spikes
        split_candidates = (
            (df['daily_return'] < -self.split_detection_threshold) |  # Large price drop
            (df['daily_return'] > self.split_detection_threshold)     # Large price increase (reverse split)
        ) & (df['volume_ratio'] > self.volume_spike_threshold)        # Volume spike
        
        # FIXED: Added TODO for NSE API integration
        # TODO: Integrate NSE API for verification
        
        for date, row in df[split_candidates].iterrows():
            daily_return = row['daily_return']
            volume_ratio = row['volume_ratio']
            
            # Determine split ratio
            if daily_return < -0.4:  # ~50% drop suggests 2:1 split
                ratio = 2.0
                confidence = 0.8
            elif daily_return < -0.6:  # ~67% drop suggests 3:1 split
                ratio = 3.0
                confidence = 0.7
            elif daily_return > 0.8:  # ~100% increase suggests 1:2 reverse split
                ratio = 0.5
                confidence = 0.7
            else:
                # Estimate ratio from price change
                ratio = 1.0 / (1.0 + daily_return) if daily_return < 0 else (1.0 + daily_return)
                confidence = 0.6
            
            # Adjust confidence based on volume spike
            if volume_ratio > 5.0:
                confidence += 0.1
            elif volume_ratio < 2.0:
                confidence -= 0.2
            
            confidence = max(0.1, min(1.0, confidence))
            
            split = CorporateAction(
                date=date,
                action_type=CorporateActionType.STOCK_SPLIT,
                ratio=ratio,
                dividend_amount=0.0,
                description=f"Detected {ratio}:1 stock split on {date.date()}",
                confidence=confidence,
                detected_automatically=True
            )
            splits.append(split)
        
        return splits
    
    def _detect_dividends(self, df: pd.DataFrame, symbol: str) -> List[CorporateAction]:
        """Detect dividend payments from price patterns"""
        dividends = []
        
        # Look for moderate price drops with volume spikes (ex-dividend effect)
        dividend_candidates = (
            (df['daily_return'] < -self.dividend_detection_threshold) &  # Price drop
            (df['daily_return'] > -0.15) &                               # But not too large (not a split)
            (df['volume_ratio'] > self.volume_spike_threshold)           # Volume spike
        )
        
        for date, row in df[dividend_candidates].iterrows():
            daily_return = row['daily_return']
            volume_ratio = row['volume_ratio']
            close_price = row['close']
            
            # Estimate dividend amount from price drop
            estimated_dividend = abs(daily_return) * close_price
            
            # Calculate confidence based on pattern strength
            confidence = 0.5  # Base confidence for dividend detection
            
            # Adjust confidence based on price drop magnitude
            if 0.02 <= abs(daily_return) <= 0.05:  # Typical dividend range
                confidence += 0.2
            
            # Adjust confidence based on volume spike
            if volume_ratio > 3.0:
                confidence += 0.2
            elif volume_ratio < 2.0:
                confidence -= 0.1
            
            confidence = max(0.1, min(1.0, confidence))
            
            dividend = CorporateAction(
                date=date,
                action_type=CorporateActionType.CASH_DIVIDEND,
                ratio=1.0,
                dividend_amount=estimated_dividend,
                description=f"Detected ${estimated_dividend:.2f} dividend payment on {date.date()}",
                confidence=confidence,
                detected_automatically=True
            )
            dividends.append(dividend)
        
        return dividends
    
    def _apply_adjustments(self, df: pd.DataFrame, actions: List[CorporateAction]) -> Tuple[pd.DataFrame, List[CorporateAction], List[str]]:
        """Apply corporate action adjustments to the data"""
        adjusted_df = df.copy()
        applied_actions = []
        warnings = []
        
        # Sort actions by date (oldest first) for proper sequential adjustment
        sorted_actions = sorted(actions, key=lambda x: x.date)
        
        for action in sorted_actions:
            if action.confidence >= self.confidence_threshold:
                try:
                    if action.action_type == CorporateActionType.STOCK_SPLIT:
                        adjusted_df = self._apply_stock_split_adjustment(adjusted_df, action)
                        applied_actions.append(action)
                    elif action.action_type == CorporateActionType.CASH_DIVIDEND:
                        adjusted_df = self._apply_dividend_adjustment(adjusted_df, action)
                        applied_actions.append(action)
                    else:
                        warnings.append(f"Unsupported corporate action type: {action.action_type}")
                except Exception as e:
                    warnings.append(f"Failed to apply {action.action_type} adjustment: {str(e)}")
            else:
                warnings.append(f"Skipped {action.action_type} adjustment due to low confidence: {action.confidence:.2f}")
        
        return adjusted_df, applied_actions, warnings
    
    def _apply_stock_split_adjustment(self, df: pd.DataFrame, action: CorporateAction) -> pd.DataFrame:
        """Apply stock split adjustment to historical data"""
        adjusted_df = df.copy()
        
        # Adjust all data before the split date
        pre_split_mask = adjusted_df.index < action.date
        
        if pre_split_mask.any():
            # Adjust prices (divide by split ratio)
            price_fields = ['open', 'high', 'low', 'close']
            for field in price_fields:
                adjusted_df.loc[pre_split_mask, field] = adjusted_df.loc[pre_split_mask, field] / action.ratio
            
            # Adjust volume (multiply by split ratio)
            adjusted_df.loc[pre_split_mask, 'volume'] = adjusted_df.loc[pre_split_mask, 'volume'] * action.ratio
        
        return adjusted_df
    
    def _apply_dividend_adjustment(self, df: pd.DataFrame, action: CorporateAction) -> pd.DataFrame:
        """Apply dividend adjustment to historical data"""
        # FIXED: Subtract dividend from post-ex-date prices
        adjusted_df = df.copy()
        
        # Adjust all data after the ex-dividend date (subtract dividend)
        post_dividend_mask = adjusted_df.index >= action.date
        
        if post_dividend_mask.any():
            # Subtract dividend amount from post-ex-date prices
            price_fields = ['open', 'high', 'low', 'close']
            for field in price_fields:
                adjusted_df.loc[post_dividend_mask, field] = adjusted_df.loc[post_dividend_mask, field] - action.dividend_amount
        
        return adjusted_df
    
    def _calculate_adjustment_quality(self, original_df: pd.DataFrame, 
                                    adjusted_df: pd.DataFrame, 
                                    applied_actions: List[CorporateAction]) -> float:
        """Calculate quality score for the adjustments"""
        if original_df.empty or adjusted_df.empty:
            return 0.0
        
        # Base quality score
        quality = 1.0
        
        # Check for data consistency after adjustments
        if len(original_df) != len(adjusted_df):
            quality -= 0.2  # Penalize if data length changed
        
        # Check for reasonable price continuity
        if len(adjusted_df) > 1:
            adjusted_returns = adjusted_df['close'].pct_change(fill_method=None).dropna()
            extreme_returns = (abs(adjusted_returns) > 0.5).sum()  # More than 50% daily change
            
            if extreme_returns > 0:
                quality -= min(0.3, extreme_returns * 0.1)  # Penalize extreme returns
        
        # Reward high-confidence adjustments
        if applied_actions:
            avg_confidence = sum(action.confidence for action in applied_actions) / len(applied_actions)
            quality = quality * (0.7 + 0.3 * avg_confidence)  # Scale by average confidence
        
        # Check for negative prices (should not happen with good adjustments)
        price_fields = ['open', 'high', 'low', 'close']
        for field in price_fields:
            if (adjusted_df[field] <= 0).any():
                quality -= 0.4  # Heavy penalty for negative prices
        
        return max(0.0, quality)
    
    def _convert_to_historical_data(self, df: pd.DataFrame, symbol: str) -> HistoricalData:
        """Convert DataFrame back to HistoricalData format"""
        data = []
        for date, row in df.iterrows():
            stock_data = create_stock_data(
                symbol=symbol,
                date=date.strftime("%Y-%m-%d"),
                open_price=float(row['open']) if pd.notna(row['open']) else None,
                high=float(row['high']) if pd.notna(row['high']) else None,
                low=float(row['low']) if pd.notna(row['low']) else None,
                close=float(row['close']) if pd.notna(row['close']) else None,
                volume=int(row['volume']) if pd.notna(row['volume']) else 0
            )
            data.append(stock_data)
        
        return data
    
    def validate_adjustments(self, original_data: HistoricalData, 
                           adjusted_data: HistoricalData) -> Dict[str, Any]:
        """Validate that adjustments were applied correctly"""
        validation_result = {
            'is_valid': True,
            'issues': [],
            'metrics': {}
        }
        
        if not original_data or not adjusted_data:
            validation_result['is_valid'] = False
            validation_result['issues'].append("Missing data for validation")
            return validation_result
        
        # Convert to DataFrames for analysis
        original_df = self._convert_to_dataframe(original_data)
        adjusted_df = self._convert_to_dataframe(adjusted_data)
        
        # Check data length consistency
        if len(original_df) != len(adjusted_df):
            validation_result['issues'].append(f"Data length mismatch: {len(original_df)} vs {len(adjusted_df)}")
        
        # Check for negative prices
        price_fields = ['open', 'high', 'low', 'close']
        for field in price_fields:
            if (adjusted_df[field] <= 0).any():
                validation_result['is_valid'] = False
                validation_result['issues'].append(f"Negative {field} prices detected after adjustment")
        
        # Check OHLC consistency
        ohlc_issues = 0
        for _, row in adjusted_df.iterrows():
            if pd.notna(row['high']) and pd.notna(row['low']) and row['high'] < row['low']:
                ohlc_issues += 1
        
        if ohlc_issues > 0:
            validation_result['issues'].append(f"OHLC inconsistencies in {ohlc_issues} records")
        
        # Calculate adjustment metrics
        if len(adjusted_df) > 1:
            original_volatility = original_df['close'].pct_change(fill_method=None).std()
            adjusted_volatility = adjusted_df['close'].pct_change(fill_method=None).std()
            
            validation_result['metrics'] = {
                'original_volatility': float(original_volatility) if pd.notna(original_volatility) else 0.0,
                'adjusted_volatility': float(adjusted_volatility) if pd.notna(adjusted_volatility) else 0.0,
                'volatility_ratio': float(adjusted_volatility / original_volatility) if original_volatility > 0 else 1.0
            }
        
        # Overall validation
        if len(validation_result['issues']) > 2:
            validation_result['is_valid'] = False
        
        return validation_result


def adjust_for_corporate_actions(data: HistoricalData, symbol: str,
                               known_actions: Optional[List[CorporateAction]] = None,
                               adjuster: Optional[CorporateActionAdjuster] = None) -> AdjustedData:
    """
    Convenience function to adjust data for corporate actions
    
    Args:
        data: Historical stock data to adjust
        symbol: Stock symbol
        known_actions: Optional list of known corporate actions
        adjuster: Optional custom adjuster instance
        
    Returns:
        AdjustedData with corporate action adjustments applied
    """
    if adjuster is None:
        adjuster = CorporateActionAdjuster()
    
    return adjuster.adjust_for_corporate_actions(data, symbol, known_actions)