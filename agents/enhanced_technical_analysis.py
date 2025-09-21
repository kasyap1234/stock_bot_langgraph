"""
Enhanced Technical Analysis Engine with Signal Quality Control and Performance Tracking

This module provides advanced technical analysis capabilities with:
- Signal quality scoring and filtering
- Multi-timeframe analysis with consistency checks
- Dynamic parameter adjustment based on market conditions
- Performance-based indicator weighting
- Historical performance tracking for continuous improvement
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """Represents a trading signal with quality metrics"""
    indicator: str
    direction: str  # 'buy', 'sell', 'neutral'
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    timeframe: str
    metadata: Dict[str, Any]


@dataclass
class IndicatorPerformance:
    """Tracks performance metrics for individual indicators"""
    indicator_name: str
    total_signals: int
    correct_signals: int
    accuracy: float
    avg_return: float
    sharpe_ratio: float
    last_updated: datetime
    market_regime_performance: Dict[str, float]  # Performance by market regime


class SignalQualityFilter:
    """Filters and scores signals based on quality metrics"""
    
    def __init__(self, min_confidence: float = 0.3, min_strength: float = 0.2):
        self.min_confidence = min_confidence
        self.min_strength = min_strength
        self.noise_threshold = 0.1
        self.historical_performance = {}  # Track signal performance for scoring
        
    def filter_signals(self, signals: List[Signal]) -> List[Signal]:
        """Filter out low-quality signals with noise filtering"""
        filtered_signals = []
        
        for signal in signals:
            if self._is_high_quality(signal) and not self._is_noise(signal):
                filtered_signals.append(signal)
            else:
                logger.debug(f"Filtered out low-quality/noisy signal: {signal.indicator} - "
                           f"confidence: {signal.confidence}, strength: {signal.strength}")
                
        return filtered_signals
    
    def _is_high_quality(self, signal: Signal) -> bool:
        """Determine if a signal meets quality thresholds"""
        return (signal.confidence >= self.min_confidence and 
                signal.strength >= self.min_strength and
                signal.direction != 'neutral')
    
    def _is_noise(self, signal: Signal) -> bool:
        """Detect if signal is likely noise based on historical patterns"""
        # Check for rapid signal reversals (noise indicator)
        if hasattr(signal, 'metadata') and 'recent_signals' in signal.metadata:
            recent_signals = signal.metadata['recent_signals']
            if len(recent_signals) >= 3:
                # Check for alternating signals (buy-sell-buy or sell-buy-sell)
                directions = [s['direction'] for s in recent_signals[-3:]]
                if len(set(directions)) > 1 and directions[0] == directions[2] != directions[1]:
                    logger.debug(f"Detected noise pattern in {signal.indicator}: {directions}")
                    return True
        
        # Check signal strength consistency
        if signal.strength < self.noise_threshold:
            return True
            
        # Check for conflicting metadata indicators
        if 'volatility_adjusted' in signal.metadata:
            vol_adjustment = signal.metadata['volatility_adjusted']
            if vol_adjustment < 0.5:  # Heavy volatility penalty suggests noise
                return True
                
        return False
    
    def score_signal_quality(self, signal: Signal, market_context: Dict[str, Any]) -> float:
        """Score signal quality based on multiple factors including historical performance"""
        base_score = (signal.confidence + signal.strength) / 2
        
        # Adjust for market volatility
        volatility = market_context.get('volatility', 0.02)
        volatility_adjustment = 1.0
        if volatility > 0.05:  # High volatility
            volatility_adjustment = 0.85  # Reduce confidence in high volatility
        elif volatility < 0.01:  # Low volatility
            volatility_adjustment = 1.15  # Increase confidence in stable markets
            
        # Adjust for trend consistency
        trend_adjustment = 1.0
        trend_strength = market_context.get('trend_strength', 0.5)
        if trend_strength > 0.7 and signal.direction in ['buy', 'sell']:
            trend_adjustment = 1.1  # Boost signals aligned with strong trends
        elif trend_strength < 0.3 and signal.direction in ['buy', 'sell']:
            trend_adjustment = 0.9  # Reduce trend signals in ranging markets
            
        # Historical performance adjustment
        historical_adjustment = self._get_historical_performance_score(signal.indicator, market_context)
        
        # Signal freshness (newer signals get slight boost)
        freshness_adjustment = 1.0
        if hasattr(signal, 'timestamp'):
            age_hours = (datetime.now() - signal.timestamp).total_seconds() / 3600
            if age_hours < 1:
                freshness_adjustment = 1.05
            elif age_hours > 24:
                freshness_adjustment = 0.95
                
        # Combine all adjustments
        final_score = (base_score * volatility_adjustment * trend_adjustment * 
                      historical_adjustment * freshness_adjustment)
        
        # Add noise penalty
        if self._is_noise(signal):
            final_score *= 0.7
            
        return min(final_score, 1.0)
    
    def _get_historical_performance_score(self, indicator: str, market_context: Dict[str, Any]) -> float:
        """Get historical performance adjustment for indicator"""
        if indicator not in self.historical_performance:
            return 1.0  # Neutral for unknown indicators
            
        perf_data = self.historical_performance[indicator]
        base_performance = perf_data.get('accuracy', 0.5)
        
        # Adjust for current market regime if available
        current_regime = market_context.get('regime', 'unknown')
        regime_performance = perf_data.get('regime_performance', {}).get(current_regime, base_performance)
        
        # Convert performance to adjustment factor (0.5 accuracy = 1.0 adjustment)
        adjustment = 0.5 + regime_performance
        return min(max(adjustment, 0.3), 1.7)  # Clamp between 0.3 and 1.7
    
    def update_historical_performance(self, indicator: str, accuracy: float, 
                                    market_regime: str = 'unknown'):
        """Update historical performance data for quality scoring"""
        if indicator not in self.historical_performance:
            self.historical_performance[indicator] = {
                'accuracy': accuracy,
                'regime_performance': {}
            }
        else:
            # Exponential moving average update
            alpha = 0.1
            current_acc = self.historical_performance[indicator]['accuracy']
            self.historical_performance[indicator]['accuracy'] = (
                alpha * accuracy + (1 - alpha) * current_acc
            )
            
        # Update regime-specific performance
        regime_perf = self.historical_performance[indicator]['regime_performance']
        if market_regime not in regime_perf:
            regime_perf[market_regime] = accuracy
        else:
            alpha = 0.15  # Slightly faster adaptation for regime-specific performance
            regime_perf[market_regime] = (
                alpha * accuracy + (1 - alpha) * regime_perf[market_regime]
            )
    
    def get_noise_filtered_signals(self, signals: List[Signal], 
                                 lookback_window: int = 10) -> List[Signal]:
        """Apply advanced noise filtering based on signal patterns"""
        if len(signals) < 2:
            return signals
            
        filtered_signals = []
        
        for i, signal in enumerate(signals):
            # Get recent signals for pattern analysis
            start_idx = max(0, i - lookback_window)
            recent_signals = signals[start_idx:i]
            
            # Add recent signals to metadata for noise detection
            signal.metadata['recent_signals'] = [
                {'direction': s.direction, 'strength': s.strength, 'timestamp': s.timestamp}
                for s in recent_signals
            ]
            
            # Apply noise filtering
            if not self._is_noise(signal):
                filtered_signals.append(signal)
            else:
                logger.debug(f"Filtered noise signal: {signal.indicator} at {signal.timestamp}")
                
        return filtered_signals


class IndicatorPerformanceTracker:
    """Tracks and analyzes performance of individual indicators"""
    
    def __init__(self, performance_file: str = "data/indicator_performance.json"):
        self.performance_file = performance_file
        self.performance_data: Dict[str, IndicatorPerformance] = {}
        self.load_performance_data()
        
    def load_performance_data(self):
        """Load historical performance data from file"""
        if os.path.exists(self.performance_file):
            try:
                with open(self.performance_file, 'r') as f:
                    data = json.load(f)
                    for indicator_name, perf_dict in data.items():
                        self.performance_data[indicator_name] = IndicatorPerformance(
                            indicator_name=perf_dict['indicator_name'],
                            total_signals=perf_dict['total_signals'],
                            correct_signals=perf_dict['correct_signals'],
                            accuracy=perf_dict['accuracy'],
                            avg_return=perf_dict['avg_return'],
                            sharpe_ratio=perf_dict['sharpe_ratio'],
                            last_updated=datetime.fromisoformat(perf_dict['last_updated']),
                            market_regime_performance=perf_dict.get('market_regime_performance', {})
                        )
            except Exception as e:
                logger.error(f"Error loading performance data: {e}")
                
    def save_performance_data(self):
        """Save performance data to file"""
        try:
            os.makedirs(os.path.dirname(self.performance_file), exist_ok=True)
            data = {}
            for indicator_name, perf in self.performance_data.items():
                data[indicator_name] = {
                    'indicator_name': perf.indicator_name,
                    'total_signals': perf.total_signals,
                    'correct_signals': perf.correct_signals,
                    'accuracy': perf.accuracy,
                    'avg_return': perf.avg_return,
                    'sharpe_ratio': perf.sharpe_ratio,
                    'last_updated': perf.last_updated.isoformat(),
                    'market_regime_performance': perf.market_regime_performance
                }
            with open(self.performance_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving performance data: {e}")
            
    def update_performance(self, indicator_name: str, signal_correct: bool, 
                         signal_return: float, market_regime: str = 'unknown'):
        """Update performance metrics for an indicator"""
        if indicator_name not in self.performance_data:
            self.performance_data[indicator_name] = IndicatorPerformance(
                indicator_name=indicator_name,
                total_signals=0,
                correct_signals=0,
                accuracy=0.0,
                avg_return=0.0,
                sharpe_ratio=0.0,
                last_updated=datetime.now(),
                market_regime_performance={}
            )
            
        perf = self.performance_data[indicator_name]
        perf.total_signals += 1
        if signal_correct:
            perf.correct_signals += 1
            
        # Update accuracy
        perf.accuracy = perf.correct_signals / perf.total_signals
        
        # Update average return (exponential moving average)
        alpha = 0.1  # Smoothing factor
        perf.avg_return = alpha * signal_return + (1 - alpha) * perf.avg_return
        
        # Update market regime performance
        if market_regime not in perf.market_regime_performance:
            perf.market_regime_performance[market_regime] = 0.0
        perf.market_regime_performance[market_regime] = (
            alpha * signal_return + (1 - alpha) * perf.market_regime_performance[market_regime]
        )
        
        perf.last_updated = datetime.now()
        self.save_performance_data()
        
    def get_indicator_weight(self, indicator_name: str, market_regime: str = 'unknown') -> float:
        """Get dynamic weight for indicator based on performance"""
        if indicator_name not in self.performance_data:
            return 0.5  # Default weight for new indicators
            
        perf = self.performance_data[indicator_name]
        
        # Base weight from overall accuracy
        base_weight = perf.accuracy
        
        # Adjust for market regime performance
        regime_performance = perf.market_regime_performance.get(market_regime, perf.avg_return)
        regime_adjustment = 1.0 + (regime_performance * 0.5)  # Scale regime impact
        
        # Adjust for recency (decay older performance)
        days_since_update = (datetime.now() - perf.last_updated).days
        recency_factor = max(0.5, 1.0 - (days_since_update / 365))  # Decay over a year
        
        final_weight = base_weight * regime_adjustment * recency_factor
        return min(max(final_weight, 0.1), 1.0)  # Clamp between 0.1 and 1.0


class MultiTimeframeAnalyzer:
    """Enhanced multi-timeframe analysis with consistency scoring and cross-timeframe validation"""
    
    def __init__(self, timeframes: List[str] = None):
        self.timeframes = timeframes or ['5m', '15m', '1H', '4H', '1D', '1W']
        self.consistency_threshold = 0.7
        self.timeframe_weights = {
            '5m': 0.1, '15m': 0.15, '1H': 0.2, '4H': 0.25, '1D': 0.2, '1W': 0.1
        }
        
    def analyze_multi_timeframe(self, df: pd.DataFrame, 
                              analysis_func, 
                              symbol: str) -> Dict[str, Any]:
        """Analyze signals across multiple timeframes with enhanced consistency scoring"""
        timeframe_signals = {}
        consistency_scores = {}
        validation_results = {}
        
        for timeframe in self.timeframes:
            try:
                resampled_df = self._resample_data(df, timeframe)
                if len(resampled_df) >= 20:  # Minimum data requirement
                    signals = analysis_func(resampled_df, symbol)
                    timeframe_signals[timeframe] = signals
                    
            except Exception as e:
                logger.error(f"Error analyzing timeframe {timeframe}: {e}")
                continue
                
        # Calculate consistency scores with enhanced logic
        for indicator in self._get_common_indicators(timeframe_signals):
            consistency_scores[indicator] = self._calculate_enhanced_consistency_score(
                indicator, timeframe_signals
            )
            
        # Perform cross-timeframe validation
        validation_results = self._cross_timeframe_validation(timeframe_signals)
            
        return {
            'timeframe_signals': timeframe_signals,
            'consistency_scores': consistency_scores,
            'validation_results': validation_results,
            'consensus_signals': self._generate_consensus_signals(
                timeframe_signals, consistency_scores
            ),
            'timeframe_strength': self._calculate_timeframe_strength(timeframe_signals)
        }
    
    def _resample_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample data to specified timeframe with comprehensive support"""
        agg_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }
        
        try:
            if timeframe == '5m':
                return df.resample('5min').agg(agg_dict).dropna()
            elif timeframe == '15m':
                return df.resample('15min').agg(agg_dict).dropna()
            elif timeframe == '1H':
                return df.resample('1h').agg(agg_dict).dropna()
            elif timeframe == '4H':
                return df.resample('4h').agg(agg_dict).dropna()
            elif timeframe == '1D':
                return df.resample('1D').agg(agg_dict).dropna()
            elif timeframe == '1W':
                return df.resample('W').agg(agg_dict).dropna()
            elif timeframe == '1M':
                return df.resample('ME').agg(agg_dict).dropna()
            else:  # Default to daily
                return df
        except Exception as e:
            logger.warning(f"Error resampling to {timeframe}, using original data: {e}")
            return df
            
    def _get_common_indicators(self, timeframe_signals: Dict[str, Dict]) -> List[str]:
        """Get indicators that appear in all timeframes"""
        if not timeframe_signals:
            return []
            
        common_indicators = set(next(iter(timeframe_signals.values())).keys())
        for signals in timeframe_signals.values():
            common_indicators &= set(signals.keys())
            
        return list(common_indicators)
    
    def _calculate_consistency_score(self, indicator: str, 
                                   timeframe_signals: Dict[str, Dict]) -> float:
        """Calculate consistency score for an indicator across timeframes"""
        signals = []
        for tf_signals in timeframe_signals.values():
            if indicator in tf_signals:
                signal = tf_signals[indicator]
                if hasattr(signal, 'direction'):
                    signals.append(signal.direction)
                elif isinstance(signal, str):
                    signals.append(signal)
                    
        if len(signals) < 2:
            return 0.0
            
        # Calculate agreement percentage
        buy_count = signals.count('buy')
        sell_count = signals.count('sell')
        neutral_count = signals.count('neutral')
        
        max_agreement = max(buy_count, sell_count, neutral_count)
        return max_agreement / len(signals)
    
    def _calculate_enhanced_consistency_score(self, indicator: str, 
                                            timeframe_signals: Dict[str, Dict]) -> float:
        """Calculate enhanced consistency score with timeframe weighting and strength consideration"""
        weighted_signals = []
        total_weight = 0
        
        for timeframe, tf_signals in timeframe_signals.items():
            if indicator in tf_signals:
                signal = tf_signals[indicator]
                weight = self.timeframe_weights.get(timeframe, 0.1)
                
                if hasattr(signal, 'direction') and hasattr(signal, 'strength'):
                    # Weight by both timeframe importance and signal strength
                    effective_weight = weight * signal.strength
                    weighted_signals.append({
                        'direction': signal.direction,
                        'weight': effective_weight,
                        'timeframe': timeframe
                    })
                    total_weight += effective_weight
                elif isinstance(signal, str):
                    weighted_signals.append({
                        'direction': signal,
                        'weight': weight,
                        'timeframe': timeframe
                    })
                    total_weight += weight
                    
        if len(weighted_signals) < 2 or total_weight == 0:
            return 0.0
            
        # Calculate weighted agreement
        direction_weights = {'buy': 0, 'sell': 0, 'neutral': 0}
        for signal_data in weighted_signals:
            direction = signal_data['direction']
            weight = signal_data['weight']
            if direction in direction_weights:
                direction_weights[direction] += weight
                
        # Find the direction with highest weighted agreement
        max_weighted_agreement = max(direction_weights.values())
        consistency_score = max_weighted_agreement / total_weight
        
        # Bonus for agreement across different timeframe categories
        timeframe_categories = self._categorize_timeframes(weighted_signals)
        if len(timeframe_categories) >= 2:  # Agreement across short and long term
            consistency_score *= 1.1
            
        return min(consistency_score, 1.0)
    
    def _categorize_timeframes(self, weighted_signals: List[Dict]) -> set:
        """Categorize timeframes into short-term, medium-term, and long-term"""
        categories = set()
        for signal_data in weighted_signals:
            timeframe = signal_data['timeframe']
            if timeframe in ['5m', '15m']:
                categories.add('short_term')
            elif timeframe in ['1H', '4H']:
                categories.add('medium_term')
            elif timeframe in ['1D', '1W', '1M']:
                categories.add('long_term')
        return categories
    
    def _cross_timeframe_validation(self, timeframe_signals: Dict[str, Dict]) -> Dict[str, Any]:
        """Perform cross-timeframe signal validation"""
        validation_results = {
            'trend_alignment': {},
            'momentum_confirmation': {},
            'support_resistance_confluence': {},
            'overall_validation_score': 0.0
        }
        
        try:
            # Check trend alignment across timeframes
            trend_alignment = self._validate_trend_alignment(timeframe_signals)
            validation_results['trend_alignment'] = trend_alignment
            
            # Check momentum confirmation
            momentum_confirmation = self._validate_momentum_confirmation(timeframe_signals)
            validation_results['momentum_confirmation'] = momentum_confirmation
            
            # Check support/resistance confluence
            sr_confluence = self._validate_sr_confluence(timeframe_signals)
            validation_results['support_resistance_confluence'] = sr_confluence
            
            # Calculate overall validation score
            scores = [
                trend_alignment.get('score', 0),
                momentum_confirmation.get('score', 0),
                sr_confluence.get('score', 0)
            ]
            validation_results['overall_validation_score'] = np.mean(scores)
            
        except Exception as e:
            logger.error(f"Error in cross-timeframe validation: {e}")
            
        return validation_results
    
    def _validate_trend_alignment(self, timeframe_signals: Dict[str, Dict]) -> Dict[str, Any]:
        """Validate trend alignment across timeframes"""
        trend_indicators = ['MACD', 'Ichimoku']  # Trend-following indicators
        alignment_score = 0.0
        aligned_signals = 0
        total_signals = 0
        
        for indicator in trend_indicators:
            timeframe_directions = []
            for tf_signals in timeframe_signals.values():
                if indicator in tf_signals:
                    signal = tf_signals[indicator]
                    direction = signal.direction if hasattr(signal, 'direction') else signal
                    if direction != 'neutral':
                        timeframe_directions.append(direction)
                        
            if len(timeframe_directions) >= 2:
                # Check if majority agree
                buy_count = timeframe_directions.count('buy')
                sell_count = timeframe_directions.count('sell')
                majority_agreement = max(buy_count, sell_count) / len(timeframe_directions)
                
                if majority_agreement >= 0.6:  # 60% agreement threshold
                    aligned_signals += 1
                    alignment_score += majority_agreement
                    
                total_signals += 1
                
        final_score = alignment_score / total_signals if total_signals > 0 else 0.0
        
        return {
            'score': final_score,
            'aligned_indicators': aligned_signals,
            'total_indicators': total_signals,
            'alignment_strength': 'strong' if final_score > 0.8 else 'moderate' if final_score > 0.6 else 'weak'
        }
    
    def _validate_momentum_confirmation(self, timeframe_signals: Dict[str, Dict]) -> Dict[str, Any]:
        """Validate momentum confirmation across timeframes"""
        momentum_indicators = ['RSI', 'MACD']
        confirmation_score = 0.0
        confirmed_signals = 0
        
        # Check if short-term momentum aligns with longer-term trend
        short_term_tfs = ['5m', '15m', '1H']
        long_term_tfs = ['4H', '1D', '1W']
        
        for indicator in momentum_indicators:
            short_term_signals = []
            long_term_signals = []
            
            for tf, tf_signals in timeframe_signals.items():
                if indicator in tf_signals:
                    signal = tf_signals[indicator]
                    direction = signal.direction if hasattr(signal, 'direction') else signal
                    
                    if tf in short_term_tfs:
                        short_term_signals.append(direction)
                    elif tf in long_term_tfs:
                        long_term_signals.append(direction)
                        
            # Check for confirmation between short and long term
            if short_term_signals and long_term_signals:
                # Get dominant direction in each timeframe group
                short_dominant = max(set(short_term_signals), key=short_term_signals.count)
                long_dominant = max(set(long_term_signals), key=long_term_signals.count)
                
                if short_dominant == long_dominant and short_dominant != 'neutral':
                    confirmed_signals += 1
                    confirmation_score += 1.0
                    
        final_score = confirmation_score / len(momentum_indicators) if momentum_indicators else 0.0
        
        return {
            'score': final_score,
            'confirmed_indicators': confirmed_signals,
            'confirmation_strength': 'strong' if final_score > 0.7 else 'moderate' if final_score > 0.4 else 'weak'
        }
    
    def _validate_sr_confluence(self, timeframe_signals: Dict[str, Dict]) -> Dict[str, Any]:
        """Validate support/resistance confluence across timeframes"""
        sr_score = 0.0
        
        # Check if support/resistance signals align across timeframes
        sr_signals = []
        for tf_signals in timeframe_signals.values():
            if 'SupportResistance' in tf_signals:
                signal = tf_signals['SupportResistance']
                direction = signal.direction if hasattr(signal, 'direction') else signal
                if direction != 'neutral':
                    sr_signals.append(direction)
                    
        if len(sr_signals) >= 2:
            # Calculate agreement
            buy_count = sr_signals.count('buy')
            sell_count = sr_signals.count('sell')
            agreement = max(buy_count, sell_count) / len(sr_signals)
            sr_score = agreement
            
        return {
            'score': sr_score,
            'signal_count': len(sr_signals),
            'confluence_strength': 'strong' if sr_score > 0.8 else 'moderate' if sr_score > 0.6 else 'weak'
        }
    
    def _calculate_timeframe_strength(self, timeframe_signals: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate overall signal strength for each timeframe"""
        timeframe_strength = {}
        
        for timeframe, tf_signals in timeframe_signals.items():
            total_strength = 0.0
            signal_count = 0
            
            for signal in tf_signals.values():
                if hasattr(signal, 'strength') and hasattr(signal, 'confidence'):
                    # Combine strength and confidence
                    combined_strength = (signal.strength + signal.confidence) / 2
                    total_strength += combined_strength
                    signal_count += 1
                    
            avg_strength = total_strength / signal_count if signal_count > 0 else 0.0
            timeframe_strength[timeframe] = avg_strength
            
        return timeframe_strength
    
    def _generate_consensus_signals(self, timeframe_signals: Dict[str, Dict],
                                  consistency_scores: Dict[str, float]) -> Dict[str, Signal]:
        """Generate consensus signals based on timeframe agreement"""
        consensus_signals = {}
        
        for indicator, consistency in consistency_scores.items():
            if consistency >= self.consistency_threshold:
                # Find the most common signal
                signals = []
                strengths = []
                confidences = []
                
                for tf_signals in timeframe_signals.values():
                    if indicator in tf_signals:
                        signal = tf_signals[indicator]
                        if hasattr(signal, 'direction'):
                            signals.append(signal.direction)
                            strengths.append(signal.strength)
                            confidences.append(signal.confidence)
                        elif isinstance(signal, str):
                            signals.append(signal)
                            strengths.append(0.5)  # Default strength
                            confidences.append(consistency)  # Use consistency as confidence
                            
                if signals:
                    # Find most common signal
                    from collections import Counter
                    signal_counts = Counter(signals)
                    most_common_signal = signal_counts.most_common(1)[0][0]
                    
                    # Average strength and confidence
                    avg_strength = np.mean(strengths) if strengths else 0.5
                    avg_confidence = np.mean(confidences) if confidences else consistency
                    
                    consensus_signals[indicator] = Signal(
                        indicator=indicator,
                        direction=most_common_signal,
                        strength=avg_strength,
                        confidence=avg_confidence * consistency,  # Boost by consistency
                        timestamp=datetime.now(),
                        timeframe='consensus',
                        metadata={'consistency_score': consistency}
                    )
                    
        return consensus_signals


class EnhancedTechnicalAnalysisEngine:
    """Main engine for enhanced technical analysis with quality control"""
    
    def __init__(self):
        self.signal_quality_filter = SignalQualityFilter()
        self.performance_tracker = IndicatorPerformanceTracker()
        self.multi_timeframe_analyzer = MultiTimeframeAnalyzer()
        
        # Import existing technical analysis components
        from agents.technical_analysis import (
            AdaptiveParameterCalculator,
            TrendStrengthScorer,
            IchimokuCloud,
            FibonacciRetracement,
            SupportResistanceCalculator,
            VWAPAnalyzer
        )
        
        self.adaptive_calc = AdaptiveParameterCalculator()
        self.trend_scorer = TrendStrengthScorer()
        self.ichimoku = IchimokuCloud()
        self.fibonacci = FibonacciRetracement()
        self.support_resistance = SupportResistanceCalculator()
        self.vwap_analyzer = VWAPAnalyzer()
        
    def analyze_with_quality_control(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Perform technical analysis with quality control and filtering"""
        try:
            # Get market context for quality scoring
            market_context = self._get_market_context(df)
            
            # Perform multi-timeframe analysis
            mtf_results = self.multi_timeframe_analyzer.analyze_multi_timeframe(
                df, lambda d, s: self._analyze_single_timeframe(d, s), symbol
            )
            
            # Filter signals based on quality
            filtered_signals = {}
            for indicator, signal in mtf_results['consensus_signals'].items():
                quality_score = self.signal_quality_filter.score_signal_quality(
                    signal, market_context
                )
                if quality_score >= 0.5:  # Quality threshold
                    signal.metadata['quality_score'] = quality_score
                    filtered_signals[indicator] = signal
                    
            # Get dynamic weights based on performance
            dynamic_weights = self.get_dynamic_weights(
                market_context.get('regime', 'unknown'),
                market_context.get('volatility', 0.02)
            )
            
            return {
                'signals': filtered_signals,
                'market_context': market_context,
                'dynamic_weights': dynamic_weights,
                'timeframe_analysis': mtf_results,
                'quality_metrics': self._calculate_quality_metrics(filtered_signals)
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced technical analysis: {e}")
            return {
                'signals': {},
                'market_context': {},
                'dynamic_weights': {},
                'error': str(e)
            }
    
    def _analyze_single_timeframe(self, df: pd.DataFrame, symbol: str) -> Dict[str, Signal]:
        """Analyze a single timeframe and return Signal objects"""
        signals = {}
        
        try:
            # RSI Analysis
            rsi_signal = self._calculate_rsi_signal(df)
            if rsi_signal:
                signals['RSI'] = rsi_signal
                
            # MACD Analysis
            macd_signal = self._calculate_macd_signal(df)
            if macd_signal:
                signals['MACD'] = macd_signal
                
            # Ichimoku Analysis
            ichimoku_signal = self._calculate_ichimoku_signal(df)
            if ichimoku_signal:
                signals['Ichimoku'] = ichimoku_signal
                
            # Support/Resistance Analysis
            sr_signal = self._calculate_sr_signal(df)
            if sr_signal:
                signals['SupportResistance'] = sr_signal
                
            # VWAP Analysis
            vwap_signal = self._calculate_vwap_signal(df)
            if vwap_signal:
                signals['VWAP'] = vwap_signal
                
        except Exception as e:
            logger.error(f"Error in single timeframe analysis: {e}")
            
        return signals
    
    def _get_market_context(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get market context for quality scoring"""
        context = {}
        
        try:
            # Calculate volatility (ATR normalized)
            atr = self.adaptive_calc.calculate_atr(df)
            if len(atr) > 0 and not pd.isna(atr.iloc[-1]):
                context['volatility'] = atr.iloc[-1] / df['Close'].iloc[-1]
            else:
                context['volatility'] = 0.02
                
            # Calculate trend strength
            context['trend_strength'] = self.trend_scorer.score_trend_strength(df)
            
            # Determine market regime (simplified)
            if context['volatility'] > 0.05:
                context['regime'] = 'high_volatility'
            elif context['trend_strength'] > 0.7:
                context['regime'] = 'trending'
            elif context['trend_strength'] < 0.3:
                context['regime'] = 'ranging'
            else:
                context['regime'] = 'normal'
                
        except Exception as e:
            logger.error(f"Error calculating market context: {e}")
            context = {'volatility': 0.02, 'trend_strength': 0.5, 'regime': 'unknown'}
            
        return context
    
    def get_dynamic_weights(self, market_regime: str, volatility: float) -> Dict[str, float]:
        """Get dynamic weights for indicators based on performance and market conditions"""
        base_indicators = ['RSI', 'MACD', 'Ichimoku', 'SupportResistance', 'VWAP']
        weights = {}
        
        for indicator in base_indicators:
            # Get performance-based weight
            perf_weight = self.performance_tracker.get_indicator_weight(indicator, market_regime)
            
            # Apply market condition adjustments
            condition_weight = self._calculate_market_condition_weight(
                indicator, market_regime, volatility
            )
            
            # Combine performance and condition weights
            combined_weight = perf_weight * condition_weight
            weights[indicator] = combined_weight
            
        # Apply advanced weighting algorithms
        weights = self._apply_advanced_weighting_algorithms(weights, market_regime, volatility)
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
            
        return weights
    
    def _calculate_market_condition_weight(self, indicator: str, market_regime: str, 
                                         volatility: float) -> float:
        """Calculate market condition-based weight adjustments"""
        base_weight = 1.0
        
        # Regime-based adjustments
        if market_regime == 'trending':
            if indicator in ['MACD', 'Ichimoku']:
                base_weight *= 1.3  # Strong boost for trend-following indicators
            elif indicator == 'RSI':
                base_weight *= 0.7  # Reduce oscillator weight in trends
            elif indicator == 'VWAP':
                base_weight *= 1.1  # Moderate boost for VWAP in trends
                
        elif market_regime == 'ranging':
            if indicator == 'RSI':
                base_weight *= 1.4  # Strong boost for oscillators in ranging markets
            elif indicator in ['MACD', 'Ichimoku']:
                base_weight *= 0.6  # Reduce trend indicators
            elif indicator == 'SupportResistance':
                base_weight *= 1.2  # Boost S/R in ranging markets
                
        elif market_regime == 'high_volatility':
            if indicator in ['SupportResistance', 'VWAP']:
                base_weight *= 1.2  # Boost level-based indicators
            elif indicator == 'RSI':
                base_weight *= 0.8  # Reduce RSI in high volatility
            elif indicator == 'MACD':
                base_weight *= 0.9  # Slightly reduce MACD
                
        elif market_regime == 'low_volatility':
            if indicator in ['MACD', 'Ichimoku']:
                base_weight *= 1.1  # Boost trend indicators in stable markets
            elif indicator == 'RSI':
                base_weight *= 1.0  # Neutral for RSI
                
        # Volatility-based fine-tuning
        if volatility > 0.05:  # High volatility
            if indicator in ['SupportResistance', 'VWAP']:
                base_weight *= 1.1
            else:
                base_weight *= 0.95
        elif volatility < 0.01:  # Low volatility
            if indicator in ['MACD', 'Ichimoku']:
                base_weight *= 1.05
                
        return base_weight
    
    def _apply_advanced_weighting_algorithms(self, weights: Dict[str, float], 
                                           market_regime: str, volatility: float) -> Dict[str, float]:
        """Apply advanced weighting algorithms including ensemble methods"""
        enhanced_weights = weights.copy()
        
        # Apply momentum-based weighting
        enhanced_weights = self._apply_momentum_weighting(enhanced_weights)
        
        # Apply correlation-based adjustments
        enhanced_weights = self._apply_correlation_adjustments(enhanced_weights)
        
        # Apply recency weighting
        enhanced_weights = self._apply_recency_weighting(enhanced_weights)
        
        # Apply ensemble diversity weighting
        enhanced_weights = self._apply_diversity_weighting(enhanced_weights)
        
        return enhanced_weights
    
    def _apply_momentum_weighting(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply momentum-based weighting - boost indicators with recent good performance"""
        momentum_weights = weights.copy()
        
        for indicator in weights.keys():
            if indicator in self.performance_tracker.performance_data:
                perf_data = self.performance_tracker.performance_data[indicator]
                
                # Check recent performance trend (last 30 days)
                days_since_update = (datetime.now() - perf_data.last_updated).days
                if days_since_update <= 30:
                    # Recent good performance gets momentum boost
                    if perf_data.accuracy > 0.6:
                        momentum_boost = 1.0 + (perf_data.accuracy - 0.6) * 0.5
                        momentum_weights[indicator] *= momentum_boost
                    elif perf_data.accuracy < 0.4:
                        # Recent poor performance gets penalty
                        momentum_penalty = 1.0 - (0.4 - perf_data.accuracy) * 0.3
                        momentum_weights[indicator] *= max(momentum_penalty, 0.5)
                        
        return momentum_weights
    
    def _apply_correlation_adjustments(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply correlation-based adjustments to reduce redundancy"""
        correlation_weights = weights.copy()
        
        # Define indicator correlations (simplified)
        high_correlation_pairs = [
            ('MACD', 'Ichimoku'),  # Both trend-following
            ('RSI', 'SupportResistance')  # Both mean-reversion oriented
        ]
        
        for indicator1, indicator2 in high_correlation_pairs:
            if indicator1 in correlation_weights and indicator2 in correlation_weights:
                # If both indicators have high weights, reduce the weaker one
                if correlation_weights[indicator1] > 0.2 and correlation_weights[indicator2] > 0.2:
                    if correlation_weights[indicator1] > correlation_weights[indicator2]:
                        correlation_weights[indicator2] *= 0.8
                    else:
                        correlation_weights[indicator1] *= 0.8
                        
        return correlation_weights
    
    def _apply_recency_weighting(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply recency weighting - favor indicators with recent updates"""
        recency_weights = weights.copy()
        
        for indicator in weights.keys():
            if indicator in self.performance_tracker.performance_data:
                perf_data = self.performance_tracker.performance_data[indicator]
                days_since_update = (datetime.now() - perf_data.last_updated).days
                
                # Apply recency decay
                if days_since_update > 90:  # Very old data
                    recency_weights[indicator] *= 0.7
                elif days_since_update > 30:  # Moderately old data
                    recency_weights[indicator] *= 0.85
                elif days_since_update <= 7:  # Very recent data
                    recency_weights[indicator] *= 1.1
                    
        return recency_weights
    
    def _apply_diversity_weighting(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply diversity weighting to ensure balanced indicator types"""
        diversity_weights = weights.copy()
        
        # Categorize indicators by type
        trend_indicators = ['MACD', 'Ichimoku']
        oscillator_indicators = ['RSI']
        level_indicators = ['SupportResistance', 'VWAP']
        
        # Calculate total weight by category
        trend_weight = sum(diversity_weights.get(ind, 0) for ind in trend_indicators)
        oscillator_weight = sum(diversity_weights.get(ind, 0) for ind in oscillator_indicators)
        level_weight = sum(diversity_weights.get(ind, 0) for ind in level_indicators)
        
        total_weight = trend_weight + oscillator_weight + level_weight
        
        if total_weight > 0:
            # Target distribution: 40% trend, 30% oscillator, 30% level
            target_trend = 0.4
            target_oscillator = 0.3
            target_level = 0.3
            
            # Adjust if any category is over/under-represented
            if trend_weight / total_weight > target_trend + 0.1:
                # Reduce trend indicators
                for ind in trend_indicators:
                    if ind in diversity_weights:
                        diversity_weights[ind] *= 0.9
                        
            elif oscillator_weight / total_weight < target_oscillator - 0.1:
                # Boost oscillator indicators
                for ind in oscillator_indicators:
                    if ind in diversity_weights:
                        diversity_weights[ind] *= 1.2
                        
            elif level_weight / total_weight < target_level - 0.1:
                # Boost level indicators
                for ind in level_indicators:
                    if ind in diversity_weights:
                        diversity_weights[ind] *= 1.1
                        
        return diversity_weights
    
    def calculate_adaptive_weights(self, signals: Dict[str, Signal], 
                                 market_context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate adaptive weights based on current signals and market context"""
        if not signals:
            return {}
            
        adaptive_weights = {}
        
        # Base weights from dynamic weighting
        base_weights = self.get_dynamic_weights(
            market_context.get('regime', 'unknown'),
            market_context.get('volatility', 0.02)
        )
        
        for indicator, signal in signals.items():
            base_weight = base_weights.get(indicator, 0.2)
            
            # Adjust based on signal quality
            quality_score = signal.metadata.get('quality_score', 0.5)
            quality_adjustment = 0.5 + quality_score  # Range: 0.5 to 1.5
            
            # Adjust based on signal strength and confidence
            signal_strength_adjustment = (signal.strength + signal.confidence) / 2
            
            # Combine adjustments
            final_weight = base_weight * quality_adjustment * signal_strength_adjustment
            adaptive_weights[indicator] = final_weight
            
        # Normalize
        total_weight = sum(adaptive_weights.values())
        if total_weight > 0:
            adaptive_weights = {k: v / total_weight for k, v in adaptive_weights.items()}
            
        return adaptive_weights
    
    def _calculate_quality_metrics(self, signals: Dict[str, Signal]) -> Dict[str, Any]:
        """Calculate overall quality metrics for the signal set"""
        if not signals:
            return {'overall_quality': 0.0, 'signal_count': 0}
            
        quality_scores = [signal.metadata.get('quality_score', 0.5) for signal in signals.values()]
        confidence_scores = [signal.confidence for signal in signals.values()]
        
        return {
            'overall_quality': np.mean(quality_scores),
            'avg_confidence': np.mean(confidence_scores),
            'signal_count': len(signals),
            'high_quality_signals': sum(1 for score in quality_scores if score > 0.7)
        }
    
    # Signal calculation methods (simplified implementations)
    def _calculate_rsi_signal(self, df: pd.DataFrame) -> Optional[Signal]:
        """Calculate RSI signal with quality metrics"""
        try:
            # Use adaptive RSI period
            rsi_period = self.adaptive_calc.adaptive_rsi_period(df)
            
            # Calculate RSI (simplified implementation)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            if len(rsi) == 0 or pd.isna(rsi.iloc[-1]):
                return None
                
            current_rsi = rsi.iloc[-1]
            
            # Determine signal
            if current_rsi < 30:
                direction = 'buy'
                strength = (30 - current_rsi) / 30
            elif current_rsi > 70:
                direction = 'sell'
                strength = (current_rsi - 70) / 30
            else:
                direction = 'neutral'
                strength = 0.0
                
            # Calculate confidence based on RSI extremes
            confidence = min(abs(current_rsi - 50) / 50, 1.0)
            
            return Signal(
                indicator='RSI',
                direction=direction,
                strength=strength,
                confidence=confidence,
                timestamp=datetime.now(),
                timeframe='current',
                metadata={'rsi_value': current_rsi, 'period': rsi_period}
            )
            
        except Exception as e:
            logger.error(f"Error calculating RSI signal: {e}")
            return None
    
    def _calculate_macd_signal(self, df: pd.DataFrame, symbol: str = "") -> Optional[Signal]:
        """Calculate MACD signal with quality metrics, matching TALib"""
        try:
            # Use adaptive MACD periods, apply India params if .NS
            is_indian = symbol.endswith('.NS')
            if is_indian:
                periods = {'fast': 8, 'slow': 17, 'signal': 9}
            else:
                periods = self.adaptive_calc.adaptive_macd_periods(df)
            
            # Calculate MACD using TALib-matching EMA
            close_series = df['Close']
            ema_fast = calculate_ema_talib(close_series, periods['fast'])
            ema_slow = calculate_ema_talib(close_series, periods['slow'])
            macd_line = ema_fast - ema_slow
            signal_line = calculate_ema_talib(macd_line, periods['signal'])
            histogram = macd_line - signal_line
            
            if len(histogram) < 2 or pd.isna(histogram.iloc[-1]) or pd.isna(histogram.iloc[-2]):
                return None
                
            current_hist = histogram.iloc[-1]
            prev_hist = histogram.iloc[-2]
            
            # Determine signal based on histogram crossover
            if current_hist > 0 and prev_hist <= 0:
                direction = 'buy'
                strength = min(abs(current_hist) / df['Close'].iloc[-1] * 100, 1.0)
            elif current_hist < 0 and prev_hist >= 0:
                direction = 'sell'
                strength = min(abs(current_hist) / df['Close'].iloc[-1] * 100, 1.0)
            else:
                direction = 'neutral'
                strength = 0.0
                
            # Confidence based on histogram magnitude, penalize if NaN propagated
            if pd.isna(current_hist):
                confidence = 0.0
            else:
                confidence = min(abs(current_hist) / (df['Close'].iloc[-1] * 0.01), 1.0)
            
            return Signal(
                indicator='MACD',
                direction=direction,
                strength=strength,
                confidence=confidence,
                timestamp=datetime.now(),
                timeframe='current',
                metadata={
                    'macd_line': macd_line.iloc[-1],
                    'signal_line': signal_line.iloc[-1],
                    'histogram': current_hist,
                    'periods': periods,
                    'is_indian': is_indian
                }
            )
            
        except Exception as e:
            logger.error(f"Error calculating MACD signal: {e}")
            return None
    
    def _calculate_ichimoku_signal(self, df: pd.DataFrame) -> Optional[Signal]:
        """Calculate Ichimoku signal with quality metrics"""
        try:
            ichimoku_signal = self.ichimoku.get_ichimoku_signal(df)
            
            if ichimoku_signal == 'neutral':
                return None
                
            # Calculate strength based on cloud thickness and price position
            ichimoku_data = self.ichimoku.calculate_ichimoku(df)
            if not ichimoku_data:
                return None
                
            close = df['Close'].iloc[-1]
            senkou_a = ichimoku_data['senkou_span_a'].iloc[-1]
            senkou_b = ichimoku_data['senkou_span_b'].iloc[-1]
            
            if pd.isna(senkou_a) or pd.isna(senkou_b):
                return None
                
            cloud_thickness = abs(senkou_a - senkou_b) / close
            strength = min(cloud_thickness * 10, 1.0)  # Scale cloud thickness
            
            # Confidence based on price distance from cloud
            cloud_top = max(senkou_a, senkou_b)
            cloud_bottom = min(senkou_a, senkou_b)
            
            if close > cloud_top:
                confidence = min((close - cloud_top) / close, 1.0)
            elif close < cloud_bottom:
                confidence = min((cloud_bottom - close) / close, 1.0)
            else:
                confidence = 0.3  # Inside cloud - lower confidence
                
            return Signal(
                indicator='Ichimoku',
                direction=ichimoku_signal,
                strength=strength,
                confidence=confidence,
                timestamp=datetime.now(),
                timeframe='current',
                metadata={
                    'cloud_thickness': cloud_thickness,
                    'price_position': 'above' if close > cloud_top else 'below' if close < cloud_bottom else 'inside'
                }
            )
            
        except Exception as e:
            logger.error(f"Error calculating Ichimoku signal: {e}")
            return None
    
    def _calculate_sr_signal(self, df: pd.DataFrame) -> Optional[Signal]:
        """Calculate Support/Resistance signal with quality metrics"""
        try:
            sr_signal = self.support_resistance.get_sr_signal(df)
            
            if sr_signal == 'neutral':
                return None
                
            # Get support/resistance levels
            sr_levels = self.support_resistance.calculate_support_resistance(df)
            if not sr_levels:
                return None
                
            close = df['Close'].iloc[-1]
            
            # Find closest level
            all_levels = list(sr_levels.values())
            closest_level = min(all_levels, key=lambda x: abs(x - close))
            distance_pct = abs(close - closest_level) / close
            
            # Strength based on proximity to level
            strength = max(0, 1.0 - (distance_pct * 50))  # Closer = stronger
            
            # Confidence based on level significance
            confidence = 0.7 if distance_pct < 0.02 else 0.4  # High confidence near levels
            
            return Signal(
                indicator='SupportResistance',
                direction=sr_signal,
                strength=strength,
                confidence=confidence,
                timestamp=datetime.now(),
                timeframe='current',
                metadata={
                    'closest_level': closest_level,
                    'distance_pct': distance_pct,
                    'levels': sr_levels
                }
            )
            
        except Exception as e:
            logger.error(f"Error calculating S/R signal: {e}")
            return None
    
    def _calculate_vwap_signal(self, df: pd.DataFrame) -> Optional[Signal]:
        """Calculate VWAP signal with quality metrics"""
        try:
            vwap_signal = self.vwap_analyzer.get_vwap_signal(df)
            
            if vwap_signal == 'neutral':
                return None
                
            # Calculate VWAP
            vwap = self.vwap_analyzer.calculate_vwap(df)
            if len(vwap) == 0 or pd.isna(vwap.iloc[-1]):
                return None
                
            close = df['Close'].iloc[-1]
            vwap_value = vwap.iloc[-1]
            deviation_pct = abs(close - vwap_value) / vwap_value
            
            # Strength based on deviation from VWAP
            strength = min(deviation_pct * 20, 1.0)  # Scale deviation
            
            # Confidence based on volume and deviation
            avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
            current_volume = df['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            confidence = min(0.5 + (volume_ratio * 0.3), 1.0)  # Higher volume = higher confidence
            
            return Signal(
                indicator='VWAP',
                direction=vwap_signal,
                strength=strength,
                confidence=confidence,
                timestamp=datetime.now(),
                timeframe='current',
                metadata={
                    'vwap_value': vwap_value,
                    'deviation_pct': deviation_pct,
                    'volume_ratio': volume_ratio
                }
            )
            
        except Exception as e:
            logger.error(f"Error calculating VWAP signal: {e}")
            return None