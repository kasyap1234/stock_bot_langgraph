"""
Enhanced Performance Monitoring System

This module provides real-time performance tracking, analytics, and monitoring
capabilities for the stock trading bot with adaptive feedback mechanisms.
"""

import logging
import asyncio
import threading
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

from analysis.performance_analyzer import PerformanceAnalyzer
from integration.market_analysis_integration import AdaptiveMarketAnalyzer, AdaptiveSignal
from config.config import DEBUG_RECOMMENDATION_LOGGING

logger = logging.getLogger(__name__)


class PerformanceMetric(Enum):
    """Performance metric types"""
    WIN_RATE = "win_rate"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    PROFIT_FACTOR = "profit_factor"
    EXPECTANCY = "expectancy"
    CALMAR_RATIO = "calmar_ratio"


@dataclass
class SignalPerformance:
    """Performance tracking for individual signals"""
    signal_id: str
    symbol: str
    action: str
    entry_price: float
    entry_time: datetime
    confidence: float
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    holding_period: Optional[timedelta] = None
    market_regime: str = "unknown"
    volatility_regime: str = "unknown"
    signal_strength: str = "unknown"
    success: Optional[bool] = None
    notes: str = ""


@dataclass
class PredictionOutcome:
    """Tracks actual vs predicted outcomes"""
    prediction_id: str
    symbol: str
    predicted_action: str
    predicted_price_target: Optional[float]
    predicted_probability: float
    prediction_time: datetime
    actual_price: Optional[float] = None
    actual_outcome: Optional[str] = None  # 'correct', 'incorrect', 'partial'
    outcome_time: Optional[datetime] = None
    accuracy_score: Optional[float] = None  # 0.0 to 1.0
    prediction_error: Optional[float] = None
    market_conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelDriftMetrics:
    """Tracks model drift indicators"""
    model_name: str
    baseline_accuracy: float
    current_accuracy: float
    accuracy_trend: List[float] = field(default_factory=list)
    drift_score: float = 0.0
    last_retrain_date: Optional[datetime] = None
    drift_detected: bool = False
    drift_threshold: float = 0.05  # 5% accuracy drop threshold


@dataclass
class BiasDetectionResult:
    """Results from systematic bias detection"""
    bias_type: str
    affected_symbols: List[str]
    bias_magnitude: float
    confidence: float
    detection_time: datetime
    corrective_action: str
    description: str


@dataclass
class StrategyPerformance:
    """Performance metrics for trading strategies"""
    strategy_name: str
    total_signals: int = 0
    winning_signals: int = 0
    losing_signals: int = 0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    expectancy: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    # Enhanced tracking
    prediction_accuracy: float = 0.0
    model_drift_score: float = 0.0
    bias_indicators: Dict[str, float] = field(default_factory=dict)
    performance_trend: List[float] = field(default_factory=list)


class EnhancedPerformanceMonitor:
    """Enhanced performance monitoring with real-time analytics"""
    
    def __init__(self):
        self.signals: Dict[str, List[SignalPerformance]] = {}
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        self.performance_analyzer = PerformanceAnalyzer()
        self.market_analyzer: Optional[AdaptiveMarketAnalyzer] = None
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.callbacks: List[Callable] = []
        self.performance_cache: Dict[str, Any] = {}
        self.alert_thresholds = self._initialize_alert_thresholds()
        self.performance_history: List[Dict[str, Any]] = []
        
        # Enhanced tracking capabilities
        self.prediction_outcomes: Dict[str, List[PredictionOutcome]] = {}
        self.model_drift_metrics: Dict[str, ModelDriftMetrics] = {}
        self.bias_detection_results: List[BiasDetectionResult] = []
        self.retraining_triggers: Dict[str, datetime] = {}
        self.performance_decline_threshold = 0.1  # 10% decline triggers alert
        self.drift_detection_window = 100  # Number of predictions to analyze for drift
        
    def _initialize_alert_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize alert thresholds for performance monitoring"""
        return {
            'win_rate': {'low': 0.4, 'high': 0.7},
            'sharpe_ratio': {'low': 0.5, 'high': 2.0},
            'max_drawdown': {'low': -0.05, 'high': -0.15},
            'profit_factor': {'low': 1.0, 'high': 2.0},
            'expectancy': {'low': 0.001, 'high': 0.01}
        }
    
    def set_market_analyzer(self, market_analyzer: AdaptiveMarketAnalyzer) -> None:
        """Set the market analyzer for enhanced performance tracking"""
        self.market_analyzer = market_analyzer
        if market_analyzer:
            market_analyzer.add_callback(self._process_adaptive_signal)
    
    def start_monitoring(self) -> None:
        """Start performance monitoring"""
        if self.monitoring_active:
            logger.warning("Performance monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Started enhanced performance monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Stopped performance monitoring")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Update performance metrics every 60 seconds
                self._update_performance_metrics()
                self._check_alert_conditions()
                time.sleep(60)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    def _process_adaptive_signal(self, signal: AdaptiveSignal) -> None:
        """Process adaptive signals for performance tracking"""
        try:
            # Create signal performance entry
            signal_perf = SignalPerformance(
                signal_id=f"{signal.symbol}_{int(signal.timestamp.timestamp())}",
                symbol=signal.symbol,
                action=signal.action,
                entry_price=0.0,  # Will be updated when trade is executed
                entry_time=signal.timestamp,
                confidence=signal.confidence,
                market_regime=signal.market_regime,
                volatility_regime=signal.volatility_regime,
                signal_strength=signal.signal_strength
            )
            
            # Store signal
            if signal.symbol not in self.signals:
                self.signals[signal.symbol] = []
            self.signals[signal.symbol].append(signal_perf)
            
            # Update strategy performance
            self._update_strategy_performance(signal.symbol, signal.action)
            
            logger.debug(f"Processed adaptive signal for {signal.symbol}: {signal.action}")
            
        except Exception as e:
            logger.error(f"Error processing adaptive signal: {e}")
    
    def record_trade_execution(self, symbol: str, action: str, price: float, 
                               quantity: int, timestamp: Optional[datetime] = None) -> None:
        """Record trade execution for performance tracking"""
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            # Find corresponding signal (most recent unexecuted signal)
            signal_perf = self._find_corresponding_signal(symbol, action, timestamp)
            
            if signal_perf:
                signal_perf.entry_price = price
                signal_perf.entry_time = timestamp
                logger.info(f"Recorded trade execution: {symbol} {action} @ {price}")
            else:
                # Create new signal performance entry for manual trades
                signal_perf = SignalPerformance(
                    signal_id=f"{symbol}_manual_{int(timestamp.timestamp())}",
                    symbol=symbol,
                    action=action,
                    entry_price=price,
                    entry_time=timestamp,
                    confidence=0.5,  # Default confidence for manual trades
                    notes="Manual trade"
                )
                
                if symbol not in self.signals:
                    self.signals[symbol] = []
                self.signals[symbol].append(signal_perf)
                
                logger.info(f"Recorded manual trade: {symbol} {action} @ {price}")
                
        except Exception as e:
            logger.error(f"Error recording trade execution: {e}")
    
    def record_trade_exit(self, symbol: str, exit_price: float, 
                         exit_time: Optional[datetime] = None) -> None:
        """Record trade exit for performance tracking"""
        try:
            if exit_time is None:
                exit_time = datetime.now()
            
            # Find most recent open position
            open_signals = [s for s in self.signals.get(symbol, []) if s.exit_price is None]
            
            if not open_signals:
                logger.warning(f"No open positions found for {symbol}")
                return
            
            # Update the most recent open position
            signal = open_signals[-1]
            signal.exit_price = exit_price
            signal.exit_time = exit_time
            signal.holding_period = exit_time - signal.entry_time
            
            # Calculate P&L
            if signal.action.upper() == "BUY":
                signal.pnl = exit_price - signal.entry_price
                signal.pnl_percent = (exit_price - signal.entry_price) / signal.entry_price
                signal.success = signal.pnl > 0
            else:  # SELL
                signal.pnl = signal.entry_price - exit_price
                signal.pnl_percent = (signal.entry_price - exit_price) / signal.entry_price
                signal.success = signal.pnl > 0
            
            logger.info(f"Recorded trade exit: {symbol} @ {exit_price}, P&L: {signal.pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error recording trade exit: {e}")
    
    def _find_corresponding_signal(self, symbol: str, action: str, 
                                  timestamp: datetime) -> Optional[SignalPerformance]:
        """Find the most recent unexecuted signal that matches the trade"""
        try:
            # Get recent signals (within last hour)
            recent_signals = [
                s for s in self.signals.get(symbol, [])
                if s.action.upper() == action.upper() and 
                s.entry_price == 0.0 and  # Not yet executed
                (timestamp - s.entry_time) <= timedelta(hours=1)
            ]
            
            # Return the most recent matching signal
            return max(recent_signals, key=lambda x: x.entry_time) if recent_signals else None
            
        except Exception as e:
            logger.error(f"Error finding corresponding signal: {e}")
            return None
    
    def _update_strategy_performance(self, symbol: str, action: str) -> None:
        """Update strategy performance metrics"""
        try:
            strategy_key = f"{symbol}_{action}"
            
            if strategy_key not in self.strategy_performance:
                self.strategy_performance[strategy_key] = StrategyPerformance(
                    strategy_name=strategy_key
                )
            
            strategy = self.strategy_performance[strategy_key]
            strategy.total_signals += 1
            strategy.last_updated = datetime.now()
            
            # Update win/loss counts and P&L from completed trades
            completed_signals = [
                s for s in self.signals.get(symbol, [])
                if s.action.upper() == action.upper() and s.success is not None
            ]
            
            if completed_signals:
                strategy.winning_signals = sum(1 for s in completed_signals if s.success)
                strategy.losing_signals = sum(1 for s in completed_signals if not s.success)
                strategy.total_pnl = sum(s.pnl for s in completed_signals if s.pnl is not None)
                
                # Calculate averages
                wins = [s.pnl for s in completed_signals if s.success and s.pnl is not None]
                losses = [s.pnl for s in completed_signals if not s.success and s.pnl is not None]
                
                strategy.avg_win = np.mean(wins) if wins else 0.0
                strategy.avg_loss = np.mean(losses) if losses else 0.0
                strategy.win_rate = strategy.winning_signals / len(completed_signals) if completed_signals else 0.0
                
                # Calculate profit factor
                total_wins = sum(wins) if wins else 0.0
                total_losses = abs(sum(losses)) if losses else 0.0
                strategy.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
                
                # Calculate expectancy
                strategy.expectancy = (strategy.win_rate * strategy.avg_win) - ((1 - strategy.win_rate) * abs(strategy.avg_loss))
            
        except Exception as e:
            logger.error(f"Error updating strategy performance: {e}")
    
    def _update_performance_metrics(self) -> None:
        """Update comprehensive performance metrics"""
        try:
            all_signals = []
            for symbol_signals in self.signals.values():
                all_signals.extend([s for s in symbol_signals if s.success is not None])
            
            if not all_signals:
                return
            
            # Calculate overall metrics
            total_signals = len(all_signals)
            winning_signals = sum(1 for s in all_signals if s.success)
            losing_signals = total_signals - winning_signals
            
            win_rate = winning_signals / total_signals if total_signals > 0 else 0.0
            total_pnl = sum(s.pnl for s in all_signals if s.pnl is not None)
            
            # Calculate Sharpe ratio (simplified)
            returns = [s.pnl_percent for s in all_signals if s.pnl_percent is not None]
            if len(returns) > 1:
                sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
            else:
                sharpe_ratio = 0.0
            
            # Calculate max drawdown
            cumulative_returns = np.cumsum(returns) if returns else np.array([])
            max_drawdown = self._calculate_max_drawdown(cumulative_returns)
            
            # Update performance cache
            self.performance_cache = {
                'timestamp': datetime.now(),
                'total_signals': total_signals,
                'winning_signals': winning_signals,
                'losing_signals': losing_signals,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'avg_holding_period': np.mean([s.holding_period.total_seconds()/3600 for s in all_signals if s.holding_period]) if all_signals else 0.0
            }
            
            # Add to performance history
            self.performance_history.append(self.performance_cache.copy())
            
            # Keep only last 1000 entries
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown from cumulative returns"""
        if len(cumulative_returns) == 0:
            return 0.0
        
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        return np.min(drawdown) if len(drawdown) > 0 else 0.0
    
    def _check_alert_conditions(self) -> None:
        """Check for alert conditions and notify callbacks"""
        try:
            if not self.performance_cache:
                return
            
            alerts = []
            
            # Check win rate
            win_rate = self.performance_cache.get('win_rate', 0.0)
            if win_rate < self.alert_thresholds['win_rate']['low']:
                alerts.append({
                    'type': 'low_win_rate',
                    'value': win_rate,
                    'threshold': self.alert_thresholds['win_rate']['low'],
                    'message': f"Low win rate: {win_rate:.2%}"
                })
            
            # Check Sharpe ratio
            sharpe = self.performance_cache.get('sharpe_ratio', 0.0)
            if sharpe < self.alert_thresholds['sharpe_ratio']['low']:
                alerts.append({
                    'type': 'low_sharpe_ratio',
                    'value': sharpe,
                    'threshold': self.alert_thresholds['sharpe_ratio']['low'],
                    'message': f"Low Sharpe ratio: {sharpe:.2f}"
                })
            
            # Check max drawdown
            drawdown = self.performance_cache.get('max_drawdown', 0.0)
            if drawdown < self.alert_thresholds['max_drawdown']['low']:
                alerts.append({
                    'type': 'high_drawdown',
                    'value': drawdown,
                    'threshold': self.alert_thresholds['max_drawdown']['low'],
                    'message': f"High drawdown: {drawdown:.2%}"
                })
            
            # Notify callbacks of alerts
            for alert in alerts:
                self._notify_alert(alert)
                
        except Exception as e:
            logger.error(f"Error checking alert conditions: {e}")
    
    def _notify_alert(self, alert: Dict[str, Any]) -> None:
        """Notify callbacks of alerts"""
        for callback in self.callbacks:
            try:
                callback('alert', alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def add_callback(self, callback: Callable) -> None:
        """Add callback for performance events"""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable) -> None:
        """Remove callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            'current_metrics': self.performance_cache,
            'strategy_performance': {k: v.__dict__ for k, v in self.strategy_performance.items()},
            'signal_summary': self._get_signal_summary(),
            'performance_history': self.performance_history[-50:],  # Last 50 entries
            'alert_thresholds': self.alert_thresholds
        }
    
    def _get_signal_summary(self) -> Dict[str, Any]:
        """Get summary of signal performance by various dimensions"""
        all_signals = []
        for symbol_signals in self.signals.values():
            all_signals.extend([s for s in symbol_signals if s.success is not None])
        
        if not all_signals:
            return {}
        
        # Summary by market regime
        regime_summary = {}
        for regime in ['bull', 'bear', 'sideways', 'high_volatility', 'low_volatility']:
            regime_signals = [s for s in all_signals if s.market_regime == regime]
            if regime_signals:
                regime_summary[regime] = {
                    'count': len(regime_signals),
                    'win_rate': sum(1 for s in regime_signals if s.success) / len(regime_signals),
                    'avg_pnl': np.mean([s.pnl for s in regime_signals if s.pnl is not None])
                }
        
        # Summary by signal strength
        strength_summary = {}
        for strength in ['strong', 'moderate', 'weak', 'uncertain']:
            strength_signals = [s for s in all_signals if s.signal_strength == strength]
            if strength_signals:
                strength_summary[strength] = {
                    'count': len(strength_signals),
                    'win_rate': sum(1 for s in strength_signals if s.success) / len(strength_signals),
                    'avg_pnl': np.mean([s.pnl for s in strength_signals if s.pnl is not None])
                }
        
        return {
            'regime_summary': regime_summary,
            'strength_summary': strength_summary,
            'total_completed_signals': len(all_signals)
        }
    
    def export_performance_data(self, filename: str = None) -> str:
        """Export performance data to CSV file"""
        try:
            if filename is None:
                filename = f"performance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            # Collect all signal data
            all_signals = []
            for symbol_signals in self.signals.values():
                all_signals.extend(symbol_signals)
            
            if not all_signals:
                logger.warning("No signal data to export")
                return ""
            
            # Convert to DataFrame
            data = []
            for signal in all_signals:
                data.append({
                    'signal_id': signal.signal_id,
                    'symbol': signal.symbol,
                    'action': signal.action,
                    'entry_price': signal.entry_price,
                    'entry_time': signal.entry_time,
                    'exit_price': signal.exit_price,
                    'exit_time': signal.exit_time,
                    'pnl': signal.pnl,
                    'pnl_percent': signal.pnl_percent,
                    'holding_period_hours': signal.holding_period.total_seconds()/3600 if signal.holding_period else None,
                    'confidence': signal.confidence,
                    'market_regime': signal.market_regime,
                    'volatility_regime': signal.volatility_regime,
                    'signal_strength': signal.signal_strength,
                    'success': signal.success,
                    'notes': signal.notes
                })
            
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            
            logger.info(f"Exported performance data to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting performance data: {e}")
            return ""
    
    # Enhanced Performance Tracking Methods
    
    def record_prediction(self, prediction_id: str, symbol: str, predicted_action: str,
                         predicted_price_target: Optional[float], predicted_probability: float,
                         market_conditions: Optional[Dict[str, Any]] = None) -> None:
        """Record a prediction for later outcome tracking"""
        try:
            prediction = PredictionOutcome(
                prediction_id=prediction_id,
                symbol=symbol,
                predicted_action=predicted_action,
                predicted_price_target=predicted_price_target,
                predicted_probability=predicted_probability,
                prediction_time=datetime.now(),
                market_conditions=market_conditions or {}
            )
            
            if symbol not in self.prediction_outcomes:
                self.prediction_outcomes[symbol] = []
            
            self.prediction_outcomes[symbol].append(prediction)
            
            # Keep only recent predictions (last 1000 per symbol)
            if len(self.prediction_outcomes[symbol]) > 1000:
                self.prediction_outcomes[symbol] = self.prediction_outcomes[symbol][-1000:]
            
            logger.debug(f"Recorded prediction {prediction_id} for {symbol}: {predicted_action}")
            
        except Exception as e:
            logger.error(f"Error recording prediction: {e}")
    
    def update_prediction_outcome(self, prediction_id: str, actual_price: float,
                                 actual_outcome: str, outcome_time: Optional[datetime] = None) -> None:
        """Update the actual outcome of a prediction"""
        try:
            if outcome_time is None:
                outcome_time = datetime.now()
            
            # Find the prediction
            prediction = None
            for symbol_predictions in self.prediction_outcomes.values():
                for pred in symbol_predictions:
                    if pred.prediction_id == prediction_id:
                        prediction = pred
                        break
                if prediction:
                    break
            
            if not prediction:
                logger.warning(f"Prediction {prediction_id} not found for outcome update")
                return
            
            # Update outcome
            prediction.actual_price = actual_price
            prediction.actual_outcome = actual_outcome
            prediction.outcome_time = outcome_time
            
            # Calculate accuracy score
            if prediction.predicted_price_target is not None:
                prediction.prediction_error = abs(actual_price - prediction.predicted_price_target)
                # Accuracy based on price prediction error (closer = higher accuracy)
                max_error = prediction.predicted_price_target * 0.1  # 10% max error for full accuracy
                prediction.accuracy_score = max(0.0, 1.0 - (prediction.prediction_error / max_error))
            else:
                # Binary accuracy for action predictions
                prediction.accuracy_score = 1.0 if actual_outcome == 'correct' else 0.0
            
            # Update model drift metrics
            self._update_model_drift_metrics(prediction)
            
            # Check for systematic biases
            self._check_systematic_biases(prediction.symbol)
            
            logger.info(f"Updated prediction outcome {prediction_id}: {actual_outcome}, accuracy: {prediction.accuracy_score:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating prediction outcome: {e}")
    
    def _update_model_drift_metrics(self, prediction: PredictionOutcome) -> None:
        """Update model drift metrics based on prediction outcomes"""
        try:
            model_name = f"{prediction.symbol}_model"
            
            if model_name not in self.model_drift_metrics:
                # Initialize with baseline accuracy (assume 70% baseline)
                self.model_drift_metrics[model_name] = ModelDriftMetrics(
                    model_name=model_name,
                    baseline_accuracy=0.7,
                    current_accuracy=0.7
                )
            
            drift_metrics = self.model_drift_metrics[model_name]
            
            # Add to accuracy trend
            if prediction.accuracy_score is not None:
                drift_metrics.accuracy_trend.append(prediction.accuracy_score)
                
                # Keep only recent accuracy scores for drift calculation
                if len(drift_metrics.accuracy_trend) > self.drift_detection_window:
                    drift_metrics.accuracy_trend = drift_metrics.accuracy_trend[-self.drift_detection_window:]
                
                # Calculate current accuracy (moving average of recent predictions)
                if len(drift_metrics.accuracy_trend) >= 10:  # Need minimum samples
                    drift_metrics.current_accuracy = np.mean(drift_metrics.accuracy_trend[-50:])  # Last 50 predictions
                    
                    # Calculate drift score
                    drift_metrics.drift_score = drift_metrics.baseline_accuracy - drift_metrics.current_accuracy
                    
                    # Check if drift threshold exceeded
                    if drift_metrics.drift_score > drift_metrics.drift_threshold:
                        if not drift_metrics.drift_detected:
                            drift_metrics.drift_detected = True
                            self._trigger_model_retraining(model_name, "Model drift detected")
                            logger.warning(f"Model drift detected for {model_name}: {drift_metrics.drift_score:.3f}")
            
        except Exception as e:
            logger.error(f"Error updating model drift metrics: {e}")
    
    def _check_systematic_biases(self, symbol: str) -> None:
        """Check for systematic biases in predictions"""
        try:
            symbol_predictions = self.prediction_outcomes.get(symbol, [])
            completed_predictions = [p for p in symbol_predictions if p.actual_outcome is not None]
            
            if len(completed_predictions) < 20:  # Need minimum samples
                return
            
            # Check for various bias types
            biases_detected = []
            
            # 1. Overconfidence bias (high confidence but low accuracy)
            high_conf_predictions = [p for p in completed_predictions if p.predicted_probability > 0.8]
            if len(high_conf_predictions) >= 10:
                high_conf_accuracy = np.mean([p.accuracy_score for p in high_conf_predictions if p.accuracy_score is not None])
                if high_conf_accuracy < 0.6:  # High confidence but low accuracy
                    biases_detected.append(BiasDetectionResult(
                        bias_type="overconfidence",
                        affected_symbols=[symbol],
                        bias_magnitude=0.8 - high_conf_accuracy,
                        confidence=0.8,
                        detection_time=datetime.now(),
                        corrective_action="Reduce confidence calibration",
                        description=f"High confidence predictions ({len(high_conf_predictions)}) have low accuracy ({high_conf_accuracy:.2f})"
                    ))
            
            # 2. Directional bias (consistently wrong direction)
            buy_predictions = [p for p in completed_predictions if p.predicted_action.upper() == 'BUY']
            sell_predictions = [p for p in completed_predictions if p.predicted_action.upper() == 'SELL']
            
            if len(buy_predictions) >= 10 and len(sell_predictions) >= 10:
                buy_accuracy = np.mean([p.accuracy_score for p in buy_predictions if p.accuracy_score is not None])
                sell_accuracy = np.mean([p.accuracy_score for p in sell_predictions if p.accuracy_score is not None])
                
                if abs(buy_accuracy - sell_accuracy) > 0.2:  # Significant difference
                    bias_direction = "buy" if buy_accuracy < sell_accuracy else "sell"
                    biases_detected.append(BiasDetectionResult(
                        bias_type="directional_bias",
                        affected_symbols=[symbol],
                        bias_magnitude=abs(buy_accuracy - sell_accuracy),
                        confidence=0.7,
                        detection_time=datetime.now(),
                        corrective_action=f"Adjust {bias_direction} signal calibration",
                        description=f"Significant accuracy difference: BUY {buy_accuracy:.2f} vs SELL {sell_accuracy:.2f}"
                    ))
            
            # 3. Market regime bias (poor performance in specific regimes)
            regime_accuracies = {}
            for regime in ['bull', 'bear', 'sideways', 'high_volatility']:
                regime_preds = [p for p in completed_predictions 
                              if p.market_conditions.get('regime') == regime]
                if len(regime_preds) >= 5:
                    regime_accuracies[regime] = np.mean([p.accuracy_score for p in regime_preds if p.accuracy_score is not None])
            
            if len(regime_accuracies) >= 2:
                min_regime = min(regime_accuracies, key=regime_accuracies.get)
                max_regime = max(regime_accuracies, key=regime_accuracies.get)
                
                if regime_accuracies[max_regime] - regime_accuracies[min_regime] > 0.25:
                    biases_detected.append(BiasDetectionResult(
                        bias_type="regime_bias",
                        affected_symbols=[symbol],
                        bias_magnitude=regime_accuracies[max_regime] - regime_accuracies[min_regime],
                        confidence=0.6,
                        detection_time=datetime.now(),
                        corrective_action=f"Improve {min_regime} market regime handling",
                        description=f"Poor performance in {min_regime} regime: {regime_accuracies[min_regime]:.2f}"
                    ))
            
            # Store detected biases
            for bias in biases_detected:
                self.bias_detection_results.append(bias)
                self._notify_bias_detection(bias)
                logger.warning(f"Systematic bias detected for {symbol}: {bias.bias_type}")
            
            # Keep only recent bias results
            if len(self.bias_detection_results) > 100:
                self.bias_detection_results = self.bias_detection_results[-100:]
            
        except Exception as e:
            logger.error(f"Error checking systematic biases: {e}")
    
    def _trigger_model_retraining(self, model_name: str, reason: str) -> None:
        """Trigger model retraining"""
        try:
            self.retraining_triggers[model_name] = datetime.now()
            
            # Notify callbacks about retraining trigger
            retraining_event = {
                'type': 'retraining_triggered',
                'model_name': model_name,
                'reason': reason,
                'timestamp': datetime.now()
            }
            
            for callback in self.callbacks:
                try:
                    callback('retraining', retraining_event)
                except Exception as e:
                    logger.error(f"Retraining callback error: {e}")
            
            logger.info(f"Triggered retraining for {model_name}: {reason}")
            
        except Exception as e:
            logger.error(f"Error triggering model retraining: {e}")
    
    def _notify_bias_detection(self, bias: BiasDetectionResult) -> None:
        """Notify callbacks about bias detection"""
        try:
            bias_event = {
                'type': 'bias_detected',
                'bias_result': bias.__dict__,
                'timestamp': datetime.now()
            }
            
            for callback in self.callbacks:
                try:
                    callback('bias', bias_event)
                except Exception as e:
                    logger.error(f"Bias callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Error notifying bias detection: {e}")
    
    def check_performance_decline(self) -> List[Dict[str, Any]]:
        """Check for performance decline and trigger alerts"""
        try:
            decline_alerts = []
            
            for strategy_name, strategy in self.strategy_performance.items():
                if len(strategy.performance_trend) < 10:  # Need minimum history
                    continue
                
                # Calculate recent vs historical performance
                recent_performance = np.mean(strategy.performance_trend[-10:])  # Last 10 periods
                historical_performance = np.mean(strategy.performance_trend[:-10])  # Earlier periods
                
                if historical_performance > 0:  # Avoid division by zero
                    decline_ratio = (historical_performance - recent_performance) / historical_performance
                    
                    if decline_ratio > self.performance_decline_threshold:
                        alert = {
                            'strategy': strategy_name,
                            'decline_ratio': decline_ratio,
                            'recent_performance': recent_performance,
                            'historical_performance': historical_performance,
                            'timestamp': datetime.now()
                        }
                        decline_alerts.append(alert)
                        
                        # Trigger retraining
                        self._trigger_model_retraining(
                            strategy_name, 
                            f"Performance decline detected: {decline_ratio:.2%}"
                        )
            
            return decline_alerts
            
        except Exception as e:
            logger.error(f"Error checking performance decline: {e}")
            return []
    
    def get_prediction_accuracy_report(self, symbol: Optional[str] = None, 
                                     days: int = 30) -> Dict[str, Any]:
        """Get comprehensive prediction accuracy report"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            if symbol:
                symbol_predictions = self.prediction_outcomes.get(symbol, [])
                all_predictions = [p for p in symbol_predictions 
                                 if p.prediction_time >= cutoff_time and p.actual_outcome is not None]
            else:
                all_predictions = []
                for symbol_preds in self.prediction_outcomes.values():
                    all_predictions.extend([p for p in symbol_preds 
                                          if p.prediction_time >= cutoff_time and p.actual_outcome is not None])
            
            if not all_predictions:
                return {'error': 'No completed predictions found'}
            
            # Calculate overall accuracy
            accuracies = [p.accuracy_score for p in all_predictions if p.accuracy_score is not None]
            overall_accuracy = np.mean(accuracies) if accuracies else 0.0
            
            # Accuracy by action type
            buy_predictions = [p for p in all_predictions if p.predicted_action.upper() == 'BUY']
            sell_predictions = [p for p in all_predictions if p.predicted_action.upper() == 'SELL']
            
            buy_accuracy = np.mean([p.accuracy_score for p in buy_predictions if p.accuracy_score is not None]) if buy_predictions else 0.0
            sell_accuracy = np.mean([p.accuracy_score for p in sell_predictions if p.accuracy_score is not None]) if sell_predictions else 0.0
            
            # Accuracy by confidence level
            high_conf = [p for p in all_predictions if p.predicted_probability > 0.8]
            med_conf = [p for p in all_predictions if 0.5 <= p.predicted_probability <= 0.8]
            low_conf = [p for p in all_predictions if p.predicted_probability < 0.5]
            
            return {
                'period_days': days,
                'total_predictions': len(all_predictions),
                'overall_accuracy': overall_accuracy,
                'accuracy_by_action': {
                    'buy': {'count': len(buy_predictions), 'accuracy': buy_accuracy},
                    'sell': {'count': len(sell_predictions), 'accuracy': sell_accuracy}
                },
                'accuracy_by_confidence': {
                    'high': {'count': len(high_conf), 'accuracy': np.mean([p.accuracy_score for p in high_conf if p.accuracy_score is not None]) if high_conf else 0.0},
                    'medium': {'count': len(med_conf), 'accuracy': np.mean([p.accuracy_score for p in med_conf if p.accuracy_score is not None]) if med_conf else 0.0},
                    'low': {'count': len(low_conf), 'accuracy': np.mean([p.accuracy_score for p in low_conf if p.accuracy_score is not None]) if low_conf else 0.0}
                },
                'model_drift_status': {name: metrics.__dict__ for name, metrics in self.model_drift_metrics.items()},
                'recent_biases': [bias.__dict__ for bias in self.bias_detection_results[-10:]],
                'retraining_triggers': {name: timestamp.isoformat() for name, timestamp in self.retraining_triggers.items()}
            }
            
        except Exception as e:
            logger.error(f"Error generating prediction accuracy report: {e}")
            return {'error': str(e)}