"""
Accuracy Improvement Module for Stock Bot
Implements feedback loops, adaptive weighting, and performance tracking
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class FeatureCorrelationAnalyzer:
    """Analyzes and removes highly correlated features"""
    
    def __init__(self, correlation_threshold: float = 0.95):
        self.correlation_threshold = correlation_threshold
        self.correlation_matrix = None
        self.features_to_drop = []
    
    def analyze_correlations(self, X: pd.DataFrame) -> pd.DataFrame:
        """Analyze feature correlations and identify redundant features"""
        self.correlation_matrix = X.corr().abs()
        
        # Find highly correlated feature pairs
        upper = self.correlation_matrix.where(
            np.triu(np.ones(self.correlation_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns 
                   if any(upper[column] > self.correlation_threshold)]
        
        self.features_to_drop = to_drop
        logger.info(f"Identified {len(to_drop)} highly correlated features to drop")
        
        return self.correlation_matrix
    
    def filter_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features"""
        if not self.features_to_drop:
            self.analyze_correlations(X)
        
        X_filtered = X.drop(columns=[col for col in self.features_to_drop if col in X.columns])
        logger.info(f"Filtered features: {X.shape[1]} -> {X_filtered.shape[1]}")
        
        return X_filtered
    
    def get_correlation_report(self) -> Dict[str, Any]:
        """Get detailed correlation analysis report"""
        if self.correlation_matrix is None:
            return {}
        
        return {
            'features_dropped': self.features_to_drop,
            'correlation_threshold': self.correlation_threshold,
            'num_features_dropped': len(self.features_to_drop)
        }


class ImprovedTargetDefinition:
    """Creates better target definitions for classification"""
    
    def __init__(self, horizons: List[int] = None, risk_adjustment: bool = True):
        self.horizons = horizons or [3, 5, 10]
        self.risk_adjustment = risk_adjustment
        self.target_stats = {}
    
    def create_multi_horizon_targets(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Create targets for multiple prediction horizons"""
        targets = {}
        
        for horizon in self.horizons:
            future_close = df['Close'].shift(-horizon)
            returns = (future_close - df['Close']) / df['Close']
            
            # Use adaptive threshold based on volatility
            volatility = df['Close'].pct_change().rolling(20).std().mean()
            threshold = max(0.01, volatility * 1.5)  # At least 1%, or 1.5x volatility
            
            target = (returns > threshold).astype(int)
            targets[f'target_{horizon}d'] = target
            
            # Log class distribution
            class_dist = target.value_counts()
            self.target_stats[f'target_{horizon}d'] = {
                'threshold': threshold,
                'class_distribution': class_dist.to_dict(),
                'positive_ratio': class_dist.get(1, 0) / len(target)
            }
        
        logger.info(f"Created {len(targets)} target variables with adaptive thresholds")
        return targets
    
    def create_risk_adjusted_target(self, df: pd.DataFrame, horizon: int = 5) -> pd.Series:
        """Create risk-adjusted return target (Sharpe-like metric)"""
        returns = df['Close'].pct_change(horizon)
        volatility = df['Close'].pct_change().rolling(20).std()
        
        # Risk-adjusted returns
        risk_adjusted = returns / (volatility + 1e-6)
        
        # Threshold at 75th percentile for positive signal
        threshold = risk_adjusted.quantile(0.75)
        target = (risk_adjusted > threshold).astype(int)
        
        logger.info(f"Created risk-adjusted target with threshold: {threshold:.3f}")
        return target
    
    def handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Handle class imbalance using stratified sampling"""
        try:
            from imblearn.over_sampling import SMOTE
            
            # Apply SMOTE only if there's significant imbalance
            class_dist = y.value_counts()
            imbalance_ratio = min(class_dist) / max(class_dist)
            
            if imbalance_ratio < 0.3:  # Significant imbalance
                smote = SMOTE(random_state=42, k_neighbors=3)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                logger.info(f"Applied SMOTE: {len(y)} -> {len(y_resampled)} samples")
                return X_resampled, y_resampled
        except ImportError:
            logger.warning("imbalanced-learn not available, using class weights instead")
        
        return X, y


class AdaptiveEnsembleWeighter:
    """Adaptively adjusts ensemble weights based on recent performance"""
    
    def __init__(self, lookback_periods: int = 20, min_history: int = 5):
        self.lookback_periods = lookback_periods
        self.min_history = min_history
        self.performance_history = {}
        self.current_weights = {}
    
    def track_prediction(self, symbol: str, model_name: str, prediction: float, 
                        actual: float, confidence: float):
        """Track individual model predictions and outcomes"""
        if symbol not in self.performance_history:
            self.performance_history[symbol] = {}
        if model_name not in self.performance_history[symbol]:
            self.performance_history[symbol][model_name] = []
        
        # Calculate if prediction was correct
        pred_direction = 1 if prediction > 0.5 else 0
        actual_direction = 1 if actual > 0.5 else 0
        correct = pred_direction == actual_direction
        
        self.performance_history[symbol][model_name].append({
            'timestamp': datetime.now(),
            'prediction': prediction,
            'actual': actual,
            'correct': correct,
            'confidence': confidence,
            'error': abs(prediction - actual)
        })
        
        # Keep only recent history
        if len(self.performance_history[symbol][model_name]) > self.lookback_periods:
            self.performance_history[symbol][model_name] = \
                self.performance_history[symbol][model_name][-self.lookback_periods:]
    
    def calculate_adaptive_weights(self, symbol: str) -> Dict[str, float]:
        """Calculate weights based on recent performance"""
        if symbol not in self.performance_history:
            return {}
        
        weights = {}
        
        for model_name, history in self.performance_history[symbol].items():
            if len(history) < self.min_history:
                weights[model_name] = 1.0 / len(self.performance_history[symbol])
                continue
            
            # Calculate accuracy
            accuracy = sum(1 for h in history if h['correct']) / len(history)
            
            # Calculate confidence calibration
            avg_confidence = np.mean([h['confidence'] for h in history])
            confidence_calibration = accuracy / (avg_confidence + 0.1)
            
            # Calculate prediction error
            avg_error = np.mean([h['error'] for h in history])
            error_score = 1 / (1 + avg_error)
            
            # Combine metrics
            weight = (accuracy * 0.5 + confidence_calibration * 0.3 + error_score * 0.2)
            weights[model_name] = max(0.1, weight)  # Minimum weight of 0.1
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        self.current_weights[symbol] = weights
        logger.info(f"Updated adaptive weights for {symbol}: {weights}")
        
        return weights
    
    def get_performance_report(self, symbol: str) -> Dict[str, Any]:
        """Get detailed performance report for a symbol"""
        if symbol not in self.performance_history:
            return {}
        
        report = {}
        for model_name, history in self.performance_history[symbol].items():
            if not history:
                continue
            
            accuracies = [h['correct'] for h in history]
            confidences = [h['confidence'] for h in history]
            errors = [h['error'] for h in history]
            
            report[model_name] = {
                'accuracy': sum(accuracies) / len(accuracies),
                'avg_confidence': np.mean(confidences),
                'avg_error': np.mean(errors),
                'num_predictions': len(history),
                'weight': self.current_weights.get(symbol, {}).get(model_name, 0)
            }
        
        return report


class PerformanceTracker:
    """Tracks overall system performance and generates reports"""
    
    def __init__(self, history_file: Optional[str] = None):
        self.history_file = history_file or "bot_performance_history.json"
        self.predictions = []
        self.load_history()
    
    def load_history(self):
        """Load historical predictions from file"""
        if Path(self.history_file).exists():
            try:
                with open(self.history_file, 'r') as f:
                    self.predictions = json.load(f)
                logger.info(f"Loaded {len(self.predictions)} historical predictions")
            except Exception as e:
                logger.warning(f"Could not load history: {e}")
                self.predictions = []
    
    def save_history(self):
        """Save predictions to file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.predictions, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Could not save history: {e}")
    
    def record_prediction(self, symbol: str, action: str, confidence: float,
                         reasoning: str, timestamp: Optional[datetime] = None):
        """Record a prediction"""
        self.predictions.append({
            'symbol': symbol,
            'action': action,
            'confidence': confidence,
            'reasoning': reasoning,
            'timestamp': (timestamp or datetime.now()).isoformat()
        })
        
        # Save periodically
        if len(self.predictions) % 10 == 0:
            self.save_history()
    
    def record_outcome(self, symbol: str, timestamp: datetime, actual_return: float,
                      predicted_action: str):
        """Record the outcome of a prediction"""
        # Find matching prediction
        for pred in reversed(self.predictions):
            if pred['symbol'] == symbol:
                pred['actual_return'] = actual_return
                pred['outcome_timestamp'] = timestamp.isoformat()
                pred['correct'] = (actual_return > 0 and predicted_action == 'BUY') or \
                                 (actual_return < 0 and predicted_action == 'SELL')
                break
        
        self.save_history()
    
    def get_accuracy_metrics(self, lookback_days: int = 30) -> Dict[str, Any]:
        """Calculate accuracy metrics for recent predictions"""
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        
        recent_preds = [p for p in self.predictions 
                       if 'outcome_timestamp' in p and 
                       datetime.fromisoformat(p['outcome_timestamp']) > cutoff_date]
        
        if not recent_preds:
            return {'message': 'No recent predictions with outcomes'}
        
        correct = sum(1 for p in recent_preds if p.get('correct', False))
        accuracy = correct / len(recent_preds)
        
        # Calculate by action
        buy_preds = [p for p in recent_preds if p['action'] == 'BUY']
        sell_preds = [p for p in recent_preds if p['action'] == 'SELL']
        hold_preds = [p for p in recent_preds if p['action'] == 'HOLD']
        
        metrics = {
            'total_predictions': len(recent_preds),
            'accuracy': accuracy,
            'correct_predictions': correct,
            'avg_confidence': np.mean([p['confidence'] for p in recent_preds]),
            'by_action': {
                'BUY': {
                    'count': len(buy_preds),
                    'accuracy': sum(1 for p in buy_preds if p.get('correct')) / len(buy_preds) if buy_preds else 0
                },
                'SELL': {
                    'count': len(sell_preds),
                    'accuracy': sum(1 for p in sell_preds if p.get('correct')) / len(sell_preds) if sell_preds else 0
                },
                'HOLD': {
                    'count': len(hold_preds),
                    'accuracy': sum(1 for p in hold_preds if p.get('correct')) / len(hold_preds) if hold_preds else 0
                }
            }
        }
        
        return metrics
    
    def get_symbol_metrics(self, symbol: str) -> Dict[str, Any]:
        """Get metrics for a specific symbol"""
        symbol_preds = [p for p in self.predictions if p['symbol'] == symbol]
        
        if not symbol_preds:
            return {}
        
        with_outcomes = [p for p in symbol_preds if 'outcome_timestamp' in p]
        
        if not with_outcomes:
            return {
                'total_predictions': len(symbol_preds),
                'avg_confidence': np.mean([p['confidence'] for p in symbol_preds])
            }
        
        correct = sum(1 for p in with_outcomes if p.get('correct', False))
        
        return {
            'total_predictions': len(symbol_preds),
            'predictions_with_outcomes': len(with_outcomes),
            'accuracy': correct / len(with_outcomes),
            'avg_confidence': np.mean([p['confidence'] for p in with_outcomes]),
            'avg_actual_return': np.mean([p.get('actual_return', 0) for p in with_outcomes])
        }


class AccuracyImprovementAgent:
    """Main agent for accuracy improvements"""
    
    def __init__(self):
        self.correlation_analyzer = FeatureCorrelationAnalyzer()
        self.target_improver = ImprovedTargetDefinition()
        self.weight_adjuster = AdaptiveEnsembleWeighter()
        self.performance_tracker = PerformanceTracker()
    
    def improve_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature improvements"""
        self.correlation_analyzer.analyze_correlations(X)
        X_filtered = self.correlation_analyzer.filter_features(X)
        return X_filtered
    
    def improve_targets(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Create improved target definitions"""
        return self.target_improver.create_multi_horizon_targets(df)
    
    def get_adaptive_weights(self, symbol: str) -> Dict[str, float]:
        """Get adaptive weights for ensemble"""
        return self.weight_adjuster.calculate_adaptive_weights(symbol)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return {
            'overall_metrics': self.performance_tracker.get_accuracy_metrics(),
            'feature_analysis': self.correlation_analyzer.get_correlation_report(),
            'target_stats': self.target_improver.target_stats
        }
