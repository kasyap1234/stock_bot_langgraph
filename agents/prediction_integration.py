

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from config.trading_config import ENSEMBLE_THRESHOLD, PROBABILITY_THRESHOLD
from data.models import State

logger = logging.getLogger(__name__)


class PredictionAggregator:
    

    def __init__(self):
        self.ml_weights = {
            'random_forest': 0.15,
            'gradient_boosting': 0.15,
            'xgboost': 0.15,
            'lightgbm': 0.15,
            'cnn': 0.10,
            'bilstm': 0.10,
            'gru': 0.10,
            'transformer': 0.10
        }
        self.technical_weight = 0.3
        self.anomaly_penalty = 0.2  # Reduce confidence if anomalies detected

    def aggregate_ml_predictions(self, ml_predictions: Dict[str, Any],
                               nn_predictions: Dict[str, Any]) -> Dict[str, Any]:
        
        aggregated = {
            'individual_predictions': {},
            'ensemble_prediction': 0.5,
            'confidence_score': 0.0,
            'prediction_sources': 0,
            'anomaly_detected': False
        }

        total_weight = 0
        weighted_sum = 0

        # Process traditional ML models
        if ml_predictions and 'latest_prediction' in ml_predictions:
            latest_pred = ml_predictions['latest_prediction']
            individual_preds = latest_pred.get('predictions', {})

            for model_name, pred_data in individual_preds.items():
                if model_name in self.ml_weights and isinstance(pred_data, dict):
                    prediction = pred_data.get('prediction', [0.5])[0]
                    weight = self.ml_weights[model_name]

                    aggregated['individual_predictions'][model_name] = prediction
                    weighted_sum += prediction * weight
                    total_weight += weight
                    aggregated['prediction_sources'] += 1

        # Process neural network models
        if nn_predictions and 'predictions' in nn_predictions:
            nn_pred_data = nn_predictions['predictions']
            individual_nn_preds = nn_pred_data.get('individual_predictions', {})

            for model_name, prediction in individual_nn_preds.items():
                full_name = f"nn_{model_name}"
                if model_name in self.ml_weights:
                    weight = self.ml_weights[model_name]

                    aggregated['individual_predictions'][full_name] = prediction
                    weighted_sum += prediction * weight
                    total_weight += weight
                    aggregated['prediction_sources'] += 1

            # Check for anomalies
            anomalies = nn_pred_data.get('anomalies', {})
            if anomalies and anomalies.get('n_anomalies', 0) > 0:
                aggregated['anomaly_detected'] = True

        # Calculate ensemble prediction
        if total_weight > 0:
            aggregated['ensemble_prediction'] = weighted_sum / total_weight

            # Calculate confidence based on agreement and number of sources
            predictions_list = list(aggregated['individual_predictions'].values())
            if len(predictions_list) > 1:
                pred_std = np.std(predictions_list)
                agreement_score = 1 - pred_std  # Lower variance = higher agreement
                source_score = min(aggregated['prediction_sources'] / 4, 1)  # Max confidence at 4+ sources
                aggregated['confidence_score'] = agreement_score * source_score
            else:
                aggregated['confidence_score'] = 0.5

            # Apply anomaly penalty
            if aggregated['anomaly_detected']:
                aggregated['confidence_score'] *= (1 - self.anomaly_penalty)

        return aggregated

    def combine_with_technical_analysis(self, ml_aggregated: Dict[str, Any],
                                      technical_signals: Dict[str, str],
                                      ensemble_signal: str) -> Dict[str, Any]:
        
        combined = ml_aggregated.copy()

        # Convert technical ensemble signal to numeric
        tech_prediction = 0.5
        if ensemble_signal == "buy":
            tech_prediction = 0.8
        elif ensemble_signal == "sell":
            tech_prediction = 0.2

        # Weighted combination
        ml_pred = ml_aggregated['ensemble_prediction']
        combined_pred = (ml_pred * (1 - self.technical_weight) +
                        tech_prediction * self.technical_weight)

        combined['combined_prediction'] = combined_pred
        combined['technical_prediction'] = tech_prediction
        combined['technical_signals'] = technical_signals
        combined['ensemble_signal'] = ensemble_signal

        # Adjust confidence based on technical-ML agreement
        pred_diff = abs(ml_pred - tech_prediction)
        agreement_bonus = max(0, 0.2 - pred_diff)  # Bonus for close agreement
        combined['confidence_score'] = min(1.0, combined['confidence_score'] + agreement_bonus)

        return combined

    def generate_final_signal(self, combined_prediction: Dict[str, Any]) -> Dict[str, Any]:
        
        final_signal = {
            'signal': 'neutral',
            'confidence': combined_prediction.get('confidence_score', 0.0),
            'prediction_value': combined_prediction.get('combined_prediction', 0.5),
            'fallback_used': False,
            'reasoning': []
        }

        pred_value = final_signal['prediction_value']
        confidence = final_signal['confidence']

        # Primary decision logic
        if confidence >= PROBABILITY_THRESHOLD:
            if pred_value >= (1 + ENSEMBLE_THRESHOLD) / 2:  # Above neutral threshold
                final_signal['signal'] = 'buy'
                final_signal['reasoning'].append("Strong ML and technical agreement for upward movement")
            elif pred_value <= (1 - ENSEMBLE_THRESHOLD) / 2:  # Below neutral threshold
                final_signal['signal'] = 'sell'
                final_signal['reasoning'].append("Strong ML and technical agreement for downward movement")
            else:
                final_signal['signal'] = 'neutral'
                final_signal['reasoning'].append("Mixed signals, maintaining neutral position")
        else:
            # Low confidence - use fallback
            final_signal = self._apply_fallback_logic(combined_prediction, final_signal)

        return final_signal

    def _apply_fallback_logic(self, combined_prediction: Dict[str, Any],
                            final_signal: Dict[str, Any]) -> Dict[str, Any]:
        
        final_signal['fallback_used'] = True

        # Fallback 1: Use technical analysis only
        tech_signal = combined_prediction.get('ensemble_signal', 'neutral')
        if tech_signal != 'neutral':
            final_signal['signal'] = tech_signal
            final_signal['reasoning'].append("Low ML confidence, using technical analysis fallback")
            final_signal['confidence'] = 0.6  # Moderate confidence for technical-only
            return final_signal

        # Fallback 2: Use majority vote of individual ML models
        individual_preds = combined_prediction.get('individual_predictions', {})
        if individual_preds:
            buy_votes = sum(1 for pred in individual_preds.values() if pred > 0.6)
            sell_votes = sum(1 for pred in individual_preds.values() if pred < 0.4)

            if buy_votes > sell_votes:
                final_signal['signal'] = 'buy'
                final_signal['reasoning'].append("Low confidence, using majority vote of ML models")
                final_signal['confidence'] = 0.5
            elif sell_votes > buy_votes:
                final_signal['signal'] = 'sell'
                final_signal['reasoning'].append("Low confidence, using majority vote of ML models")
                final_signal['confidence'] = 0.5
            else:
                final_signal['signal'] = 'neutral'
                final_signal['reasoning'].append("No clear majority, maintaining neutral")
                final_signal['confidence'] = 0.3

        return final_signal


class RealTimePredictor:
    

    def __init__(self):
        self.prediction_cache = {}
        self.cache_timeout = timedelta(minutes=15)  # Cache predictions for 15 minutes

    def should_update_prediction(self, symbol: str) -> bool:
        
        if symbol not in self.prediction_cache:
            return True

        cached_time = self.prediction_cache[symbol].get('timestamp')
        if cached_time is None:
            return True

        return datetime.now() - cached_time > self.cache_timeout

    def cache_prediction(self, symbol: str, prediction: Dict[str, Any]):
        
        self.prediction_cache[symbol] = {
            'prediction': prediction,
            'timestamp': datetime.now()
        }

    def get_cached_prediction(self, symbol: str) -> Optional[Dict[str, Any]]:
        
        if symbol not in self.prediction_cache:
            return None

        cached_data = self.prediction_cache[symbol]
        if datetime.now() - cached_data['timestamp'] > self.cache_timeout:
            return None

        return cached_data['prediction']

    def clear_expired_cache(self):
        
        current_time = datetime.now()
        expired_symbols = []

        for symbol, data in self.prediction_cache.items():
            if current_time - data['timestamp'] > self.cache_timeout:
                expired_symbols.append(symbol)

        for symbol in expired_symbols:
            del self.prediction_cache[symbol]

        if expired_symbols:
            logger.info(f"Cleared expired predictions for {len(expired_symbols)} symbols")


class PredictionValidator:
    

    def __init__(self):
        self.performance_history = {}
        self.max_history_length = 100

    def validate_prediction(self, symbol: str, prediction: Dict[str, Any],
                          recent_performance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        
        validated = prediction.copy()
        validated['validation_warnings'] = []

        # Check recent model performance
        if recent_performance:
            recent_accuracy = recent_performance.get('accuracy', 0.5)
            if recent_accuracy < 0.5:
                validated['validation_warnings'].append(
                    f"Recent model accuracy low: {recent_accuracy:.2f}"
                )
                validated['confidence'] *= 0.8  # Reduce confidence

        # Check for overfitting indicators
        train_val_gap = recent_performance.get('train_val_gap', 0) if recent_performance else 0
        if train_val_gap > 0.2:  # Large gap indicates potential overfitting
            validated['validation_warnings'].append(
                f"Potential overfitting detected (train-val gap: {train_val_gap:.2f})"
            )
            validated['confidence'] *= 0.9

        # Update performance history
        self._update_performance_history(symbol, prediction)

        return validated

    def _update_performance_history(self, symbol: str, prediction: Dict[str, Any]):
        
        if symbol not in self.performance_history:
            self.performance_history[symbol] = []

        self.performance_history[symbol].append({
            'timestamp': datetime.now(),
            'signal': prediction.get('signal'),
            'confidence': prediction.get('confidence')
        })

        # Keep only recent history
        if len(self.performance_history[symbol]) > self.max_history_length:
            self.performance_history[symbol] = self.performance_history[symbol][-self.max_history_length:]


class RiskAdjustedPredictor:
    

    def __init__(self):
        self.volatility_threshold = 0.05  # 5% annualized volatility
        self.anomaly_risk_penalty = 0.3

    def apply_risk_adjustments(self, prediction: Dict[str, Any],
                             market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        
        adjusted = prediction.copy()

        # Check volatility
        current_volatility = market_conditions.get('volatility', 0)
        if current_volatility > self.volatility_threshold:
            vol_multiplier = 1 - min((current_volatility - self.volatility_threshold) / 0.02, 0.5)
            adjusted['confidence'] *= vol_multiplier
            adjusted['risk_adjustments'] = adjusted.get('risk_adjustments', [])
            adjusted['risk_adjustments'].append({
                'type': 'high_volatility',
                'factor': vol_multiplier,
                'reason': f"High volatility: {current_volatility:.2%}"
            })

        # Check for anomalies
        if prediction.get('anomaly_detected', False):
            adjusted['confidence'] *= (1 - self.anomaly_risk_penalty)
            adjusted['risk_adjustments'].append({
                'type': 'anomaly_detected',
                'factor': 1 - self.anomaly_risk_penalty,
                'reason': "Market anomalies detected"
            })

        # Check market regime
        regime = market_conditions.get('regime', 'neutral')
        if regime in ['bear_regime', 'sideways_regime']:
            regime_penalty = 0.1 if regime == 'bear_regime' else 0.05
            adjusted['confidence'] *= (1 - regime_penalty)
            adjusted['risk_adjustments'].append({
                'type': 'market_regime',
                'factor': 1 - regime_penalty,
                'reason': f"Adverse market regime: {regime}"
            })

        return adjusted


class PredictionIntegrationAgent:
    

    def __init__(self):
        self.aggregator = PredictionAggregator()
        self.realtime_predictor = RealTimePredictor()
        self.validator = PredictionValidator()
        self.risk_adjuster = RiskAdjustedPredictor()

    def generate_integrated_prediction(self, state: State, symbol: str) -> Dict[str, Any]:
        
        # Check cache first
        if not self.realtime_predictor.should_update_prediction(symbol):
            cached = self.realtime_predictor.get_cached_prediction(symbol)
            if cached:
                logger.info(f"Using cached prediction for {symbol}")
                return cached

        # Extract relevant data from state
        ml_predictions = state.get('ml_predictions', {}).get(symbol, {})
        nn_predictions = state.get('nn_predictions', {}).get(symbol, {})
        technical_signals = state.get('technical_signals', {}).get(symbol, {})
        engineered_features = state.get('engineered_features', {}).get(symbol, pd.DataFrame())

        # Aggregate ML predictions
        ml_aggregated = self.aggregator.aggregate_ml_predictions(ml_predictions, nn_predictions)

        # Combine with technical analysis
        ensemble_signal = technical_signals.get('EnsembleSignal', 'neutral')
        combined_prediction = self.aggregator.combine_with_technical_analysis(
            ml_aggregated, technical_signals, ensemble_signal
        )

        # Generate final signal
        final_signal = self.aggregator.generate_final_signal(combined_prediction)

        # Apply risk adjustments
        market_conditions = self._extract_market_conditions(state, symbol)
        risk_adjusted = self.risk_adjuster.apply_risk_adjustments(final_signal, market_conditions)

        # Validate prediction
        validated = self.validator.validate_prediction(symbol, risk_adjusted)

        # Add metadata
        validated.update({
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'model_versions': self._get_model_versions(ml_predictions, nn_predictions),
            'feature_count': len(engineered_features.columns) if not engineered_features.empty else 0
        })

        # Cache prediction
        self.realtime_predictor.cache_prediction(symbol, validated)

        logger.info(f"Generated integrated prediction for {symbol}: {validated['signal']} "
                   f"(confidence: {validated['confidence']:.2f})")

        return validated

    def _extract_market_conditions(self, state: State, symbol: str) -> Dict[str, Any]:
        
        conditions = {}

        # Get volatility from engineered features
        engineered_features = state.get('engineered_features', {}).get(symbol, pd.DataFrame())
        if not engineered_features.empty and 'volatility_20' in engineered_features.columns:
            conditions['volatility'] = engineered_features['volatility_20'].iloc[-1]

        # Get regime from technical signals
        technical_signals = state.get('technical_signals', {}).get(symbol, {})
        conditions['regime'] = technical_signals.get('HMMSignal', 'neutral')

        return conditions

    def _get_model_versions(self, ml_predictions: Dict, nn_predictions: Dict) -> Dict[str, str]:
        
        versions = {}

        if ml_predictions.get('trained_models'):
            versions['ml_models'] = list(ml_predictions['trained_models'].keys())

        if nn_predictions.get('training_results'):
            versions['nn_models'] = list(nn_predictions['training_results'].keys())

        return versions

    def batch_predict(self, state: State, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        
        predictions = {}

        for symbol in symbols:
            try:
                prediction = self.generate_integrated_prediction(state, symbol)
                predictions[symbol] = prediction
            except Exception as e:
                logger.error(f"Error generating prediction for {symbol}: {e}")
                predictions[symbol] = {
                    'error': str(e),
                    'signal': 'neutral',
                    'confidence': 0.0
                }

        # Clear expired cache
        self.realtime_predictor.clear_expired_cache()

        return predictions


def prediction_integration_agent(state: State) -> State:
    
    logging.info("Starting prediction integration agent")

    stock_data = state.get("stock_data", {})
    if not stock_data:
        logger.warning("No stock data available for prediction integration")
        return state

    symbols = list(stock_data.keys())
    integration_agent = PredictionIntegrationAgent()

    integrated_predictions = integration_agent.batch_predict(state, symbols)

    state["integrated_predictions"] = integrated_predictions
    logger.info(f"Completed prediction integration for {len(integrated_predictions)} symbols")

    return state