"""
Tests for the prediction integration module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from agents.prediction_integration import (
    PredictionAggregator, RealTimePredictor, PredictionValidator,
    RiskAdjustedPredictor, PredictionIntegrationAgent
)
from data.models import State


class TestPredictionAggregator:
    """Test cases for PredictionAggregator."""

    @pytest.fixture
    def aggregator(self):
        """Create aggregator instance."""
        return PredictionAggregator()

    def test_aggregate_ml_predictions(self, aggregator):
        """Test ML prediction aggregation."""
        ml_predictions = {
            'latest_prediction': {
                'predictions': {
                    'random_forest': {'prediction': np.array([1]), 'probability': np.array([0.8])},
                    'xgboost': {'prediction': np.array([1]), 'probability': np.array([0.7])}
                }
            }
        }

        nn_predictions = {
            'predictions': {
                'individual_predictions': {
                    'cnn': 0.75,
                    'bilstm': 0.85
                },
                'anomalies': {'n_anomalies': 0}
            }
        }

        result = aggregator.aggregate_ml_predictions(ml_predictions, nn_predictions)

        assert 'ensemble_prediction' in result
        assert 'confidence_score' in result
        assert 'prediction_sources' in result
        assert result['prediction_sources'] == 4  # 2 ML + 2 NN models
        assert 0 <= result['ensemble_prediction'] <= 1
        assert 0 <= result['confidence_score'] <= 1

    def test_combine_with_technical_analysis(self, aggregator):
        """Test combination with technical analysis."""
        ml_aggregated = {
            'ensemble_prediction': 0.7,
            'confidence_score': 0.8,
            'individual_predictions': {'rf': 0.7},
            'prediction_sources': 1
        }

        technical_signals = {'RSI': 'buy', 'MACD': 'buy'}
        ensemble_signal = 'buy'

        result = aggregator.combine_with_technical_analysis(
            ml_aggregated, technical_signals, ensemble_signal
        )

        assert 'combined_prediction' in result
        assert 'technical_prediction' in result
        assert 'technical_signals' in result
        assert result['ensemble_signal'] == 'buy'

    def test_generate_final_signal(self, aggregator):
        """Test final signal generation."""
        # High confidence case
        combined = {
            'combined_prediction': 0.8,
            'confidence_score': 0.9,
            'prediction_sources': 3
        }

        result = aggregator.generate_final_signal(combined)

        assert 'signal' in result
        assert 'confidence' in result
        assert result['signal'] == 'buy'
        assert result['confidence'] == 0.9

        # Low confidence case - should use fallback
        combined_low = {
            'combined_prediction': 0.6,
            'confidence_score': 0.3,
            'ensemble_signal': 'buy'
        }

        result_low = aggregator.generate_final_signal(combined_low)

        assert result_low['fallback_used'] is True
        assert result_low['signal'] == 'buy'  # Should use technical signal

    def test_fallback_logic(self, aggregator):
        """Test fallback logic when confidence is low."""
        # Test technical analysis fallback
        combined = {
            'combined_prediction': 0.5,
            'confidence_score': 0.2,
            'ensemble_signal': 'sell',
            'individual_predictions': {}
        }

        result = aggregator.generate_final_signal(combined)

        assert result['fallback_used'] is True
        assert result['signal'] == 'sell'

        # Test majority vote fallback
        combined_mv = {
            'combined_prediction': 0.5,
            'confidence_score': 0.2,
            'ensemble_signal': 'neutral',
            'individual_predictions': {
                'model1': 0.8,  # buy
                'model2': 0.3,  # sell
                'model3': 0.9   # buy
            }
        }

        result_mv = aggregator.generate_final_signal(combined_mv)

        assert result_mv['fallback_used'] is True
        assert result_mv['signal'] == 'buy'  # Majority vote


class TestRealTimePredictor:
    """Test cases for RealTimePredictor."""

    @pytest.fixture
    def rt_predictor(self):
        """Create real-time predictor instance."""
        return RealTimePredictor()

    def test_initialization(self, rt_predictor):
        """Test initialization."""
        assert rt_predictor.prediction_cache == {}
        assert rt_predictor.cache_timeout.seconds == 900  # 15 minutes

    def test_should_update_prediction(self, rt_predictor):
        """Test cache validation logic."""
        # No cache - should update
        assert rt_predictor.should_update_prediction('TEST') is True

        # Fresh cache - should not update
        rt_predictor.cache_prediction('TEST', {'signal': 'buy'})
        assert rt_predictor.should_update_prediction('TEST') is False

    def test_cache_prediction(self, rt_predictor):
        """Test prediction caching."""
        prediction = {'signal': 'buy', 'confidence': 0.8}

        rt_predictor.cache_prediction('TEST', prediction)

        assert 'TEST' in rt_predictor.prediction_cache
        assert rt_predictor.prediction_cache['TEST']['prediction'] == prediction

    def test_get_cached_prediction(self, rt_predictor):
        """Test cached prediction retrieval."""
        prediction = {'signal': 'buy', 'confidence': 0.8}
        rt_predictor.cache_prediction('TEST', prediction)

        cached = rt_predictor.get_cached_prediction('TEST')
        assert cached == prediction

        # Test expired cache
        rt_predictor.prediction_cache['TEST']['timestamp'] = pd.Timestamp.now() - pd.Timedelta(minutes=20)
        expired = rt_predictor.get_cached_prediction('TEST')
        assert expired is None


class TestPredictionValidator:
    """Test cases for PredictionValidator."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return PredictionValidator()

    def test_validate_prediction(self, validator):
        """Test prediction validation."""
        prediction = {
            'signal': 'buy',
            'confidence': 0.8,
            'reasoning': ['Test reason']
        }

        # Test with good performance
        recent_performance = {'accuracy': 0.8}
        validated = validator.validate_prediction('TEST', prediction, recent_performance)

        assert 'validation_warnings' in validated
        assert len(validated['validation_warnings']) == 0  # No warnings for good performance

        # Test with poor performance
        recent_performance_bad = {'accuracy': 0.4}
        validated_bad = validator.validate_prediction('TEST', prediction, recent_performance_bad)

        assert len(validated_bad['validation_warnings']) > 0
        assert validated_bad['confidence'] < prediction['confidence']  # Should reduce confidence

    def test_update_performance_history(self, validator):
        """Test performance history updates."""
        prediction1 = {'signal': 'buy', 'confidence': 0.8}
        prediction2 = {'signal': 'sell', 'confidence': 0.7}

        validator._update_performance_history('TEST', prediction1)
        validator._update_performance_history('TEST', prediction2)

        assert 'TEST' in validator.performance_history
        assert len(validator.performance_history['TEST']) == 2


class TestRiskAdjustedPredictor:
    """Test cases for RiskAdjustedPredictor."""

    @pytest.fixture
    def risk_adjuster(self):
        """Create risk adjuster instance."""
        return RiskAdjustedPredictor()

    def test_apply_risk_adjustments(self, risk_adjuster):
        """Test risk adjustment application."""
        prediction = {
            'signal': 'buy',
            'confidence': 0.8,
            'prediction_value': 0.7
        }

        # Test with high volatility
        market_conditions = {'volatility': 0.1}  # High volatility
        adjusted = risk_adjuster.apply_risk_adjustments(prediction, market_conditions)

        assert 'risk_adjustments' in adjusted
        assert len(adjusted['risk_adjustments']) > 0
        assert adjusted['confidence'] < prediction['confidence']  # Should reduce confidence

        # Test with anomalies
        prediction_anomaly = prediction.copy()
        prediction_anomaly['anomaly_detected'] = True
        adjusted_anomaly = risk_adjuster.apply_risk_adjustments(prediction_anomaly, {})

        assert adjusted_anomaly['confidence'] < prediction['confidence']

    def test_market_regime_adjustment(self, risk_adjuster):
        """Test market regime-based adjustments."""
        prediction = {
            'signal': 'buy',
            'confidence': 0.8
        }

        # Test bear regime
        market_conditions = {'regime': 'bear_regime'}
        adjusted = risk_adjuster.apply_risk_adjustments(prediction, market_conditions)

        assert adjusted['confidence'] < prediction['confidence']
        assert any('bear_regime' in adj['reason'] for adj in adjusted['risk_adjustments'])


class TestPredictionIntegrationAgent:
    """Test cases for PredictionIntegrationAgent."""

    @pytest.fixture
    def agent(self):
        """Create integration agent instance."""
        return PredictionIntegrationAgent()

    def test_generate_integrated_prediction(self, agent):
        """Test integrated prediction generation."""
        state = State()
        state['stock_data'] = {'TEST.NS': pd.DataFrame({'close': [100, 101, 102]})}
        state['ml_predictions'] = {
            'TEST.NS': {
                'latest_prediction': {
                    'predictions': {
                        'random_forest': {'prediction': np.array([1]), 'probability': np.array([0.8])}
                    }
                }
            }
        }
        state['nn_predictions'] = {
            'TEST.NS': {
                'predictions': {
                    'individual_predictions': {'cnn': 0.7},
                    'anomalies': {'n_anomalies': 0}
                }
            }
        }
        state['technical_signals'] = {'TEST.NS': {'EnsembleSignal': 'buy'}}
        state['engineered_features'] = {'TEST.NS': pd.DataFrame({'close': [100, 101, 102]})}

        prediction = agent.generate_integrated_prediction(state, 'TEST.NS')

        assert 'signal' in prediction
        assert 'confidence' in prediction
        assert 'timestamp' in prediction
        assert prediction['signal'] in ['buy', 'sell', 'neutral']

    def test_batch_predict(self, agent):
        """Test batch prediction."""
        state = State()
        state['stock_data'] = {
            'TEST1.NS': pd.DataFrame({'close': [100, 101, 102]}),
            'TEST2.NS': pd.DataFrame({'close': [200, 201, 202]})
        }
        # Add minimal required data
        state['ml_predictions'] = {}
        state['nn_predictions'] = {}
        state['technical_signals'] = {}
        state['engineered_features'] = {}

        predictions = agent.batch_predict(state, ['TEST1.NS', 'TEST2.NS'])

        assert isinstance(predictions, dict)
        assert len(predictions) == 2
        for symbol in ['TEST1.NS', 'TEST2.NS']:
            assert symbol in predictions
            assert 'signal' in predictions[symbol]


if __name__ == '__main__':
    pytest.main([__file__])