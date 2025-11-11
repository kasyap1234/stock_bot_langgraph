"""
Tests for Systematic Bias Detection System

Tests cover:
- Detection of various types of systematic biases
- Automatic bias correction mechanisms
- Bias monitoring and reporting
- Integration with performance monitoring
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from monitoring.bias_detection_system import (
    SystematicBiasDetector,
    BiasMonitoringConfig,
    BiasDetectionResult,
    CorrectionAction,
    BiasType,
    BiasCorrection,
    create_bias_monitoring_config
)


class TestSystematicBiasDetector:
    """Test suite for SystematicBiasDetector"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.detector = SystematicBiasDetector()
        self.model_name = "test_model"
        
        # Create monitoring configuration
        self.config = create_bias_monitoring_config(
            model_name=self.model_name,
            detection_window=50,
            auto_correction=True
        )
        self.detector.register_model(self.config)
        
        # Add helper method for testing
        self._add_run_bias_detection_with_data_method()
    
    def _create_prediction_data(self, count: int, 
                               accuracy_pattern: str = "random",
                               confidence_pattern: str = "random") -> List[Dict[str, Any]]:
        """Create test prediction data with specified patterns"""
        predictions = []
        base_time = datetime.now() - timedelta(days=count)
        
        for i in range(count):
            # Generate accuracy based on pattern
            if accuracy_pattern == "random":
                accuracy = np.random.uniform(0.4, 0.9)
            elif accuracy_pattern == "declining":
                accuracy = 0.9 - (i * 0.01)  # Decline over time
            elif accuracy_pattern == "high_for_buy":
                accuracy = 0.8 if i % 2 == 0 else 0.5  # Alternating BUY/SELL with bias
            elif accuracy_pattern == "low_confidence_high_accuracy":
                accuracy = 0.85  # High accuracy regardless of confidence
            else:
                accuracy = 0.7
            
            # Generate confidence based on pattern
            if confidence_pattern == "random":
                confidence = np.random.uniform(0.3, 0.95)
            elif confidence_pattern == "high":
                confidence = np.random.uniform(0.8, 0.95)
            elif confidence_pattern == "low":
                confidence = np.random.uniform(0.3, 0.6)
            elif confidence_pattern == "overconfident":
                confidence = 0.9  # High confidence
            else:
                confidence = 0.7
            
            # Determine action (alternating for directional bias tests)
            action = "BUY" if i % 2 == 0 else "SELL"
            
            prediction = {
                'prediction_id': f"pred_{i}",
                'predicted_action': action,
                'confidence': confidence,
                'actual_outcome': 'correct' if accuracy > 0.5 else 'incorrect',
                'accuracy': accuracy,
                'timestamp': base_time + timedelta(hours=i),
                'market_regime': ['bull', 'bear', 'sideways'][i % 3],
                'volatility_regime': ['low', 'medium', 'high'][i % 3]
            }
            predictions.append(prediction)
        
        return predictions
    
    def test_register_model(self):
        """Test model registration for bias monitoring"""
        new_config = create_bias_monitoring_config("new_model")
        self.detector.register_model(new_config)
        
        assert "new_model" in self.detector.monitoring_configs
        assert self.detector.monitoring_configs["new_model"] == new_config
    
    def test_add_prediction_data(self):
        """Test adding prediction data"""
        predictions = self._create_prediction_data(10)
        
        for pred in predictions:
            self.detector.add_prediction_data(self.model_name, pred)
        
        assert len(self.detector.prediction_data[self.model_name]) == 10
    
    def test_overconfidence_bias_detection(self):
        """Test detection of overconfidence bias"""
        # Create predictions with high confidence but low accuracy
        predictions = []
        for i in range(30):
            pred = {
                'prediction_id': f"overconf_{i}",
                'predicted_action': "BUY",
                'confidence': 0.9,  # High confidence
                'actual_outcome': 'incorrect',  # But poor outcomes
                'accuracy': 0.4,  # Low accuracy
                'timestamp': datetime.now() - timedelta(hours=i),
                'market_regime': 'bull',
                'volatility_regime': 'medium'
            }
            predictions.append(pred)
        
        # Run bias detection
        detected_biases = self.detector._run_bias_detection_with_data(
            self.model_name, predictions
        )
        
        # Should detect overconfidence bias
        overconfidence_biases = [b for b in detected_biases if b.bias_type == BiasType.OVERCONFIDENCE]
        assert len(overconfidence_biases) > 0
        
        bias = overconfidence_biases[0]
        assert bias.severity in ['medium', 'high', 'critical']
        assert BiasCorrection.CONFIDENCE_CALIBRATION in bias.recommended_corrections
    
    def test_underconfidence_bias_detection(self):
        """Test detection of underconfidence bias"""
        # Create predictions with low confidence but high accuracy
        predictions = []
        for i in range(30):
            pred = {
                'prediction_id': f"underconf_{i}",
                'predicted_action': "SELL",
                'confidence': 0.4,  # Low confidence
                'actual_outcome': 'correct',  # But good outcomes
                'accuracy': 0.85,  # High accuracy
                'timestamp': datetime.now() - timedelta(hours=i),
                'market_regime': 'bear',
                'volatility_regime': 'low'
            }
            predictions.append(pred)
        
        detected_biases = self.detector._run_bias_detection_with_data(
            self.model_name, predictions
        )
        
        # Should detect underconfidence bias
        underconfidence_biases = [b for b in detected_biases if b.bias_type == BiasType.UNDERCONFIDENCE]
        assert len(underconfidence_biases) > 0
        
        bias = underconfidence_biases[0]
        assert bias.severity in ['medium', 'high', 'critical']  # Large gap can be critical
        assert BiasCorrection.CONFIDENCE_CALIBRATION in bias.recommended_corrections
    
    def test_directional_bias_detection(self):
        """Test detection of directional bias"""
        predictions = []
        
        # Create BUY predictions with high accuracy
        for i in range(20):
            pred = {
                'prediction_id': f"buy_{i}",
                'predicted_action': "BUY",
                'confidence': 0.7,
                'actual_outcome': 'correct',
                'accuracy': 0.85,  # High accuracy for BUY
                'timestamp': datetime.now() - timedelta(hours=i),
                'market_regime': 'bull',
                'volatility_regime': 'medium'
            }
            predictions.append(pred)
        
        # Create SELL predictions with low accuracy
        for i in range(20):
            pred = {
                'prediction_id': f"sell_{i}",
                'predicted_action': "SELL",
                'confidence': 0.7,
                'actual_outcome': 'incorrect',
                'accuracy': 0.45,  # Low accuracy for SELL
                'timestamp': datetime.now() - timedelta(hours=i + 20),
                'market_regime': 'bear',
                'volatility_regime': 'medium'
            }
            predictions.append(pred)
        
        detected_biases = self.detector._run_bias_detection_with_data(
            self.model_name, predictions
        )
        
        # Should detect directional bias
        directional_biases = [b for b in detected_biases if b.bias_type == BiasType.DIRECTIONAL]
        assert len(directional_biases) > 0
        
        bias = directional_biases[0]
        assert bias.severity in ['medium', 'high', 'critical']  # 40% difference is critical
        assert BiasCorrection.SIGNAL_ADJUSTMENT in bias.recommended_corrections
    
    def test_regime_dependent_bias_detection(self):
        """Test detection of regime-dependent bias"""
        predictions = []
        
        # Create predictions with good performance in bull market
        for i in range(15):
            pred = {
                'prediction_id': f"bull_{i}",
                'predicted_action': "BUY",
                'confidence': 0.7,
                'actual_outcome': 'correct',
                'accuracy': 0.85,
                'timestamp': datetime.now() - timedelta(hours=i),
                'market_regime': 'bull',
                'volatility_regime': 'medium'
            }
            predictions.append(pred)
        
        # Create predictions with poor performance in bear market
        for i in range(15):
            pred = {
                'prediction_id': f"bear_{i}",
                'predicted_action': "SELL",
                'confidence': 0.7,
                'actual_outcome': 'incorrect',
                'accuracy': 0.35,
                'timestamp': datetime.now() - timedelta(hours=i + 15),
                'market_regime': 'bear',
                'volatility_regime': 'medium'
            }
            predictions.append(pred)
        
        detected_biases = self.detector._run_bias_detection_with_data(
            self.model_name, predictions
        )
        
        # Should detect regime-dependent bias
        regime_biases = [b for b in detected_biases if b.bias_type == BiasType.REGIME_DEPENDENT]
        assert len(regime_biases) > 0
        
        bias = regime_biases[0]
        assert bias.severity in ['medium', 'high', 'critical']
        assert BiasCorrection.FEATURE_ENGINEERING in bias.recommended_corrections
    
    def test_temporal_bias_detection(self):
        """Test detection of temporal bias (performance degradation over time)"""
        predictions = []
        
        # Create predictions with declining accuracy over time
        for i in range(40):
            accuracy = 0.9 - (i * 0.015)  # Decline from 90% to 30%
            pred = {
                'prediction_id': f"temporal_{i}",
                'predicted_action': "BUY" if i % 2 == 0 else "SELL",
                'confidence': 0.7,
                'actual_outcome': 'correct' if accuracy > 0.5 else 'incorrect',
                'accuracy': max(0.1, accuracy),
                'timestamp': datetime.now() - timedelta(hours=40-i),  # Chronological order
                'market_regime': 'bull',
                'volatility_regime': 'medium'
            }
            predictions.append(pred)
        
        detected_biases = self.detector._run_bias_detection_with_data(
            self.model_name, predictions
        )
        
        # Should detect temporal bias
        temporal_biases = [b for b in detected_biases if b.bias_type == BiasType.TEMPORAL]
        assert len(temporal_biases) > 0
        
        bias = temporal_biases[0]
        assert bias.severity in ['medium', 'high', 'critical']
        assert BiasCorrection.MODEL_RETRAINING in bias.recommended_corrections
    
    def test_volatility_dependent_bias_detection(self):
        """Test detection of volatility-dependent bias"""
        predictions = []
        
        # Good performance in low volatility
        for i in range(15):
            pred = {
                'prediction_id': f"low_vol_{i}",
                'predicted_action': "BUY",
                'confidence': 0.7,
                'actual_outcome': 'correct',
                'accuracy': 0.85,
                'timestamp': datetime.now() - timedelta(hours=i),
                'market_regime': 'bull',
                'volatility_regime': 'low'
            }
            predictions.append(pred)
        
        # Poor performance in high volatility
        for i in range(15):
            pred = {
                'prediction_id': f"high_vol_{i}",
                'predicted_action': "SELL",
                'confidence': 0.7,
                'actual_outcome': 'incorrect',
                'accuracy': 0.35,
                'timestamp': datetime.now() - timedelta(hours=i + 15),
                'market_regime': 'bear',
                'volatility_regime': 'high'
            }
            predictions.append(pred)
        
        detected_biases = self.detector._run_bias_detection_with_data(
            self.model_name, predictions
        )
        
        # Should detect volatility-dependent bias
        vol_biases = [b for b in detected_biases if b.bias_type == BiasType.VOLATILITY_DEPENDENT]
        assert len(vol_biases) > 0
        
        bias = vol_biases[0]
        assert bias.severity in ['medium', 'high', 'critical']
        assert BiasCorrection.THRESHOLD_ADJUSTMENT in bias.recommended_corrections
    
    def test_recency_bias_detection(self):
        """Test detection of recency bias"""
        predictions = []
        
        # Create predictions with stable long-term performance but recent spike
        for i in range(50):
            if i < 40:
                # Long-term stable performance
                accuracy = 0.7
            else:
                # Recent spike in performance
                accuracy = 0.95
            
            pred = {
                'prediction_id': f"recency_{i}",
                'predicted_action': "BUY" if i % 2 == 0 else "SELL",
                'confidence': 0.7,
                'actual_outcome': 'correct' if accuracy > 0.5 else 'incorrect',
                'accuracy': accuracy,
                'timestamp': datetime.now() - timedelta(hours=50-i),
                'market_regime': 'bull',
                'volatility_regime': 'medium'
            }
            predictions.append(pred)
        
        detected_biases = self.detector._run_bias_detection_with_data(
            self.model_name, predictions
        )
        
        # Should detect recency bias
        recency_biases = [b for b in detected_biases if b.bias_type == BiasType.RECENCY]
        assert len(recency_biases) > 0
        
        bias = recency_biases[0]
        assert BiasCorrection.WEIGHT_REBALANCING in bias.recommended_corrections
    
    def test_automatic_correction_application(self):
        """Test automatic application of bias corrections"""
        # Create a high-confidence bias detection
        bias_result = BiasDetectionResult(
            bias_id="test_bias_001",
            bias_type=BiasType.OVERCONFIDENCE,
            severity="high",
            confidence=0.9,  # High confidence should trigger auto-correction
            detection_time=datetime.now(),
            affected_models=[self.model_name],
            affected_symbols=[],
            evidence=[],
            impact_assessment={'accuracy_loss': 0.15},
            recommended_corrections=[BiasCorrection.CONFIDENCE_CALIBRATION],
            description="Test overconfidence bias"
        )
        
        # Apply automatic corrections
        self.detector._apply_automatic_corrections(bias_result)
        
        # Check that correction was applied
        assert len(self.detector.correction_history) > 0
        correction = self.detector.correction_history[0]
        assert correction.bias_id == bias_result.bias_id
        assert correction.correction_type == BiasCorrection.CONFIDENCE_CALIBRATION
        assert correction.status == 'applied'
        
        # Check that model calibration was updated
        assert self.model_name in self.detector.model_calibrations
        assert 'confidence_multiplier' in self.detector.model_calibrations[self.model_name]
    
    def test_confidence_calibration_correction(self):
        """Test confidence calibration correction"""
        bias_result = BiasDetectionResult(
            bias_id="test_overconf",
            bias_type=BiasType.OVERCONFIDENCE,
            severity="medium",
            confidence=0.8,
            detection_time=datetime.now(),
            affected_models=[self.model_name],
            affected_symbols=[],
            evidence=[],
            impact_assessment={},
            recommended_corrections=[BiasCorrection.CONFIDENCE_CALIBRATION],
            description="Test"
        )
        
        action = CorrectionAction(
            action_id="test_action",
            bias_id=bias_result.bias_id,
            correction_type=BiasCorrection.CONFIDENCE_CALIBRATION,
            target_models=[self.model_name],
            parameters={},
            expected_impact={},
            implementation_time=datetime.now(),
            status='pending'
        )
        
        # Apply correction
        success = self.detector._apply_confidence_calibration(action, bias_result)
        assert success is True
        
        # Check calibration was applied
        calibration = self.detector.model_calibrations[self.model_name]
        assert 'confidence_multiplier' in calibration
        assert calibration['confidence_multiplier'] == 0.9  # Reduced for overconfidence
    
    def test_signal_adjustment_correction(self):
        """Test signal adjustment correction"""
        bias_result = BiasDetectionResult(
            bias_id="test_directional",
            bias_type=BiasType.DIRECTIONAL,
            severity="medium",
            confidence=0.8,
            detection_time=datetime.now(),
            affected_models=[self.model_name],
            affected_symbols=[],
            evidence=[],
            impact_assessment={'bias_direction': 'BUY'},
            recommended_corrections=[BiasCorrection.SIGNAL_ADJUSTMENT],
            description="Test"
        )
        
        action = CorrectionAction(
            action_id="test_signal_action",
            bias_id=bias_result.bias_id,
            correction_type=BiasCorrection.SIGNAL_ADJUSTMENT,
            target_models=[self.model_name],
            parameters={},
            expected_impact={},
            implementation_time=datetime.now(),
            status='pending'
        )
        
        # Apply correction
        success = self.detector._apply_signal_adjustment(action, bias_result)
        assert success is True
        
        # Check adjustment was applied
        calibration = self.detector.model_calibrations[self.model_name]
        assert 'buy_threshold_adjustment' in calibration
    
    def test_callback_notifications(self):
        """Test callback notifications for bias detection"""
        callback_events = []
        
        def test_callback(event_type, event_data):
            callback_events.append((event_type, event_data))
        
        self.detector.add_callback(test_callback)
        
        # Create a bias detection that should trigger callback
        bias_result = BiasDetectionResult(
            bias_id="callback_test",
            bias_type=BiasType.OVERCONFIDENCE,
            severity="medium",
            confidence=0.8,
            detection_time=datetime.now(),
            affected_models=[self.model_name],
            affected_symbols=[],
            evidence=[],
            impact_assessment={},
            recommended_corrections=[],
            description="Test callback"
        )
        
        # Trigger notification
        self.detector._notify_bias_detection(bias_result)
        
        # Check callback was called
        assert len(callback_events) == 1
        event_type, event_data = callback_events[0]
        assert event_type == 'bias_detected'
        assert event_data['bias_id'] == 'callback_test'
    
    def test_bias_report_generation(self):
        """Test comprehensive bias report generation"""
        # Add some detection history
        for i in range(5):
            bias = BiasDetectionResult(
                bias_id=f"report_test_{i}",
                bias_type=list(BiasType)[i % len(BiasType)],
                severity=['low', 'medium', 'high'][i % 3],
                confidence=0.8,
                detection_time=datetime.now() - timedelta(days=i),
                affected_models=[self.model_name],
                affected_symbols=[],
                evidence=[],
                impact_assessment={},
                recommended_corrections=[],
                description=f"Test bias {i}"
            )
            self.detector.detection_history.append(bias)
        
        # Generate report
        report = self.detector.get_bias_report(model_name=self.model_name, days=30)
        
        # Check report structure
        assert 'total_detections' in report
        assert 'detections_by_type' in report
        assert 'severity_distribution' in report
        assert 'recent_detections' in report
        assert 'bias_trends' in report
        
        assert report['total_detections'] == 5
        assert len(report['recent_detections']) <= 10
    
    def test_bias_monitoring_config_creation(self):
        """Test bias monitoring configuration creation"""
        config = create_bias_monitoring_config(
            model_name="config_test_model",
            detection_window=200,
            enabled_bias_types=[BiasType.OVERCONFIDENCE, BiasType.DIRECTIONAL],
            auto_correction=False
        )
        
        assert config.model_name == "config_test_model"
        assert config.detection_window == 200
        assert len(config.enabled_bias_types) == 2
        assert BiasType.OVERCONFIDENCE in config.enabled_bias_types
        assert BiasType.DIRECTIONAL in config.enabled_bias_types
        assert config.correction_enabled is False
    
    def test_severity_calculation(self):
        """Test bias severity calculation"""
        config = BiasMonitoringConfig(model_name="test")
        
        # Test different deviation levels
        assert self.detector._calculate_severity(0.01, config.severity_thresholds) == 'low'
        assert self.detector._calculate_severity(0.07, config.severity_thresholds) == 'medium'
        assert self.detector._calculate_severity(0.15, config.severity_thresholds) == 'high'
        assert self.detector._calculate_severity(0.25, config.severity_thresholds) == 'critical'
    
    def test_prediction_data_validation(self):
        """Test validation of prediction data"""
        # Test with incomplete data
        incomplete_pred = {
            'prediction_id': 'incomplete',
            'predicted_action': 'BUY'
            # Missing required fields
        }
        
        # Should not crash, but should log warning
        self.detector.add_prediction_data(self.model_name, incomplete_pred)
        
        # Prediction should not be added
        assert len(self.detector.prediction_data[self.model_name]) == 0
    
    def test_bias_trend_calculation(self):
        """Test bias trend calculation"""
        # Create detections over multiple weeks
        detections = []
        base_time = datetime.now() - timedelta(weeks=4)
        
        for week in range(4):
            for day in range(7):
                if week < 2:  # First 2 weeks: few detections
                    count = 1 if day < 2 else 0
                else:  # Last 2 weeks: more detections
                    count = 1 if day < 5 else 0
                
                for _ in range(count):
                    detection = BiasDetectionResult(
                        bias_id=f"trend_test_{week}_{day}",
                        bias_type=BiasType.OVERCONFIDENCE,
                        severity="medium",
                        confidence=0.8,
                        detection_time=base_time + timedelta(weeks=week, days=day),
                        affected_models=[self.model_name],
                        affected_symbols=[],
                        evidence=[],
                        impact_assessment={},
                        recommended_corrections=[],
                        description="Trend test"
                    )
                    detections.append(detection)
        
        # Calculate trends
        trends = self.detector._calculate_bias_trends(detections)
        
        assert 'trend_direction' in trends
        assert 'weekly_counts' in trends
        assert trends['trend_direction'] in ['increasing', 'decreasing', 'stable']
    
    # Helper method for testing (would be added to the main class for testing)
    def _add_run_bias_detection_with_data_method(self):
        """Add helper method to detector for testing"""
        def _run_bias_detection_with_data(self, model_name: str, predictions: List[Dict]) -> List[BiasDetectionResult]:
            """Helper method for testing - run bias detection with provided data"""
            # Temporarily replace prediction data
            original_data = self.prediction_data[model_name].copy()
            self.prediction_data[model_name].clear()
            self.prediction_data[model_name].extend(predictions)
            
            # Run detection
            result = self._run_bias_detection(model_name)
            
            # Restore original data
            self.prediction_data[model_name] = original_data
            
            return result
        
        # Bind method to detector
        import types
        self.detector._run_bias_detection_with_data = types.MethodType(_run_bias_detection_with_data, self.detector)


if __name__ == "__main__":
    pytest.main([__file__])