"""
Systematic Bias Detection and Correction System

This module implements comprehensive bias detection and correction including:
- Multiple types of systematic bias detection
- Automated corrective measures
- Bias monitoring and alerting
- Model calibration adjustments
- Performance impact assessment
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import statistics
from scipy import stats
from collections import defaultdict, deque
import warnings

logger = logging.getLogger(__name__)


class BiasType(Enum):
    OVERCONFIDENCE = "overconfidence"
    UNDERCONFIDENCE = "underconfidence"
    DIRECTIONAL = "directional"
    REGIME_DEPENDENT = "regime_dependent"
    VOLATILITY_DEPENDENT = "volatility_dependent"
    TEMPORAL = "temporal"
    CONFIRMATION = "confirmation"
    ANCHORING = "anchoring"
    RECENCY = "recency"
    SURVIVORSHIP = "survivorship"
    SELECTION = "selection"


class BiasCorrection(Enum):
    CONFIDENCE_CALIBRATION = "confidence_calibration"
    SIGNAL_ADJUSTMENT = "signal_adjustment"
    WEIGHT_REBALANCING = "weight_rebalancing"
    THRESHOLD_ADJUSTMENT = "threshold_adjustment"
    MODEL_RETRAINING = "model_retraining"
    FEATURE_ENGINEERING = "feature_engineering"
    ENSEMBLE_REWEIGHTING = "ensemble_reweighting"


@dataclass
class BiasEvidence:
    evidence_type: str
    metric_name: str
    expected_value: float
    observed_value: float
    deviation: float
    statistical_significance: float
    sample_size: int
    confidence_interval: Tuple[float, float]
    description: str


@dataclass
class BiasDetectionResult:
    bias_id: str
    bias_type: BiasType
    severity: str  # 'low', 'medium', 'high', 'critical'
    confidence: float  # 0.0 to 1.0
    detection_time: datetime
    affected_models: List[str]
    affected_symbols: List[str]
    evidence: List[BiasEvidence]
    impact_assessment: Dict[str, float]
    recommended_corrections: List[BiasCorrection]
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CorrectionAction:
    action_id: str
    bias_id: str
    correction_type: BiasCorrection
    target_models: List[str]
    parameters: Dict[str, Any]
    expected_impact: Dict[str, float]
    implementation_time: datetime
    status: str  # 'pending', 'applied', 'failed', 'reverted'
    actual_impact: Optional[Dict[str, float]] = None
    completion_time: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class BiasMonitoringConfig:
    model_name: str
    detection_window: int = 100  # Number of predictions to analyze
    min_sample_size: int = 20    # Minimum samples for bias detection
    significance_threshold: float = 0.05  # Statistical significance threshold
    severity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'low': 0.02, 'medium': 0.05, 'high': 0.10, 'critical': 0.20
    })
    enabled_bias_types: List[BiasType] = field(default_factory=lambda: list(BiasType))
    correction_enabled: bool = True
    auto_correction_threshold: float = 0.7  # Auto-apply corrections above this confidence


class SystematicBiasDetector:
    def __init__(self):
        self.monitoring_configs: Dict[str, BiasMonitoringConfig] = {}
        self.detection_history: List[BiasDetectionResult] = []
        self.correction_history: List[CorrectionAction] = []
        self.prediction_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.model_calibrations: Dict[str, Dict[str, Any]] = {}
        self.callbacks: List[Callable] = []
        
        # Bias detection methods registry
        self.bias_detectors = {
            BiasType.OVERCONFIDENCE: self._detect_overconfidence_bias,
            BiasType.UNDERCONFIDENCE: self._detect_underconfidence_bias,
            BiasType.DIRECTIONAL: self._detect_directional_bias,
            BiasType.REGIME_DEPENDENT: self._detect_regime_dependent_bias,
            BiasType.VOLATILITY_DEPENDENT: self._detect_volatility_dependent_bias,
            BiasType.TEMPORAL: self._detect_temporal_bias,
            BiasType.CONFIRMATION: self._detect_confirmation_bias,
            BiasType.ANCHORING: self._detect_anchoring_bias,
            BiasType.RECENCY: self._detect_recency_bias,
            BiasType.SURVIVORSHIP: self._detect_survivorship_bias,
            BiasType.SELECTION: self._detect_selection_bias
        }
        
        # Correction methods registry
        self.correction_methods = {
            BiasCorrection.CONFIDENCE_CALIBRATION: self._apply_confidence_calibration,
            BiasCorrection.SIGNAL_ADJUSTMENT: self._apply_signal_adjustment,
            BiasCorrection.WEIGHT_REBALANCING: self._apply_weight_rebalancing,
            BiasCorrection.THRESHOLD_ADJUSTMENT: self._apply_threshold_adjustment,
            BiasCorrection.ENSEMBLE_REWEIGHTING: self._apply_ensemble_reweighting
        }
    
    def register_model(self, config: BiasMonitoringConfig) -> None:
        try:
            self.monitoring_configs[config.model_name] = config
            logger.info(f"Registered model for bias monitoring: {config.model_name}")
            
        except Exception as e:
            logger.error(f"Error registering model for bias monitoring: {e}")
    
    def add_prediction_data(self, model_name: str, prediction_data: Dict[str, Any]) -> None:
        try:
            # Ensure required fields are present
            required_fields = ['prediction_id', 'predicted_action', 'confidence', 
                             'actual_outcome', 'accuracy', 'timestamp']
            
            if not all(field in prediction_data for field in required_fields):
                logger.warning(f"Incomplete prediction data for {model_name}")
                return
            
            # Add to prediction data store
            self.prediction_data[model_name].append(prediction_data)
            
            # Check if we should run bias detection
            config = self.monitoring_configs.get(model_name)
            if config and len(self.prediction_data[model_name]) >= config.detection_window:
                self._run_bias_detection(model_name)
                
        except Exception as e:
            logger.error(f"Error adding prediction data for {model_name}: {e}")
    
    def _run_bias_detection(self, model_name: str) -> List[BiasDetectionResult]:
        try:
            config = self.monitoring_configs.get(model_name)
            if not config:
                return []
            
            predictions = list(self.prediction_data[model_name])
            if len(predictions) < config.min_sample_size:
                return []
            
            detected_biases = []
            
            # Run each enabled bias detector
            for bias_type in config.enabled_bias_types:
                if bias_type in self.bias_detectors:
                    try:
                        bias_result = self.bias_detectors[bias_type](model_name, predictions, config)
                        if bias_result:
                            detected_biases.append(bias_result)
                            
                    except Exception as e:
                        logger.error(f"Error detecting {bias_type.value} bias for {model_name}: {e}")
            
            # Store detection results
            for bias in detected_biases:
                self.detection_history.append(bias)
                self._notify_bias_detection(bias)
                
                # Apply automatic corrections if enabled and confidence is high
                if config.correction_enabled and bias.confidence >= config.auto_correction_threshold:
                    self._apply_automatic_corrections(bias)
            
            # Keep only recent detection history
            if len(self.detection_history) > 1000:
                self.detection_history = self.detection_history[-1000:]
            
            return detected_biases
            
        except Exception as e:
            logger.error(f"Error running bias detection for {model_name}: {e}")
            return []
    
    def _detect_overconfidence_bias(self, model_name: str, predictions: List[Dict],
                                   config: BiasMonitoringConfig) -> Optional[BiasDetectionResult]:
        try:
            # Filter high confidence predictions
            high_conf_predictions = [p for p in predictions if p['confidence'] > 0.8]
            
            if len(high_conf_predictions) < config.min_sample_size:
                return None
            
            # Calculate accuracy for high confidence predictions
            high_conf_accuracy = np.mean([p['accuracy'] for p in high_conf_predictions])
            
            # Expected accuracy should be close to confidence level
            expected_accuracy = np.mean([p['confidence'] for p in high_conf_predictions])
            
            # Calculate deviation
            deviation = expected_accuracy - high_conf_accuracy
            
            # Statistical test
            accuracies = [p['accuracy'] for p in high_conf_predictions]
            confidences = [p['confidence'] for p in high_conf_predictions]
            
            # One-sample t-test against expected accuracy
            t_stat, p_value = stats.ttest_1samp(accuracies, expected_accuracy)
            
            # Check if bias is significant
            if p_value < config.significance_threshold and deviation > 0.1:  # 10% overconfidence
                severity = self._calculate_severity(deviation, config.severity_thresholds)
                
                evidence = BiasEvidence(
                    evidence_type="accuracy_confidence_gap",
                    metric_name="high_confidence_accuracy",
                    expected_value=expected_accuracy,
                    observed_value=high_conf_accuracy,
                    deviation=deviation,
                    statistical_significance=p_value,
                    sample_size=len(high_conf_predictions),
                    confidence_interval=stats.t.interval(0.95, len(accuracies)-1, 
                                                       loc=np.mean(accuracies), 
                                                       scale=stats.sem(accuracies)),
                    description=f"High confidence predictions ({len(high_conf_predictions)}) "
                               f"have {deviation:.2%} lower accuracy than expected"
                )
                
                return BiasDetectionResult(
                    bias_id=f"{model_name}_overconfidence_{int(datetime.now().timestamp())}",
                    bias_type=BiasType.OVERCONFIDENCE,
                    severity=severity,
                    confidence=1.0 - p_value,  # Higher significance = higher confidence
                    detection_time=datetime.now(),
                    affected_models=[model_name],
                    affected_symbols=[],  # Would need symbol info in predictions
                    evidence=[evidence],
                    impact_assessment={'accuracy_loss': deviation, 'affected_predictions': len(high_conf_predictions)},
                    recommended_corrections=[BiasCorrection.CONFIDENCE_CALIBRATION],
                    description=f"Overconfidence bias detected: {deviation:.2%} accuracy gap"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting overconfidence bias: {e}")
            return None
    
    def _detect_underconfidence_bias(self, model_name: str, predictions: List[Dict],
                                     config: BiasMonitoringConfig) -> Optional[BiasDetectionResult]:
        try:
            # Filter low confidence predictions
            low_conf_predictions = [p for p in predictions if p['confidence'] < 0.6]
            
            if len(low_conf_predictions) < config.min_sample_size:
                return None
            
            # Calculate accuracy for low confidence predictions
            low_conf_accuracy = np.mean([p['accuracy'] for p in low_conf_predictions])
            expected_accuracy = np.mean([p['confidence'] for p in low_conf_predictions])
            
            # Calculate deviation (positive means underconfident)
            deviation = low_conf_accuracy - expected_accuracy
            
            # Statistical test
            accuracies = [p['accuracy'] for p in low_conf_predictions]
            t_stat, p_value = stats.ttest_1samp(accuracies, expected_accuracy)
            
            # Check if bias is significant
            if p_value < config.significance_threshold and deviation > 0.1:  # 10% underconfidence
                severity = self._calculate_severity(deviation, config.severity_thresholds)
                
                evidence = BiasEvidence(
                    evidence_type="accuracy_confidence_gap",
                    metric_name="low_confidence_accuracy",
                    expected_value=expected_accuracy,
                    observed_value=low_conf_accuracy,
                    deviation=deviation,
                    statistical_significance=p_value,
                    sample_size=len(low_conf_predictions),
                    confidence_interval=stats.t.interval(0.95, len(accuracies)-1, 
                                                       loc=np.mean(accuracies), 
                                                       scale=stats.sem(accuracies)),
                    description=f"Low confidence predictions ({len(low_conf_predictions)}) "
                               f"have {deviation:.2%} higher accuracy than expected"
                )
                
                return BiasDetectionResult(
                    bias_id=f"{model_name}_underconfidence_{int(datetime.now().timestamp())}",
                    bias_type=BiasType.UNDERCONFIDENCE,
                    severity=severity,
                    confidence=1.0 - p_value,
                    detection_time=datetime.now(),
                    affected_models=[model_name],
                    affected_symbols=[],
                    evidence=[evidence],
                    impact_assessment={'missed_opportunities': deviation, 'affected_predictions': len(low_conf_predictions)},
                    recommended_corrections=[BiasCorrection.CONFIDENCE_CALIBRATION],
                    description=f"Underconfidence bias detected: {deviation:.2%} accuracy gap"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting underconfidence bias: {e}")
            return None
    
    def _detect_directional_bias(self, model_name: str, predictions: List[Dict],
                                 config: BiasMonitoringConfig) -> Optional[BiasDetectionResult]:
        try:
            # Separate predictions by action
            buy_predictions = [p for p in predictions if p['predicted_action'].upper() == 'BUY']
            sell_predictions = [p for p in predictions if p['predicted_action'].upper() == 'SELL']
            
            if len(buy_predictions) < config.min_sample_size or len(sell_predictions) < config.min_sample_size:
                return None
            
            # Calculate accuracies
            buy_accuracy = np.mean([p['accuracy'] for p in buy_predictions])
            sell_accuracy = np.mean([p['accuracy'] for p in sell_predictions])
            
            # Calculate difference
            accuracy_difference = abs(buy_accuracy - sell_accuracy)
            
            # Statistical test (two-sample t-test)
            buy_accuracies = [p['accuracy'] for p in buy_predictions]
            sell_accuracies = [p['accuracy'] for p in sell_predictions]
            
            t_stat, p_value = stats.ttest_ind(buy_accuracies, sell_accuracies)
            
            # Check if bias is significant
            if p_value < config.significance_threshold and accuracy_difference > 0.1:  # 10% difference
                severity = self._calculate_severity(accuracy_difference, config.severity_thresholds)
                
                bias_direction = "BUY" if buy_accuracy > sell_accuracy else "SELL"
                
                evidence = BiasEvidence(
                    evidence_type="directional_accuracy_difference",
                    metric_name="buy_vs_sell_accuracy",
                    expected_value=0.0,  # Expected no difference
                    observed_value=accuracy_difference,
                    deviation=accuracy_difference,
                    statistical_significance=p_value,
                    sample_size=len(buy_predictions) + len(sell_predictions),
                    confidence_interval=(accuracy_difference - 0.05, accuracy_difference + 0.05),
                    description=f"Significant accuracy difference: BUY {buy_accuracy:.2%} vs SELL {sell_accuracy:.2%}"
                )
                
                return BiasDetectionResult(
                    bias_id=f"{model_name}_directional_{int(datetime.now().timestamp())}",
                    bias_type=BiasType.DIRECTIONAL,
                    severity=severity,
                    confidence=1.0 - p_value,
                    detection_time=datetime.now(),
                    affected_models=[model_name],
                    affected_symbols=[],
                    evidence=[evidence],
                    impact_assessment={'accuracy_imbalance': accuracy_difference, 'bias_direction': bias_direction},
                    recommended_corrections=[BiasCorrection.SIGNAL_ADJUSTMENT, BiasCorrection.WEIGHT_REBALANCING],
                    description=f"Directional bias detected: {bias_direction} predictions are {accuracy_difference:.2%} more accurate"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting directional bias: {e}")
            return None
    
    def _detect_regime_dependent_bias(self, model_name: str, predictions: List[Dict],
                                      config: BiasMonitoringConfig) -> Optional[BiasDetectionResult]:
        try:
            # Group predictions by market regime (if available)
            regime_groups = defaultdict(list)
            for p in predictions:
                regime = p.get('market_regime', 'unknown')
                regime_groups[regime].append(p)
            
            # Need at least 2 regimes with sufficient data
            valid_regimes = {regime: preds for regime, preds in regime_groups.items() 
                           if len(preds) >= config.min_sample_size}
            
            if len(valid_regimes) < 2:
                return None
            
            # Calculate accuracy by regime
            regime_accuracies = {}
            for regime, preds in valid_regimes.items():
                regime_accuracies[regime] = np.mean([p['accuracy'] for p in preds])
            
            # Find regime with worst performance
            worst_regime = min(regime_accuracies, key=regime_accuracies.get)
            best_regime = max(regime_accuracies, key=regime_accuracies.get)
            
            accuracy_gap = regime_accuracies[best_regime] - regime_accuracies[worst_regime]
            
            # Statistical test (ANOVA across regimes)
            regime_accuracy_lists = [[p['accuracy'] for p in preds] for preds in valid_regimes.values()]
            f_stat, p_value = stats.f_oneway(*regime_accuracy_lists)
            
            # Check if bias is significant
            if p_value < config.significance_threshold and accuracy_gap > 0.15:  # 15% gap
                severity = self._calculate_severity(accuracy_gap, config.severity_thresholds)
                
                evidence = BiasEvidence(
                    evidence_type="regime_performance_gap",
                    metric_name="regime_accuracy_variance",
                    expected_value=0.0,
                    observed_value=accuracy_gap,
                    deviation=accuracy_gap,
                    statistical_significance=p_value,
                    sample_size=sum(len(preds) for preds in valid_regimes.values()),
                    confidence_interval=(accuracy_gap - 0.05, accuracy_gap + 0.05),
                    description=f"Performance varies significantly across regimes: "
                               f"{worst_regime} ({regime_accuracies[worst_regime]:.2%}) vs "
                               f"{best_regime} ({regime_accuracies[best_regime]:.2%})"
                )
                
                return BiasDetectionResult(
                    bias_id=f"{model_name}_regime_{int(datetime.now().timestamp())}",
                    bias_type=BiasType.REGIME_DEPENDENT,
                    severity=severity,
                    confidence=1.0 - p_value,
                    detection_time=datetime.now(),
                    affected_models=[model_name],
                    affected_symbols=[],
                    evidence=[evidence],
                    impact_assessment={'worst_regime': worst_regime, 'accuracy_gap': accuracy_gap},
                    recommended_corrections=[BiasCorrection.FEATURE_ENGINEERING, BiasCorrection.MODEL_RETRAINING],
                    description=f"Regime-dependent bias: poor performance in {worst_regime} regime"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting regime-dependent bias: {e}")
            return None
    
    def _detect_volatility_dependent_bias(self, model_name: str, predictions: List[Dict],
                                          config: BiasMonitoringConfig) -> Optional[BiasDetectionResult]:
        try:
            # Group predictions by volatility level (if available)
            volatility_groups = {'low': [], 'medium': [], 'high': []}
            
            for p in predictions:
                volatility = p.get('volatility_regime', 'medium')
                if volatility in volatility_groups:
                    volatility_groups[volatility].append(p)
            
            # Filter groups with sufficient data
            valid_groups = {vol: preds for vol, preds in volatility_groups.items() 
                          if len(preds) >= config.min_sample_size}
            
            if len(valid_groups) < 2:
                return None
            
            # Calculate accuracy by volatility
            vol_accuracies = {}
            for vol, preds in valid_groups.items():
                vol_accuracies[vol] = np.mean([p['accuracy'] for p in preds])
            
            # Check for significant differences
            accuracy_values = list(vol_accuracies.values())
            accuracy_gap = max(accuracy_values) - min(accuracy_values)
            
            # Statistical test
            vol_accuracy_lists = [[p['accuracy'] for p in preds] for preds in valid_groups.values()]
            f_stat, p_value = stats.f_oneway(*vol_accuracy_lists)
            
            if p_value < config.significance_threshold and accuracy_gap > 0.12:  # 12% gap
                severity = self._calculate_severity(accuracy_gap, config.severity_thresholds)
                
                worst_vol = min(vol_accuracies, key=vol_accuracies.get)
                best_vol = max(vol_accuracies, key=vol_accuracies.get)
                
                evidence = BiasEvidence(
                    evidence_type="volatility_performance_gap",
                    metric_name="volatility_accuracy_variance",
                    expected_value=0.0,
                    observed_value=accuracy_gap,
                    deviation=accuracy_gap,
                    statistical_significance=p_value,
                    sample_size=sum(len(preds) for preds in valid_groups.values()),
                    confidence_interval=(accuracy_gap - 0.05, accuracy_gap + 0.05),
                    description=f"Performance varies by volatility: "
                               f"{worst_vol} ({vol_accuracies[worst_vol]:.2%}) vs "
                               f"{best_vol} ({vol_accuracies[best_vol]:.2%})"
                )
                
                return BiasDetectionResult(
                    bias_id=f"{model_name}_volatility_{int(datetime.now().timestamp())}",
                    bias_type=BiasType.VOLATILITY_DEPENDENT,
                    severity=severity,
                    confidence=1.0 - p_value,
                    detection_time=datetime.now(),
                    affected_models=[model_name],
                    affected_symbols=[],
                    evidence=[evidence],
                    impact_assessment={'worst_volatility': worst_vol, 'accuracy_gap': accuracy_gap},
                    recommended_corrections=[BiasCorrection.FEATURE_ENGINEERING, BiasCorrection.THRESHOLD_ADJUSTMENT],
                    description=f"Volatility-dependent bias: poor performance in {worst_vol} volatility"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting volatility-dependent bias: {e}")
            return None
    
    def _detect_temporal_bias(self, model_name: str, predictions: List[Dict],
                              config: BiasMonitoringConfig) -> Optional[BiasDetectionResult]:
        try:
            # Sort predictions by timestamp
            sorted_predictions = sorted(predictions, key=lambda x: x['timestamp'])
            
            if len(sorted_predictions) < config.min_sample_size * 2:
                return None
            
            # Split into early and recent periods
            split_point = len(sorted_predictions) // 2
            early_predictions = sorted_predictions[:split_point]
            recent_predictions = sorted_predictions[split_point:]
            
            # Calculate accuracies
            early_accuracy = np.mean([p['accuracy'] for p in early_predictions])
            recent_accuracy = np.mean([p['accuracy'] for p in recent_predictions])
            
            # Calculate temporal drift
            temporal_drift = early_accuracy - recent_accuracy  # Positive means degradation
            
            # Statistical test
            early_accuracies = [p['accuracy'] for p in early_predictions]
            recent_accuracies = [p['accuracy'] for p in recent_predictions]
            
            t_stat, p_value = stats.ttest_ind(early_accuracies, recent_accuracies)
            
            if p_value < config.significance_threshold and temporal_drift > 0.08:  # 8% degradation
                severity = self._calculate_severity(temporal_drift, config.severity_thresholds)
                
                evidence = BiasEvidence(
                    evidence_type="temporal_performance_drift",
                    metric_name="accuracy_over_time",
                    expected_value=early_accuracy,
                    observed_value=recent_accuracy,
                    deviation=temporal_drift,
                    statistical_significance=p_value,
                    sample_size=len(sorted_predictions),
                    confidence_interval=stats.t.interval(0.95, len(recent_accuracies)-1, 
                                                       loc=np.mean(recent_accuracies), 
                                                       scale=stats.sem(recent_accuracies)),
                    description=f"Performance degraded over time: {early_accuracy:.2%} -> {recent_accuracy:.2%}"
                )
                
                return BiasDetectionResult(
                    bias_id=f"{model_name}_temporal_{int(datetime.now().timestamp())}",
                    bias_type=BiasType.TEMPORAL,
                    severity=severity,
                    confidence=1.0 - p_value,
                    detection_time=datetime.now(),
                    affected_models=[model_name],
                    affected_symbols=[],
                    evidence=[evidence],
                    impact_assessment={'performance_drift': temporal_drift, 'degradation_rate': temporal_drift / split_point},
                    recommended_corrections=[BiasCorrection.MODEL_RETRAINING],
                    description=f"Temporal bias: {temporal_drift:.2%} performance degradation over time"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting temporal bias: {e}")
            return None
    
    def _detect_confirmation_bias(self, model_name: str, predictions: List[Dict],
                                   config: BiasMonitoringConfig) -> Optional[BiasDetectionResult]:
        # This would require more complex analysis of prediction patterns
        # For now, return None as this is a complex bias to detect automatically
        return None
    
    def _detect_anchoring_bias(self, model_name: str, predictions: List[Dict],
                                config: BiasMonitoringConfig) -> Optional[BiasDetectionResult]:
        # This would require analysis of how predictions change with new information
        # For now, return None as this requires more sophisticated analysis
        return None
    
    def _detect_recency_bias(self, model_name: str, predictions: List[Dict],
                              config: BiasMonitoringConfig) -> Optional[BiasDetectionResult]:
        try:
            # Sort predictions by timestamp
            sorted_predictions = sorted(predictions, key=lambda x: x['timestamp'])
            
            if len(sorted_predictions) < config.min_sample_size:
                return None
            
            # Calculate rolling accuracy to see if recent performance is overweighted
            window_size = min(20, len(sorted_predictions) // 4)
            rolling_accuracies = []
            
            for i in range(window_size, len(sorted_predictions)):
                window_accuracy = np.mean([p['accuracy'] for p in sorted_predictions[i-window_size:i]])
                rolling_accuracies.append(window_accuracy)
            
            if len(rolling_accuracies) < 10:
                return None
            
            # Check if recent accuracy is significantly different from long-term average
            recent_accuracy = np.mean(rolling_accuracies[-5:])  # Last 5 windows
            long_term_accuracy = np.mean(rolling_accuracies[:-5])  # Earlier windows
            
            recency_bias = abs(recent_accuracy - long_term_accuracy)
            
            # Statistical test
            t_stat, p_value = stats.ttest_ind(rolling_accuracies[-5:], rolling_accuracies[:-5])
            
            if p_value < config.significance_threshold and recency_bias > 0.1:  # 10% bias
                severity = self._calculate_severity(recency_bias, config.severity_thresholds)
                
                evidence = BiasEvidence(
                    evidence_type="recency_performance_bias",
                    metric_name="recent_vs_longterm_accuracy",
                    expected_value=long_term_accuracy,
                    observed_value=recent_accuracy,
                    deviation=recency_bias,
                    statistical_significance=p_value,
                    sample_size=len(rolling_accuracies),
                    confidence_interval=(recency_bias - 0.05, recency_bias + 0.05),
                    description=f"Recent performance differs significantly from long-term: "
                               f"{recent_accuracy:.2%} vs {long_term_accuracy:.2%}"
                )
                
                return BiasDetectionResult(
                    bias_id=f"{model_name}_recency_{int(datetime.now().timestamp())}",
                    bias_type=BiasType.RECENCY,
                    severity=severity,
                    confidence=1.0 - p_value,
                    detection_time=datetime.now(),
                    affected_models=[model_name],
                    affected_symbols=[],
                    evidence=[evidence],
                    impact_assessment={'recency_bias': recency_bias},
                    recommended_corrections=[BiasCorrection.WEIGHT_REBALANCING],
                    description=f"Recency bias: {recency_bias:.2%} overweighting of recent performance"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting recency bias: {e}")
            return None
    
    def _detect_survivorship_bias(self, model_name: str, predictions: List[Dict],
                                   config: BiasMonitoringConfig) -> Optional[BiasDetectionResult]:
        # This would require analysis of what data is being excluded
        # For now, return None as this requires access to excluded data
        return None
    
    def _detect_selection_bias(self, model_name: str, predictions: List[Dict],
                                config: BiasMonitoringConfig) -> Optional[BiasDetectionResult]:
        # This would require analysis of data selection patterns
        # For now, return None as this requires more sophisticated analysis
        return None
    
    def _calculate_severity(self, deviation: float, thresholds: Dict[str, float]) -> str:
        abs_deviation = abs(deviation)
        
        if abs_deviation >= thresholds['critical']:
            return 'critical'
        elif abs_deviation >= thresholds['high']:
            return 'high'
        elif abs_deviation >= thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _apply_automatic_corrections(self, bias_result: BiasDetectionResult) -> None:
        try:
            for correction_type in bias_result.recommended_corrections:
                if correction_type in self.correction_methods:
                    action = CorrectionAction(
                        action_id=f"{bias_result.bias_id}_correction_{correction_type.value}",
                        bias_id=bias_result.bias_id,
                        correction_type=correction_type,
                        target_models=bias_result.affected_models,
                        parameters={},  # Would be populated based on bias specifics
                        expected_impact={'bias_reduction': 0.5},  # Estimated
                        implementation_time=datetime.now(),
                        status='pending'
                    )
                    
                    # Apply correction
                    success = self.correction_methods[correction_type](action, bias_result)
                    
                    if success:
                        action.status = 'applied'
                        action.completion_time = datetime.now()
                        logger.info(f"Applied automatic correction {correction_type.value} for bias {bias_result.bias_id}")
                    else:
                        action.status = 'failed'
                        action.error_message = "Correction application failed"
                    
                    self.correction_history.append(action)
                    
        except Exception as e:
            logger.error(f"Error applying automatic corrections: {e}")
    
    def _apply_confidence_calibration(self, action: CorrectionAction,
                                      bias_result: BiasDetectionResult) -> bool:
        try:
            for model_name in action.target_models:
                if model_name not in self.model_calibrations:
                    self.model_calibrations[model_name] = {}
                
                # Adjust confidence calibration based on bias type
                if bias_result.bias_type == BiasType.OVERCONFIDENCE:
                    # Reduce confidence for high-confidence predictions
                    self.model_calibrations[model_name]['confidence_multiplier'] = 0.9
                elif bias_result.bias_type == BiasType.UNDERCONFIDENCE:
                    # Increase confidence for low-confidence predictions
                    self.model_calibrations[model_name]['confidence_multiplier'] = 1.1
                
                logger.info(f"Applied confidence calibration for {model_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying confidence calibration: {e}")
            return False
    
    def _apply_signal_adjustment(self, action: CorrectionAction,
                                  bias_result: BiasDetectionResult) -> bool:
        try:
            for model_name in action.target_models:
                if model_name not in self.model_calibrations:
                    self.model_calibrations[model_name] = {}
                
                # Adjust signals based on bias type
                if bias_result.bias_type == BiasType.DIRECTIONAL:
                    # Adjust thresholds for the biased direction
                    bias_direction = bias_result.impact_assessment.get('bias_direction')
                    if bias_direction == 'BUY':
                        self.model_calibrations[model_name]['buy_threshold_adjustment'] = 0.05
                    else:
                        self.model_calibrations[model_name]['sell_threshold_adjustment'] = 0.05
                
                logger.info(f"Applied signal adjustment for {model_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying signal adjustment: {e}")
            return False
    
    def _apply_weight_rebalancing(self, action: CorrectionAction,
                                   bias_result: BiasDetectionResult) -> bool:
        try:
            for model_name in action.target_models:
                if model_name not in self.model_calibrations:
                    self.model_calibrations[model_name] = {}
                
                # Rebalance weights based on bias
                if bias_result.bias_type == BiasType.RECENCY:
                    # Reduce weight of recent predictions
                    self.model_calibrations[model_name]['recency_weight_reduction'] = 0.1
                
                logger.info(f"Applied weight rebalancing for {model_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying weight rebalancing: {e}")
            return False
    
    def _apply_threshold_adjustment(self, action: CorrectionAction,
                                     bias_result: BiasDetectionResult) -> bool:
        try:
            for model_name in action.target_models:
                if model_name not in self.model_calibrations:
                    self.model_calibrations[model_name] = {}
                
                # Adjust decision thresholds
                if bias_result.bias_type == BiasType.VOLATILITY_DEPENDENT:
                    worst_vol = bias_result.impact_assessment.get('worst_volatility')
                    if worst_vol:
                        self.model_calibrations[model_name][f'{worst_vol}_volatility_threshold'] = 0.05
                
                logger.info(f"Applied threshold adjustment for {model_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying threshold adjustment: {e}")
            return False
    
    def _apply_ensemble_reweighting(self, action: CorrectionAction,
                                     bias_result: BiasDetectionResult) -> bool:
        try:
            # This would integrate with ensemble systems to reweight models
            logger.info("Applied ensemble reweighting correction")
            return True
            
        except Exception as e:
            logger.error(f"Error applying ensemble reweighting: {e}")
            return False
    
    def _notify_bias_detection(self, bias_result: BiasDetectionResult) -> None:
        for callback in self.callbacks:
            try:
                callback('bias_detected', bias_result.__dict__)
            except Exception as e:
                logger.error(f"Bias detection callback error: {e}")
    
    def add_callback(self, callback: Callable) -> None:
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable) -> None:
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def get_bias_report(self, model_name: Optional[str] = None,
                         days: int = 30) -> Dict[str, Any]:
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            # Filter recent detections
            recent_detections = [
                bias for bias in self.detection_history
                if bias.detection_time >= cutoff_time and 
                (model_name is None or model_name in bias.affected_models)
            ]
            
            # Group by bias type
            bias_by_type = defaultdict(list)
            for bias in recent_detections:
                bias_by_type[bias.bias_type.value].append(bias)
            
            # Calculate statistics
            total_detections = len(recent_detections)
            severity_counts = defaultdict(int)
            for bias in recent_detections:
                severity_counts[bias.severity] += 1
            
            # Recent corrections
            recent_corrections = [
                action for action in self.correction_history
                if action.implementation_time >= cutoff_time and
                (model_name is None or any(m in action.target_models for m in [model_name] if model_name))
            ]
            
            return {
                'period_days': days,
                'model_name': model_name,
                'total_detections': total_detections,
                'detections_by_type': {bias_type: len(detections) for bias_type, detections in bias_by_type.items()},
                'severity_distribution': dict(severity_counts),
                'recent_detections': [bias.__dict__ for bias in recent_detections[-10:]],
                'recent_corrections': [action.__dict__ for action in recent_corrections[-10:]],
                'active_calibrations': self.model_calibrations.get(model_name, {}) if model_name else self.model_calibrations,
                'bias_trends': self._calculate_bias_trends(recent_detections)
            }
            
        except Exception as e:
            logger.error(f"Error generating bias report: {e}")
            return {'error': str(e)}
    
    def _calculate_bias_trends(self, detections: List[BiasDetectionResult]) -> Dict[str, Any]:
        try:
            if not detections:
                return {}
            
            # Group by week
            weekly_counts = defaultdict(int)
            for bias in detections:
                week = bias.detection_time.strftime('%Y-W%U')
                weekly_counts[week] += 1
            
            # Calculate trend
            weeks = sorted(weekly_counts.keys())
            counts = [weekly_counts[week] for week in weeks]
            
            if len(counts) > 1:
                # Simple linear trend
                x = np.arange(len(counts))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, counts)
                trend_direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
            else:
                trend_direction = 'insufficient_data'
                slope = 0
            
            return {
                'trend_direction': trend_direction,
                'trend_slope': slope,
                'weekly_counts': dict(weekly_counts),
                'total_weeks': len(weeks)
            }
            
        except Exception as e:
            logger.error(f"Error calculating bias trends: {e}")
            return {}


# Convenience functions

def create_bias_monitoring_config(model_name: str,
                                 detection_window: int = 100,
                                 enabled_bias_types: Optional[List[BiasType]] = None,
                                 auto_correction: bool = True) -> BiasMonitoringConfig:
    if enabled_bias_types is None:
        enabled_bias_types = [
            BiasType.OVERCONFIDENCE,
            BiasType.UNDERCONFIDENCE,
            BiasType.DIRECTIONAL,
            BiasType.REGIME_DEPENDENT,
            BiasType.VOLATILITY_DEPENDENT,
            BiasType.TEMPORAL,
            BiasType.RECENCY
        ]
    
    return BiasMonitoringConfig(
        model_name=model_name,
        detection_window=detection_window,
        enabled_bias_types=enabled_bias_types,
        correction_enabled=auto_correction
    )