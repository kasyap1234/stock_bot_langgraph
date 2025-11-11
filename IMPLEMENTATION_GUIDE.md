# Stock Bot Accuracy Improvements - Implementation Guide

## Overview

This guide documents the accuracy improvements implemented in the stock bot. The improvements focus on three key areas:

1. **Feature Engineering Enhancement** - Reduce multicollinearity and improve feature quality
2. **Target Definition Improvement** - Use adaptive thresholds and risk-adjusted returns
3. **Adaptive Ensemble Weighting** - Track performance and adjust model weights dynamically

---

## Changes Made

### 1. Feature Engineering Improvements

**File**: `agents/feature_engineering.py`

#### Changes:
- Added correlation filtering to remove highly correlated features (threshold: 0.95)
- Implemented adaptive target thresholds based on volatility
- Better class distribution logging

#### Key Methods:
```python
def filter_correlated_features(self, X: pd.DataFrame, correlation_threshold: float = 0.95)
    """Remove highly correlated features to reduce multicollinearity"""

def prepare_training_data(self, features: pd.DataFrame, target_horizon: int = 5)
    """Create targets with adaptive thresholds based on volatility"""
```

#### Benefits:
- Reduces model overfitting by removing redundant features
- Adaptive thresholds better capture market conditions
- Improved feature quality leads to better model generalization

---

### 2. Accuracy Improvement Module

**File**: `agents/accuracy_improvement.py` (NEW)

A comprehensive module for tracking and improving prediction accuracy with four main components:

#### A. FeatureCorrelationAnalyzer
Analyzes and removes highly correlated features:
```python
analyzer = FeatureCorrelationAnalyzer(correlation_threshold=0.95)
X_filtered = analyzer.filter_features(X)
report = analyzer.get_correlation_report()
```

**Use Cases**:
- Identify redundant features
- Reduce multicollinearity
- Improve model interpretability

#### B. ImprovedTargetDefinition
Creates better classification targets:
```python
target_improver = ImprovedTargetDefinition(horizons=[3, 5, 10])
targets = target_improver.create_multi_horizon_targets(df)
risk_adjusted_target = target_improver.create_risk_adjusted_target(df)
X_resampled, y_resampled = target_improver.handle_class_imbalance(X, y)
```

**Features**:
- Multi-horizon targets (3, 5, 10 days)
- Risk-adjusted returns (Sharpe-like metric)
- Class imbalance handling with SMOTE
- Adaptive thresholds based on volatility

#### C. AdaptiveEnsembleWeighter
Tracks predictions and adjusts weights based on performance:
```python
weighter = AdaptiveEnsembleWeighter(lookback_periods=20)
weighter.track_prediction(symbol, model_name, prediction, actual, confidence)
weights = weighter.calculate_adaptive_weights(symbol)
report = weighter.get_performance_report(symbol)
```

**Metrics Tracked**:
- Prediction accuracy
- Confidence calibration
- Prediction error
- Historical performance

#### D. PerformanceTracker
Records and analyzes system performance:
```python
tracker = PerformanceTracker(history_file="bot_performance_history.json")
tracker.record_prediction(symbol, action, confidence, reasoning)
tracker.record_outcome(symbol, timestamp, actual_return, predicted_action)
metrics = tracker.get_accuracy_metrics(lookback_days=30)
symbol_metrics = tracker.get_symbol_metrics(symbol)
```

**Capabilities**:
- Persistent prediction history
- Accuracy metrics by action (BUY/SELL/HOLD)
- Per-symbol performance tracking
- Outcome recording and analysis

---

## Integration Points

### 1. Feature Engineering Pipeline

The correlation filtering is automatically applied in `feature_engineering_agent`:

```python
# In feature_engineering_agent()
X_filtered = engineer.filter_correlated_features(X, correlation_threshold=0.95)
features_with_target = X_filtered.copy()
features_with_target['target'] = y
```

**Impact**: Features are now cleaner and less redundant, improving model performance.

---

### 2. Advanced ML Training

The improved targets and class imbalance handling can be integrated:

```python
from agents.accuracy_improvement import ImprovedTargetDefinition

target_improver = ImprovedTargetDefinition()
targets = target_improver.create_multi_horizon_targets(df)
X_resampled, y_resampled = target_improver.handle_class_imbalance(X, y)
```

**Impact**: Better target definitions and balanced training data improve model accuracy.

---

### 3. Ensemble Weighting

Adaptive weights can be used in the recommendation engine:

```python
from agents.accuracy_improvement import AdaptiveEnsembleWeighter

weighter = AdaptiveEnsembleWeighter()
for symbol in symbols:
    weights = weighter.calculate_adaptive_weights(symbol)
    # Use weights in ensemble combination
```

**Impact**: Ensemble predictions are weighted by recent performance, improving accuracy.

---

### 4. Performance Monitoring

Track system performance over time:

```python
from agents.accuracy_improvement import PerformanceTracker

tracker = PerformanceTracker()
# After making a prediction
tracker.record_prediction(symbol, action, confidence, reasoning)
# After outcome is known
tracker.record_outcome(symbol, timestamp, actual_return, predicted_action)
# Get metrics
metrics = tracker.get_accuracy_metrics(lookback_days=30)
```

**Impact**: Continuous monitoring enables feedback loops and performance optimization.

---

## Usage Examples

### Example 1: Using the Accuracy Improvement Agent

```python
from agents.accuracy_improvement import AccuracyImprovementAgent

agent = AccuracyImprovementAgent()

# Improve features
X_improved = agent.improve_features(X)

# Improve targets
targets = agent.improve_targets(df)

# Get adaptive weights
weights = agent.get_adaptive_weights(symbol)

# Get performance report
report = agent.get_performance_report()
```

### Example 2: Tracking Predictions

```python
from agents.accuracy_improvement import PerformanceTracker

tracker = PerformanceTracker()

# Record a prediction
tracker.record_prediction(
    symbol='RELIANCE.NS',
    action='BUY',
    confidence=0.85,
    reasoning='Strong technical signals with positive ML consensus'
)

# Later, record the outcome
tracker.record_outcome(
    symbol='RELIANCE.NS',
    timestamp=datetime.now(),
    actual_return=0.05,  # 5% return
    predicted_action='BUY'
)

# Get metrics
metrics = tracker.get_accuracy_metrics(lookback_days=30)
print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"BUY accuracy: {metrics['by_action']['BUY']['accuracy']:.2%}")
```

### Example 3: Adaptive Ensemble Weighting

```python
from agents.accuracy_improvement import AdaptiveEnsembleWeighter

weighter = AdaptiveEnsembleWeighter(lookback_periods=20)

# Track predictions from different models
for model_name in ['random_forest', 'xgboost', 'neural_net']:
    prediction = model.predict(X)[0]
    actual = y_test[0]
    confidence = model.predict_proba(X)[0].max()
    
    weighter.track_prediction(
        symbol='RELIANCE.NS',
        model_name=model_name,
        prediction=prediction,
        actual=actual,
        confidence=confidence
    )

# Get adaptive weights based on recent performance
weights = weighter.calculate_adaptive_weights('RELIANCE.NS')
print(f"Adaptive weights: {weights}")

# Get detailed performance report
report = weighter.get_performance_report('RELIANCE.NS')
for model, metrics in report.items():
    print(f"{model}: accuracy={metrics['accuracy']:.2%}, weight={metrics['weight']:.3f}")
```

---

## Key Improvements Summary

| Component | Improvement | Impact |
|-----------|------------|--------|
| **Feature Engineering** | Correlation filtering | Reduced multicollinearity, better generalization |
| **Target Definition** | Adaptive thresholds + multi-horizon | Better signal quality, captures market conditions |
| **Ensemble Weighting** | Adaptive based on performance | More accurate predictions, self-improving |
| **Performance Tracking** | Persistent history + metrics | Enables feedback loops and optimization |

---

## Next Steps

1. **Integrate into Workflow**: Add accuracy improvement steps to the main trading workflow
2. **Monitor Performance**: Use PerformanceTracker to monitor system accuracy over time
3. **Tune Parameters**: Adjust correlation threshold, lookback periods, and target horizons based on results
4. **Feedback Loop**: Implement automatic weight adjustment based on tracked performance
5. **Backtesting**: Validate improvements with comprehensive backtesting

---

## Configuration

### Feature Correlation Filtering
```python
correlation_threshold = 0.95  # Drop features with >95% correlation
```

### Target Definition
```python
horizons = [3, 5, 10]  # Prediction horizons in days
risk_adjustment = True  # Use risk-adjusted returns
```

### Adaptive Weighting
```python
lookback_periods = 20  # Use last 20 predictions for weight calculation
min_history = 5  # Minimum predictions before calculating weights
```

### Performance Tracking
```python
history_file = "bot_performance_history.json"  # File to store history
lookback_days = 30  # Days for accuracy metrics
```

---

## Troubleshooting

### Issue: Too many features dropped
**Solution**: Increase correlation_threshold (e.g., 0.98 instead of 0.95)

### Issue: Class imbalance not handled
**Solution**: Ensure imbalanced-learn is installed: `pip install imbalanced-learn`

### Issue: Weights not updating
**Solution**: Ensure min_history threshold is met (default: 5 predictions)

### Issue: Performance history not persisting
**Solution**: Check file permissions and disk space for history_file location

---

## References

- Feature correlation analysis: Reduces multicollinearity and improves model stability
- Adaptive thresholds: Better capture market regime changes
- SMOTE: Handles class imbalance in training data
- Performance tracking: Enables continuous improvement and feedback loops
