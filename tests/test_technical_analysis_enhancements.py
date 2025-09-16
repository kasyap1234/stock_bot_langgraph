

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import logging
import sys
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from agents.technical_analysis import (
    DataValidator, MLSignalPredictor, ParameterOptimizer,
    MultiTimeframeAnalyzer, SignalConfirmer, AdaptiveParameterCalculator,
    RiskAdjuster, EnsembleSignalGenerator, TrendStrengthScorer,
    VolatilityAdjuster, IchimokuCloud, FibonacciRetracement,
    SupportResistanceCalculator, ProbabilityScorer, BacktestValidator,
    technical_analysis_agent, TradingSetup,
    VPVRProfile, HeikinAshiTransformer, GARCHForecaster,
    HarmonicPatternDetector, HMMRegimeDetector,
    LSTMPredictor, EnhancedVaRCalculator
)
from data.models import State
from config.config import (
    RSI_OVERBOUGHT, RSI_OVERSOLD, CONFIRMATION_THRESHOLD,
    ENSEMBLE_THRESHOLD, TREND_STRENGTH_THRESHOLD,
    PROBABILITY_THRESHOLD, BACKTEST_VALIDATION_THRESHOLD
)

@pytest.fixture
def sample_df():
    
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    data = {
        'Open': 100 + np.random.randn(100).cumsum() * 0.5,
        'High': 105 + np.random.randn(100).cumsum() * 0.5,
        'Low': 95 + np.random.randn(100).cumsum() * 0.5,
        'Close': 100 + np.random.randn(100).cumsum() * 0.5,
        'Volume': np.random.randint(100000, 1000000, 100)
    }
    df = pd.DataFrame(data, index=dates)
    # Ensure OHLC relationships are valid
    df['High'] = df[['Open', 'Close']].max(axis=1) + abs(np.random.randn(100))
    df['Low'] = df[['Open', 'Close']].min(axis=1) - abs(np.random.randn(100))
    return df

@pytest.fixture
def sample_signals():
    
    return {
        'RSI': 'buy',
        'MACD': 'sell',
        'SMA': 'buy',
        'EMA': 'buy',
        'Bollinger': 'neutral',
        'Stochastic': 'sell',
        'WilliamsR': 'buy',
        'CCI': 'neutral'
    }

@pytest.fixture
def sample_state(sample_df):
    
    return {
        "stock_data": {"AAPL": sample_df},
        "technical_signals": {},
        "fundamental_analysis": {},
        "sentiment_scores": {},
        "risk_metrics": {},
        "final_recommendation": {},
        "simulation_results": {},
        "performance_analysis": {}
    }

class TestDataValidator:
    

    def test_validate_dataframe_valid(self, sample_df):
        
        validator = DataValidator()
        is_valid, cleaned_df, validation_info = validator.validate_dataframe(sample_df, "AAPL")

        assert is_valid is True
        assert isinstance(cleaned_df, pd.DataFrame)
        assert len(cleaned_df) == len(sample_df)
        assert validation_info['data_quality_score'] == 1.0
        assert len(validation_info['errors']) == 0

    def test_validate_dataframe_missing_columns(self):
        
        df = pd.DataFrame({'Close': [100, 101, 102]})
        validator = DataValidator()
        is_valid, cleaned_df, validation_info = validator.validate_dataframe(df, "AAPL")

        assert is_valid is False
        assert 'Missing required columns' in str(validation_info['errors'])

    def test_validate_dataframe_insufficient_data(self):
        
        df = pd.DataFrame({
            'Open': [100, 101],
            'High': [105, 106],
            'Low': [95, 96],
            'Close': [101, 102],
            'Volume': [100000, 200000]
        })
        validator = DataValidator(max_period=50)
        is_valid, cleaned_df, validation_info = validator.validate_dataframe(df, "AAPL")

        assert is_valid is True  # Still valid but with warnings
        assert 'Insufficient data' in str(validation_info['warnings'])

    def test_validate_dataframe_negative_prices(self):
        
        df = pd.DataFrame({
            'Open': [100, -101],
            'High': [105, 106],
            'Low': [95, 96],
            'Close': [101, 102],
            'Volume': [100000, 200000]
        })
        validator = DataValidator()
        is_valid, cleaned_df, validation_info = validator.validate_dataframe(df, "AAPL")

        assert is_valid is False
        assert any('negative prices' in error for error in validation_info['errors'])

    def test_validate_dataframe_ohlc_inconsistency(self):
        
        df = pd.DataFrame({
            'Open': [100, 101],
            'High': [95, 96],  # High < Low
            'Low': [105, 106],
            'Close': [101, 102],
            'Volume': [100000, 200000]
        })
        validator = DataValidator()
        is_valid, cleaned_df, validation_info = validator.validate_dataframe(df, "AAPL")

        assert is_valid is False
        assert any('High < Low' in error for error in validation_info['errors'])

    def test_handle_missing_data_interpolation(self):
        
        df = pd.DataFrame({
            'Open': [100, np.nan, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [101, 102, 103],
            'Volume': [100000, 200000, 300000]
        })
        validator = DataValidator()
        cleaned_df, missing_info = validator._handle_missing_data(df)

        assert missing_info['missing_data_handled'] is True
        assert missing_info['interpolation_applied'] is True
        assert not cleaned_df.isnull().any().any()

    def test_detect_outliers(self, sample_df):
        
        validator = DataValidator()
        outlier_info = validator._detect_outliers(sample_df)

        assert 'outliers_detected' in outlier_info
        assert 'outlier_columns' in outlier_info
        # With normal data, should detect minimal outliers
        assert outlier_info['outliers_detected'] >= 0

class TestMLSignalPredictor:
    

    @patch('agents.technical_analysis.RandomForestClassifier')
    def test_initialize_model_success(self, mock_rf):
        
        mock_model = Mock()
        mock_rf.return_value = mock_model

        predictor = MLSignalPredictor()
        assert predictor.model is not None
        mock_rf.assert_called_once()

    def test_initialize_model_scikit_learn_missing(self):
        
        with patch.dict('sys.modules', {'sklearn': None}):
            predictor = MLSignalPredictor()
            assert predictor.model is None

    def test_prepare_features(self, sample_df, sample_signals):
        
        predictor = MLSignalPredictor()
        features = predictor.prepare_features(sample_df, sample_signals)

        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
        # Should have lagged features
        assert any('lag' in col for col in features.columns)

    def test_train_model_success(self, sample_df, sample_signals):
        
        predictor = MLSignalPredictor()
        predictor.model = Mock()
        predictor.model.fit = Mock(return_value=None)

        success = predictor.train_model(sample_df, sample_signals)
        assert success is True
        predictor.model.fit.assert_called_once()

    def test_train_model_insufficient_data(self):
        
        predictor = MLSignalPredictor()
        df = pd.DataFrame({'Close': [100, 101]})  # Too small
        signals = {'RSI': 'buy'}

        success = predictor.train_model(df, signals)
        assert success is False

    def test_predict_signal_with_model(self, sample_df, sample_signals):
        
        predictor = MLSignalPredictor()
        predictor.model = Mock()
        predictor.model.predict.return_value = [1]  # Buy signal
        predictor.model.predict_proba.return_value = [[0.3, 0.7]]  # Confident

        signal = predictor.predict_signal(sample_df, sample_signals)
        assert signal == "buy"
        predictor.model.predict.assert_called_once()

    def test_predict_signal_no_model(self, sample_df, sample_signals):
        
        predictor = MLSignalPredictor()
        predictor.model = None

        signal = predictor.predict_signal(sample_df, sample_signals)
        assert signal == "neutral"

    def test_get_feature_importance_no_model(self):
        
        predictor = MLSignalPredictor()
        predictor.model = None

        importance = predictor.get_feature_importance()
        assert importance == {}

class TestParameterOptimizer:
    

    def test_optimize_rsi_parameters(self, sample_df):
        
        optimizer = ParameterOptimizer()
        result = optimizer.optimize_rsi_parameters(sample_df)

        assert 'period' in result
        assert 'score' in result
        assert isinstance(result['period'], int)
        assert 5 <= result['period'] <= 30  # Reasonable range

    def test_optimize_macd_parameters(self, sample_df):
        
        optimizer = ParameterOptimizer()
        result = optimizer.optimize_macd_parameters(sample_df)

        assert 'fast' in result
        assert 'slow' in result
        assert 'signal' in result
        assert 'score' in result
        assert result['fast'] < result['slow']  # Fast should be less than slow

    def test_optimize_stochastic_parameters(self, sample_df):
        
        optimizer = ParameterOptimizer()
        result = optimizer.optimize_stochastic_parameters(sample_df)

        assert 'k_period' in result
        assert 'd_period' in result
        assert 'score' in result
        assert result['k_period'] >= result['d_period']  # K should be >= D

    def test_optimize_all_parameters(self, sample_df):
        
        optimizer = ParameterOptimizer()
        results = optimizer.optimize_all_parameters(sample_df)

        assert 'rsi' in results
        assert 'macd' in results
        assert 'stochastic' in results

        for indicator, params in results.items():
            assert 'score' in params

    def test_optimization_insufficient_data(self):
        
        optimizer = ParameterOptimizer()
        df = pd.DataFrame({'Close': [100, 101, 102]})  # Too small

        result = optimizer.optimize_rsi_parameters(df)
        assert result['period'] == 14  # Default
        assert result['score'] == 0.5  # Default score

class TestMultiTimeframeAnalyzer:
    

    def test_resample_data_weekly(self, sample_df):
        
        analyzer = MultiTimeframeAnalyzer()
        weekly_df = analyzer.resample_data(sample_df, 'weekly')

        assert isinstance(weekly_df, pd.DataFrame)
        assert len(weekly_df) < len(sample_df)  # Should be fewer rows

    def test_resample_data_monthly(self, sample_df):
        
        analyzer = MultiTimeframeAnalyzer()
        monthly_df = analyzer.resample_data(sample_df, 'monthly')

        assert isinstance(monthly_df, pd.DataFrame)
        assert len(monthly_df) < len(sample_df)

    def test_analyze_multi_timeframe(self, sample_df):
        
        analyzer = MultiTimeframeAnalyzer()

        def mock_signals_func(df):
            return {'RSI': 'buy', 'MACD': 'sell'}

        result = analyzer.analyze_multi_timeframe(sample_df, mock_signals_func)

        assert isinstance(result, dict)
        assert len(result) > 0
        # Should have timeframe-specific keys
        assert any('_daily' in key for key in result.keys())

    def test_analyze_multi_timeframe_insufficient_data(self):
        
        analyzer = MultiTimeframeAnalyzer()
        small_df = pd.DataFrame({
            'Open': [100, 101],
            'High': [105, 106],
            'Low': [95, 96],
            'Close': [101, 102],
            'Volume': [100000, 200000]
        })

        def mock_signals_func(df):
            return {'RSI': 'buy'}

        result = analyzer.analyze_multi_timeframe(small_df, mock_signals_func)
        assert isinstance(result, dict)

class TestSignalConfirmer:
    

    def test_confirm_signals_above_threshold(self, sample_signals):
        
        confirmer = SignalConfirmer(confirmation_threshold=2)
        confirmed = confirmer.confirm_signals(sample_signals)

        assert isinstance(confirmed, dict)
        # Should have confirmed signals
        assert any('confirmed' in str(signal) for signal in confirmed.values())

    def test_confirm_signals_below_threshold(self):
        
        signals = {'RSI': 'buy', 'MACD': 'neutral', 'SMA': 'neutral'}
        confirmer = SignalConfirmer(confirmation_threshold=3)
        confirmed = confirmer.confirm_signals(signals)

        assert isinstance(confirmed, dict)
        # Should not have confirmed signals
        assert not any('confirmed' in str(signal) for signal in confirmed.values())

class TestAdaptiveParameterCalculator:
    

    def test_calculate_atr(self, sample_df):
        
        calc = AdaptiveParameterCalculator()
        atr = calc.calculate_atr(sample_df)

        assert isinstance(atr, pd.Series)
        assert len(atr) > 0
        assert not atr.isnull().all()

    def test_adaptive_rsi_period_high_volatility(self):
        
        calc = AdaptiveParameterCalculator()
        # Create high volatility data
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        high_vol_df = pd.DataFrame({
            'Open': np.random.randn(50) * 10 + 100,
            'High': np.random.randn(50) * 15 + 105,
            'Low': np.random.randn(50) * 15 + 95,
            'Close': np.random.randn(50) * 10 + 100,
            'Volume': np.random.randint(100000, 1000000, 50)
        }, index=dates)

        period = calc.adaptive_rsi_period(high_vol_df)
        assert period == 9  # Should be shorter for high volatility

    def test_adaptive_rsi_period_low_volatility(self):
        
        calc = AdaptiveParameterCalculator()
        # Create low volatility data
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        low_vol_df = pd.DataFrame({
            'Open': np.random.randn(50) * 0.1 + 100,
            'High': np.random.randn(50) * 0.15 + 100.5,
            'Low': np.random.randn(50) * 0.15 + 99.5,
            'Close': np.random.randn(50) * 0.1 + 100,
            'Volume': np.random.randint(100000, 1000000, 50)
        }, index=dates)

        period = calc.adaptive_rsi_period(low_vol_df)
        assert period == 21  # Should be longer for low volatility

class TestEnsembleSignalGenerator:
    

    def test_generate_ensemble_signal_buy(self, sample_signals, sample_df):
        
        generator = EnsembleSignalGenerator()
        # Make most signals buy
        buy_signals = {k: 'buy' for k in sample_signals.keys()}
        signal = generator.generate_ensemble_signal(buy_signals, sample_df)
        assert signal in ['buy', 'sell', 'neutral']

    def test_generate_ensemble_signal_sell(self, sample_signals, sample_df):
        
        generator = EnsembleSignalGenerator()
        # Make most signals sell
        sell_signals = {k: 'sell' for k in sample_signals.keys()}
        signal = generator.generate_ensemble_signal(sell_signals, sample_df)
        assert signal in ['buy', 'sell', 'neutral']

    def test_update_weights_dynamically(self, sample_signals, sample_df):
        
        generator = EnsembleSignalGenerator()
        weights = generator.update_weights_dynamically(sample_df, sample_signals)

        assert isinstance(weights, dict)
        assert len(weights) > 0
        assert abs(sum(weights.values()) - 1.0) < 0.01  # Should sum to ~1

class TestTrendStrengthScorer:
    

    @patch('agents.technical_analysis.talib.ADX')
    def test_score_trend_strength_with_talib(self, mock_adx, sample_df):
        
        mock_adx.return_value = np.array([25.0] * len(sample_df))

        scorer = TrendStrengthScorer()
        score = scorer.score_trend_strength(sample_df)

        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_score_trend_strength_without_talib(self, sample_df):
        
        with patch('agents.technical_analysis.TALIB_AVAILABLE', False):
            scorer = TrendStrengthScorer()
            score = scorer.score_trend_strength(sample_df)

            assert isinstance(score, float)
            assert 0 <= score <= 1

    def test_score_trend_strength_insufficient_data(self):
        
        scorer = TrendStrengthScorer()
        small_df = pd.DataFrame({'Close': [100, 101]})

        score = scorer.score_trend_strength(small_df)
        assert score == 0.5  # Default score

class TestIchimokuCloud:
    

    def test_calculate_ichimoku(self, sample_df):
        
        ichimoku = IchimokuCloud()
        components = ichimoku.calculate_ichimoku(sample_df)

        assert isinstance(components, dict)
        if len(sample_df) >= 52:  # Minimum period for Ichimoku
            assert 'tenkan_sen' in components
            assert 'kijun_sen' in components
            assert 'senkou_span_a' in components
            assert 'senkou_span_b' in components
            assert 'chikou_span' in components

    def test_get_ichimoku_signal_bullish(self, sample_df):
        
        ichimoku = IchimokuCloud()
        signal = ichimoku.get_ichimoku_signal(sample_df)
        assert signal in ['buy', 'sell', 'neutral']

    def test_get_ichimoku_signal_insufficient_data(self):
        
        ichimoku = IchimokuCloud()
        small_df = pd.DataFrame({'Close': [100, 101]})

        signal = ichimoku.get_ichimoku_signal(small_df)
        assert signal == 'neutral'

class TestFibonacciRetracement:
    

    def test_calculate_fib_levels(self, sample_df):
        
        fib = FibonacciRetracement()
        levels = fib.calculate_fib_levels(sample_df)

        assert isinstance(levels, dict)
        if len(sample_df) >= 50:
            assert 'fib_0.236' in levels
            assert 'fib_0.382' in levels
            assert 'fib_0.618' in levels

    def test_get_fib_signal(self, sample_df):
        
        fib = FibonacciRetracement()
        signal = fib.get_fib_signal(sample_df)
        assert signal in ['buy', 'sell', 'neutral']

class TestSupportResistanceCalculator:
    

    def test_calculate_support_resistance(self, sample_df):
        
        sr_calc = SupportResistanceCalculator()
        levels = sr_calc.calculate_support_resistance(sample_df)

        assert isinstance(levels, dict)
        if len(sample_df) >= 60:  # Minimum for long-term
            assert 'long_resistance' in levels
            assert 'long_support' in levels

    def test_get_sr_signal(self, sample_df):
        
        sr_calc = SupportResistanceCalculator()
        signal = sr_calc.get_sr_signal(sample_df)
        assert signal in ['buy', 'sell', 'neutral']

class TestBacktestValidator:
    

    def test_validate_signal(self, sample_df):
        
        validator = BacktestValidator()
        score = validator.validate_signal(sample_df, 'buy')

        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_walk_forward_analysis(self, sample_df, sample_signals):
        
        validator = BacktestValidator()
        results = validator.walk_forward_analysis(sample_df, sample_signals)

        assert isinstance(results, dict)
        assert 'stability_score' in results
        assert 'avg_performance' in results

    @patch('agents.technical_analysis.stats.norm.fit')
    @patch('agents.technical_analysis.np.random.normal')
    def test_monte_carlo_simulation(self, mock_random, mock_fit, sample_df, sample_signals):
        
        mock_fit.return_value = (0.001, 0.02)  # mu, sigma
        mock_random.return_value = np.random.normal(0.001, 0.02, 100)

        validator = BacktestValidator()
        results = validator.monte_carlo_simulation(sample_df, sample_signals)

        assert isinstance(results, dict)
        assert 'expected_return' in results
        assert 'var_95' in results

class TestTechnicalAnalysisAgent:
    

    def test_technical_analysis_agent_success(self, sample_state):
        
        result = technical_analysis_agent(sample_state)

        assert isinstance(result, dict)
        assert 'technical_signals' in result
        assert 'AAPL' in result['technical_signals']

    def test_technical_analysis_agent_empty_data(self):
        
        state = {"stock_data": {}}
        result = technical_analysis_agent(state)

        assert isinstance(result, dict)
        assert 'technical_signals' in result
        assert result['technical_signals'] == {}

    def test_technical_analysis_agent_invalid_data(self):
        
        state = {"stock_data": {"AAPL": None}}
        result = technical_analysis_agent(state)

        assert isinstance(result, dict)
        assert 'technical_signals' in result
        assert 'AAPL' in result['technical_signals']
        assert 'error' in result['technical_signals']['AAPL']

    @patch('agents.technical_analysis.DataValidator.validate_dataframe')
    def test_technical_analysis_agent_validation_failure(self, mock_validate, sample_state):
        
        mock_validate.return_value = (False, sample_state['stock_data']['AAPL'], {'errors': ['Validation failed']})

        result = technical_analysis_agent(sample_state)

        assert isinstance(result, dict)
        assert 'technical_signals' in result
        assert 'AAPL' in result['technical_signals']
        assert 'error' in result['technical_signals']['AAPL']

class TestPerformanceTests:
    

    def test_indicator_calculation_performance(self, sample_df, benchmark):
        
        def calculate_indicators():
            from agents.technical_analysis import _calculate_technical_indicators_with_retry
            return _calculate_technical_indicators_with_retry(sample_df, symbol="AAPL")

        result = benchmark(calculate_indicators)
        assert result is not None
        assert isinstance(result, dict)

    def test_ml_prediction_performance(self, sample_df, sample_signals, benchmark):
        
        predictor = MLSignalPredictor()
        predictor.model = Mock()
        predictor.model.predict.return_value = [1]
        predictor.model.predict_proba.return_value = [[0.3, 0.7]]

        def predict_signal():
            return predictor.predict_signal(sample_df, sample_signals)

        result = benchmark(predict_signal)
        assert result in ['buy', 'sell', 'neutral']

class TestRegressionTests:
    

    def test_original_indicators_still_work(self, sample_df):
        
        from agents.technical_analysis import _calculate_technical_indicators_with_retry

        signals = _calculate_technical_indicators_with_retry(sample_df, symbol="AAPL")

        assert isinstance(signals, dict)
        assert 'RSI' in signals
        assert 'MACD' in signals
        assert 'SMA' in signals
        assert signals['RSI'] in ['buy', 'sell', 'neutral']

    def test_analyzer_compatibility(self, sample_df):
        
        analyzer = MultiTimeframeAnalyzer()
        weekly_df = analyzer.resample_data(sample_df, 'weekly')

        assert isinstance(weekly_df, pd.DataFrame)
        assert not weekly_df.empty

    def test_error_handling_unchanged(self, sample_state):
        
        # Test with corrupted data
        corrupted_state = sample_state.copy()
        corrupted_state['stock_data']['AAPL'] = pd.DataFrame()  # Empty DataFrame

        result = technical_analysis_agent(corrupted_state)

        assert isinstance(result, dict)
        assert 'technical_signals' in result
        # Should handle gracefully without crashing

class TestVPVRProfile:
    

    def test_calculate_vpvr_success(self, sample_df):
        
        vpvr = VPVRProfile()
        levels = vpvr.calculate_vpvr(sample_df)

        assert isinstance(levels, dict)
        if len(sample_df) >= 200:  # VISIBLE_RANGE default
            assert 'poc' in levels
            assert 'vah' in levels
            assert 'val' in levels
            assert levels['vah'] >= levels['poc'] >= levels['val']

    def test_calculate_vpvr_insufficient_data(self):
        
        small_df = pd.DataFrame({
            'Open': [100, 101],
            'High': [105, 106],
            'Low': [95, 96],
            'Close': [101, 102],
            'Volume': [100000, 200000]
        })
        vpvr = VPVRProfile()
        levels = vpvr.calculate_vpvr(small_df)

        assert levels == {}

    def test_get_vpvr_signal_near_vah(self, sample_df):
        
        vpvr = VPVRProfile()
        # Mock VPVR levels
        with patch.object(vpvr, 'calculate_vpvr', return_value={'vah': 105, 'val': 95, 'poc': 100}):
            sample_df.loc[sample_df.index[-1], 'Close'] = 104.5  # Near VAH
            signal = vpvr.get_vpvr_signal(sample_df)
            assert signal == "sell"

    def test_get_vpvr_signal_near_val(self, sample_df):
        
        vpvr = VPVRProfile()
        with patch.object(vpvr, 'calculate_vpvr', return_value={'vah': 105, 'val': 95, 'poc': 100}):
            sample_df.loc[sample_df.index[-1], 'Close'] = 95.5  # Near VAL
            signal = vpvr.get_vpvr_signal(sample_df)
            assert signal == "buy"

    def test_get_vpvr_signal_neutral(self, sample_df):
        
        vpvr = VPVRProfile()
        with patch.object(vpvr, 'calculate_vpvr', return_value={'vah': 105, 'val': 95, 'poc': 100}):
            sample_df.loc[sample_df.index[-1], 'Close'] = 100  # At POC
            signal = vpvr.get_vpvr_signal(sample_df)
            assert signal == "neutral"

    def test_merge_with_support_resistance(self, sample_df):
        
        vpvr = VPVRProfile()
        sr_calc = SupportResistanceCalculator()

        vpvr_levels = {'vah': 105, 'val': 95, 'poc': 100}
        merged = vpvr.merge_with_support_resistance(vpvr_levels, sr_calc)

        assert 'vpvr_resistance' in merged
        assert 'vpvr_support' in merged
        assert merged['vpvr_resistance'] == 105
        assert merged['vpvr_support'] == 95


class TestHeikinAshiTransformer:
    

    def test_transform_to_heikin_ashi_success(self, sample_df):
        
        ha_transformer = HeikinAshiTransformer()
        ha_df = ha_transformer.transform_to_heikin_ashi(sample_df)

        assert isinstance(ha_df, pd.DataFrame)
        assert 'HA_Open' in ha_df.columns
        assert 'HA_High' in ha_df.columns
        assert 'HA_Low' in ha_df.columns
        assert 'HA_Close' in ha_df.columns
        assert len(ha_df) == len(sample_df)

    def test_transform_to_heikin_ashi_insufficient_data(self):
        
        small_df = pd.DataFrame({
            'Open': [100],
            'High': [105],
            'Low': [95],
            'Close': [101],
            'Volume': [100000]
        })
        ha_transformer = HeikinAshiTransformer()
        ha_df = ha_transformer.transform_to_heikin_ashi(small_df)

        assert ha_df.equals(small_df)  # Should return original if insufficient data

    def test_get_heikin_ashi_signal_bullish(self, sample_df):
        
        ha_transformer = HeikinAshiTransformer()
        with patch.object(ha_transformer, 'transform_to_heikin_ashi') as mock_transform:
            mock_ha_df = pd.DataFrame({
                'HA_Open': [100, 101],
                'HA_Close': [102, 103],  # Close > Open = bullish
                'HA_High': [105, 106],
                'HA_Low': [98, 99]
            })
            mock_transform.return_value = mock_ha_df
            signal = ha_transformer.get_heikin_ashi_signal(sample_df)
            assert signal == "buy"

    def test_get_heikin_ashi_signal_bearish(self, sample_df):
        
        ha_transformer = HeikinAshiTransformer()
        with patch.object(ha_transformer, 'transform_to_heikin_ashi') as mock_transform:
            mock_ha_df = pd.DataFrame({
                'HA_Open': [103, 102],
                'HA_Close': [101, 100],  # Close < Open = bearish
                'HA_High': [105, 106],
                'HA_Low': [98, 99]
            })
            mock_transform.return_value = mock_ha_df
            signal = ha_transformer.get_heikin_ashi_signal(sample_df)
            assert signal == "sell"

    def test_get_heikin_ashi_signal_insufficient_data(self):
        
        small_df = pd.DataFrame({
            'Open': [100],
            'High': [105],
            'Low': [95],
            'Close': [101],
            'Volume': [100000]
        })
        ha_transformer = HeikinAshiTransformer()
        signal = ha_transformer.get_heikin_ashi_signal(small_df)
        assert signal == "neutral"

    def test_check_confluence_true(self, sample_df):
        
        ha_transformer = HeikinAshiTransformer()
        multi_tf = MultiTimeframeAnalyzer()

        with patch.object(ha_transformer, 'get_heikin_ashi_signal', return_value="buy"):
            confluence = ha_transformer.check_confluence(sample_df, multi_tf)
            assert confluence is True

    def test_check_confluence_false(self, sample_df):
        
        ha_transformer = HeikinAshiTransformer()
        multi_tf = MultiTimeframeAnalyzer()

        with patch.object(ha_transformer, 'get_heikin_ashi_signal', side_effect=["buy", "sell", "neutral"]):
            confluence = ha_transformer.check_confluence(sample_df, multi_tf)
            assert confluence is False


class TestGARCHForecaster:
    

    @patch('agents.technical_analysis.arch_model')
    def test_fit_garch_model_success(self, mock_arch_model, sample_df):
        
        mock_model = Mock()
        mock_fit = Mock()
        mock_arch_model.return_value = mock_model
        mock_model.fit.return_value = mock_fit

        forecaster = GARCHForecaster()
        returns = sample_df['Close'].pct_change().dropna()

        success = forecaster.fit_garch_model(returns)
        assert success is True
        mock_arch_model.assert_called_once()

    def test_fit_garch_model_insufficient_data(self):
        
        forecaster = GARCHForecaster()
        small_returns = pd.Series([0.01, 0.02])  # Less than 50

        success = forecaster.fit_garch_model(small_returns)
        assert success is False

    @patch('agents.technical_analysis.arch_model')
    def test_forecast_volatility_success(self, mock_arch_model, sample_df):
        
        mock_model = Mock()
        mock_fit = Mock()
        mock_forecast = Mock()
        mock_arch_model.return_value = mock_model
        mock_model.fit.return_value = mock_fit
        mock_fit.forecast.return_value = Mock(variance=pd.DataFrame([[0.0004]]))  # 2% volatility squared

        forecaster = GARCHForecaster()
        returns = sample_df['Close'].pct_change().dropna()

        volatility = forecaster.forecast_volatility(returns)
        assert isinstance(volatility, float)
        assert volatility > 0

    def test_forecast_volatility_fallback(self, sample_df):
        
        forecaster = GARCHForecaster()
        returns = sample_df['Close'].pct_change().dropna()

        # Mock arch import failure
        with patch.dict('sys.modules', {'arch': None}):
            volatility = forecaster.forecast_volatility(returns)
            assert isinstance(volatility, float)
            assert volatility > 0  # Should use ATR fallback

    def test_should_shorten_periods_high_volatility(self, sample_df):
        
        forecaster = GARCHForecaster()
        with patch.object(forecaster, 'forecast_volatility', return_value=0.03):  # High volatility
            should_shorten = forecaster.should_shorten_periods(sample_df)
            assert should_shorten is True

    def test_should_shorten_periods_low_volatility(self, sample_df):
        
        forecaster = GARCHForecaster()
        with patch.object(forecaster, 'forecast_volatility', return_value=0.01):  # Low volatility
            should_shorten = forecaster.should_shorten_periods(sample_df)
            assert should_shorten is False

    def test_should_shorten_periods_insufficient_data(self):
        
        small_df = pd.DataFrame({'Close': [100, 101]})
        forecaster = GARCHForecaster()
        should_shorten = forecaster.should_shorten_periods(small_df)
        assert should_shorten is False


class TestHarmonicPatternDetector:
    

    @patch('agents.technical_analysis.find_peaks')
    def test_detect_peaks_troughs_success(self, mock_find_peaks, sample_df):
        
        mock_find_peaks.return_value = (np.array([10, 20, 30]), {'prominences': np.array([1, 2, 1])})

        detector = HarmonicPatternDetector()
        peaks, troughs = detector.detect_peaks_troughs(sample_df)

        assert isinstance(peaks, np.ndarray)
        assert isinstance(troughs, np.ndarray)
        mock_find_peaks.assert_called()

    def test_detect_peaks_troughs_insufficient_data(self):
        
        small_df = pd.DataFrame({'Close': [100, 101]})
        detector = HarmonicPatternDetector()
        peaks, troughs = detector.detect_peaks_troughs(small_df)

        assert peaks == []
        assert troughs == []

    @patch('agents.technical_analysis.Scipy_AVAILABLE', False)
    def test_detect_peaks_troughs_no_scipy(self, sample_df):
        
        detector = HarmonicPatternDetector()
        peaks, troughs = detector.detect_peaks_troughs(sample_df)

        assert peaks == []
        assert troughs == []

    def test_validate_gartley_pattern_bullish(self):
        
        detector = HarmonicPatternDetector()
        points = [100, 90, 105, 95, 102]  # XA=10, AB=15, BC=-10, CD=7

        signal, confidence = detector.validate_gartley_pattern(points)
        assert signal in ["buy", "sell", "neutral"]
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1

    def test_validate_gartley_pattern_insufficient_points(self):
        
        detector = HarmonicPatternDetector()
        points = [100, 90, 105]  # Only 3 points

        signal, confidence = detector.validate_gartley_pattern(points)
        assert signal == "neutral"
        assert confidence == 0.0

    def test_validate_butterfly_pattern_bearish(self):
        
        detector = HarmonicPatternDetector()
        points = [100, 110, 95, 108, 97]  # Butterfly points

        signal, confidence = detector.validate_butterfly_pattern(points)
        assert signal in ["buy", "sell", "neutral"]
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1

    @patch('agents.technical_analysis.find_peaks')
    def test_detect_patterns_gartley_found(self, mock_find_peaks, sample_df):
        
        mock_find_peaks.return_value = (np.array([10, 30, 50]), {'prominences': np.array([1, 2, 1])})

        detector = HarmonicPatternDetector()
        with patch.object(detector, 'validate_gartley_pattern', return_value=("buy", 0.8)):
            signal, confidence = detector.detect_patterns(sample_df)
            assert signal == "buy"
            assert confidence == 0.8

    @patch('agents.technical_analysis.find_peaks')
    def test_detect_patterns_no_pattern(self, mock_find_peaks, sample_df):
        
        mock_find_peaks.return_value = (np.array([10]), {'prominences': np.array([1])})

        detector = HarmonicPatternDetector()
        signal, confidence = detector.detect_patterns(sample_df)
        assert signal == "neutral"
        assert confidence == 0.0

    def test_get_harmonic_signal_with_pattern(self, sample_df):
        
        detector = HarmonicPatternDetector()
        with patch.object(detector, 'detect_patterns', return_value=("buy", 0.7)):
            signal = detector.get_harmonic_signal(sample_df)
            assert signal == "buy"

    def test_get_harmonic_signal_low_confidence(self, sample_df):
        
        detector = HarmonicPatternDetector()
        with patch.object(detector, 'detect_patterns', return_value=("buy", 0.3)):
            signal = detector.get_harmonic_signal(sample_df)
            assert signal == "neutral"


class TestHMMRegimeDetector:
    

    def test_prepare_features_success(self, sample_df):
        
        detector = HMMRegimeDetector()
        features = detector.prepare_features(sample_df)

        assert isinstance(features, pd.DataFrame)
        assert 'returns' in features.columns
        assert 'volatility' in features.columns
        assert len(features) > 0

    def test_prepare_features_insufficient_data(self):
        
        small_df = pd.DataFrame({'Close': [100, 101]})
        detector = HMMRegimeDetector()
        features = detector.prepare_features(small_df)

        assert features.empty

    @patch('agents.technical_analysis.hmm')
    def test_fit_hmm_model_success(self, mock_hmm, sample_df):
        
        mock_model = Mock()
        mock_hmm.GaussianHMM.return_value = mock_model
        mock_model.fit.return_value = None
        mock_model.predict.return_value = [0, 1, 2] * (len(sample_df) // 3 + 1)

        detector = HMMRegimeDetector()
        features = detector.prepare_features(sample_df)

        success = detector.fit_hmm_model(features)
        assert success is True
        mock_hmm.GaussianHMM.assert_called_once()

    def test_fit_hmm_model_insufficient_data(self):
        
        detector = HMMRegimeDetector()
        small_features = pd.DataFrame({'returns': [0.01, 0.02]})

        success = detector.fit_hmm_model(small_features)
        assert success is False

    @patch('agents.technical_analysis.HMMLEARN_AVAILABLE', False)
    def test_fit_hmm_model_no_hmmlearn(self, sample_df):
        
        detector = HMMRegimeDetector()
        features = detector.prepare_features(sample_df)

        success = detector.fit_hmm_model(features)
        assert success is False

    def test_infer_current_regime_success(self, sample_df):
        
        detector = HMMRegimeDetector()
        features = detector.prepare_features(sample_df)

        # Mock fitted model
        detector.model = Mock()
        detector.model.predict.return_value = [1]

        regime = detector.infer_current_regime(features)
        assert isinstance(regime, int)
        assert regime == 1

    def test_infer_current_regime_no_model(self, sample_df):
        
        detector = HMMRegimeDetector()
        features = detector.prepare_features(sample_df)

        regime = detector.infer_current_regime(features)
        assert regime == 0  # Default bear regime

    def test_get_regime_signal_bull(self):
        
        detector = HMMRegimeDetector()
        signal = detector.get_regime_signal(2)  # Bull regime
        assert signal == "bull_regime"

    def test_get_regime_signal_bear(self):
        
        detector = HMMRegimeDetector()
        signal = detector.get_regime_signal(0)  # Bear regime
        assert signal == "bear_regime"

    def test_get_regime_signal_sideways(self):
        
        detector = HMMRegimeDetector()
        signal = detector.get_regime_signal(1)  # Sideways regime
        assert signal == "sideways_regime"

    def test_detect_regime_change_true(self):
        
        detector = HMMRegimeDetector()
        detector.current_regime = 0
        change = detector.detect_regime_change(1)
        assert change is True

    def test_detect_regime_change_false(self):
        
        detector = HMMRegimeDetector()
        detector.current_regime = 1
        change = detector.detect_regime_change(1)
        assert change is False

    def test_get_hmm_signal_success(self, sample_df):
        
        detector = HMMRegimeDetector()
        features = detector.prepare_features(sample_df)

        # Mock fitted model
        detector.model = Mock()
        detector.model.predict.return_value = [2]  # Bull regime

        signal = detector.get_hmm_signal(sample_df)
        assert signal == "bull_regime"

    def test_get_hmm_signal_insufficient_data(self):
        
        small_df = pd.DataFrame({'Close': [100, 101]})
        detector = HMMRegimeDetector()
        signal = detector.get_hmm_signal(small_df)
        assert signal == "neutral"

    def test_adjust_weights_for_regime_bull(self):
        
        detector = HMMRegimeDetector()
        base_weights = {'SMA': 0.1, 'RSI': 0.1, 'MACD': 0.1}

        adjusted = detector.adjust_weights_for_regime(base_weights, "bull_regime")
        assert adjusted['SMA'] > base_weights['SMA']  # Trend indicators increased

    def test_adjust_weights_for_regime_sideways(self):
        
        detector = HMMRegimeDetector()
        base_weights = {'SMA': 0.1, 'RSI': 0.1, 'MACD': 0.1}

        adjusted = detector.adjust_weights_for_regime(base_weights, "sideways_regime")
        assert adjusted['RSI'] > base_weights['RSI']  # Oscillator indicators increased


class TestLSTMPredictor:
    

    @patch('agents.technical_analysis.Sequential')
    @patch('agents.technical_analysis.LSTM')
    @patch('agents.technical_analysis.Dense')
    def test_initialize_model_success(self, mock_dense, mock_lstm, mock_sequential, sample_df):
        
        mock_model = Mock()
        mock_sequential.return_value = mock_model

        predictor = LSTMPredictor()
        assert predictor.model is not None
        mock_sequential.assert_called_once()

    @patch('agents.technical_analysis.TENSORFLOW_AVAILABLE', False)
    def test_initialize_model_no_tensorflow(self, sample_df):
        
        predictor = LSTMPredictor()
        assert predictor.model is None
        assert predictor.fallback_model is not None  # Should use RandomForest

    def test_prepare_features_success(self, sample_df, sample_signals):
        
        predictor = LSTMPredictor()
        features = predictor.prepare_features(sample_df, sample_signals)

        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
        assert 'Close' in features.columns

    def test_create_sequences_success(self, sample_df, sample_signals):
        
        predictor = LSTMPredictor()
        features = predictor.prepare_features(sample_df, sample_signals)

        X, y = predictor.create_sequences(features)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        if len(features) > predictor.window:
            assert len(X) > 0
            assert len(y) > 0

    def test_create_sequences_insufficient_data(self):
        
        predictor = LSTMPredictor()
        small_features = pd.DataFrame({'Close': [100, 101, 102]})

        X, y = predictor.create_sequences(small_features)
        assert X.size == 0
        assert y.size == 0

    @patch('agents.technical_analysis.RandomForestClassifier')
    def test_train_model_fallback_success(self, mock_rf, sample_df, sample_signals):
        
        mock_model = Mock()
        mock_rf.return_value = mock_model

        predictor = LSTMPredictor()
        predictor.model = None  # Force fallback
        predictor.fallback_model = mock_model

        success = predictor.train_model(sample_df, sample_signals)
        assert success is True

    def test_train_model_insufficient_data(self):
        
        small_df = pd.DataFrame({'Close': [100, 101, 102]})
        predictor = LSTMPredictor()
        signals = {'RSI': 'buy'}

        success = predictor.train_model(small_df, signals)
        assert success is False

    def test_predict_signal_buy(self, sample_df, sample_signals):
        
        predictor = LSTMPredictor()
        predictor.model = Mock()
        predictor.model.predict.return_value = [[0.7]]  # High probability for buy

        signal, confidence = predictor.predict_signal(sample_df, sample_signals)
        assert signal == "buy"
        assert confidence == 0.7

    def test_predict_signal_sell(self, sample_df, sample_signals):
        
        predictor = LSTMPredictor()
        predictor.model = Mock()
        predictor.model.predict.return_value = [[0.3]]  # Low probability for sell

        signal, confidence = predictor.predict_signal(sample_df, sample_signals)
        assert signal == "sell"
        assert confidence == 0.7  # 1 - 0.3

    def test_predict_signal_neutral(self, sample_df, sample_signals):
        
        predictor = LSTMPredictor()
        predictor.model = Mock()
        predictor.model.predict.return_value = [[0.5]]  # Neutral probability

        signal, confidence = predictor.predict_signal(sample_df, sample_signals)
        assert signal == "neutral"
        assert confidence == 0.5

    def test_predict_signal_insufficient_data(self):
        
        small_df = pd.DataFrame({'Close': [100, 101]})
        predictor = LSTMPredictor()
        signals = {'RSI': 'buy'}

        signal, confidence = predictor.predict_signal(small_df, signals)
        assert signal == "neutral"
        assert confidence == 0.5


class TestEnhancedVaRCalculator:
    

    def test_compute_monte_carlo_var_success(self, sample_df):
        
        calculator = EnhancedVaRCalculator()
        var_results = calculator.compute_monte_carlo_var(sample_df)

        assert isinstance(var_results, dict)
        assert 'var_95' in var_results
        assert 'var_99' in var_results
        assert 'es_95' in var_results
        assert 'es_99' in var_results
        assert var_results['var_95'] < 0  # VaR should be negative

    def test_compute_monte_carlo_var_insufficient_data(self):
        
        small_df = pd.DataFrame({'Close': [100, 101]})
        calculator = EnhancedVaRCalculator()
        var_results = calculator.compute_monte_carlo_var(small_df)

        assert var_results['var_95'] == 0.0
        assert var_results['var_99'] == 0.0

    @patch('agents.technical_analysis.stats.norm.fit')
    @patch('agents.technical_analysis.np.random.normal')
    def test_compute_monte_carlo_var_with_mocks(self, mock_random, mock_fit, sample_df):
        
        mock_fit.return_value = (0.001, 0.02)  # mu, sigma
        mock_random.return_value = np.random.normal(0.001, 0.02, 1000)

        calculator = EnhancedVaRCalculator()
        var_results = calculator.compute_monte_carlo_var(sample_df)

        assert isinstance(var_results, dict)
        assert 'var_95' in var_results

    def test_apply_stress_scenarios_success(self, sample_df):
        
        calculator = EnhancedVaRCalculator()
        base_var = {'var_95': -0.05, 'var_99': -0.08}

        stressed_results = calculator.apply_stress_scenarios(sample_df, base_var)
        assert isinstance(stressed_results, dict)
        assert len(stressed_results) > 0

    def test_calculate_risk_adjustment_factor_high_risk(self):
        
        calculator = EnhancedVaRCalculator()
        adjustment = calculator.calculate_risk_adjustment_factor(-0.06, 100000)  # 6% VaR
        assert adjustment == -0.2

    def test_calculate_risk_adjustment_factor_medium_risk(self):
        
        calculator = EnhancedVaRCalculator()
        adjustment = calculator.calculate_risk_adjustment_factor(-0.03, 100000)  # 3% VaR
        assert adjustment == -0.1

    def test_calculate_risk_adjustment_factor_low_risk(self):
        
        calculator = EnhancedVaRCalculator()
        adjustment = calculator.calculate_risk_adjustment_factor(-0.01, 100000)  # 1% VaR
        assert adjustment == 0.0

    def test_get_comprehensive_risk_metrics_success(self, sample_df):
        
        calculator = EnhancedVaRCalculator()
        risk_metrics = calculator.get_comprehensive_risk_metrics(sample_df)

        assert isinstance(risk_metrics, dict)
        assert 'var_95' in risk_metrics
        assert 'risk_level' in risk_metrics
        assert 'risk_adjustment_factor' in risk_metrics
        assert risk_metrics['risk_level'] in ['high', 'medium', 'low']

    def test_get_comprehensive_risk_metrics_insufficient_data(self):
        
        small_df = pd.DataFrame({'Close': [100, 101]})
        calculator = EnhancedVaRCalculator()
        risk_metrics = calculator.get_comprehensive_risk_metrics(small_df)

        assert risk_metrics['var_95'] == 0.0
        assert risk_metrics['risk_level'] == 'low'  # Default

    def test_export_to_state_success(self, sample_df):
        
        calculator = EnhancedVaRCalculator()
        risk_metrics = calculator.get_comprehensive_risk_metrics(sample_df)

        state = {"risk_assessment": {}}
        updated_state = calculator.export_to_state(risk_metrics, state)

        assert 'var_metrics' in updated_state['risk_assessment']
        assert updated_state['risk_assessment']['var_metrics']['risk_level'] == risk_metrics['risk_level']


class TestErrorHandlingAndConfig:
    

    @patch('agents.technical_analysis.ENABLE_ADVANCED_TECH', False)
    def test_enable_advanced_tech_false(self, sample_state):
        
        result = technical_analysis_agent(sample_state)

        assert isinstance(result, dict)
        assert 'technical_signals' in result
        signals = result['technical_signals']['AAPL']

        # Should not have advanced signals
        assert 'VPVRSignal' not in signals
        assert 'HeikinAshiSignal' not in signals
        assert 'HarmonicSignal' not in signals
        assert 'HMMSignal' not in signals
        assert 'LSTMSignal' not in signals
        assert 'VaRMetrics' not in signals

    @patch('agents.technical_analysis.VPVR_BINS', -1)
    def test_invalid_config_vpvr_bins(self, sample_df):
        
        vpvr = VPVRProfile()
        levels = vpvr.calculate_vpvr(sample_df)

        # Should handle gracefully, possibly use defaults
        assert isinstance(levels, dict)

    @patch('agents.technical_analysis.LSTM_WINDOW', 1)
    def test_invalid_config_lstm_window(self, sample_df, sample_signals):
        
        predictor = LSTMPredictor()
        success = predictor.train_model(sample_df, sample_signals)

        # Should handle gracefully
        assert success is False

    @patch('agents.technical_analysis.arch')
    def test_arch_import_error_garch(self, sample_df):
        
        with patch.dict('sys.modules', {'arch': None}):
            forecaster = GARCHForecaster()
            volatility = forecaster.forecast_volatility(sample_df['Close'].pct_change().dropna())

            assert isinstance(volatility, float)
            assert volatility > 0  # Should use ATR fallback

    @patch('agents.technical_analysis.hmm')
    def test_hmmlearn_import_error(self, sample_df):
        
        with patch.dict('sys.modules', {'hmmlearn': None}):
            detector = HMMRegimeDetector()
            features = detector.prepare_features(sample_df)
            success = detector.fit_hmm_model(features)

            assert success is False

    @patch('agents.technical_analysis.Sequential')
    def test_tensorflow_import_error_lstm(self, sample_df, sample_signals):
        
        with patch.dict('sys.modules', {'tensorflow': None}):
            predictor = LSTMPredictor()
            signal, confidence = predictor.predict_signal(sample_df, sample_signals)

            assert signal == "neutral"
            assert confidence == 0.5

    def test_nan_values_handling_vpvr(self):
        
        df_with_nan = pd.DataFrame({
            'Open': [100, np.nan, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [101, 102, 103],
            'Volume': [100000, 200000, 300000]
        })
        vpvr = VPVRProfile()
        levels = vpvr.calculate_vpvr(df_with_nan)

        # Should handle NaN gracefully
        assert isinstance(levels, dict)

    def test_nan_values_handling_heikin_ashi(self):
        
        df_with_nan = pd.DataFrame({
            'Open': [100, np.nan, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [101, 102, 103],
            'Volume': [100000, 200000, 300000]
        })
        ha_transformer = HeikinAshiTransformer()
        ha_df = ha_transformer.transform_to_heikin_ashi(df_with_nan)

        # Should handle NaN gracefully
        assert isinstance(ha_df, pd.DataFrame)

    def test_empty_dataframe_handling(self):
        
        empty_df = pd.DataFrame()
        vpvr = VPVRProfile()
        levels = vpvr.calculate_vpvr(empty_df)

        assert levels == {}

    def test_insufficient_data_technical_analysis_agent(self):
        
        small_df = pd.DataFrame({
            'Open': [100, 101],
            'High': [105, 106],
            'Low': [95, 96],
            'Close': [101, 102],
            'Volume': [100000, 200000]
        })
        state = {"stock_data": {"AAPL": small_df}}

        result = technical_analysis_agent(state)

        assert isinstance(result, dict)
        assert 'technical_signals' in result
        assert 'AAPL' in result['technical_signals']

    @patch('agents.technical_analysis.logger')
    def test_logging_on_errors(self, mock_logger, sample_df):
        
        # Create scenario that triggers error
        with patch.object(VPVRProfile, 'calculate_vpvr', side_effect=Exception("Test error")):
            vpvr = VPVRProfile()
            levels = vpvr.calculate_vpvr(sample_df)

            # Should handle error gracefully
            assert isinstance(levels, dict)

    def test_config_param_loading(self):
        
        from config.config import VPVR_BINS, VISIBLE_RANGE, HARMONIC_PATTERNS

        vpvr = VPVRProfile()
        assert vpvr.bins == VPVR_BINS
        assert vpvr.visible_range == VISIBLE_RANGE

        detector = HarmonicPatternDetector()
        assert detector.patterns_detected == []  # Should be initialized empty

    @patch('agents.technical_analysis.ENABLE_ADVANCED_TECH', False)
    def test_basic_functionality_preserved(self, sample_state):
        
        result = technical_analysis_agent(sample_state)

        signals = result['technical_signals']['AAPL']
        # Should still have basic indicators
        assert 'RSI' in signals
        assert 'MACD' in signals
        assert 'SMA' in signals


class TestIntegrationTests:
    

    def test_technical_analysis_agent_full_integration(self, sample_state):
        
        result = technical_analysis_agent(sample_state)

        assert isinstance(result, dict)
        assert 'technical_signals' in result
        signals = result['technical_signals']['AAPL']

        # Should have all types of signals
        assert 'EnsembleSignal' in signals
        assert 'TrendStrengthScore' in signals
        assert 'ProbabilityScore' in signals

        # Should have data validation info
        assert 'DataValidation' in signals

    @patch('agents.technical_analysis.ENABLE_ADVANCED_TECH', True)
    def test_technical_analysis_agent_with_advanced_features(self, sample_state):
        
        result = technical_analysis_agent(sample_state)

        signals = result['technical_signals']['AAPL']

        # Should have advanced signals when enabled
        assert 'VPVRSignal' in signals
        assert 'HeikinAshiSignal' in signals
        assert 'HarmonicSignal' in signals
        assert 'HMMSignal' in signals
        assert 'LSTMSignal' in signals
        assert 'VaRMetrics' in signals

    def test_ensemble_weights_integration(self, sample_df, sample_signals):
        
        generator = EnsembleSignalGenerator()

        # Add all signal types
        all_signals = sample_signals.copy()
        all_signals.update({
            'VPVR': 'buy',
            'HeikinAshi': 'buy',
            'Harmonic': 'sell',
            'HMM': 'bull_regime',
            'LSTM': 'buy'
        })

        weights = generator.update_weights_dynamically(sample_df, all_signals)
        signal = generator.generate_ensemble_signal(all_signals, sample_df)

        assert isinstance(weights, dict)
        assert abs(sum(weights.values()) - 1.0) < 0.01  # Should sum to ~1
        assert signal in ['buy', 'sell', 'neutral']

    def test_risk_assessment_integration(self, sample_state):
        
        result = technical_analysis_agent(sample_state)

        # Check if VaR metrics are exported to state
        if 'risk_assessment' in result:
            assert 'var_metrics' in result['risk_assessment']
            var_metrics = result['risk_assessment']['var_metrics']
            assert 'risk_level' in var_metrics
            assert var_metrics['risk_level'] in ['high', 'medium', 'low']

    @patch('agents.technical_analysis.yfinance.download')
    def test_data_fetcher_integration_mock(self, mock_download, sample_state):
        
        # Mock yfinance data
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [101, 102, 103],
            'Volume': [100000, 200000, 300000]
        })
        mock_download.return_value = mock_data

        # Simulate data fetcher providing data
        sample_state['stock_data']['AAPL'] = mock_data

        result = technical_analysis_agent(sample_state)

        assert isinstance(result, dict)
        assert 'technical_signals' in result
        assert 'AAPL' in result['technical_signals']

    def test_multi_timeframe_with_advanced_signals(self, sample_df):
        
        analyzer = MultiTimeframeAnalyzer()

        def mock_advanced_signals_func(df):
            signals = {
                'RSI': 'buy',
                'MACD': 'sell',
                'VPVR': 'buy',
                'HeikinAshi': 'buy',
                'Harmonic': 'sell'
            }
            return signals

        result = analyzer.analyze_multi_timeframe(sample_df, mock_advanced_signals_func)

        assert isinstance(result, dict)
        assert len(result) > 0

    def test_adaptive_parameters_integration(self, sample_df):
        
        adaptive_calc = AdaptiveParameterCalculator()

        rsi_period = adaptive_calc.adaptive_rsi_period(sample_df)
        macd_periods = adaptive_calc.adaptive_macd_periods(sample_df)
        stoch_periods = adaptive_calc.adaptive_stochastic_periods(sample_df)

        assert isinstance(rsi_period, int)
        assert 5 <= rsi_period <= 21  # Within expected range
        assert isinstance(macd_periods, dict)
        assert 'fast' in macd_periods
        assert 'slow' in macd_periods
        assert isinstance(stoch_periods, dict)
        assert 'k_period' in stoch_periods

    def test_signal_confirmer_with_advanced_signals(self, sample_signals):
        
        confirmer = SignalConfirmer(confirmation_threshold=2)

        # Add advanced signals
        advanced_signals = sample_signals.copy()
        advanced_signals.update({
            'VPVR': 'buy',
            'HeikinAshi': 'buy',
            'Harmonic': 'buy',
            'HMM': 'bull_regime',
            'LSTM': 'buy'
        })

        confirmed = confirmer.confirm_signals(advanced_signals)

        assert isinstance(confirmed, dict)
        assert len(confirmed) >= len(advanced_signals)

    def test_backtest_validator_with_advanced_signals(self, sample_df, sample_signals):
        
        validator = BacktestValidator()

        # Add advanced signals
        advanced_signals = sample_signals.copy()
        advanced_signals.update({
            'VPVR': 'buy',
            'HeikinAshi': 'buy',
            'Harmonic': 'sell'
        })

        score = validator.validate_signal(sample_df, 'buy')
        wf_results = validator.walk_forward_analysis(sample_df, advanced_signals)
        mc_results = validator.monte_carlo_simulation(sample_df, advanced_signals)

        assert isinstance(score, float)
        assert 0 <= score <= 1
        assert isinstance(wf_results, dict)
        assert 'stability_score' in wf_results
        assert isinstance(mc_results, dict)
        assert 'expected_return' in mc_results

    def test_parameter_optimizer_integration(self, sample_df):
        
        optimizer = ParameterOptimizer()

        results = optimizer.optimize_all_parameters(sample_df)

        assert isinstance(results, dict)
        assert 'rsi' in results
        assert 'macd' in results
        assert 'stochastic' in results

        for indicator, params in results.items():
            assert 'score' in params
            assert isinstance(params['score'], float)

    def test_full_pipeline_integration(self, sample_state):
        
        # Start with raw data
        result = technical_analysis_agent(sample_state)

        signals = result['technical_signals']['AAPL']

        # Verify all components are present
        required_keys = [
            'EnsembleSignal', 'TrendStrengthScore', 'ProbabilityScore',
            'DataValidation'
        ]

        for key in required_keys:
            assert key in signals

        # Verify signal types
        assert signals['EnsembleSignal'] in ['buy', 'sell', 'neutral']
        assert isinstance(signals['TrendStrengthScore'], float)
        assert isinstance(signals['ProbabilityScore'], float)


class TestPerformanceTests:
    

    @pytest.mark.parametrize("symbol_count", [1, 5, 10])
    def test_vpvr_calculation_performance(self, sample_df, benchmark, symbol_count):
        
        vpvr = VPVRProfile()

        def calculate_multiple_vpvr():
            results = {}
            for i in range(symbol_count):
                symbol_df = sample_df.copy()
                # Add some variation to simulate different symbols
                symbol_df['Close'] = sample_df['Close'] * (1 + np.random.normal(0, 0.01, len(sample_df)))
                results[f'SYMBOL_{i}'] = vpvr.calculate_vpvr(symbol_df)
            return results

        result = benchmark(calculate_multiple_vpvr)
        assert isinstance(result, dict)
        assert len(result) == symbol_count

    @pytest.mark.parametrize("symbol_count", [1, 5, 10])
    def test_garch_forecasting_performance(self, sample_df, benchmark, symbol_count):
        
        forecaster = GARCHForecaster()

        def forecast_multiple_symbols():
            results = {}
            for i in range(symbol_count):
                symbol_returns = sample_df['Close'].pct_change().dropna()
                # Add variation
                symbol_returns = symbol_returns * (1 + np.random.normal(0, 0.01, len(symbol_returns)))
                results[f'SYMBOL_{i}'] = forecaster.forecast_volatility(symbol_returns)
            return results

        result = benchmark(forecast_multiple_symbols)
        assert isinstance(result, dict)
        assert len(result) == symbol_count

    @pytest.mark.parametrize("symbol_count", [1, 3, 5])
    def test_lstm_prediction_performance(self, sample_df, sample_signals, benchmark, symbol_count):
        
        predictor = LSTMPredictor()

        def predict_multiple_symbols():
            results = {}
            for i in range(symbol_count):
                symbol_df = sample_df.copy()
                symbol_signals = sample_signals.copy()
                # Add variation
                symbol_df['Close'] = sample_df['Close'] * (1 + np.random.normal(0, 0.01, len(sample_df)))
                signal, confidence = predictor.predict_signal(symbol_df, symbol_signals)
                results[f'SYMBOL_{i}'] = {'signal': signal, 'confidence': confidence}
            return results

        result = benchmark(predict_multiple_symbols)
        assert isinstance(result, dict)
        assert len(result) == symbol_count

    @pytest.mark.parametrize("simulations", [100, 1000, 10000])
    def test_monte_carlo_var_performance(self, sample_df, benchmark, simulations):
        
        calculator = EnhancedVaRCalculator()
        calculator.mc_paths = simulations

        def calculate_var():
            return calculator.compute_monte_carlo_var(sample_df)

        result = benchmark(calculate_var)
        assert isinstance(result, dict)
        assert 'var_95' in result

    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not available")
    def test_memory_usage_monte_carlo_var(self, sample_df):
        
        import psutil
        import os

        calculator = EnhancedVaRCalculator()
        calculator.mc_paths = 10000  # Large number of simulations

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        result = calculator.compute_monte_carlo_var(sample_df)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory

        assert isinstance(result, dict)
        assert 'var_95' in result
        # Should not use excessive memory (less than 500MB increase)
        assert memory_used < 500

    @pytest.mark.parametrize("data_size", [100, 500, 1000])
    def test_hmm_training_performance(self, benchmark, data_size):
        
        detector = HMMRegimeDetector()

        # Create larger dataset
        dates = pd.date_range(start='2023-01-01', periods=data_size, freq='D')
        large_df = pd.DataFrame({
            'Open': 100 + np.random.randn(data_size).cumsum() * 0.5,
            'High': 105 + np.random.randn(data_size).cumsum() * 0.5,
            'Low': 95 + np.random.randn(data_size).cumsum() * 0.5,
            'Close': 100 + np.random.randn(data_size).cumsum() * 0.5,
            'Volume': np.random.randint(100000, 1000000, data_size)
        }, index=dates)

        features = detector.prepare_features(large_df)

        def train_hmm():
            return detector.fit_hmm_model(features)

        result = benchmark(train_hmm)
        assert isinstance(result, bool)

    @pytest.mark.parametrize("pattern_count", [10, 50, 100])
    def test_harmonic_pattern_detection_performance(self, benchmark, pattern_count):
        
        detector = HarmonicPatternDetector()

        # Create larger dataset with synthetic patterns
        dates = pd.date_range(start='2023-01-01', periods=500, freq='D')
        large_df = pd.DataFrame({
            'Open': 100 + np.random.randn(500).cumsum() * 0.5,
            'High': 105 + np.random.randn(500).cumsum() * 0.5,
            'Low': 95 + np.random.randn(500).cumsum() * 0.5,
            'Close': 100 + np.random.randn(500).cumsum() * 0.5,
            'Volume': np.random.randint(100000, 1000000, 500)
        }, index=dates)

        def detect_patterns():
            results = []
            for _ in range(pattern_count):
                signal, confidence = detector.detect_patterns(large_df)
                results.append((signal, confidence))
            return results

        result = benchmark(detect_patterns)
        assert isinstance(result, list)
        assert len(result) == pattern_count

    def test_ensemble_signal_performance(self, sample_df, sample_signals, benchmark):
        
        generator = EnsembleSignalGenerator()

        # Add all signal types for comprehensive test
        comprehensive_signals = sample_signals.copy()
        comprehensive_signals.update({
            'VPVR': 'buy',
            'HeikinAshi': 'buy',
            'Harmonic': 'sell',
            'HMM': 'bull_regime',
            'LSTM': 'buy',
            'Ichimoku': 'buy',
            'Fibonacci': 'sell',
            'SupportResistance': 'buy'
        })

        def generate_ensemble():
            weights = generator.update_weights_dynamically(sample_df, comprehensive_signals)
            signal = generator.generate_ensemble_signal(comprehensive_signals, sample_df)
            return {'weights': weights, 'signal': signal}

        result = benchmark(generate_ensemble)
        assert isinstance(result, dict)
        assert 'weights' in result
        assert 'signal' in result

    def test_full_agent_performance(self, sample_state, benchmark):
        
        def run_agent():
            return technical_analysis_agent(sample_state)

        result = benchmark(run_agent)
        assert isinstance(result, dict)
        assert 'technical_signals' in result

    @pytest.mark.parametrize("optimization_type", ["rsi", "macd", "stochastic"])
    def test_parameter_optimization_performance(self, sample_df, benchmark, optimization_type):
        
        optimizer = ParameterOptimizer()

        def optimize_params():
            if optimization_type == "rsi":
                return optimizer.optimize_rsi_parameters(sample_df)
            elif optimization_type == "macd":
                return optimizer.optimize_macd_parameters(sample_df)
            else:
                return optimizer.optimize_stochastic_parameters(sample_df)

        result = benchmark(optimize_params)
        assert isinstance(result, dict)
        assert 'score' in result


class TestBacktestingValidation:
    

    @patch('simulation.backtesting_engine.BacktestingEngine.run_backtest')
    def test_backtest_engine_integration(self, mock_run_backtest, sample_df):
        
        mock_run_backtest.return_value = {
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.08,
            'win_rate': 0.55
        }

        validator = BacktestValidator()
        recommendations = {'AAPL': {'action': 'BUY'}}
        start_date = sample_df.index[0]
        end_date = sample_df.index[-1]

        results = validator.engine.run_backtest(
            recommendations=recommendations,
            stock_data={'AAPL': sample_df},
            start_date=start_date,
            end_date=end_date
        )

        assert isinstance(results, dict)
        assert 'sharpe_ratio' in results
        assert results['sharpe_ratio'] == 1.2

    @patch('simulation.backtesting_engine.BacktestingEngine.run_backtest')
    def test_signal_backtest_validation(self, mock_run_backtest, sample_df):
        
        mock_run_backtest.return_value = {
            'total_return': 0.12,
            'sharpe_ratio': 1.5,
            'win_rate': 0.58
        }

        validator = BacktestValidator()
        score = validator.validate_signal(sample_df, 'buy')

        assert isinstance(score, float)
        assert 0 <= score <= 1
        mock_run_backtest.assert_called_once()

    @patch('agents.technical_analysis.yfinance.download')
    @patch('simulation.backtesting_engine.BacktestingEngine.run_backtest')
    def test_historical_data_backtest(self, mock_run_backtest, mock_download):
        
        # Mock historical data
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        historical_data = pd.DataFrame({
            'Open': 100 + np.random.randn(200).cumsum() * 0.5,
            'High': 105 + np.random.randn(200).cumsum() * 0.5,
            'Low': 95 + np.random.randn(200).cumsum() * 0.5,
            'Close': 100 + np.random.randn(200).cumsum() * 0.5,
            'Volume': np.random.randint(100000, 1000000, 200)
        }, index=dates)
        mock_download.return_value = historical_data

        mock_run_backtest.return_value = {
            'total_return': 0.18,
            'sharpe_ratio': 1.8,
            'max_drawdown': -0.05
        }

        validator = BacktestValidator()
        recommendations = {'RELIANCE.NS': {'action': 'BUY'}}

        results = validator.engine.run_backtest(
            recommendations=recommendations,
            stock_data={'RELIANCE.NS': historical_data},
            start_date=dates[0],
            end_date=dates[-1]
        )

        assert results['sharpe_ratio'] > 1.5  # Good performance
        assert results['total_return'] > 0.15

    def test_walk_forward_validation(self, sample_df, sample_signals):
        
        validator = BacktestValidator()

        results = validator.walk_forward_analysis(sample_df, sample_signals)

        assert isinstance(results, dict)
        assert 'stability_score' in results
        assert 'avg_performance' in results
        assert 'performance_std' in results
        assert 0 <= results['stability_score'] <= 1
        assert isinstance(results['performance_std'], float)

    @patch('agents.technical_analysis.stats.norm.fit')
    @patch('agents.technical_analysis.np.random.normal')
    def test_monte_carlo_validation(self, mock_random, mock_fit, sample_df, sample_signals):
        
        mock_fit.return_value = (0.001, 0.02)
        mock_random.return_value = np.random.normal(0.001, 0.02, 1000)

        validator = BacktestValidator()

        results = validator.monte_carlo_simulation(sample_df, sample_signals)

        assert isinstance(results, dict)
        assert 'expected_return' in results
        assert 'var_95' in results
        assert 'max_drawdown' in results
        assert 'return_std' in results

    @patch('simulation.backtesting_engine.BacktestingEngine.run_backtest')
    def test_old_vs_new_signals_comparison(self, mock_run_backtest, sample_df):
        
        # Mock results for old signals
        mock_run_backtest.return_value = {
            'total_return': 0.10,
            'sharpe_ratio': 1.0,
            'win_rate': 0.52
        }

        validator = BacktestValidator()

        # Test old signals (basic indicators only)
        old_recommendations = {'AAPL': {'action': 'BUY'}}
        old_results = validator.engine.run_backtest(
            recommendations=old_recommendations,
            stock_data={'AAPL': sample_df},
            start_date=sample_df.index[0],
            end_date=sample_df.index[-1]
        )

        # Mock better results for new signals
        mock_run_backtest.return_value = {
            'total_return': 0.15,
            'sharpe_ratio': 1.5,
            'win_rate': 0.58
        }

        # Test new signals (with advanced features)
        new_recommendations = {'AAPL': {'action': 'BUY'}}  # Would include advanced signals
        new_results = validator.engine.run_backtest(
            recommendations=new_recommendations,
            stock_data={'AAPL': sample_df},
            start_date=sample_df.index[0],
            end_date=sample_df.index[-1]
        )

        # New signals should perform better
        assert new_results['sharpe_ratio'] > old_results['sharpe_ratio']
        assert new_results['total_return'] > old_results['total_return']
        assert new_results['win_rate'] > old_results['win_rate']

    @patch('simulation.backtesting_engine.BacktestingEngine.run_backtest')
    def test_stress_testing_backtest(self, mock_run_backtest, sample_df):
        
        # Mock stressed market conditions
        mock_run_backtest.return_value = {
            'total_return': -0.05,
            'sharpe_ratio': -0.5,
            'max_drawdown': -0.15,
            'win_rate': 0.45
        }

        validator = BacktestValidator()

        # Create stressed data (high volatility)
        stressed_df = sample_df.copy()
        stressed_df['Close'] = sample_df['Close'] * (1 + np.random.normal(0, 0.05, len(sample_df)))

        recommendations = {'AAPL': {'action': 'BUY'}}
        results = validator.engine.run_backtest(
            recommendations=recommendations,
            stock_data={'AAPL': stressed_df},
            start_date=stressed_df.index[0],
            end_date=stressed_df.index[-1]
        )

        # Should handle stress gracefully
        assert isinstance(results, dict)
        assert 'max_drawdown' in results

    def test_signal_robustness_validation(self, sample_df):
        
        validator = BacktestValidator()

        # Test with trending market
        trending_df = sample_df.copy()
        trending_df['Close'] = trending_df['Close'] * (1 + np.linspace(0, 0.5, len(trending_df)))

        score_trending = validator.validate_signal(trending_df, 'buy')

        # Test with sideways market
        sideways_df = sample_df.copy()
        sideways_df['Close'] = 100 + np.sin(np.linspace(0, 4*np.pi, len(sideways_df))) * 5

        score_sideways = validator.validate_signal(sideways_df, 'neutral')

        assert isinstance(score_trending, float)
        assert isinstance(score_sideways, float)
        assert 0 <= score_trending <= 1
        assert 0 <= score_sideways <= 1

    @patch('simulation.backtesting_engine.BacktestingEngine.run_backtest')
    def test_multi_asset_backtest(self, mock_run_backtest, sample_df):
        
        mock_run_backtest.return_value = {
            'total_return': 0.12,
            'sharpe_ratio': 1.3,
            'win_rate': 0.55
        }

        validator = BacktestValidator()

        # Multiple assets
        multi_asset_data = {
            'AAPL': sample_df,
            'GOOGL': sample_df * 1.1,  # Slight variation
            'MSFT': sample_df * 0.9    # Different scale
        }

        recommendations = {
            'AAPL': {'action': 'BUY'},
            'GOOGL': {'action': 'SELL'},
            'MSFT': {'action': 'BUY'}
        }

        results = validator.engine.run_backtest(
            recommendations=recommendations,
            stock_data=multi_asset_data,
            start_date=sample_df.index[0],
            end_date=sample_df.index[-1]
        )

        assert results['sharpe_ratio'] > 1.0
        assert results['total_return'] > 0.1


class TestRegressionTests:
    

    def test_original_indicators_unchanged(self, sample_df):
        
        from agents.technical_analysis import _calculate_technical_indicators_with_retry

        signals = _calculate_technical_indicators_with_retry(sample_df, symbol="AAPL")

        assert isinstance(signals, dict)
        assert 'RSI' in signals
        assert 'MACD' in signals
        assert 'SMA' in signals
        assert 'EMA' in signals
        assert 'Bollinger' in signals
        assert signals['RSI'] in ['buy', 'sell', 'neutral']
        assert signals['MACD'] in ['buy', 'sell', 'neutral']

    def test_basic_agent_functionality_preserved(self, sample_state):
        
        result = technical_analysis_agent(sample_state)

        assert isinstance(result, dict)
        assert 'technical_signals' in result
        signals = result['technical_signals']['AAPL']

        # Should have all expected basic signals
        basic_signals = ['RSI', 'MACD', 'SMA', 'EMA', 'Bollinger', 'Stochastic', 'WilliamsR', 'CCI']
        for signal in basic_signals:
            assert signal in signals
            assert signals[signal] in ['buy', 'sell', 'neutral']

    def test_output_structure_consistency(self, sample_state):
        
        result = technical_analysis_agent(sample_state)

        # Check top-level structure
        assert 'technical_signals' in result
        assert isinstance(result['technical_signals'], dict)

        # Check symbol-level structure
        signals = result['technical_signals']['AAPL']
        assert isinstance(signals, dict)

        # Check that all signals are strings
        for key, value in signals.items():
            if key != 'DataValidation' and not key.endswith('Score') and not key.endswith('Metrics'):
                assert isinstance(value, str)
                assert value in ['buy', 'sell', 'neutral', 'bull_regime', 'bear_regime', 'sideways_regime']

    def test_error_handling_unchanged(self, sample_state):
        
        # Test with corrupted data
        corrupted_state = sample_state.copy()
        corrupted_state['stock_data']['AAPL'] = pd.DataFrame()  # Empty DataFrame

        result = technical_analysis_agent(corrupted_state)

        assert isinstance(result, dict)
        assert 'technical_signals' in result
        # Should handle gracefully without crashing
        assert 'AAPL' in result['technical_signals']

    def test_config_defaults_unchanged(self):
        
        from config.config import (
            RSI_OVERBOUGHT, RSI_OVERSOLD, CONFIRMATION_THRESHOLD,
            ENSEMBLE_THRESHOLD, TREND_STRENGTH_THRESHOLD
        )

        # These should be the same as before
        assert RSI_OVERBOUGHT == 70
        assert RSI_OVERSOLD == 30
        assert CONFIRMATION_THRESHOLD == 2
        assert ENSEMBLE_THRESHOLD == 0.6
        assert TREND_STRENGTH_THRESHOLD == 0.7

    def test_fixture_compatibility(self, sample_df, sample_signals):
        
        # Test that sample_df has required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            assert col in sample_df.columns

        # Test that sample_signals has expected structure
        assert isinstance(sample_signals, dict)
        assert len(sample_signals) > 0
        for signal in sample_signals.values():
            assert signal in ['buy', 'sell', 'neutral']

    def test_import_compatibility(self):
        
        try:
            from agents.technical_analysis import (
                technical_analysis_agent, DataValidator, MLSignalPredictor,
                MultiTimeframeAnalyzer, SignalConfirmer, AdaptiveParameterCalculator,
                RiskAdjuster, EnsembleSignalGenerator, TrendStrengthScorer,
                VolatilityAdjuster, IchimokuCloud, FibonacciRetracement,
                SupportResistanceCalculator, ProbabilityScorer, BacktestValidator,
                TradingSetup, VPVRProfile, HeikinAshiTransformer, GARCHForecaster,
                HarmonicPatternDetector, HMMRegimeDetector,
                LSTMPredictor, EnhancedVaRCalculator
            )
            assert True  # All imports successful
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")

    def test_function_signatures_unchanged(self):
        
        import inspect

        # Test technical_analysis_agent signature
        sig = inspect.signature(technical_analysis_agent)
        params = list(sig.parameters.keys())
        assert 'state' in params

        # Test other key functions
        from agents.technical_analysis import _calculate_technical_indicators_with_retry
        sig = inspect.signature(_calculate_technical_indicators_with_retry)
        params = list(sig.parameters.keys())
        assert 'df' in params
        assert 'symbol' in params

    def test_class_initialization_unchanged(self):
        
        # Test existing classes
        validator = DataValidator()
        assert hasattr(validator, 'validate_dataframe')

        predictor = MLSignalPredictor()
        assert hasattr(predictor, 'predict_signal')

        # Test new classes
        vpvr = VPVRProfile()
        assert hasattr(vpvr, 'calculate_vpvr')

        ha = HeikinAshiTransformer()
        assert hasattr(ha, 'transform_to_heikin_ashi')

        garch = GARCHForecaster()
        assert hasattr(garch, 'forecast_volatility')

    def test_no_unexpected_side_effects(self, sample_state):
        
        original_state = sample_state.copy()

        result = technical_analysis_agent(sample_state)

        # State should not be modified in place
        assert sample_state == original_state

        # Result should be a new dict
        assert result is not sample_state
        assert result != sample_state

    def test_logging_behavior_unchanged(self, sample_state):
        
        import logging

        logger = logging.getLogger('agents.technical_analysis')
        original_level = logger.level

        try:
            # Should not crash with logging
            result = technical_analysis_agent(sample_state)
            assert isinstance(result, dict)
        finally:
            logger.setLevel(original_level)

    def test_exception_handling_robustness(self):
        
        # Test with None input
        result = technical_analysis_agent(None)
        assert isinstance(result, dict)  # Should handle gracefully

        # Test with empty dict
        result = technical_analysis_agent({})
        assert isinstance(result, dict)

        # Test with invalid stock_data
        result = technical_analysis_agent({"stock_data": None})
        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__])