

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from agents.feature_engineering import FeatureEngineer, feature_engineering_agent
from data.models import State


class TestFeatureEngineer:
    

    @pytest.fixture
    def sample_data(self):
        
        dates = pd.date_range('2023-01-01', periods=250, freq='D')  # More data for rolling calculations
        np.random.seed(42)

        # Create more realistic price data
        base_price = 100
        price_changes = np.random.randn(250) * 2
        prices = base_price + price_changes.cumsum()

        data = {
            'Open': prices + np.random.randn(250) * 0.5,
            'High': prices + abs(np.random.randn(250)) * 2,
            'Low': prices - abs(np.random.randn(250)) * 2,
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, 250)
        }

        # Ensure OHLC relationships
        for i in range(len(data['Close'])):
            data['High'][i] = max(data['Open'][i], data['High'][i], data['Close'][i])
            data['Low'][i] = min(data['Open'][i], data['Low'][i], data['Close'][i])

        df = pd.DataFrame(data, index=dates)
        return df</search>

    @pytest.fixture
    def feature_engineer(self):
        
        return FeatureEngineer()

    def test_initialization(self, feature_engineer):
        
        assert feature_engineer.sector_mapping is not None
        assert isinstance(feature_engineer.sector_mapping, dict)
        assert 'RELIANCE.NS' in feature_engineer.sector_mapping

    def test_create_technical_features(self, feature_engineer, sample_data):
        
        features = feature_engineer.create_technical_features(sample_data)

        # Check that features DataFrame is created
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0

        # Check essential features exist
        expected_features = ['close', 'high', 'low', 'open', 'volume', 'returns', 'rsi', 'macd']
        for feature in expected_features:
            assert feature in features.columns, f"Missing feature: {feature}"

        # Check RSI values are in valid range
        assert features['rsi'].dropna().between(0, 100).all()

    def test_create_sentiment_features(self, feature_engineer):
        
        # Test with valid sentiment data
        sentiment_data = {
            'positive': 0.6,
            'negative': 0.2,
            'compound': 0.4,
            'articles_analyzed': 10
        }
        features = feature_engineer.create_sentiment_features(sentiment_data)

        assert isinstance(features, pd.Series)
        assert features['sentiment_positive'] == 0.6
        assert features['sentiment_compound'] == 0.4

        # Test with missing sentiment data
        features_empty = feature_engineer.create_sentiment_features({})
        assert features_empty['sentiment_positive'] == 0.5  # Default neutral

    def test_create_macro_features(self, feature_engineer):
        
        macro_data = {
            'RBI_REPO': -0.5,
            'INDIA_UNRATE': 0.2,
            'INDIA_GDP': 0.8,
            'composite': 0.1
        }
        features = feature_engineer.create_macro_features(macro_data)

        assert isinstance(features, pd.Series)
        assert features['macro_rbi_repo'] == -0.5
        assert features['macro_composite'] == 0.1

    def test_create_cross_sectional_features(self, feature_engineer):
        
        features = feature_engineer.create_cross_sectional_features('RELIANCE.NS')

        assert isinstance(features, pd.Series)
        assert 'market_cap' in features.index
        assert 'sector_energy' in features.index  # RELIANCE is in Energy sector

        # Check that sector features are binary
        sector_features = [f for f in features.index if f.startswith('sector_')]
        for feat in sector_features:
            assert features[feat] in [0, 1]

    def test_create_temporal_features(self, feature_engineer, sample_data):
        
        features = feature_engineer.create_temporal_features(sample_data)

        assert isinstance(features, pd.DataFrame)
        expected_cols = ['day_of_week', 'month', 'quarter', 'day_sin', 'day_cos']
        for col in expected_cols:
            assert col in features.columns

        # Check cyclical encoding
        assert features['day_sin'].between(-1, 1).all()
        assert features['day_cos'].between(-1, 1).all()

    def test_create_fibonacci_features(self, feature_engineer, sample_data):
        
        features = feature_engineer.create_fibonacci_features(sample_data)

        assert isinstance(features, pd.DataFrame)
        # Should have fib levels and distance features
        fib_cols = [col for col in features.columns if 'fib' in col]
        assert len(fib_cols) > 0

    def test_create_support_resistance_features(self, feature_engineer, sample_data):
        
        features = feature_engineer.create_support_resistance_features(sample_data)

        assert isinstance(features, pd.DataFrame)
        sr_cols = [col for col in features.columns if 'support' in col or 'resistance' in col]
        assert len(sr_cols) > 0

    def test_create_all_features(self, feature_engineer):
        
        # Create mock state
        state = State()
        state['stock_data'] = {
            'TEST.NS': pd.DataFrame({
                'Open': [100, 101, 102],
                'High': [105, 106, 107],
                'Low': [95, 96, 97],
                'Close': [102, 103, 104],
                'Volume': [1000, 1100, 1200]
            })
        }
        state['sentiment_scores'] = {'TEST.NS': {'positive': 0.5, 'compound': 0.1}}
        state['macro_scores'] = {'RBI_REPO': 0.0, 'composite': 0.2}

        features = feature_engineer.create_all_features(state, 'TEST.NS')

        assert isinstance(features, pd.DataFrame)
        assert len(features.columns) > 10  # Should have many features
        assert len(features) == 3  # Same as input data length


class TestFeatureEngineeringAgent:
    

    def test_agent_with_valid_data(self):
        
        state = State()
        state['stock_data'] = {
            'TEST.NS': pd.DataFrame({
                'Open': [100] * 50,
                'High': [105] * 50,
                'Low': [95] * 50,
                'Close': [102] * 50,
                'Volume': [1000] * 50
            })
        }
        state['sentiment_scores'] = {'TEST.NS': {'positive': 0.5}}
        state['macro_scores'] = {'composite': 0.0}

        result_state = feature_engineering_agent(state)

        assert 'engineered_features' in result_state
        assert 'TEST.NS' in result_state['engineered_features']
        assert isinstance(result_state['engineered_features']['TEST.NS'], pd.DataFrame)

    def test_agent_with_no_data(self):
        
        state = State()

        result_state = feature_engineering_agent(state)

        # Should return state unchanged
        assert result_state == state

    def test_agent_with_insufficient_data(self):
        
        state = State()
        state['stock_data'] = {
            'TEST.NS': pd.DataFrame({
                'Open': [100, 101],
                'High': [105, 106],
                'Low': [95, 96],
                'Close': [102, 103],
                'Volume': [1000, 1100]
            })
        }

        result_state = feature_engineering_agent(state)

        # Should still create features but with limited data
        assert 'engineered_features' in result_state


if __name__ == '__main__':
    pytest.main([__file__])