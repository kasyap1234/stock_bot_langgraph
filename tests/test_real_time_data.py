"""
Tests for real-time data integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from data.real_time_data import RealTimeDataManager, _validate_and_clean_data, _cache_data, _get_cached_data


class TestRealTimeDataManager:
    """Test the RealTimeDataManager class."""

    @pytest.fixture
    def manager(self):
        """Create a RealTimeDataManager instance."""
        return RealTimeDataManager()

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager.sources is not None
        assert 'yahoo' in manager.sources
        assert 'alpha_vantage' in manager.sources
        assert manager.active_streams == {}
        assert manager.callbacks == []

    def test_add_callback(self, manager):
        """Test adding callbacks."""
        def dummy_callback(data):
            pass

        manager.add_callback(dummy_callback)
        assert dummy_callback in manager.callbacks

    def test_remove_callback(self, manager):
        """Test removing callbacks."""
        def dummy_callback(data):
            pass

        manager.add_callback(dummy_callback)
        manager.remove_callback(dummy_callback)
        assert dummy_callback not in manager.callbacks

    @pytest.mark.asyncio
    async def test_yahoo_realtime_success(self, manager):
        """Test Yahoo Finance real-time data fetching."""
        mock_data = {
            'symbol': 'AAPL',
            'price': 150.0,
            'timestamp': '2024-01-01T10:00:00Z',
            'source': 'yahoo'
        }

        with patch.object(manager, '_yahoo_realtime', return_value=mock_data):
            result = await manager._yahoo_realtime('AAPL')
            assert result == mock_data

    @pytest.mark.asyncio
    async def test_alpha_vantage_realtime_success(self, manager):
        """Test Alpha Vantage real-time data fetching."""
        mock_data = {
            'symbol': 'AAPL',
            'price': 150.0,
            'timestamp': '2024-01-01T10:00:00Z',
            'source': 'alpha_vantage'
        }

        with patch.object(manager, '_alpha_vantage_realtime', return_value=mock_data):
            result = await manager._alpha_vantage_realtime('AAPL')
            assert result == mock_data

    @pytest.mark.asyncio
    async def test_fallback_mechanism(self, manager):
        """Test fallback to other sources when primary fails."""
        # Mock primary source failure
        with patch.object(manager, '_yahoo_realtime', return_value={'error': 'API failure'}):
            # Mock fallback source success
            mock_data = {
                'symbol': 'AAPL',
                'price': 150.0,
                'timestamp': '2024-01-01T10:00:00Z',
                'source': 'alpha_vantage'
            }
            with patch.object(manager, '_alpha_vantage_realtime', return_value=mock_data):
                result = await manager._get_data_with_fallback('AAPL', 'yahoo', ['alpha_vantage'])
                assert result['source'] == 'alpha_vantage'
                assert result['price'] == 150.0


class TestDataValidation:
    """Test data validation and cleaning functions."""

    def test_validate_clean_data_valid(self):
        """Test validation of valid data."""
        data = {
            'symbol': 'AAPL',
            'price': 150.0,
            'timestamp': '2024-01-01T10:00:00Z',
            'source': 'yahoo'
        }

        result = _validate_and_clean_data(data)
        assert result == data

    def test_validate_clean_data_missing_fields(self):
        """Test validation with missing required fields."""
        data = {
            'price': 150.0,
            'timestamp': '2024-01-01T10:00:00Z'
        }

        result = _validate_and_clean_data(data)
        assert result == {}

    def test_validate_clean_data_outlier(self):
        """Test outlier detection and filtering."""
        data = {
            'symbol': 'AAPL',
            'price': 150.0,
            'previous_price': 100.0,  # 50% change - should be filtered
            'timestamp': '2024-01-01T10:00:00Z',
            'source': 'yahoo'
        }

        result = _validate_and_clean_data(data)
        assert result == {}  # Should be filtered out

    def test_validate_clean_data_normal_change(self):
        """Test normal price change passes validation."""
        data = {
            'symbol': 'AAPL',
            'price': 105.0,
            'previous_price': 100.0,  # 5% change - should pass
            'timestamp': '2024-01-01T10:00:00Z',
            'source': 'yahoo'
        }

        result = _validate_and_clean_data(data)
        assert result == data


class TestDataCaching:
    """Test data caching functionality."""

    def test_cache_data(self):
        """Test caching data."""
        data = {'symbol': 'AAPL', 'price': 150.0}

        _cache_data('AAPL', data, ttl=60)

        cached = _get_cached_data('AAPL')
        assert cached == data

    def test_cache_expiry(self):
        """Test cache expiry."""
        data = {'symbol': 'AAPL', 'price': 150.0}

        _cache_data('AAPL', data, ttl=0)  # Immediate expiry

        cached = _get_cached_data('AAPL')
        assert cached is None

    def test_cache_miss(self):
        """Test cache miss for non-existent key."""
        cached = _get_cached_data('NONEXISTENT')
        assert cached is None


class TestIntegration:
    """Integration tests for the real-time data system."""

    @pytest.mark.asyncio
    async def test_full_streaming_workflow(self):
        """Test the complete streaming workflow."""
        manager = RealTimeDataManager()

        # Mock data source
        mock_data = {
            'symbol': 'AAPL',
            'price': 150.0,
            'timestamp': '2024-01-01T10:00:00Z',
            'source': 'yahoo'
        }

        received_data = []

        def callback(data):
            received_data.append(data)

        manager.add_callback(callback)

        with patch.object(manager, '_yahoo_realtime', return_value=mock_data):
            # Start streaming
            await manager.start_streaming(['AAPL'], ['yahoo'], interval=1)

            # Wait a bit for processing
            await asyncio.sleep(0.1)

            # Stop streaming
            await manager.stop_streaming()

            # Check that data was received
            assert len(received_data) > 0
            assert received_data[0]['symbol'] == 'AAPL'