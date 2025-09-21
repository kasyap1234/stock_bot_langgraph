"""
Tests for Model Retraining Manager

Tests cover:
- Automatic retraining triggers based on performance decline
- Model drift detection and retraining
- Scheduled retraining functionality
- Manual retraining requests
- Retraining queue management and prioritization
"""

import pytest
import time
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any

from monitoring.model_retraining_manager import (
    ModelRetrainingManager,
    ModelConfiguration,
    RetrainingRequest,
    RetrainingResult,
    RetrainingTrigger,
    RetrainingStatus,
    create_ml_model_config,
    create_neural_network_config
)


class TestModelRetrainingManager:
    """Test suite for ModelRetrainingManager"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.model_dir = Path(self.temp_dir) / "models"
        self.backup_dir = Path(self.temp_dir) / "backups"
        
        self.manager = ModelRetrainingManager(
            max_concurrent_retraining=2,
            model_storage_path=str(self.model_dir),
            backup_storage_path=str(self.backup_dir)
        )
        
        # Mock training and validation functions
        self.mock_training_function = Mock(return_value={
            'model_path': str(self.model_dir / "test_model.pkl"),
            'training_metrics': {'accuracy': 0.85, 'loss': 0.15}
        })
        
        self.mock_validation_function = Mock(return_value={
            'performance': 0.85,
            'metrics': {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.88}
        })
        
    def teardown_method(self):
        """Clean up test fixtures"""
        self.manager.stop_manager()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_register_model(self):
        """Test model registration"""
        config = ModelConfiguration(
            model_name="test_model",
            model_class="sklearn",
            training_function=self.mock_training_function,
            validation_function=self.mock_validation_function
        )
        
        self.manager.register_model(config)
        
        assert "test_model" in self.manager.model_configurations
        assert self.manager.model_configurations["test_model"] == config
    
    def test_request_retraining_manual(self):
        """Test manual retraining request"""
        # Register model first
        config = create_ml_model_config(
            model_name="test_model",
            training_function=self.mock_training_function,
            validation_function=self.mock_validation_function
        )
        self.manager.register_model(config)
        
        # Request retraining
        request_id = self.manager.request_retraining(
            model_name="test_model",
            trigger_type=RetrainingTrigger.MANUAL,
            reason="Manual test retraining",
            priority=1
        )
        
        assert request_id != ""
        assert len(self.manager.retraining_queue) == 1
        
        request = self.manager.retraining_queue[0]
        assert request.model_name == "test_model"
        assert request.trigger_type == RetrainingTrigger.MANUAL
        assert request.priority == 1
    
    def test_request_retraining_performance_decline(self):
        """Test retraining request due to performance decline"""
        config = create_ml_model_config(
            model_name="declining_model",
            training_function=self.mock_training_function,
            performance_threshold=0.05
        )
        self.manager.register_model(config)
        
        request_id = self.manager.request_retraining(
            model_name="declining_model",
            trigger_type=RetrainingTrigger.PERFORMANCE_DECLINE,
            reason="Performance dropped by 8%",
            priority=2
        )
        
        assert request_id != ""
        request = self.manager.retraining_queue[0]
        assert request.trigger_type == RetrainingTrigger.PERFORMANCE_DECLINE
        assert request.priority == 2
    
    def test_retraining_queue_prioritization(self):
        """Test that retraining queue is properly prioritized"""
        config = create_ml_model_config(
            model_name="test_model",
            training_function=self.mock_training_function
        )
        self.manager.register_model(config)
        
        # Add requests with different priorities
        self.manager.request_retraining("test_model", RetrainingTrigger.MANUAL, "Low priority", priority=8)
        self.manager.request_retraining("test_model", RetrainingTrigger.PERFORMANCE_DECLINE, "High priority", priority=1)
        self.manager.request_retraining("test_model", RetrainingTrigger.SCHEDULED, "Medium priority", priority=5)
        
        # Check queue is sorted by priority (lowest number = highest priority)
        priorities = [req.priority for req in self.manager.retraining_queue]
        assert priorities == [1, 5, 8]
    
    def test_minimum_retraining_interval(self):
        """Test minimum retraining interval enforcement"""
        config = create_ml_model_config(
            model_name="interval_test_model",
            training_function=self.mock_training_function
        )
        config.min_retraining_interval = timedelta(hours=1)
        self.manager.register_model(config)
        
        # First request should succeed
        request_id1 = self.manager.request_retraining(
            "interval_test_model", RetrainingTrigger.MANUAL, "First request"
        )
        assert request_id1 != ""
        
        # Simulate completion of first retraining
        result = RetrainingResult(
            request_id=request_id1,
            model_name="interval_test_model",
            success=True,
            training_time=timedelta(minutes=30)
        )
        self.manager.retraining_history.append(result)
        
        # Second request within interval should be rejected
        request_id2 = self.manager.request_retraining(
            "interval_test_model", RetrainingTrigger.MANUAL, "Second request too soon"
        )
        assert request_id2 == ""  # Should be rejected
    
    def test_performance_monitor_integration(self):
        """Test integration with performance monitor"""
        # Mock performance monitor
        mock_monitor = Mock()
        self.manager.set_performance_monitor(mock_monitor)
        
        # Verify callback was added
        mock_monitor.add_callback.assert_called_once()
        
        # Test performance event handling
        config = create_ml_model_config(
            model_name="monitored_model",
            training_function=self.mock_training_function
        )
        self.manager.register_model(config)
        
        # Simulate performance decline event
        event_data = {
            'model_name': 'monitored_model',
            'reason': 'Performance declined by 12%'
        }
        
        self.manager._handle_performance_event('retraining', event_data)
        
        # Should have queued retraining request
        assert len(self.manager.retraining_queue) == 1
        request = self.manager.retraining_queue[0]
        assert request.model_name == 'monitored_model'
        assert request.trigger_type == RetrainingTrigger.PERFORMANCE_DECLINE
    
    def test_bias_detection_trigger(self):
        """Test retraining trigger from bias detection"""
        config = create_ml_model_config(
            model_name="AAPL_model",
            training_function=self.mock_training_function
        )
        self.manager.register_model(config)
        
        # Simulate bias detection event
        bias_event_data = {
            'bias_result': {
                'bias_type': 'overconfidence',
                'affected_symbols': ['AAPL', 'GOOGL']
            }
        }
        
        self.manager._handle_performance_event('bias', bias_event_data)
        
        # Should trigger retraining for AAPL_model
        assert len(self.manager.retraining_queue) == 1
        request = self.manager.retraining_queue[0]
        assert request.model_name == 'AAPL_model'
        assert request.trigger_type == RetrainingTrigger.BIAS_DETECTED
    
    @patch('threading.Thread')
    def test_manager_lifecycle(self, mock_thread):
        """Test manager start/stop lifecycle"""
        # Test start
        self.manager.start_manager()
        assert self.manager.manager_active is True
        mock_thread.assert_called_once()
        
        # Test stop
        self.manager.stop_manager()
        assert self.manager.manager_active is False
    
    def test_model_backup_and_restore(self):
        """Test model backup and restore functionality"""
        # Create a dummy model file
        model_path = self.model_dir / "test_model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(model_path, 'w') as f:
            f.write("dummy model data")
        
        # Test backup
        backup_path = self.manager._backup_current_model("test_model")
        assert backup_path is not None
        assert Path(backup_path).exists()
        
        # Modify original model
        with open(model_path, 'w') as f:
            f.write("modified model data")
        
        # Test restore
        success = self.manager._restore_model_backup("test_model", backup_path)
        assert success is True
        
        # Verify restoration
        with open(model_path, 'r') as f:
            content = f.read()
        assert content == "dummy model data"
    
    def test_retraining_execution_success(self):
        """Test successful retraining execution"""
        config = create_ml_model_config(
            model_name="execution_test_model",
            training_function=self.mock_training_function,
            validation_function=self.mock_validation_function
        )
        self.manager.register_model(config)
        
        # Create retraining request
        request = RetrainingRequest(
            request_id="test_request_001",
            model_name="execution_test_model",
            trigger_type=RetrainingTrigger.MANUAL,
            trigger_reason="Test execution",
            priority=1,
            requested_time=datetime.now()
        )
        
        # Execute retraining
        result = self.manager._execute_retraining(request)
        
        # Verify result
        assert result.success is True
        assert result.model_name == "execution_test_model"
        assert result.training_time > timedelta(0)
        assert result.new_performance == 0.85
        
        # Verify training function was called
        self.mock_training_function.assert_called_once()
        self.mock_validation_function.assert_called_once()
    
    def test_retraining_execution_failure(self):
        """Test retraining execution with failure"""
        # Mock training function that raises exception
        failing_training_function = Mock(side_effect=Exception("Training failed"))
        
        config = create_ml_model_config(
            model_name="failing_model",
            training_function=failing_training_function
        )
        self.manager.register_model(config)
        
        request = RetrainingRequest(
            request_id="test_request_002",
            model_name="failing_model",
            trigger_type=RetrainingTrigger.MANUAL,
            trigger_reason="Test failure",
            priority=1,
            requested_time=datetime.now()
        )
        
        # Execute retraining
        result = self.manager._execute_retraining(request)
        
        # Verify failure handling
        assert result.success is False
        assert "Training failed" in result.error_details
        assert result.training_time > timedelta(0)
    
    def test_performance_degradation_handling(self):
        """Test handling of performance degradation after retraining"""
        # Mock validation function that returns worse performance
        worse_validation_function = Mock(return_value={
            'performance': 0.60,  # Worse than baseline
            'metrics': {'accuracy': 0.60}
        })
        
        config = create_ml_model_config(
            model_name="degrading_model",
            training_function=self.mock_training_function,
            validation_function=worse_validation_function
        )
        self.manager.register_model(config)
        
        # Mock current performance as higher
        with patch.object(self.manager, '_get_current_performance', return_value=0.80):
            # Create dummy model file for backup/restore
            model_path = self.model_dir / "degrading_model.pkl"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_path, 'w') as f:
                f.write("original model")
            
            request = RetrainingRequest(
                request_id="test_request_003",
                model_name="degrading_model",
                trigger_type=RetrainingTrigger.MANUAL,
                trigger_reason="Test degradation",
                priority=1,
                requested_time=datetime.now()
            )
            
            # Execute retraining
            result = self.manager._execute_retraining(request)
            
            # Should still succeed but restore backup due to performance degradation
            assert result.success is True
            assert result.performance_improvement == 0.0  # Should be reset to 0
    
    def test_scheduled_retraining_check(self):
        """Test scheduled retraining functionality"""
        config = create_ml_model_config(
            model_name="scheduled_model",
            training_function=self.mock_training_function,
            retraining_schedule="daily"
        )
        self.manager.register_model(config)
        
        # Mock last retraining time as 2 days ago
        with patch.object(self.manager, '_get_last_retraining_time', 
                         return_value=datetime.now() - timedelta(days=2)):
            
            self.manager._check_scheduled_retraining()
            
            # Should have queued a scheduled retraining
            assert len(self.manager.retraining_queue) == 1
            request = self.manager.retraining_queue[0]
            assert request.trigger_type == RetrainingTrigger.SCHEDULED
            assert "daily" in request.trigger_reason.lower()
    
    def test_callback_notifications(self):
        """Test callback notifications for retraining events"""
        callback_events = []
        
        def test_callback(event_type, event_data):
            callback_events.append((event_type, event_data))
        
        self.manager.add_callback(test_callback)
        
        # Test retraining request notification
        config = create_ml_model_config(
            model_name="callback_test_model",
            training_function=self.mock_training_function
        )
        self.manager.register_model(config)
        
        request_id = self.manager.request_retraining(
            "callback_test_model", RetrainingTrigger.MANUAL, "Test callback"
        )
        
        # Should have received callback
        assert len(callback_events) == 1
        event_type, event_data = callback_events[0]
        assert event_type == 'retraining_requested'
        assert event_data['model_name'] == 'callback_test_model'
        assert event_data['request_id'] == request_id
    
    def test_retraining_status_reporting(self):
        """Test retraining status reporting"""
        config = create_ml_model_config(
            model_name="status_test_model",
            training_function=self.mock_training_function
        )
        self.manager.register_model(config)
        
        # Add some requests and history
        self.manager.request_retraining(
            "status_test_model", RetrainingTrigger.MANUAL, "Test status"
        )
        
        result = RetrainingResult(
            request_id="completed_001",
            model_name="status_test_model",
            success=True,
            training_time=timedelta(minutes=30)
        )
        self.manager.retraining_history.append(result)
        
        # Test overall status
        overall_status = self.manager.get_retraining_status()
        assert 'queued_requests_count' in overall_status
        assert 'registered_models' in overall_status
        assert overall_status['queued_requests_count'] == 1
        assert 'status_test_model' in overall_status['registered_models']
        
        # Test model-specific status
        model_status = self.manager.get_retraining_status("status_test_model")
        assert model_status['model_name'] == "status_test_model"
        assert len(model_status['queued_requests']) == 1
        assert len(model_status['recent_history']) == 1
    
    def test_cancel_retraining(self):
        """Test retraining cancellation"""
        config = create_ml_model_config(
            model_name="cancel_test_model",
            training_function=self.mock_training_function
        )
        self.manager.register_model(config)
        
        # Request retraining
        request_id = self.manager.request_retraining(
            "cancel_test_model", RetrainingTrigger.MANUAL, "Test cancellation"
        )
        
        assert len(self.manager.retraining_queue) == 1
        
        # Cancel retraining
        success = self.manager.cancel_retraining(request_id)
        assert success is True
        assert len(self.manager.retraining_queue) == 0
        
        # Try to cancel non-existent request
        success = self.manager.cancel_retraining("non_existent")
        assert success is False
    
    def test_model_configuration_helpers(self):
        """Test model configuration helper functions"""
        # Test ML model config
        ml_config = create_ml_model_config(
            model_name="test_ml_model",
            training_function=self.mock_training_function,
            performance_threshold=0.08,
            retraining_schedule="weekly"
        )
        
        assert ml_config.model_name == "test_ml_model"
        assert ml_config.model_class == "sklearn"
        assert ml_config.performance_threshold == 0.08
        assert ml_config.retraining_schedule == "weekly"
        assert ml_config.min_retraining_interval == timedelta(hours=12)
        
        # Test neural network config
        nn_config = create_neural_network_config(
            model_name="test_nn_model",
            training_function=self.mock_training_function,
            performance_threshold=0.03
        )
        
        assert nn_config.model_name == "test_nn_model"
        assert nn_config.model_class == "neural_network"
        assert nn_config.performance_threshold == 0.03
        assert nn_config.min_retraining_interval == timedelta(hours=24)
        assert nn_config.max_training_time == timedelta(hours=6)
    
    def test_concurrent_retraining_limit(self):
        """Test concurrent retraining limit enforcement"""
        # Set up manager with limit of 1 concurrent retraining
        limited_manager = ModelRetrainingManager(max_concurrent_retraining=1)
        
        config = create_ml_model_config(
            model_name="concurrent_test_model",
            training_function=self.mock_training_function
        )
        limited_manager.register_model(config)
        
        # Add multiple requests
        limited_manager.request_retraining(
            "concurrent_test_model", RetrainingTrigger.MANUAL, "Request 1"
        )
        limited_manager.request_retraining(
            "concurrent_test_model", RetrainingTrigger.MANUAL, "Request 2"
        )
        
        # Simulate one active retraining
        mock_request = RetrainingRequest(
            request_id="active_001",
            model_name="concurrent_test_model",
            trigger_type=RetrainingTrigger.MANUAL,
            trigger_reason="Active request",
            priority=1,
            requested_time=datetime.now()
        )
        limited_manager.active_retraining["active_001"] = mock_request
        
        # Process queue - should not start new retraining due to limit
        queue_length_before = len(limited_manager.retraining_queue)
        limited_manager._process_retraining_queue()
        
        # Queue should remain unchanged due to concurrent limit
        assert len(limited_manager.retraining_queue) == queue_length_before
        
        limited_manager.stop_manager()


if __name__ == "__main__":
    pytest.main([__file__])