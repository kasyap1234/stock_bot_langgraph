"""
Automatic Model Retraining Manager

This module implements automatic model retraining triggers based on:
- Performance decline detection
- Model drift detection
- Data distribution changes
- Scheduled retraining intervals
- Manual retraining requests
"""

import logging
import asyncio
import threading
import time
import pickle
import joblib
import os
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class RetrainingTrigger(Enum):
    PERFORMANCE_DECLINE = "performance_decline"
    MODEL_DRIFT = "model_drift"
    DATA_DISTRIBUTION_CHANGE = "data_distribution_change"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    NEW_DATA_AVAILABLE = "new_data_available"
    BIAS_DETECTED = "bias_detected"


class RetrainingStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RetrainingRequest:
    request_id: str
    model_name: str
    trigger_type: RetrainingTrigger
    trigger_reason: str
    priority: int  # 1 (highest) to 10 (lowest)
    requested_time: datetime
    scheduled_time: Optional[datetime] = None
    status: RetrainingStatus = RetrainingStatus.PENDING
    progress: float = 0.0
    error_message: Optional[str] = None
    completion_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfiguration:
    model_name: str
    model_class: str  # Class name or module path
    training_function: Callable
    validation_function: Optional[Callable] = None
    data_requirements: Dict[str, Any] = field(default_factory=dict)
    retraining_schedule: Optional[str] = None  # Cron-like schedule
    performance_threshold: float = 0.05  # Trigger retraining if performance drops by 5%
    drift_threshold: float = 0.1  # Trigger retraining if drift exceeds 10%
    min_retraining_interval: timedelta = field(default_factory=lambda: timedelta(hours=24))
    max_training_time: timedelta = field(default_factory=lambda: timedelta(hours=6))
    backup_model_path: Optional[str] = None


@dataclass
class RetrainingResult:
    request_id: str
    model_name: str
    success: bool
    training_time: timedelta
    old_performance: Optional[float] = None
    new_performance: Optional[float] = None
    performance_improvement: Optional[float] = None
    model_path: Optional[str] = None
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    error_details: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelRetrainingManager:
    """
    Manages automatic model retraining based on various triggers
    """
    
    def __init__(self, 
                 max_concurrent_retraining: int = 2,
                 model_storage_path: str = "models",
                 backup_storage_path: str = "model_backups"):
        
        self.max_concurrent_retraining = max_concurrent_retraining
        self.model_storage_path = Path(model_storage_path)
        self.backup_storage_path = Path(backup_storage_path)
        
        # Ensure directories exist
        self.model_storage_path.mkdir(exist_ok=True)
        self.backup_storage_path.mkdir(exist_ok=True)
        
        # Internal state
        self.model_configurations: Dict[str, ModelConfiguration] = {}
        self.retraining_queue: List[RetrainingRequest] = []
        self.active_retraining: Dict[str, RetrainingRequest] = {}
        self.retraining_history: List[RetrainingResult] = []
        
        # Threading and execution
        self.manager_active = False
        self.manager_thread: Optional[threading.Thread] = None
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_retraining)
        
        # Callbacks
        self.callbacks: List[Callable] = []
        
        # Performance monitoring integration
        self.performance_monitor = None
        
    def register_model(self, config: ModelConfiguration) -> None:
        try:
            self.model_configurations[config.model_name] = config
            logger.info(f"Registered model for retraining: {config.model_name}")
            
        except Exception as e:
            logger.error(f"Error registering model {config.model_name}: {e}")
    
    def set_performance_monitor(self, performance_monitor) -> None:
        self.performance_monitor = performance_monitor
        if performance_monitor:
            # Add callback to performance monitor for automatic triggers
            performance_monitor.add_callback(self._handle_performance_event)
    
    def start_manager(self) -> None:
        if self.manager_active:
            logger.warning("Retraining manager is already active")
            return
        
        self.manager_active = True
        self.manager_thread = threading.Thread(target=self._manager_loop, daemon=True)
        self.manager_thread.start()
        logger.info("Started model retraining manager")
    
    def stop_manager(self) -> None:
        self.manager_active = False
        if self.manager_thread:
            self.manager_thread.join(timeout=10)
        
        # Cancel pending retraining
        for request in self.retraining_queue:
            request.status = RetrainingStatus.CANCELLED
        
        # Wait for active retraining to complete
        self.executor.shutdown(wait=True)
        logger.info("Stopped model retraining manager")
    
    def _manager_loop(self) -> None:
        while self.manager_active:
            try:
                # Process retraining queue
                self._process_retraining_queue()
                
                # Check scheduled retraining
                self._check_scheduled_retraining()
                
                # Clean up completed retraining
                self._cleanup_completed_retraining()
                
                # Sleep for a short interval
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in retraining manager loop: {e}")
                time.sleep(60)
    
    def _handle_performance_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        try:
            if event_type == 'retraining':
                # Performance monitor triggered retraining
                model_name = event_data.get('model_name')
                reason = event_data.get('reason', 'Performance decline detected')
                
                self.request_retraining(
                    model_name=model_name,
                    trigger_type=RetrainingTrigger.PERFORMANCE_DECLINE,
                    reason=reason,
                    priority=2  # High priority
                )
                
            elif event_type == 'bias':
                # Bias detected, may need retraining
                bias_result = event_data.get('bias_result', {})
                affected_symbols = bias_result.get('affected_symbols', [])
                
                for symbol in affected_symbols:
                    model_name = f"{symbol}_model"
                    if model_name in self.model_configurations:
                        self.request_retraining(
                            model_name=model_name,
                            trigger_type=RetrainingTrigger.BIAS_DETECTED,
                            reason=f"Systematic bias detected: {bias_result.get('bias_type')}",
                            priority=3
                        )
                        
        except Exception as e:
            logger.error(f"Error handling performance event: {e}")
    
    def request_retraining(self, 
                          model_name: str,
                          trigger_type: RetrainingTrigger,
                          reason: str,
                          priority: int = 5,
                          scheduled_time: Optional[datetime] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """Request model retraining"""
        try:
            if model_name not in self.model_configurations:
                raise ValueError(f"Model {model_name} not registered")
            
            config = self.model_configurations[model_name]
            
            # Check minimum retraining interval
            last_retraining = self._get_last_retraining_time(model_name)
            if last_retraining:
                time_since_last = datetime.now() - last_retraining
                if time_since_last < config.min_retraining_interval:
                    logger.warning(f"Retraining request for {model_name} ignored: "
                                 f"minimum interval not met ({time_since_last} < {config.min_retraining_interval})")
                    return ""
            
            # Create retraining request
            request_id = f"{model_name}_{int(datetime.now().timestamp())}"
            request = RetrainingRequest(
                request_id=request_id,
                model_name=model_name,
                trigger_type=trigger_type,
                trigger_reason=reason,
                priority=priority,
                requested_time=datetime.now(),
                scheduled_time=scheduled_time,
                metadata=metadata or {}
            )
            
            # Add to queue (sorted by priority)
            self.retraining_queue.append(request)
            self.retraining_queue.sort(key=lambda x: x.priority)
            
            logger.info(f"Queued retraining request {request_id} for {model_name}: {reason}")
            
            # Notify callbacks
            self._notify_callbacks('retraining_requested', {
                'request_id': request_id,
                'model_name': model_name,
                'trigger_type': trigger_type.value,
                'reason': reason
            })
            
            return request_id
            
        except Exception as e:
            logger.error(f"Error requesting retraining for {model_name}: {e}")
            return ""
    
    def _process_retraining_queue(self) -> None:
        try:
            # Check if we can start new retraining
            if len(self.active_retraining) >= self.max_concurrent_retraining:
                return
            
            # Get next request from queue
            if not self.retraining_queue:
                return
            
            request = self.retraining_queue.pop(0)
            
            # Check if scheduled time has arrived
            if request.scheduled_time and datetime.now() < request.scheduled_time:
                # Put back in queue
                self.retraining_queue.insert(0, request)
                return
            
            # Start retraining
            self._start_retraining(request)
            
        except Exception as e:
            logger.error(f"Error processing retraining queue: {e}")
    
    def _start_retraining(self, request: RetrainingRequest) -> None:
        try:
            request.status = RetrainingStatus.IN_PROGRESS
            self.active_retraining[request.request_id] = request
            
            logger.info(f"Starting retraining for {request.model_name} (request: {request.request_id})")
            
            # Submit to thread pool
            future = self.executor.submit(self._execute_retraining, request)
            
            # Store future for tracking
            request.metadata['future'] = future
            
            # Notify callbacks
            self._notify_callbacks('retraining_started', {
                'request_id': request.request_id,
                'model_name': request.model_name
            })
            
        except Exception as e:
            logger.error(f"Error starting retraining for {request.model_name}: {e}")
            request.status = RetrainingStatus.FAILED
            request.error_message = str(e)
    
    def _execute_retraining(self, request: RetrainingRequest) -> RetrainingResult:
        start_time = datetime.now()
        
        try:
            config = self.model_configurations[request.model_name]
            
            # Create backup of current model
            backup_path = self._backup_current_model(request.model_name)
            
            # Get current performance baseline
            old_performance = self._get_current_performance(request.model_name)
            
            # Execute training function
            logger.info(f"Executing training for {request.model_name}")
            request.progress = 0.1
            
            training_result = config.training_function(
                model_name=request.model_name,
                config=config,
                progress_callback=lambda p: setattr(request, 'progress', 0.1 + 0.7 * p)
            )
            
            request.progress = 0.8
            
            # Validate new model
            new_performance = None
            validation_metrics = {}
            
            if config.validation_function:
                logger.info(f"Validating retrained model {request.model_name}")
                validation_result = config.validation_function(
                    model_name=request.model_name,
                    training_result=training_result
                )
                new_performance = validation_result.get('performance')
                validation_metrics = validation_result.get('metrics', {})
            
            request.progress = 0.9
            
            # Check if retraining improved performance
            performance_improvement = None
            if old_performance is not None and new_performance is not None:
                performance_improvement = new_performance - old_performance
                
                # If performance got worse, restore backup
                if performance_improvement < -0.02:  # 2% tolerance
                    logger.warning(f"Retraining degraded performance for {request.model_name}, restoring backup")
                    self._restore_model_backup(request.model_name, backup_path)
                    new_performance = old_performance
                    performance_improvement = 0.0
            
            request.progress = 1.0
            
            # Create result
            result = RetrainingResult(
                request_id=request.request_id,
                model_name=request.model_name,
                success=True,
                training_time=datetime.now() - start_time,
                old_performance=old_performance,
                new_performance=new_performance,
                performance_improvement=performance_improvement,
                model_path=training_result.get('model_path'),
                validation_metrics=validation_metrics,
                metadata=training_result
            )
            
            old_perf_str = f"{old_performance:.3f}" if old_performance is not None else "N/A"
            new_perf_str = f"{new_performance:.3f}" if new_performance is not None else "N/A"
            logger.info(f"Completed retraining for {request.model_name}: "
                       f"performance {old_perf_str} -> {new_perf_str}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during retraining execution for {request.model_name}: {e}")
            
            # Restore backup if available
            backup_path = request.metadata.get('backup_path')
            if backup_path:
                self._restore_model_backup(request.model_name, backup_path)
            
            return RetrainingResult(
                request_id=request.request_id,
                model_name=request.model_name,
                success=False,
                training_time=datetime.now() - start_time,
                error_details=str(e)
            )
    
    def _backup_current_model(self, model_name: str) -> Optional[str]:
        try:
            model_path = self.model_storage_path / f"{model_name}.pkl"
            if not model_path.exists():
                return None
            
            backup_filename = f"{model_name}_{int(datetime.now().timestamp())}.pkl"
            backup_path = self.backup_storage_path / backup_filename
            
            # Copy model file
            import shutil
            shutil.copy2(model_path, backup_path)
            
            logger.info(f"Created model backup: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Error creating model backup for {model_name}: {e}")
            return None
    
    def _restore_model_backup(self, model_name: str, backup_path: str) -> bool:
        try:
            model_path = self.model_storage_path / f"{model_name}.pkl"
            
            import shutil
            shutil.copy2(backup_path, model_path)
            
            logger.info(f"Restored model {model_name} from backup: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring model backup for {model_name}: {e}")
            return False
    
    def _get_current_performance(self, model_name: str) -> Optional[float]:
        try:
            if self.performance_monitor:
                # Get performance from performance monitor
                report = self.performance_monitor.get_prediction_accuracy_report(days=30)
                
                # Extract performance for this model
                # This is a simplified approach - in practice, you'd need more sophisticated logic
                return report.get('overall_accuracy', 0.0)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting current performance for {model_name}: {e}")
            return None
    
    def _get_last_retraining_time(self, model_name: str) -> Optional[datetime]:
        try:
            # Find most recent successful retraining
            model_results = [r for r in self.retraining_history 
                           if r.model_name == model_name and r.success]
            
            if model_results:
                # Get completion time from most recent result
                latest_result = max(model_results, key=lambda x: x.training_time)
                # Note: We'd need to store completion time in RetrainingResult
                return datetime.now() - latest_result.training_time
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting last retraining time for {model_name}: {e}")
            return None
    
    def _check_scheduled_retraining(self) -> None:
        try:
            current_time = datetime.now()
            
            for model_name, config in self.model_configurations.items():
                if not config.retraining_schedule:
                    continue
                
                # Simple schedule check (in practice, you'd use a proper cron parser)
                # For now, just check if it's time based on interval
                last_retraining = self._get_last_retraining_time(model_name)
                
                if last_retraining is None:
                    # Never retrained, schedule now
                    self.request_retraining(
                        model_name=model_name,
                        trigger_type=RetrainingTrigger.SCHEDULED,
                        reason="Initial scheduled retraining",
                        priority=5
                    )
                elif config.retraining_schedule == "daily":
                    if (current_time - last_retraining).days >= 1:
                        self.request_retraining(
                            model_name=model_name,
                            trigger_type=RetrainingTrigger.SCHEDULED,
                            reason="Daily scheduled retraining",
                            priority=5
                        )
                elif config.retraining_schedule == "weekly":
                    if (current_time - last_retraining).days >= 7:
                        self.request_retraining(
                            model_name=model_name,
                            trigger_type=RetrainingTrigger.SCHEDULED,
                            reason="Weekly scheduled retraining",
                            priority=6
                        )
                        
        except Exception as e:
            logger.error(f"Error checking scheduled retraining: {e}")
    
    def _cleanup_completed_retraining(self) -> None:
        try:
            completed_requests = []
            
            for request_id, request in self.active_retraining.items():
                future = request.metadata.get('future')
                
                if future and future.done():
                    try:
                        result = future.result()
                        
                        # Update request status
                        if result.success:
                            request.status = RetrainingStatus.COMPLETED
                        else:
                            request.status = RetrainingStatus.FAILED
                            request.error_message = result.error_details
                        
                        request.completion_time = datetime.now()
                        
                        # Store result
                        self.retraining_history.append(result)
                        
                        # Notify callbacks
                        self._notify_callbacks('retraining_completed', {
                            'request_id': request_id,
                            'model_name': request.model_name,
                            'success': result.success,
                            'performance_improvement': result.performance_improvement
                        })
                        
                        completed_requests.append(request_id)
                        
                        logger.info(f"Completed retraining {request_id} for {request.model_name}")
                        
                    except Exception as e:
                        logger.error(f"Error getting retraining result for {request_id}: {e}")
                        request.status = RetrainingStatus.FAILED
                        request.error_message = str(e)
                        completed_requests.append(request_id)
            
            # Remove completed requests from active list
            for request_id in completed_requests:
                del self.active_retraining[request_id]
            
            # Keep only recent history (last 100 entries)
            if len(self.retraining_history) > 100:
                self.retraining_history = self.retraining_history[-100:]
                
        except Exception as e:
            logger.error(f"Error cleaning up completed retraining: {e}")
    
    def _notify_callbacks(self, event_type: str, event_data: Dict[str, Any]) -> None:
        for callback in self.callbacks:
            try:
                callback(event_type, event_data)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def add_callback(self, callback: Callable) -> None:
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable) -> None:
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def get_retraining_status(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        try:
            if model_name:
                # Status for specific model
                active = [r for r in self.active_retraining.values() if r.model_name == model_name]
                queued = [r for r in self.retraining_queue if r.model_name == model_name]
                history = [r for r in self.retraining_history if r.model_name == model_name][-10:]
                
                return {
                    'model_name': model_name,
                    'active_retraining': [r.__dict__ for r in active],
                    'queued_requests': [r.__dict__ for r in queued],
                    'recent_history': [r.__dict__ for r in history]
                }
            else:
                # Overall status
                return {
                    'active_retraining_count': len(self.active_retraining),
                    'queued_requests_count': len(self.retraining_queue),
                    'registered_models': list(self.model_configurations.keys()),
                    'recent_completions': [r.__dict__ for r in self.retraining_history[-10:]],
                    'manager_active': self.manager_active
                }
                
        except Exception as e:
            logger.error(f"Error getting retraining status: {e}")
            return {'error': str(e)}
    
    def cancel_retraining(self, request_id: str) -> bool:
        try:
            # Check if in queue
            for i, request in enumerate(self.retraining_queue):
                if request.request_id == request_id:
                    request.status = RetrainingStatus.CANCELLED
                    self.retraining_queue.pop(i)
                    logger.info(f"Cancelled queued retraining request {request_id}")
                    return True
            
            # Check if active (can't cancel active retraining easily)
            if request_id in self.active_retraining:
                logger.warning(f"Cannot cancel active retraining {request_id}")
                return False
            
            logger.warning(f"Retraining request {request_id} not found")
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling retraining {request_id}: {e}")
            return False


# Convenience functions for common model training scenarios

def create_ml_model_config(model_name: str, 
                          training_function: Callable,
                          validation_function: Optional[Callable] = None,
                          performance_threshold: float = 0.05,
                          retraining_schedule: Optional[str] = None) -> ModelConfiguration:
    """Create a model configuration for ML models"""
    return ModelConfiguration(
        model_name=model_name,
        model_class="sklearn",
        training_function=training_function,
        validation_function=validation_function,
        performance_threshold=performance_threshold,
        retraining_schedule=retraining_schedule,
        min_retraining_interval=timedelta(hours=12),
        max_training_time=timedelta(hours=2)
    )


def create_neural_network_config(model_name: str,
                                training_function: Callable,
                                validation_function: Optional[Callable] = None,
                                performance_threshold: float = 0.03,
                                retraining_schedule: Optional[str] = None) -> ModelConfiguration:
    """Create a model configuration for neural networks"""
    return ModelConfiguration(
        model_name=model_name,
        model_class="neural_network",
        training_function=training_function,
        validation_function=validation_function,
        performance_threshold=performance_threshold,
        retraining_schedule=retraining_schedule,
        min_retraining_interval=timedelta(hours=24),
        max_training_time=timedelta(hours=6)
    )