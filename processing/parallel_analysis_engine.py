"""
Parallel Analysis Engine for Multiple Stocks

This module implements thread-safe parallel processing for analyzing multiple stocks
simultaneously with efficient task distribution and resource management.
"""

import logging
import asyncio
import threading
import time
import queue
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, Future
from threading import Lock, RLock, Event
import multiprocessing as mp
from collections import defaultdict, deque
import gc

logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available, resource monitoring will be limited")


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class AnalysisTask:
    """Represents an analysis task for a stock"""
    task_id: str
    symbol: str
    analysis_type: str  # 'technical', 'fundamental', 'sentiment', 'risk', 'full'
    priority: TaskPriority
    data: Dict[str, Any]
    callback: Optional[Callable] = None
    timeout: Optional[float] = None
    created_time: datetime = field(default_factory=datetime.now)
    started_time: Optional[datetime] = None
    completed_time: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class WorkerStats:
    """Statistics for a worker thread/process"""
    worker_id: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    last_activity: Optional[datetime] = None
    current_task: Optional[str] = None
    cpu_usage: float = 0.0
    memory_usage: float = 0.0


@dataclass
class ProcessingMetrics:
    """Overall processing metrics"""
    total_tasks_submitted: int = 0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    average_queue_time: float = 0.0
    average_processing_time: float = 0.0
    throughput_per_second: float = 0.0
    active_workers: int = 0
    queue_size: int = 0
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0


class ThreadSafeTaskQueue:
    """Thread-safe priority queue for analysis tasks"""
    
    def __init__(self, maxsize: int = 0):
        self._queue = queue.PriorityQueue(maxsize=maxsize)
        self._task_map: Dict[str, AnalysisTask] = {}
        self._lock = Lock()
        self._stats = {
            'submitted': 0,
            'completed': 0,
            'failed': 0
        }
    
    def put(self, task: AnalysisTask, block: bool = True, timeout: Optional[float] = None) -> bool:
        """Add task to queue"""
        try:
            with self._lock:
                # Use priority value and creation time for ordering
                priority_item = (task.priority.value, task.created_time.timestamp(), task)
                self._queue.put(priority_item, block=block, timeout=timeout)
                self._task_map[task.task_id] = task
                self._stats['submitted'] += 1
                return True
        except queue.Full:
            logger.warning(f"Task queue full, could not add task {task.task_id}")
            return False
        except Exception as e:
            logger.error(f"Error adding task to queue: {e}")
            return False
    
    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[AnalysisTask]:
        """Get next task from queue"""
        try:
            priority_item = self._queue.get(block=block, timeout=timeout)
            task = priority_item[2]  # Extract task from priority tuple
            return task
        except queue.Empty:
            return None
        except Exception as e:
            logger.error(f"Error getting task from queue: {e}")
            return None
    
    def task_done(self, task_id: str, success: bool = True) -> None:
        """Mark task as completed"""
        with self._lock:
            if task_id in self._task_map:
                del self._task_map[task_id]
                if success:
                    self._stats['completed'] += 1
                else:
                    self._stats['failed'] += 1
            self._queue.task_done()
    
    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics"""
        with self._lock:
            return {
                **self._stats,
                'pending': self._queue.qsize(),
                'active_tasks': len(self._task_map)
            }
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task"""
        with self._lock:
            if task_id in self._task_map:
                task = self._task_map[task_id]
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.CANCELLED
                    return True
        return False
    
    def size(self) -> int:
        """Get current queue size"""
        return self._queue.qsize()
    
    def empty(self) -> bool:
        """Check if queue is empty"""
        return self._queue.empty()


class ParallelAnalysisEngine:
    """
    Main parallel processing engine for stock analysis
    """
    
    def __init__(self, 
                 max_workers: int = None,
                 use_processes: bool = False,
                 queue_maxsize: int = 1000,
                 enable_monitoring: bool = True):
        
        # Determine optimal worker count
        if max_workers is None:
            cpu_count = mp.cpu_count()
            max_workers = min(cpu_count * 2, 16)  # Cap at 16 workers
        
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.queue_maxsize = queue_maxsize
        self.enable_monitoring = enable_monitoring
        
        # Core components
        self.task_queue = ThreadSafeTaskQueue(maxsize=queue_maxsize)
        self.executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]] = None
        self.active_futures: Dict[str, Future] = {}
        self.worker_stats: Dict[str, WorkerStats] = {}
        
        # Thread safety
        self._lock = RLock()
        self._shutdown_event = Event()
        self._monitoring_thread: Optional[threading.Thread] = None
        
        # Metrics and monitoring
        self.metrics = ProcessingMetrics()
        self.task_history: deque = deque(maxlen=1000)
        self.performance_history: deque = deque(maxlen=100)
        
        # Callbacks
        self.callbacks: List[Callable] = []
        
        # Analysis modules (to be injected)
        self.analysis_modules: Dict[str, Callable] = {}
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor() if enable_monitoring else None
    
    def register_analysis_module(self, analysis_type: str, module_func: Callable) -> None:
        """Register an analysis module"""
        self.analysis_modules[analysis_type] = module_func
        logger.info(f"Registered analysis module: {analysis_type}")
    
    def start(self) -> None:
        """Start the parallel processing engine"""
        try:
            with self._lock:
                if self.executor is not None:
                    logger.warning("Engine already started")
                    return
                
                # Create executor
                if self.use_processes:
                    self.executor = ProcessPoolExecutor(
                        max_workers=self.max_workers,
                        mp_context=mp.get_context('spawn')
                    )
                else:
                    self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
                
                # Start monitoring
                if self.enable_monitoring:
                    self._monitoring_thread = threading.Thread(
                        target=self._monitoring_loop,
                        daemon=True
                    )
                    self._monitoring_thread.start()
                
                # Start resource monitoring
                if self.resource_monitor:
                    self.resource_monitor.start()
                
                logger.info(f"Started parallel analysis engine with {self.max_workers} workers "
                           f"({'processes' if self.use_processes else 'threads'})")
                
        except Exception as e:
            logger.error(f"Error starting parallel analysis engine: {e}")
            raise
    
    def stop(self, timeout: float = 30.0) -> None:
        """Stop the parallel processing engine"""
        try:
            with self._lock:
                if self.executor is None:
                    return
                
                # Signal shutdown
                self._shutdown_event.set()
                
                # Cancel pending tasks
                self._cancel_all_pending_tasks()
                
                # Shutdown executor
                self.executor.shutdown(wait=True)
                self.executor = None
                
                # Stop monitoring
                if self._monitoring_thread:
                    self._monitoring_thread.join(timeout=5.0)
                
                # Stop resource monitoring
                if self.resource_monitor:
                    self.resource_monitor.stop()
                
                logger.info("Stopped parallel analysis engine")
                
        except Exception as e:
            logger.error(f"Error stopping parallel analysis engine: {e}")
    
    def submit_task(self, task: AnalysisTask) -> bool:
        """Submit a task for processing"""
        try:
            if self.executor is None:
                logger.error("Engine not started")
                return False
            
            # Add to queue
            if not self.task_queue.put(task, block=False):
                logger.warning(f"Could not queue task {task.task_id}")
                return False
            
            # Submit to executor
            future = self.executor.submit(self._execute_task, task)
            
            with self._lock:
                self.active_futures[task.task_id] = future
                self.metrics.total_tasks_submitted += 1
            
            # Add completion callback
            future.add_done_callback(lambda f: self._task_completed(task.task_id, f))
            
            logger.debug(f"Submitted task {task.task_id} for {task.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error submitting task {task.task_id}: {e}")
            return False
    
    def submit_batch(self, tasks: List[AnalysisTask]) -> Dict[str, bool]:
        """Submit multiple tasks as a batch"""
        results = {}
        for task in tasks:
            results[task.task_id] = self.submit_task(task)
        return results
    
    def analyze_symbol(self, 
                      symbol: str,
                      analysis_types: List[str],
                      data: Dict[str, Any],
                      priority: TaskPriority = TaskPriority.NORMAL,
                      callback: Optional[Callable] = None) -> List[str]:
        """Analyze a symbol with multiple analysis types"""
        task_ids = []
        
        for analysis_type in analysis_types:
            task_id = f"{symbol}_{analysis_type}_{int(time.time() * 1000)}"
            
            task = AnalysisTask(
                task_id=task_id,
                symbol=symbol,
                analysis_type=analysis_type,
                priority=priority,
                data=data.copy(),
                callback=callback,
                timeout=60.0  # 1 minute timeout
            )
            
            if self.submit_task(task):
                task_ids.append(task_id)
        
        return task_ids
    
    def analyze_multiple_symbols(self,
                                symbols: List[str],
                                analysis_types: List[str],
                                data_provider: Callable[[str], Dict[str, Any]],
                                priority: TaskPriority = TaskPriority.NORMAL,
                                callback: Optional[Callable] = None) -> Dict[str, List[str]]:
        """Analyze multiple symbols in parallel"""
        results = {}
        
        for symbol in symbols:
            try:
                # Get data for symbol
                symbol_data = data_provider(symbol)
                
                # Submit analysis tasks
                task_ids = self.analyze_symbol(
                    symbol=symbol,
                    analysis_types=analysis_types,
                    data=symbol_data,
                    priority=priority,
                    callback=callback
                )
                
                results[symbol] = task_ids
                
            except Exception as e:
                logger.error(f"Error analyzing symbol {symbol}: {e}")
                results[symbol] = []
        
        return results
    
    def _execute_task(self, task: AnalysisTask) -> Any:
        """Execute a single analysis task"""
        start_time = time.time()
        
        try:
            # Update task status
            task.status = TaskStatus.RUNNING
            task.started_time = datetime.now()
            
            # Get analysis module
            if task.analysis_type not in self.analysis_modules:
                raise ValueError(f"Unknown analysis type: {task.analysis_type}")
            
            analysis_func = self.analysis_modules[task.analysis_type]
            
            # Execute analysis with timeout
            if task.timeout:
                # For thread-based execution, we can't easily implement timeout
                # In a production system, you'd want to use more sophisticated timeout handling
                result = analysis_func(task.symbol, task.data)
            else:
                result = analysis_func(task.symbol, task.data)
            
            # Update task
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_time = datetime.now()
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_processing_metrics(task, processing_time, success=True)
            
            # Call callback if provided
            if task.callback:
                try:
                    task.callback(task)
                except Exception as e:
                    logger.error(f"Error in task callback: {e}")
            
            return result
            
        except Exception as e:
            # Handle task failure
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_time = datetime.now()
            
            processing_time = time.time() - start_time
            self._update_processing_metrics(task, processing_time, success=False)
            
            logger.error(f"Task {task.task_id} failed: {e}")
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count})")
                # Re-submit task (simplified - in production you'd want more sophisticated retry logic)
                return self._execute_task(task)
            
            raise
    
    def _task_completed(self, task_id: str, future: Future) -> None:
        """Handle task completion"""
        try:
            with self._lock:
                if task_id in self.active_futures:
                    del self.active_futures[task_id]
            
            # Mark task as done in queue
            success = not future.exception()
            self.task_queue.task_done(task_id, success)
            
            # Update metrics
            if success:
                self.metrics.total_tasks_completed += 1
            else:
                self.metrics.total_tasks_failed += 1
            
            # Notify callbacks
            self._notify_callbacks('task_completed', {
                'task_id': task_id,
                'success': success,
                'error': str(future.exception()) if future.exception() else None
            })
            
        except Exception as e:
            logger.error(f"Error handling task completion: {e}")
    
    def _cancel_all_pending_tasks(self) -> None:
        """Cancel all pending tasks"""
        try:
            # Cancel futures
            with self._lock:
                for task_id, future in self.active_futures.items():
                    future.cancel()
                self.active_futures.clear()
            
            # Cancel queued tasks
            while not self.task_queue.empty():
                task = self.task_queue.get(block=False)
                if task:
                    task.status = TaskStatus.CANCELLED
                    self.task_queue.task_done(task.task_id, success=False)
                    
        except Exception as e:
            logger.error(f"Error cancelling pending tasks: {e}")
    
    def _update_processing_metrics(self, task: AnalysisTask, processing_time: float, success: bool) -> None:
        """Update processing metrics"""
        try:
            # Add to history
            self.task_history.append({
                'task_id': task.task_id,
                'symbol': task.symbol,
                'analysis_type': task.analysis_type,
                'processing_time': processing_time,
                'success': success,
                'timestamp': datetime.now()
            })
            
            # Update running averages
            if len(self.task_history) > 0:
                recent_tasks = list(self.task_history)[-50:]  # Last 50 tasks
                
                # Calculate average processing time
                processing_times = [t['processing_time'] for t in recent_tasks]
                self.metrics.average_processing_time = sum(processing_times) / len(processing_times)
                
                # Calculate throughput
                if len(recent_tasks) > 1:
                    time_span = (recent_tasks[-1]['timestamp'] - recent_tasks[0]['timestamp']).total_seconds()
                    if time_span > 0:
                        self.metrics.throughput_per_second = len(recent_tasks) / time_span
            
        except Exception as e:
            logger.error(f"Error updating processing metrics: {e}")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop"""
        while not self._shutdown_event.is_set():
            try:
                # Update metrics
                self._update_system_metrics()
                
                # Check for stuck tasks
                self._check_stuck_tasks()
                
                # Cleanup completed tasks
                self._cleanup_completed_tasks()
                
                # Sleep
                time.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)
    
    def _update_system_metrics(self) -> None:
        """Update system-level metrics"""
        try:
            # Update queue size
            self.metrics.queue_size = self.task_queue.size()
            
            # Update active workers
            with self._lock:
                self.metrics.active_workers = len(self.active_futures)
            
            # Update resource utilization
            if self.resource_monitor:
                resource_stats = self.resource_monitor.get_stats()
                self.metrics.cpu_utilization = resource_stats.get('cpu_percent', 0.0)
                self.metrics.memory_utilization = resource_stats.get('memory_percent', 0.0)
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
    
    def _check_stuck_tasks(self) -> None:
        """Check for tasks that may be stuck"""
        try:
            current_time = datetime.now()
            stuck_threshold = timedelta(minutes=5)  # 5 minutes
            
            with self._lock:
                stuck_tasks = []
                for task_id, future in self.active_futures.items():
                    # This is simplified - in production you'd track task start times
                    if not future.done():
                        # Check if task has been running too long
                        # Implementation would depend on how you track task start times
                        pass
                
                # Handle stuck tasks (cancel, retry, etc.)
                for task_id in stuck_tasks:
                    logger.warning(f"Detected stuck task: {task_id}")
                    # Implementation for handling stuck tasks
                    
        except Exception as e:
            logger.error(f"Error checking stuck tasks: {e}")
    
    def _cleanup_completed_tasks(self) -> None:
        """Cleanup completed task references"""
        try:
            # Force garbage collection periodically
            if len(self.task_history) % 100 == 0:
                gc.collect()
                
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")
    
    def _notify_callbacks(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Notify registered callbacks"""
        for callback in self.callbacks:
            try:
                callback(event_type, event_data)
            except Exception as e:
                logger.error(f"Error in callback: {e}")
    
    def add_callback(self, callback: Callable) -> None:
        """Add event callback"""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable) -> None:
        """Remove event callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def get_metrics(self) -> ProcessingMetrics:
        """Get current processing metrics"""
        return self.metrics
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return self.task_queue.get_stats()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'metrics': self.metrics.__dict__,
            'queue_stats': self.get_queue_stats(),
            'active_tasks': len(self.active_futures),
            'registered_modules': list(self.analysis_modules.keys()),
            'resource_stats': self.resource_monitor.get_stats() if self.resource_monitor else {},
            'uptime': datetime.now() - getattr(self, '_start_time', datetime.now())
        }


class ResourceMonitor:
    """Monitor system resource usage"""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stats = {
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
            'memory_available': 0,
            'disk_usage': 0.0
        }
        self._lock = Lock()
    
    def start(self) -> None:
        """Start resource monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop(self) -> None:
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
    
    def _monitor_loop(self) -> None:
        """Resource monitoring loop"""
        while self.monitoring:
            try:
                if PSUTIL_AVAILABLE:
                    # Get CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    
                    # Get memory usage
                    memory = psutil.virtual_memory()
                    
                    # Get disk usage
                    disk = psutil.disk_usage('/')
                    
                    with self._lock:
                        self.stats.update({
                            'cpu_percent': cpu_percent,
                            'memory_percent': memory.percent,
                            'memory_available': memory.available,
                            'disk_usage': disk.percent
                        })
                else:
                    # Fallback when psutil is not available
                    with self._lock:
                        self.stats.update({
                            'cpu_percent': 0.0,
                            'memory_percent': 0.0,
                            'memory_available': 0,
                            'disk_usage': 0.0
                        })
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(5)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current resource statistics"""
        with self._lock:
            return self.stats.copy()


# Convenience functions for common analysis patterns

def create_technical_analysis_task(symbol: str, data: Dict[str, Any], 
                                  priority: TaskPriority = TaskPriority.NORMAL) -> AnalysisTask:
    """Create a technical analysis task"""
    return AnalysisTask(
        task_id=f"{symbol}_technical_{int(time.time() * 1000)}",
        symbol=symbol,
        analysis_type="technical",
        priority=priority,
        data=data,
        timeout=30.0
    )


def create_fundamental_analysis_task(symbol: str, data: Dict[str, Any],
                                   priority: TaskPriority = TaskPriority.NORMAL) -> AnalysisTask:
    """Create a fundamental analysis task"""
    return AnalysisTask(
        task_id=f"{symbol}_fundamental_{int(time.time() * 1000)}",
        symbol=symbol,
        analysis_type="fundamental",
        priority=priority,
        data=data,
        timeout=45.0
    )


def create_risk_analysis_task(symbol: str, data: Dict[str, Any],
                             priority: TaskPriority = TaskPriority.HIGH) -> AnalysisTask:
    """Create a risk analysis task"""
    return AnalysisTask(
        task_id=f"{symbol}_risk_{int(time.time() * 1000)}",
        symbol=symbol,
        analysis_type="risk",
        priority=priority,
        data=data,
        timeout=20.0
    )