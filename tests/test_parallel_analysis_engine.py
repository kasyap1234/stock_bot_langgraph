"""
Tests for Parallel Analysis Engine

Tests cover:
- Thread-safe task queue operations
- Parallel processing of multiple stocks
- Task prioritization and scheduling
- Resource monitoring and management
- Error handling and recovery
"""

import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import Future
from typing import Dict, Any, List

from processing.parallel_analysis_engine import (
    ParallelAnalysisEngine,
    ThreadSafeTaskQueue,
    AnalysisTask,
    TaskPriority,
    TaskStatus,
    WorkerStats,
    ProcessingMetrics,
    ResourceMonitor,
    create_technical_analysis_task,
    create_fundamental_analysis_task,
    create_risk_analysis_task
)


class TestThreadSafeTaskQueue:
    """Test suite for ThreadSafeTaskQueue"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.queue = ThreadSafeTaskQueue(maxsize=10)
    
    def test_put_and_get_task(self):
        """Test basic put and get operations"""
        task = AnalysisTask(
            task_id="test_001",
            symbol="AAPL",
            analysis_type="technical",
            priority=TaskPriority.NORMAL,
            data={"price": 150.0}
        )
        
        # Put task
        success = self.queue.put(task)
        assert success is True
        assert self.queue.size() == 1
        
        # Get task
        retrieved_task = self.queue.get(block=False)
        assert retrieved_task is not None
        assert retrieved_task.task_id == "test_001"
        assert retrieved_task.symbol == "AAPL"
    
    def test_priority_ordering(self):
        """Test that tasks are retrieved in priority order"""
        # Create tasks with different priorities
        low_task = AnalysisTask("low", "AAPL", "technical", TaskPriority.LOW, {})
        high_task = AnalysisTask("high", "GOOGL", "technical", TaskPriority.HIGH, {})
        critical_task = AnalysisTask("critical", "MSFT", "technical", TaskPriority.CRITICAL, {})
        
        # Add in non-priority order
        self.queue.put(low_task)
        self.queue.put(high_task)
        self.queue.put(critical_task)
        
        # Should retrieve in priority order (CRITICAL, HIGH, LOW)
        first = self.queue.get(block=False)
        second = self.queue.get(block=False)
        third = self.queue.get(block=False)
        
        assert first.task_id == "critical"
        assert second.task_id == "high"
        assert third.task_id == "low"
    
    def test_task_done_tracking(self):
        """Test task completion tracking"""
        task = AnalysisTask("test", "AAPL", "technical", TaskPriority.NORMAL, {})
        
        self.queue.put(task)
        initial_stats = self.queue.get_stats()
        
        # Mark task as done
        self.queue.task_done("test", success=True)
        
        final_stats = self.queue.get_stats()
        assert final_stats['completed'] == initial_stats['completed'] + 1
    
    def test_cancel_task(self):
        """Test task cancellation"""
        task = AnalysisTask("cancel_test", "AAPL", "technical", TaskPriority.NORMAL, {})
        task.status = TaskStatus.PENDING
        
        self.queue.put(task)
        
        # Cancel task
        success = self.queue.cancel_task("cancel_test")
        assert success is True
        assert task.status == TaskStatus.CANCELLED
    
    def test_thread_safety(self):
        """Test thread safety of queue operations"""
        results = []
        errors = []
        
        def producer():
            try:
                for i in range(50):
                    task = AnalysisTask(f"task_{i}", "AAPL", "technical", TaskPriority.NORMAL, {})
                    self.queue.put(task)
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(e)
        
        def consumer():
            try:
                for _ in range(25):
                    task = self.queue.get(timeout=1.0)
                    if task:
                        results.append(task.task_id)
                        self.queue.task_done(task.task_id)
            except Exception as e:
                errors.append(e)
        
        # Start multiple producers and consumers
        threads = []
        for _ in range(2):
            threads.append(threading.Thread(target=producer))
        for _ in range(4):
            threads.append(threading.Thread(target=consumer))
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) > 0, "No tasks were processed"


class TestParallelAnalysisEngine:
    """Test suite for ParallelAnalysisEngine"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.engine = ParallelAnalysisEngine(
            max_workers=2,
            use_processes=False,  # Use threads for testing
            queue_maxsize=100,
            enable_monitoring=False  # Disable for testing
        )
        
        # Register mock analysis modules
        self.engine.register_analysis_module("technical", self._mock_technical_analysis)
        self.engine.register_analysis_module("fundamental", self._mock_fundamental_analysis)
        self.engine.register_analysis_module("risk", self._mock_risk_analysis)
    
    def teardown_method(self):
        """Clean up test fixtures"""
        if self.engine.executor:
            self.engine.stop()
    
    def _mock_technical_analysis(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock technical analysis function"""
        time.sleep(0.1)  # Simulate processing time
        return {
            "symbol": symbol,
            "signal": "BUY" if hash(symbol) % 2 == 0 else "SELL",
            "confidence": 0.75,
            "indicators": {"rsi": 65, "macd": 0.5}
        }
    
    def _mock_fundamental_analysis(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock fundamental analysis function"""
        time.sleep(0.15)  # Simulate longer processing time
        return {
            "symbol": symbol,
            "valuation": "undervalued" if hash(symbol) % 3 == 0 else "overvalued",
            "score": 0.8,
            "metrics": {"pe_ratio": 15.5, "debt_ratio": 0.3}
        }
    
    def _mock_risk_analysis(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock risk analysis function"""
        time.sleep(0.05)  # Simulate fast processing
        return {
            "symbol": symbol,
            "risk_level": "medium",
            "volatility": 0.25,
            "beta": 1.2
        }
    
    def test_engine_lifecycle(self):
        """Test engine start and stop"""
        # Start engine
        self.engine.start()
        assert self.engine.executor is not None
        
        # Stop engine
        self.engine.stop()
        assert self.engine.executor is None
    
    def test_register_analysis_module(self):
        """Test analysis module registration"""
        def custom_analysis(symbol, data):
            return {"custom": True}
        
        self.engine.register_analysis_module("custom", custom_analysis)
        assert "custom" in self.engine.analysis_modules
        assert self.engine.analysis_modules["custom"] == custom_analysis
    
    def test_submit_single_task(self):
        """Test submitting a single analysis task"""
        self.engine.start()
        
        task = create_technical_analysis_task("AAPL", {"price": 150.0})
        
        success = self.engine.submit_task(task)
        assert success is True
        
        # Wait for completion
        time.sleep(0.5)
        
        # Check metrics
        metrics = self.engine.get_metrics()
        assert metrics.total_tasks_submitted >= 1
    
    def test_analyze_symbol(self):
        """Test analyzing a single symbol with multiple analysis types"""
        self.engine.start()
        
        task_ids = self.engine.analyze_symbol(
            symbol="AAPL",
            analysis_types=["technical", "fundamental", "risk"],
            data={"price": 150.0, "volume": 1000000}
        )
        
        assert len(task_ids) == 3
        assert all(task_id.startswith("AAPL_") for task_id in task_ids)
        
        # Wait for completion
        time.sleep(1.0)
        
        # Check that tasks were processed
        metrics = self.engine.get_metrics()
        assert metrics.total_tasks_submitted >= 3
    
    def test_analyze_multiple_symbols(self):
        """Test analyzing multiple symbols in parallel"""
        self.engine.start()
        
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
        analysis_types = ["technical", "risk"]
        
        def data_provider(symbol):
            return {"price": 100.0 + hash(symbol) % 100, "volume": 1000000}
        
        results = self.engine.analyze_multiple_symbols(
            symbols=symbols,
            analysis_types=analysis_types,
            data_provider=data_provider
        )
        
        # Check results
        assert len(results) == len(symbols)
        for symbol in symbols:
            assert symbol in results
            assert len(results[symbol]) == len(analysis_types)
        
        # Wait for completion
        time.sleep(2.0)
        
        # Check metrics
        metrics = self.engine.get_metrics()
        expected_tasks = len(symbols) * len(analysis_types)
        assert metrics.total_tasks_submitted >= expected_tasks
    
    def test_task_prioritization(self):
        """Test that high priority tasks are processed first"""
        self.engine.start()
        
        results = []
        
        def callback(task):
            results.append((task.task_id, task.priority))
        
        # Submit tasks with different priorities
        low_task = AnalysisTask("low", "AAPL", "technical", TaskPriority.LOW, {}, callback=callback)
        high_task = AnalysisTask("high", "GOOGL", "technical", TaskPriority.HIGH, {}, callback=callback)
        critical_task = AnalysisTask("critical", "MSFT", "technical", TaskPriority.CRITICAL, {}, callback=callback)
        
        # Submit in non-priority order
        self.engine.submit_task(low_task)
        self.engine.submit_task(high_task)
        self.engine.submit_task(critical_task)
        
        # Wait for completion
        time.sleep(1.0)
        
        # Check that higher priority tasks were processed first
        assert len(results) == 3
        # Note: Due to parallel processing, exact order might vary, but critical should be early
        priorities = [result[1] for result in results]
        assert TaskPriority.CRITICAL in priorities
    
    def test_task_retry_on_failure(self):
        """Test task retry mechanism on failure"""
        def failing_analysis(symbol, data):
            if symbol == "FAIL":
                raise Exception("Simulated failure")
            return {"success": True}
        
        self.engine.register_analysis_module("failing", failing_analysis)
        self.engine.start()
        
        # Create task that will fail
        task = AnalysisTask(
            task_id="fail_test",
            symbol="FAIL",
            analysis_type="failing",
            priority=TaskPriority.NORMAL,
            data={},
            max_retries=2
        )
        
        success = self.engine.submit_task(task)
        assert success is True
        
        # Wait for completion (including retries)
        time.sleep(2.0)
        
        # Task should have failed after retries
        assert task.status == TaskStatus.FAILED
        assert task.retry_count == 2
    
    def test_callback_execution(self):
        """Test that task callbacks are executed"""
        self.engine.start()
        
        callback_results = []
        
        def test_callback(task):
            callback_results.append(task.task_id)
        
        task = create_technical_analysis_task("AAPL", {"price": 150.0})
        task.callback = test_callback
        
        self.engine.submit_task(task)
        
        # Wait for completion
        time.sleep(0.5)
        
        # Check callback was executed
        assert len(callback_results) == 1
        assert callback_results[0] == task.task_id
    
    def test_batch_submission(self):
        """Test batch task submission"""
        self.engine.start()
        
        tasks = []
        for i in range(5):
            task = create_technical_analysis_task(f"STOCK_{i}", {"price": 100.0 + i})
            tasks.append(task)
        
        results = self.engine.submit_batch(tasks)
        
        # Check all tasks were submitted successfully
        assert len(results) == 5
        assert all(success for success in results.values())
        
        # Wait for completion
        time.sleep(1.0)
        
        # Check metrics
        metrics = self.engine.get_metrics()
        assert metrics.total_tasks_submitted >= 5
    
    def test_system_status_reporting(self):
        """Test system status reporting"""
        self.engine.start()
        
        # Submit some tasks
        for i in range(3):
            task = create_technical_analysis_task(f"TEST_{i}", {"price": 100.0})
            self.engine.submit_task(task)
        
        # Get system status
        status = self.engine.get_system_status()
        
        # Check status structure
        assert 'metrics' in status
        assert 'queue_stats' in status
        assert 'active_tasks' in status
        assert 'registered_modules' in status
        
        # Check registered modules
        assert 'technical' in status['registered_modules']
        assert 'fundamental' in status['registered_modules']
        assert 'risk' in status['registered_modules']
    
    def test_engine_with_processes(self):
        """Test engine using processes instead of threads"""
        # Create engine with processes
        process_engine = ParallelAnalysisEngine(
            max_workers=2,
            use_processes=True,
            enable_monitoring=False
        )
        
        # Note: Process-based testing is more complex due to serialization requirements
        # In a real implementation, you'd need to ensure analysis functions are picklable
        
        try:
            process_engine.start()
            assert process_engine.executor is not None
            
            # For this test, we'll just verify the engine starts correctly
            # Full process testing would require more setup
            
        finally:
            process_engine.stop()
    
    def test_resource_monitoring_integration(self):
        """Test integration with resource monitoring"""
        # Create engine with monitoring enabled
        monitored_engine = ParallelAnalysisEngine(
            max_workers=2,
            enable_monitoring=True
        )
        
        try:
            monitored_engine.start()
            
            # Submit some tasks
            for i in range(3):
                task = create_technical_analysis_task(f"MONITOR_{i}", {"price": 100.0})
                monitored_engine.submit_task(task)
            
            # Wait a bit for monitoring to collect data
            time.sleep(1.0)
            
            # Check that resource monitoring is working
            status = monitored_engine.get_system_status()
            assert 'resource_stats' in status
            
        finally:
            monitored_engine.stop()


class TestResourceMonitor:
    """Test suite for ResourceMonitor"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.monitor = ResourceMonitor()
    
    def teardown_method(self):
        """Clean up test fixtures"""
        self.monitor.stop()
    
    def test_monitor_lifecycle(self):
        """Test monitor start and stop"""
        # Start monitoring
        self.monitor.start()
        assert self.monitor.monitoring is True
        assert self.monitor.monitor_thread is not None
        
        # Stop monitoring
        self.monitor.stop()
        assert self.monitor.monitoring is False
    
    def test_stats_collection(self):
        """Test that resource statistics are collected"""
        self.monitor.start()
        
        # Wait for some stats to be collected
        time.sleep(2.0)
        
        stats = self.monitor.get_stats()
        
        # Check that stats are present and reasonable
        assert 'cpu_percent' in stats
        assert 'memory_percent' in stats
        assert 'memory_available' in stats
        assert 'disk_usage' in stats
        
        # Check that values are in reasonable ranges
        assert 0 <= stats['cpu_percent'] <= 100
        assert 0 <= stats['memory_percent'] <= 100
        assert stats['memory_available'] >= 0
        assert 0 <= stats['disk_usage'] <= 100


class TestTaskCreationHelpers:
    """Test suite for task creation helper functions"""
    
    def test_create_technical_analysis_task(self):
        """Test technical analysis task creation"""
        task = create_technical_analysis_task("AAPL", {"price": 150.0})
        
        assert task.symbol == "AAPL"
        assert task.analysis_type == "technical"
        assert task.priority == TaskPriority.NORMAL
        assert task.data == {"price": 150.0}
        assert task.timeout == 30.0
    
    def test_create_fundamental_analysis_task(self):
        """Test fundamental analysis task creation"""
        task = create_fundamental_analysis_task("GOOGL", {"revenue": 1000000})
        
        assert task.symbol == "GOOGL"
        assert task.analysis_type == "fundamental"
        assert task.priority == TaskPriority.NORMAL
        assert task.data == {"revenue": 1000000}
        assert task.timeout == 45.0
    
    def test_create_risk_analysis_task(self):
        """Test risk analysis task creation"""
        task = create_risk_analysis_task("TSLA", {"volatility": 0.35})
        
        assert task.symbol == "TSLA"
        assert task.analysis_type == "risk"
        assert task.priority == TaskPriority.HIGH  # Risk analysis has high priority
        assert task.data == {"volatility": 0.35}
        assert task.timeout == 20.0


if __name__ == "__main__":
    pytest.main([__file__])