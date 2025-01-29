import time
import psutil

class PerformanceMonitor:
    """Monitor system resources during processing"""
    def __init__(self):
        self.start_time = time.time()
        self.memory_usage = []
        
    def log_metrics(self):
        """Log current system metrics"""
        memory = psutil.Process().memory_info().rss / 1024 / 1024
        self.memory_usage.append(memory) 