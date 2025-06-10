# Timeout monitoring and optimization utilities
import time
import threading
import logging
from contextlib import contextmanager
from typing import Optional, Callable, Any
import signal
import torch

logger = logging.getLogger(__name__)

class TimeoutMonitor:
    """Monitor and handle timeouts with detailed logging"""
    
    def __init__(self, operation_name: str, timeout_seconds: int = 300):
        self.operation_name = operation_name
        self.timeout_seconds = timeout_seconds
        self.start_time = None
        self.is_running = False
        self.progress_callback: Optional[Callable[[str], None]] = None
        
    def set_progress_callback(self, callback: Callable[[str], None]):
        """Set a callback function to report progress"""
        self.progress_callback = callback
        
    def log_progress(self, message: str):
        """Log progress and call callback if set"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        progress_msg = f"[{self.operation_name}] {message} (elapsed: {elapsed:.1f}s)"
        logger.info(progress_msg)
        print(progress_msg)
        
        if self.progress_callback:
            self.progress_callback(progress_msg)
    
    @contextmanager
    def monitor(self):
        """Context manager to monitor operation timeout"""
        self.start_time = time.time()
        self.is_running = True
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitor_timeout, daemon=True)
        monitor_thread.start()
        
        try:
            self.log_progress("Operation started")
            yield self
            self.log_progress("Operation completed successfully")
        except Exception as e:
            self.log_progress(f"Operation failed: {str(e)}")
            raise
        finally:
            self.is_running = False
            elapsed = time.time() - self.start_time
            logger.info(f"[{self.operation_name}] Total time: {elapsed:.2f}s")
    
    def _monitor_timeout(self):
        """Monitor timeout in background thread"""
        check_interval = 30  # Check every 30 seconds
        last_check = self.start_time
        
        while self.is_running:
            time.sleep(check_interval)
            
            if not self.is_running:
                break
                
            elapsed = time.time() - self.start_time
            remaining = self.timeout_seconds - elapsed
            
            if elapsed > last_check + check_interval:
                if remaining > 0:
                    self.log_progress(f"Still running... {remaining:.0f}s remaining")
                else:
                    self.log_progress(f"Timeout exceeded by {-remaining:.0f}s")
                last_check = elapsed
                
                # Check GPU memory if available
                if torch.cuda.is_available():
                    try:
                        allocated = torch.cuda.memory_allocated() / 1e9
                        self.log_progress(f"GPU memory allocated: {allocated:.2f} GB")
                    except:
                        pass

def optimize_for_timeout_prevention():
    """Apply optimizations to prevent timeouts"""
    logger.info("=== Applying Timeout Prevention Optimizations ===")
    print("[OPTIMIZER] Applying timeout prevention optimizations...")
    
    try:
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
            print("[OPTIMIZER] GPU cache cleared")
            
            # Set memory fraction to prevent OOM
            try:
                # This is a conservative setting
                torch.cuda.set_per_process_memory_fraction(0.8)
                logger.info("GPU memory fraction set to 80%")
                print("[OPTIMIZER] GPU memory fraction set to 80%")
            except:
                logger.warning("Could not set GPU memory fraction")
        
        # Set torch to use optimized settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        logger.info("CUDNN optimizations enabled")
        print("[OPTIMIZER] CUDNN optimizations enabled")
        
        # Set number of threads for CPU operations
        torch.set_num_threads(min(4, torch.get_num_threads()))
        logger.info(f"Torch threads set to {torch.get_num_threads()}")
        print(f"[OPTIMIZER] Torch threads set to {torch.get_num_threads()}")
        
        return True
        
    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}")
        print(f"[OPTIMIZER] Optimization failed: {str(e)}")
        return False

def create_timeout_handler(timeout_seconds: int = 300):
    """Create a timeout handler for operations"""
    
    def timeout_decorator(func):
        def wrapper(*args, **kwargs):
            operation_name = f"{func.__name__}"
            monitor = TimeoutMonitor(operation_name, timeout_seconds)
            
            with monitor.monitor():
                return func(*args, **kwargs)
        
        return wrapper
    return timeout_decorator

# Apply optimizations on module import
optimize_for_timeout_prevention()
