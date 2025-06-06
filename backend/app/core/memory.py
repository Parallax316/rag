import torch
import gc
import logging
from ..config import DEVICE_MAP

logger = logging.getLogger(__name__)

def clear_cache():
    """
    Clear GPU memory cache for different platforms.
    """
    try:
        logger.info("Clearing memory cache...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Additional CUDA memory stats if available
            if hasattr(torch.cuda, 'memory_summary'):
                logger.info(f"CUDA Memory Summary:\n{torch.cuda.memory_summary(abbreviated=True)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        # CPU doesn't need explicit cache clearing
        gc.collect()  # Force garbage collection
        logger.info("Memory cache cleared")
    except Exception as e:
        logger.error(f"Could not clear cache: {str(e)}")

def get_device():
    """
    Get the appropriate device for tensor operations
    """
    return DEVICE_MAP