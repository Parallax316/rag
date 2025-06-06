import os
import logging
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database settings
DATABASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'image_embeddings.db')

# Model settings
MODEL_NAME = "vidore/colqwen2-v0.1"
PROCESSOR_NAME = "vidore/colqwen2.5-v0.2"

# Device settings
def get_device():
    if torch.cuda.is_available():
        logger.info("CUDA is available - using GPU")
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("MPS is available - using Apple Silicon GPU")
        return "mps"
    else:
        logger.info("No GPU available - falling back to CPU")
        return "cpu"

# Force device selection priority: CUDA GPU > MPS > CPU
DEVICE_MAP = get_device()

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000

# LLM settings
LLM_MODEL = "qwen2.5vl:3b"