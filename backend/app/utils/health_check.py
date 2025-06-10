# Health check utilities for debugging timeout issues
import time
import logging
import torch
from ..models.model_loader import ModelManager
from ..core.mongodb import embeddings_col
from PIL import Image
import io
import numpy as np

logger = logging.getLogger(__name__)

def test_model_performance():
    """Test model loading and inference performance"""
    logger.info("=== Testing Model Performance ===")
    print("[HEALTH_CHECK] Testing model performance...")
    
    try:
        start_time = time.time()
        
        # Test model loading
        logger.info("Step 1: Loading model...")
        print("[HEALTH_CHECK] Step 1: Loading model...")
        model_manager = ModelManager()
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f} seconds")
        print(f"[HEALTH_CHECK] Model loaded in {load_time:.2f} seconds")
        
        # Test image processing
        logger.info("Step 2: Testing image processing...")
        print("[HEALTH_CHECK] Step 2: Testing image processing...")
        
        # Create a simple test image
        test_image = Image.new('RGB', (224, 224), color='red')
        
        inference_start = time.time()
        embedding = model_manager.process_image(test_image)
        inference_time = time.time() - inference_start
        
        logger.info(f"Image processing completed in {inference_time:.2f} seconds")
        print(f"[HEALTH_CHECK] Image processing completed in {inference_time:.2f} seconds")
        logger.info(f"Embedding shape: {embedding.shape}")
        print(f"[HEALTH_CHECK] Embedding shape: {embedding.shape}")
        
        total_time = time.time() - start_time
        logger.info(f"Total test time: {total_time:.2f} seconds")
        print(f"[HEALTH_CHECK] Total test time: {total_time:.2f} seconds")
        
        return {
            "status": "success",
            "load_time": load_time,
            "inference_time": inference_time,
            "total_time": total_time,
            "embedding_shape": embedding.shape
        }
        
    except Exception as e:
        logger.error(f"Model performance test failed: {str(e)}", exc_info=True)
        print(f"[HEALTH_CHECK] Model test failed: {str(e)}")
        return {
            "status": "failed",
            "error": str(e)
        }

def test_mongodb_performance():
    """Test MongoDB connection and performance"""
    logger.info("=== Testing MongoDB Performance ===")
    print("[HEALTH_CHECK] Testing MongoDB performance...")
    
    try:
        start_time = time.time()
        
        # Test connection
        logger.info("Step 1: Testing connection...")
        print("[HEALTH_CHECK] Step 1: Testing connection...")
        
        # Ping the database
        result = embeddings_col.database.command("ping")
        ping_time = time.time() - start_time
        logger.info(f"MongoDB ping successful in {ping_time:.2f} seconds")
        print(f"[HEALTH_CHECK] MongoDB ping successful in {ping_time:.2f} seconds")
        
        # Test insert performance
        logger.info("Step 2: Testing insert performance...")
        print("[HEALTH_CHECK] Step 2: Testing insert performance...")
        
        test_doc = {
            "collection_name": "health_check",
            "type": "test",
            "embedding": [0.1, 0.2, 0.3] * 100,  # Small test embedding
            "data": {"test": "health_check"},
            "metadata": {"timestamp": time.time()}
        }
        
        insert_start = time.time()
        insert_result = embeddings_col.insert_one(test_doc)
        insert_time = time.time() - insert_start
        
        logger.info(f"Test document inserted in {insert_time:.2f} seconds")
        print(f"[HEALTH_CHECK] Test document inserted in {insert_time:.2f} seconds")
        
        # Clean up test document
        embeddings_col.delete_one({"_id": insert_result.inserted_id})
        
        total_time = time.time() - start_time
        logger.info(f"MongoDB test completed in {total_time:.2f} seconds")
        print(f"[HEALTH_CHECK] MongoDB test completed in {total_time:.2f} seconds")
        
        return {
            "status": "success",
            "ping_time": ping_time,
            "insert_time": insert_time,
            "total_time": total_time
        }
        
    except Exception as e:
        logger.error(f"MongoDB performance test failed: {str(e)}", exc_info=True)
        print(f"[HEALTH_CHECK] MongoDB test failed: {str(e)}")
        return {
            "status": "failed",
            "error": str(e)
        }

def test_gpu_memory():
    """Test GPU memory availability"""
    logger.info("=== Testing GPU Memory ===")
    print("[HEALTH_CHECK] Testing GPU memory...")
    
    try:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_memory = torch.cuda.memory_allocated(device)
            free_memory = total_memory - allocated_memory
            
            logger.info(f"GPU Device: {torch.cuda.get_device_name(device)}")
            logger.info(f"Total Memory: {total_memory / 1e9:.2f} GB")
            logger.info(f"Allocated Memory: {allocated_memory / 1e9:.2f} GB")
            logger.info(f"Free Memory: {free_memory / 1e9:.2f} GB")
            
            print(f"[HEALTH_CHECK] GPU Device: {torch.cuda.get_device_name(device)}")
            print(f"[HEALTH_CHECK] Total Memory: {total_memory / 1e9:.2f} GB")
            print(f"[HEALTH_CHECK] Allocated Memory: {allocated_memory / 1e9:.2f} GB")
            print(f"[HEALTH_CHECK] Free Memory: {free_memory / 1e9:.2f} GB")
            
            return {
                "status": "success",
                "gpu_available": True,
                "device_name": torch.cuda.get_device_name(device),
                "total_memory_gb": total_memory / 1e9,
                "allocated_memory_gb": allocated_memory / 1e9,
                "free_memory_gb": free_memory / 1e9
            }
        else:
            logger.info("No GPU available, using CPU")
            print("[HEALTH_CHECK] No GPU available, using CPU")
            return {
                "status": "success", 
                "gpu_available": False,
                "device": "CPU"
            }
            
    except Exception as e:
        logger.error(f"GPU memory test failed: {str(e)}", exc_info=True)
        print(f"[HEALTH_CHECK] GPU test failed: {str(e)}")
        return {
            "status": "failed",
            "error": str(e)
        }

def run_full_health_check():
    """Run complete health check"""
    logger.info("=== Starting Full Health Check ===")
    print("[HEALTH_CHECK] Starting full system health check...")
    
    results = {
        "timestamp": time.time(),
        "model": test_model_performance(),
        "mongodb": test_mongodb_performance(),
        "gpu": test_gpu_memory()
    }
    
    # Summary
    all_passed = all(result.get("status") == "success" for result in results.values() if isinstance(result, dict))
    
    if all_passed:
        logger.info("=== Health Check: ALL SYSTEMS HEALTHY ===")
        print("[HEALTH_CHECK] === ALL SYSTEMS HEALTHY ===")
    else:
        logger.warning("=== Health Check: SOME ISSUES DETECTED ===")
        print("[HEALTH_CHECK] === SOME ISSUES DETECTED ===")
    
    return results
