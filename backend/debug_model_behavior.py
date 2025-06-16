#!/usr/bin/env python3
"""
Test to see what exact dimensions the model produces and why they might differ
"""

import torch
import numpy as np
from PIL import Image
import io
import requests
from app.models.model_loader import ModelManager
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_model_behavior():
    """Debug what's happening with the model dimensions"""
    try:
        print("=== Debugging Model Behavior ===")
        
        # Initialize model manager
        print("Loading model...")
        model_manager = ModelManager()
        
        print(f"Model: {model_manager.model}")
        print(f"Processor: {model_manager.processor}")
        
        # Test with different image sizes
        test_cases = [
            ("Small image", Image.new('RGB', (224, 224), color='red')),
            ("Medium image", Image.new('RGB', (512, 512), color='blue')), 
            ("Large image", Image.new('RGB', (1024, 1024), color='green')),
        ]
        
        for name, test_image in test_cases:
            print(f"\n--- Testing {name} ({test_image.size}) ---")
            try:
                image_embedding = model_manager.process_image(test_image)
                print(f"Image embedding shape: {image_embedding.shape}")
                print(f"Image embedding dimensions: {len(image_embedding.flatten())}")
            except Exception as e:
                print(f"Error processing {name}: {e}")
        
        # Test queries
        test_queries = [
            "short query",
            "this is a medium length query with some more words",
            "this is a very long query with many words that should test if the query length affects the embedding dimensions in any way shape or form"
        ]
        
        for i, query in enumerate(test_queries):
            print(f"\n--- Testing Query {i+1} (length: {len(query.split())} words) ---")
            print(f"Query: '{query}'")
            try:
                query_embedding = model_manager.process_query(query)
                print(f"Query embedding shape: {query_embedding.shape}")
                print(f"Query embedding dimensions: {len(query_embedding.flatten())}")
            except Exception as e:
                print(f"Error processing query {i+1}: {e}")
        
        # Test the same image multiple times to see if results are consistent
        print(f"\n--- Testing Consistency (same image multiple times) ---")
        test_image = Image.new('RGB', (224, 224), color='red')
        shapes = []
        for i in range(3):
            embedding = model_manager.process_image(test_image)
            shapes.append(embedding.shape)
            print(f"Run {i+1}: Shape {embedding.shape}")
        
        if len(set(shapes)) == 1:
            print("✅ Model is consistent - same image produces same shape")
        else:
            print(f"⚠️ Model is inconsistent - got shapes: {set(shapes)}")
            
    except Exception as e:
        print(f"Error in model debugging: {e}")
        logger.error(f"Model debugging failed: {e}", exc_info=True)

if __name__ == "__main__":
    debug_model_behavior()
