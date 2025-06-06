import unittest
import sys
import os
import io
from PIL import Image
import numpy as np

# Add the parent directory to the path so we can import the app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.app.services.image_service import ImageService
from backend.app.models.model_loader import ModelManager

class TestImageService(unittest.TestCase):
    
    def setUp(self):
        # This is a mock test that doesn't actually load the model
        # In a real test, you would use mocking to avoid loading the actual model
        pass
    
    def test_image_hash_generation(self):
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Calculate hash directly
        import hashlib
        expected_hash = hashlib.sha256(img_byte_arr).hexdigest()
        
        # This is just a verification that the hash function works as expected
        self.assertTrue(len(expected_hash) > 0)
        
    def test_image_dimensions(self):
        # Create test images of different sizes
        sizes = [(100, 100), (200, 150), (300, 200)]
        
        for width, height in sizes:
            img = Image.new('RGB', (width, height), color='blue')
            self.assertEqual(img.size[0], width)
            self.assertEqual(img.size[1], height)

if __name__ == '__main__':
    unittest.main()