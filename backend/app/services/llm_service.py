import ollama
import logging
import base64
import io
from PIL import Image
from ..config import LLM_MODEL
from ..core.memory import clear_cache
import requests
import os

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.model = LLM_MODEL
    
    def generate_response(self, query, image_base64):
        """
        Generate a response from the Ollama LLM model based on the query and image
        """
        ollama_url = os.getenv('OLLAMA_API_URL', 'http://localhost:11434')
        try:
            # Prepare payload for Ollama
            payload = {
                "model": self.model,
                "messages": [
                    {
                        'role': 'user',
                        'content': f"Please answer the following question using only the information visible in the provided image. Do not use any of your own knowledge, training data, or external sources. Base your response solely on the content depicted within the image. If there is no relation with question and image, you can respond with 'Question is not related to image'.\nHere is the question: {query}",
                        'images': [image_base64]
                    }
                ],
                "stream": False
            }
            response = requests.post(f"{ollama_url}/v1/chat/completions", json=payload)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Error generating response from Ollama: {str(e)}")
            raise
        finally:
            clear_cache()