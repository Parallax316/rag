import ollama
import logging
import base64
import io
from PIL import Image
from ..config import LLM_MODEL
from ..core.memory import clear_cache

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.model = LLM_MODEL
    
    def generate_response(self, query, image_base64):
        """
        Generate a response from the LLM model based on the query and image
        """
        try:
            # Ensure GPU memory is cleared before calling Ollama
            clear_cache()
            logger.info(f"Sending query to {self.model} model")
            
            # Decode image from base64
            img_data = base64.b64decode(image_base64)
            
            # Call Ollama API
            stream = ollama.chat(
                model=self.model,
                messages=[
                    {
                        'role': 'user',
                        'content': "Please answer the following question using only the information visible in the provided image" 
                        " Do not use any of your own knowledge, training data, or external sources."
                        " Base your response solely on the content depicted within the image."
                        " If there is no relation with question and image," 
                        f" you can respond with 'Question is not related to image'.\nHere is the question: {query}",
                        'images': [img_data]
                    }
                ],
                stream=True
            )
            
            # Process the streaming response
            collected_chunks = []
            for chunk in stream:
                chunk_content = chunk['message']['content']
                collected_chunks.append(chunk_content)
            
            # Combine all chunks into the complete response
            complete_response = ''.join(collected_chunks)
            logger.info("Response generation completed")
            
            return complete_response
            
        except Exception as e:
            logger.error(f"Error generating response from Ollama: {str(e)}")
            raise
        finally:
            # Ensure memory is cleared after response generation
            clear_cache()