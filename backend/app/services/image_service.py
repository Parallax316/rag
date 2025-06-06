import base64
import io
import hashlib
import numpy as np
import logging
from PIL import Image
from pdf2image import convert_from_bytes
from ..models.model_loader import ModelManager
from ..core.database import get_db_connection, store_embedding
from ..core.memory import clear_cache
import pickle


logger = logging.getLogger(__name__)

class ImageService:
    def __init__(self):
        self.model_manager = ModelManager()
    
    def process_image_file(self, image_data):
        """
        Process a single image file
        """
        try:
            # Compute image hash
            image_hash = hashlib.sha256(image_data).hexdigest()
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # Encode image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Process and index the image
            self.process_and_index_image(image, img_str, image_hash)
            
            return image_hash
        except Exception as e:
            logger.error(f"Error processing image file: {str(e)}")
            raise
    
    def process_pdf_file(self, pdf_data):
        """
        Process a PDF file by converting it to images
        """
        try:
            logger.info("Converting PDF to images")
            images = convert_from_bytes(pdf_data)
            logger.info(f"PDF converted to {len(images)} images")
            
            image_hashes = []
            for j, image in enumerate(images):
                logger.info(f"Processing PDF page {j+1}/{len(images)}")
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                byte_data = buffer.getvalue()
                img_str = base64.b64encode(byte_data).decode('utf-8')
                
                # Compute image hash
                image_hash = hashlib.sha256(byte_data).hexdigest()
                
                # Process and index the image
                self.process_and_index_image(image, img_str, image_hash)
                image_hashes.append(image_hash)
            
            return image_hashes
        except Exception as e:
            logger.error(f"Error processing PDF file: {str(e)}")
            raise
    
    def process_and_index_image(self, image, img_str, image_hash):
        """
        Process an image and store its embedding in the database
        """
        try:
            logger.info(f"Processing image with hash: {image_hash[:8]}...")
            
            # Get database connection
            conn = get_db_connection()
            
            # Check if image already exists in database
            c = conn.cursor()
            c.execute('SELECT id FROM embeddings WHERE image_hash = ?', (image_hash,))
            result = c.fetchone()
            if result:
                # Image already indexed
                logger.info(f"Image {image_hash[:8]} already indexed, skipping")
                conn.close()
                return
            
            # Generate embedding
            logger.info("Generating embedding for image...")
            image_embedding = self.model_manager.process_image(image)
            
            # Store in database
            store_embedding(conn, img_str, image_hash, image_embedding)
            
        except Exception as e:
            logger.error(f"Error processing and indexing image: {str(e)}")
            raise
        finally:
            # Clear memory after processing
            clear_cache()
    
    def query_images(self, query_text):
        """
        Query the image database with text
        """
        try:
            # Process query to get embedding
            query_embedding = self.model_manager.process_query(query_text)
            
            # Retrieve image embeddings from database
            logger.info("Retrieving image embeddings from database")
            conn = get_db_connection()
            c = conn.cursor()
            c.execute('SELECT image_base64, embedding FROM embeddings')
            rows = c.fetchall()
            conn.close()

            if not rows:
                logger.warning("No images found in the index")
                return None, None
                
            logger.info(f"Retrieved {len(rows)} image embeddings from database")

            # Set fixed sequence length
            fixed_seq_len = 620  # Adjust based on your embeddings
            logger.info(f"Using fixed sequence length of {fixed_seq_len}")

            image_embeddings_list = []
            image_base64_list = []

            for row in rows:
                image_base64, embedding_bytes = row
                embedding = pickle.loads(embedding_bytes)
                seq_len, embedding_dim = embedding.shape

                # Adjust to fixed sequence length
                if seq_len < fixed_seq_len:
                    padding = np.zeros((fixed_seq_len - seq_len, embedding_dim), dtype=embedding.dtype)
                    embedding_fixed = np.concatenate([embedding, padding], axis=0)
                elif seq_len > fixed_seq_len:
                    embedding_fixed = embedding[:fixed_seq_len, :]
                else:
                    embedding_fixed = embedding  # No adjustment needed

                image_embeddings_list.append(embedding_fixed)
                image_base64_list.append(image_base64)

            # Stack embeddings
            retrieved_image_embeddings = np.stack(image_embeddings_list)
            logger.info(f"Prepared {len(image_embeddings_list)} image embeddings for comparison")

            # Adjust query embedding
            seq_len_q, embedding_dim_q = query_embedding.shape

            if seq_len_q < fixed_seq_len:
                padding = np.zeros((fixed_seq_len - seq_len_q, embedding_dim_q), dtype=query_embedding.dtype)
                query_embedding_fixed = np.concatenate([query_embedding, padding], axis=0)
            elif seq_len_q > fixed_seq_len:
                query_embedding_fixed = query_embedding[:fixed_seq_len, :]
            else:
                query_embedding_fixed = query_embedding
                
            # Compute similarity scores
            scores = self.model_manager.compute_similarity(query_embedding_fixed, retrieved_image_embeddings)
            
            # Combine images and scores
            similarities = list(zip(image_base64_list, scores))

            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            logger.info(f"Found {len(similarities)} matches, sorted by similarity")

            if similarities:
                top_match = similarities[0]
                return top_match[0], top_match[1]  # Return base64 and score
            else:
                return None, None
                
        except Exception as e:
            logger.error(f"Error querying images: {str(e)}")
            raise
        finally:
            clear_cache()