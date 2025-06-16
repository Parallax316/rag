import base64
import io
import hashlib
import numpy as np
import logging
import time
from PIL import Image
from pdf2image import convert_from_bytes
from ..models.model_loader import ModelManager
from ..core.memory import clear_cache
from ..core.mongodb import insert_embedding, find_embeddings
import pickle


logger = logging.getLogger(__name__)

class ImageService:
    def __init__(self):
        self.model_manager = ModelManager()
    
    def process_image_file(self, image_data, collection_name="default"):
        """
        Process a single image file
        """
        try:
            logger.info("Starting process_image_file")
            print(f"[IMAGE_SERVICE] Starting process_image_file")
            # Compute image hash
            image_hash = hashlib.sha256(image_data).hexdigest()
            logger.info(f"Image hash: {image_hash}")
            print(f"[IMAGE_SERVICE] Image hash: {image_hash}")
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            logger.info("Image loaded and converted to RGB")
            print(f"[IMAGE_SERVICE] Image loaded and converted to RGB")
            # Encode image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            logger.info("Image encoded to base64")
            print(f"[IMAGE_SERVICE] Image encoded to base64")
            # Process and index the image
            self.process_and_index_image(image, img_str, image_hash, collection_name)
            logger.info("Image processed and indexed")
            print(f"[IMAGE_SERVICE] Image processed and indexed")
            return image_hash
        except Exception as e:
            logger.error(f"Error processing image file: {str(e)}", exc_info=True)
            print(f"[IMAGE_SERVICE] Error processing image file: {str(e)}")
            raise

    def process_pdf_file(self, pdf_data, collection_name="default"):
        """
        Process a PDF file by converting it to images
        """
        try:
            logger.info("Converting PDF to images")
            print(f"[IMAGE_SERVICE] Converting PDF to images")
            images = convert_from_bytes(pdf_data)
            logger.info(f"PDF converted to {len(images)} images")
            print(f"[IMAGE_SERVICE] PDF converted to {len(images)} images")
            image_hashes = []
            for j, image in enumerate(images):
                logger.info(f"Processing PDF page {j+1}/{len(images)}")
                print(f"[IMAGE_SERVICE] Processing PDF page {j+1}/{len(images)}")
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                byte_data = buffer.getvalue()
                img_str = base64.b64encode(byte_data).decode('utf-8')
                # Compute image hash
                image_hash = hashlib.sha256(byte_data).hexdigest()
                logger.info(f"Page {j+1} hash: {image_hash}")
                print(f"[IMAGE_SERVICE] Page {j+1} hash: {image_hash}")
                # Process and index the image
                self.process_and_index_image(image, img_str, image_hash, collection_name)
                logger.info(f"Page {j+1} processed and indexed")
                print(f"[IMAGE_SERVICE] Page {j+1} processed and indexed")
                image_hashes.append(image_hash)
            logger.info("All PDF pages processed and indexed")
            print(f"[IMAGE_SERVICE] All PDF pages processed and indexed")
            return image_hashes
        except Exception as e:
            logger.error(f"Error processing PDF file: {str(e)}", exc_info=True)
            print(f"[IMAGE_SERVICE] Error processing PDF file: {str(e)}")
            raise

    def process_and_index_image(self, image, img_str, image_hash, collection_name="default"):
        """
        Process an image and store its embedding in MongoDB Atlas
        """
        try:
            logger.info(f"Starting embedding generation for hash: {image_hash[:8]}...")
            print(f"[IMAGE_SERVICE] Starting embedding generation for hash: {image_hash[:8]}...")
            
            # Generate embedding
            logger.info("Step 1: Generating embedding with ColQwen2...")
            print(f"[IMAGE_SERVICE] Step 1: Generating embedding with ColQwen2...")
            start_time = time.time()
            
            image_embedding = self.model_manager.process_image(image)
            
            embedding_time = time.time() - start_time
            logger.info(f"Step 1 Complete: Embedding generated in {embedding_time:.2f} seconds")
            print(f"[IMAGE_SERVICE] Step 1 Complete: Embedding generated in {embedding_time:.2f} seconds")
            
            # Prepare document for MongoDB
            logger.info("Step 2: Preparing MongoDB document...")
            print(f"[IMAGE_SERVICE] Step 2: Preparing MongoDB document...")
            doc = {
                "collection_name": collection_name,
                "type": "image",
                "embedding": image_embedding.tolist(),
                "data": {
                    "image_base64": img_str,
                    "image_hash": image_hash
                },
                "metadata": {}
            }
            logger.info("Step 2 Complete: Document prepared")
            print(f"[IMAGE_SERVICE] Step 2 Complete: Document prepared")
            
            logger.info("Step 3: Inserting document into MongoDB...")
            print(f"[IMAGE_SERVICE] Step 3: Inserting document into MongoDB...")
            start_insert = time.time()
            
            insert_embedding(doc)
            
            insert_time = time.time() - start_insert
            logger.info(f"Step 3 Complete: Document inserted in {insert_time:.2f} seconds")
            print(f"[IMAGE_SERVICE] Step 3 Complete: Document inserted in {insert_time:.2f} seconds")
            
            total_time = time.time() - start_time
            logger.info(f"Image processing completed in {total_time:.2f} seconds")
            print(f"[IMAGE_SERVICE] Image processing completed in {total_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error processing and indexing image: {str(e)}", exc_info=True)
            print(f"[IMAGE_SERVICE] Error processing and indexing image: {str(e)}")
            raise
        finally:
            clear_cache()

    def query_images(self, query_text, collection_name="default"):
        """
        Query the image database with text (now using MongoDB Atlas)
        """
        try:
            # Process query to get embedding
            logger.info("Processing query embedding...")
            print(f"[IMAGE_SERVICE] Processing query embedding...")
            query_embedding = self.model_manager.process_query(query_text)
            
            # Retrieve image embeddings from MongoDB
            logger.info(f"Retrieving image embeddings from MongoDB Atlas for collection: {collection_name}")
            print(f"[IMAGE_SERVICE] Retrieving image embeddings from MongoDB Atlas for collection: {collection_name}")
            
            query_filter = {"type": "image", "collection_name": collection_name}
            results = find_embeddings(query_filter)
            
            if not results:
                logger.warning(f"No images found in collection: {collection_name}")
                print(f"[IMAGE_SERVICE] No images found in collection: {collection_name}")
                return []
            
            # Use proper ColQwen2 similarity computation
            logger.info(f"Computing similarity scores for {len(results)} images...")
            print(f"[IMAGE_SERVICE] Computing similarity scores for {len(results)} images...")
            
            # Prepare image embeddings array
            image_embeddings = []
            for doc in results:
                emb = np.array(doc["embedding"])
                image_embeddings.append(emb)
            
            # Stack image embeddings
            image_embeddings_array = np.stack(image_embeddings)
            
            # Use the model's compute_similarity method
            scores = self.model_manager.compute_similarity(query_embedding, image_embeddings_array)
            
            if len(scores) > 0:
                # Get top 3 results
                top_k = min(3, len(scores))
                top_indices = np.argsort(scores)[::-1][:top_k]
                top_images = []
                for idx in top_indices:
                    top_images.append({
                        "image_base64": results[idx]["data"]["image_base64"],
                        "score": float(scores[idx])
                    })
                logger.info(f"Top {top_k} scores: {[img['score'] for img in top_images]}")
                print(f"[IMAGE_SERVICE] Top {top_k} scores: {[img['score'] for img in top_images]}")
                return top_images
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error querying images: {str(e)}")
            print(f"[IMAGE_SERVICE] Error querying images: {str(e)}")
            raise
        finally:
            clear_cache()
