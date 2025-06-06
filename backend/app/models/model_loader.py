import torch
import logging
from colpali_engine.models import ColQwen2, ColQwen2Processor
from ..config import MODEL_NAME, PROCESSOR_NAME, DEVICE_MAP
from ..core.memory import clear_cache

logger = logging.getLogger(__name__)

class ModelManager:
    _instance = None
    _model = None
    _processor = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._load_model()
        return cls._instance
    
    def _load_model(self):
        """
        Load the ColQwen2 model and processor
        """
        try:
            logger.info("Loading ColQwen2 model and processor...")
            self._model = ColQwen2.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.bfloat16,
                device_map=DEVICE_MAP,  # Use GPU if available
                low_cpu_mem_usage=True  # Optimize memory usage
            )
            
            self._processor = ColQwen2Processor.from_pretrained(PROCESSOR_NAME)
            logger.info("Model and processor loaded successfully")
            logger.info(f"Model is on device: {next(self._model.parameters()).device}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    @property
    def model(self):
        return self._model
    
    @property
    def processor(self):
        return self._processor
    
    def process_image(self, image):
        """
        Process an image and return its embedding
        """
        try:
            # Move processing to GPU if available
            batch_images = self._processor.process_images([image]).to(self._model.device)
            with torch.no_grad():
                clear_cache() if torch.cuda.is_available() else None  # Clear GPU cache before inference
                image_embeddings = self._model(**batch_images)
            return image_embeddings[0].cpu().to(torch.float32).numpy()
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise
        finally:
            clear_cache()
    
    def process_query(self, query):
        """
        Process a text query and return its embedding
        """
        try:
            with torch.no_grad():
                # Clear GPU cache before processing query
                clear_cache()
                logger.info("Processing query on device: " + str(self._model.device))
                batch_query = self._processor.process_queries([query]).to(self._model.device)
                query_embedding = self._model(**batch_query)
            return query_embedding.cpu().to(torch.float32).numpy()[0]
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
        finally:
            clear_cache()
    
    def compute_similarity(self, query_embedding, image_embeddings):
        """
        Compute similarity between query and image embeddings
        """
        try:
            # Convert to tensors and move to appropriate device
            logger.info(f"Moving tensors to device: {self._model.device}")
            query_embedding_tensor = torch.from_numpy(query_embedding).to(self._model.device).unsqueeze(0)
            image_embeddings_tensor = torch.from_numpy(image_embeddings).to(self._model.device)

            # Compute similarity scores
            logger.info("Computing similarity scores between query and images")
            with torch.no_grad():
                # Clear cache before computation
                clear_cache()
                scores = self._processor.score_multi_vector(query_embedding_tensor, image_embeddings_tensor)
            scores_np = scores.cpu().numpy().flatten()
            logger.info("Similarity scores computed successfully")
            return scores_np
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            raise
        finally:
            # Free up GPU memory
            clear_cache()