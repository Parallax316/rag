import streamlit as st
import torch
from PIL import Image
import sqlite3
import numpy as np
import pickle
import base64
import io
from colpali_engine.models import ColQwen2, ColQwen2Processor
import gc
from pdf2image import convert_from_bytes
from io import BytesIO
import hashlib  # Import hashlib for hashing
import logging  # Import logging module

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
device_map = get_device()
logger.info(f"Using device: {device_map}")

# Function to load the model and processor
@st.cache_resource
def load_model():
    logger.info("Loading ColQwen2 model and processor...")
    model = ColQwen2.from_pretrained(
        "vidore/colqwen2-v0.1",
        torch_dtype=torch.bfloat16,
        device_map=device_map,  # Use GPU if available
        low_cpu_mem_usage=True  # Optimize memory usage
    )
    
    processor = ColQwen2Processor.from_pretrained("vidore/colqwen2.5-v0.2")
    logger.info("Model and processor loaded successfully")
    logger.info(f"Model is on device: {next(model.parameters()).device}")
    
    return model, processor

# Function to get a database connection
def get_db_connection():
    conn = sqlite3.connect('image_embeddings.db')
    return conn

def process_and_index_image(image, img_str, image_hash, processor, model):
    logger.info(f"Processing image with hash: {image_hash[:8]}...")
    # Store in database
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_base64 TEXT,
            image_hash TEXT UNIQUE,
            embedding BLOB
        )
    ''')
    # Check if the image hash already exists
    c.execute('SELECT id FROM embeddings WHERE image_hash = ?', (image_hash,))
    result = c.fetchone()
    if result:
        # Image already indexed
        logger.info(f"Image {image_hash[:8]} already indexed, skipping")
        conn.close()
        return
    
    logger.info("Generating embedding for image...")
    # Process image to get embedding
    try:
        # Move processing to GPU if available
        batch_images = processor.process_images([image]).to(model.device)
        with torch.no_grad():
            torch.cuda.empty_cache() if torch.cuda.is_available() else None  # Clear GPU cache before inference
            image_embeddings = model(**batch_images)
        image_embedding = image_embeddings[0].cpu().to(torch.float32).numpy()
        logger.info("Embedding generated successfully")
        
        # Serialize the embedding
        embedding_bytes = pickle.dumps(image_embedding)
        c.execute('INSERT INTO embeddings (image_base64, image_hash, embedding) VALUES (?, ?, ?)', (img_str, image_hash, embedding_bytes))
        conn.commit()
        logger.info(f"Image {image_hash[:8]} indexed and stored in database")
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        # Rollback in case of error
        conn.rollback()
    finally:
        conn.close()
        # Clear memory after processing
        clear_cache()

def clear_cache():
    """Clear GPU memory cache for different platforms."""
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

def main():
    st.title("📷 Image RAG(Colpali + Llama Vision)")
    logger.info("Starting Image RAG application")

    model, processor = load_model()

    # Initialize session state for image hashes
    if 'image_hashes' not in st.session_state:
        st.session_state.image_hashes = set()

    # Use st.radio for tab selection
    tab = st.radio("Navigation", ["➕ Add to Index", "🔍 Query Index"])

    if tab == "➕ Add to Index":
        st.header("Add Images to Index")
        logger.info("User navigated to 'Add to Index' tab")
        # File uploader
        uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg', 'pdf'])
        if uploaded_files:
            logger.info(f"Processing {len(uploaded_files)} uploaded files")
            # Process the uploaded images
            for i, uploaded_file in enumerate(uploaded_files):
                logger.info(f"Processing file {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                
                if uploaded_file.type == 'application/pdf':
                    logger.info(f"Converting PDF to images: {uploaded_file.name}")
                    images = convert_from_bytes(uploaded_file.read())
                    logger.info(f"PDF converted to {len(images)} images")
                    
                    for j, image in enumerate(images):
                        logger.info(f"Processing PDF page {j+1}/{len(images)}")
                        buffer = BytesIO()
                        image.save(buffer, format="PNG")
                        byte_data = buffer.getvalue()
                        img_str = base64.b64encode(byte_data).decode('utf-8')
                        # Compute image hash
                        image_hash = hashlib.sha256(byte_data).hexdigest()
                        if image_hash in st.session_state.image_hashes:
                            logger.info(f"Page {j+1} already processed in this session, skipping")
                            continue  # Skip if already processed in this session
                        with st.spinner(f'Processing and embedding PDF page {j+1}/{len(images)}...'):
                            process_and_index_image(image, img_str, image_hash, processor, model)
                        st.session_state.image_hashes.add(image_hash)
                else:
                    logger.info(f"Processing image file: {uploaded_file.name}")
                    # Read image data
                    image_data = uploaded_file.read()
                    # Compute image hash
                    image_hash = hashlib.sha256(image_data).hexdigest()
                    if image_hash in st.session_state.image_hashes:
                        logger.info(f"Image already processed in this session, skipping")
                        continue  # Skip if already processed in this session
                    # Convert to PIL Image
                    image = Image.open(io.BytesIO(image_data)).convert('RGB')
                    # Encode image to base64
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    with st.spinner('Processing and embedding image...'):
                        process_and_index_image(image, img_str, image_hash, processor, model)
                    st.session_state.image_hashes.add(image_hash)
            logger.info("All files processed successfully")
            st.success("Images added to index.")

    elif tab == "🔍 Query Index":
        st.header("Query Index")
        logger.info("User navigated to 'Query Index' tab")
        query = st.text_input("Enter your query")
        if query:
            logger.info(f"Processing query: '{query}'")
            # Process query
            with torch.no_grad():
                # Clear GPU cache before processing query
                clear_cache()
                logger.info("Processing query on device: " + str(model.device))
                batch_query = processor.process_queries([query]).to(model.device)
                query_embedding = model(**batch_query)
            query_embedding_cpu = query_embedding.cpu().to(torch.float32).numpy()[0]
            logger.info("Query embedding generated successfully")
            # Free up GPU memory
            del batch_query, query_embedding
            clear_cache()

            # Retrieve image embeddings from database
            logger.info("Retrieving image embeddings from database")
            conn = get_db_connection()
            c = conn.cursor()
            c.execute('SELECT image_base64, embedding FROM embeddings')
            rows = c.fetchall()
            conn.close()

            if not rows:
                logger.warning("No images found in the index")
                st.warning("No images found in the index. Please add images first.")
                return
                
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
            seq_len_q, embedding_dim_q = query_embedding_cpu.shape

            if seq_len_q < fixed_seq_len:
                padding = np.zeros((fixed_seq_len - seq_len_q, embedding_dim_q), dtype=query_embedding_cpu.dtype)
                query_embedding_fixed = np.concatenate([query_embedding_cpu, padding], axis=0)
            elif seq_len_q > fixed_seq_len:
                query_embedding_fixed = query_embedding_cpu[:fixed_seq_len, :]
            else:
                query_embedding_fixed = query_embedding_cpu

            # Convert to tensors and move to appropriate device
            logger.info(f"Moving tensors to device: {model.device}")
            query_embedding_tensor = torch.from_numpy(query_embedding_fixed).to(model.device).unsqueeze(0)
            retrieved_image_embeddings_tensor = torch.from_numpy(retrieved_image_embeddings).to(model.device)

            # Compute similarity scores
            logger.info("Computing similarity scores between query and images")
            with torch.no_grad():
                # Clear cache before computation
                clear_cache()
                scores = processor.score_multi_vector(query_embedding_tensor, retrieved_image_embeddings_tensor)
            scores_np = scores.cpu().numpy().flatten()
            logger.info("Similarity scores computed successfully")
            
            # Free up GPU memory
            del query_embedding_tensor, retrieved_image_embeddings_tensor, scores
            clear_cache()

            # Combine images and scores
            similarities = list(zip(image_base64_list, scores_np))

            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            logger.info(f"Found {len(similarities)} matches, sorted by similarity")

            if similarities:
                st.write("Most similar image:")
                img_str, score = similarities[0]
                st.write(f"Similarity Score: {score:.4f}")
                logger.info(f"Top match has similarity score: {score:.4f}")
                # Decode image from base64
                img_data = base64.b64decode(img_str)
                image = Image.open(io.BytesIO(img_data))
                st.image(image)
            else:
                st.write("No similar images found.")
                logger.warning("No similar images found after processing")

            st.write("AI Response:")

            import ollama
            logger.info("Preparing to generate AI response using Ollama")

            response_container = st.empty()
            
            # Ensure GPU memory is cleared before calling Ollama
            clear_cache()
            logger.info("Sending query to Qwen2.5vl model")
            
            try:
                stream = ollama.chat(
                    model="qwen2.5vl:3b",
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

                collected_chunks = []
                stream_iter = iter(stream)

                with st.spinner('⏳ Generating Response...'):
                    try:
                        # Get the first chunk
                        first_chunk = next(stream_iter)
                        chunk_content = first_chunk['message']['content']
                        collected_chunks.append(chunk_content)
                        # Display the initial response
                        complete_response = ''.join(collected_chunks)
                        response_container.markdown(complete_response)
                        logger.info("Received first response chunk from Ollama")
                    except StopIteration:
                        # Handle if no chunks are received
                        logger.warning("No response received from Ollama")
                        pass

                # Continue streaming the rest of the response
                logger.info("Streaming remaining response chunks")
                for chunk in stream_iter:
                    chunk_content = chunk['message']['content']
                    collected_chunks.append(chunk_content)
                    complete_response = ''.join(collected_chunks)
                    response_container.markdown(complete_response)

                logger.info("Response generation completed")
            except Exception as e:
                logger.error(f"Error generating response from Ollama: {str(e)}")
                st.error(f"Error generating response: {str(e)}")
            finally:
                # Ensure memory is cleared after response generation
                clear_cache()
                gc.collect()
                logger.info("Memory cleared after response generation")

            clear_cache()
            gc.collect()

if __name__ == "__main__":
    main()
