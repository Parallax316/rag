# Image RAG System Architecture

## Overview

The Image RAG (Retrieval Augmented Generation) system is designed as a microservices architecture that separates concerns into distinct components. This architecture allows for better scalability, maintainability, and flexibility.

## Components

### Backend Service

The backend service is built with FastAPI and provides RESTful API endpoints for image indexing and querying. It is organized into the following components:

#### API Layer

The API layer handles HTTP requests and responses. It defines the endpoints for:
- Indexing images
- Indexing PDFs
- Querying the index

#### Core Layer

The core layer contains fundamental functionality:
- **Database**: Manages SQLite connections and operations
- **Memory**: Handles memory management and cache operations

#### Models Layer

The models layer is responsible for loading and managing the ML models:
- **ModelManager**: Singleton class that loads and manages the ColQwen2 model and processor

#### Services Layer

The services layer contains the business logic:
- **ImageService**: Handles image processing, embedding generation, and indexing
- **LLMService**: Manages interactions with the Ollama LLM for image understanding

### Demo Frontend

The demo frontend is a Streamlit application that provides a user-friendly interface for interacting with the backend API. It allows users to:
- Upload and index images and PDFs
- Query the index with natural language
- View results and AI-generated responses

## Data Flow

### Indexing Flow

1. User uploads image/PDF through the frontend
2. Frontend sends the file to the backend API
3. Backend processes the file:
   - For images: Generates embeddings directly
   - For PDFs: Converts to images and generates embeddings for each page
4. Embeddings are stored in the SQLite database with image data and hash

### Querying Flow

1. User enters a text query in the frontend
2. Frontend sends the query to the backend API
3. Backend processes the query:
   - Generates query embedding
   - Compares with stored image embeddings
   - Ranks results by similarity
4. Backend sends the most similar image and its score to the frontend
5. Backend generates an AI response using Ollama's vision model
6. Frontend displays the image and AI response to the user

## Technical Details

### Embedding Generation

The system uses ColQwen2 model from Colpali to generate embeddings for both images and text queries. These embeddings capture the semantic meaning of the content, allowing for similarity-based retrieval.

### Similarity Calculation

Similarity between query and image embeddings is calculated using the processor's score_multi_vector method, which efficiently computes similarity scores between the query embedding and multiple image embeddings.

### Database Schema

The system uses SQLite for storage with a simple schema:

```sql
CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_base64 TEXT,
    image_hash TEXT UNIQUE,
    embedding BLOB
)
```

### Memory Management

The system includes careful memory management to handle large models and embeddings:
- GPU memory is cleared after operations
- Garbage collection is triggered to free memory
- Tensor operations are moved to the appropriate device (CUDA, MPS, or CPU)

## Deployment

The system can be deployed in several ways:

### Local Development

Run the backend and frontend separately for development:

```bash
# Backend
cd backend
python -m app.main

# Frontend
cd demo
streamlit run app.py
```

### Docker Deployment

The backend includes a Dockerfile for containerized deployment:

```bash
cd backend
docker build -t image-rag-backend .
docker run -p 8000:8000 image-rag-backend
```

## Future Enhancements

- Add authentication and user management
- Implement vector database for improved scaling
- Add support for video indexing
- Implement batch processing for large datasets
- Add monitoring and logging infrastructure