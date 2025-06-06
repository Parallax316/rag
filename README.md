# 🖼️ Image RAG (Colpali + LLaMA Vision)

A powerful Retrieval-Augmented Generation (RAG) system combining Colpali's ColQwen image embeddings with LLaMA Vision via Ollama.

## 🌟 Key Features

- 🧬 ColQwen model for generating powerful image embeddings via Colpali
- 🤖 LLaMA Vision integration through Ollama for image understanding
- 📥 Intelligent image indexing with duplicate detection
- 💬 Natural language image queries
- 📄 PDF document support
- 🔍 Semantic similarity search
- 📊 Efficient SQLite storage
- 🔄 Microservices architecture for scalability

## 🏗️ Project Structure

The project has been restructured into a microservices architecture:

```
chatbot_theme_identifier/ 
├── backend/ 
│ ├── app/ 
│ │ ├── api/ - API endpoints
│ │ ├── core/ - Core functionality (database, memory management)
│ │ ├── models/ - Model loading and processing
│ │ ├── services/ - Business logic services
│ │ ├── main.py - FastAPI application entry point
│ │ └── config.py - Configuration settings
│ ├── data/ - Database and data storage
│ ├── Dockerfile - Docker configuration for backend
│ └── requirements.txt - Backend dependencies
├── docs/ - Documentation
├── tests/ - Test cases
├── demo/ - Streamlit demo application
└── README.md - Project documentation
```

## 🛠️ Technical Stack

- **Embedding Model**: ColQwen via Colpali
- **Vision Model**: LLaMA Vision via Ollama
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Database**: SQLite
- **Image Processing**: Pillow, pdf2image
- **ML Framework**: PyTorch


## ⚡ Quick Start

1. Install Poppler (required for PDF support):

   **Mac:**
   ```bash
   brew install poppler
   ```

   **Windows:**
   1. Download the latest poppler package from: https://github.com/oschwartz10612/poppler-windows/releases/
   2. Extract the downloaded zip to a location (e.g., `C:\Program Files\poppler`)
   3. Add bin directory to PATH:
      - Open System Properties > Advanced > Environment Variables
      - Under System Variables, find and select "Path"
      - Click "Edit" > "New"
      - Add the bin path (e.g., `C:\Program Files\poppler\bin`)
   4. Verify installation:
      ```bash
      pdftoppm -h
      ```

2. Clone and setup environment:
   ```bash
   git clone https://github.com/kturung/colpali-llama-vision-rag.git
   python -m venv venv
   source venv/bin/activate  # For Mac/Linux
   # or
   .\venv\Scripts\activate  # For Windows
   pip install -r requirements.txt
   ```

3. Install Ollama from https://ollama.com

4. Running the microservices:

   **Backend API:**
   ```bash
   cd backend
   pip install -r requirements.txt
   python -m app.main
   ```

   **Demo Frontend:**
   ```bash
   cd demo
   pip install -r requirements.txt
   streamlit run app.py
   ```

> Note: Restart your terminal/IDE after modifying PATH variables


## 💡 Usage

### 📤 Adding Images
1. Navigate to "➕ Add to Index"
2. Upload images/PDFs
3. System automatically:
   - Generates ColQwen embeddings
   - Checks for duplicates
   - Stores in SQLite

### 🔎 Querying
1. Go to "🔍 Query Index"
2. Enter natural language query
3. View similar images
4. Get LLaMA Vision analysis


## 💾 Database Schema

```sql
CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_base64 TEXT,
    image_hash TEXT UNIQUE,
    embedding BLOB
)
```

## 🔌 API Endpoints

### Backend API

- `POST /index/image` - Index a single image
  - Input: Image file (multipart/form-data)
  - Output: JSON with status, message, and image hash

- `POST /index/pdf` - Index a PDF document
  - Input: PDF file (multipart/form-data)
  - Output: JSON with status, message, and list of image hashes

- `POST /query` - Query the index with text
  - Input: query (form field)
  - Output: JSON with status, image (base64), similarity score, and LLM response

- `GET /` - Root endpoint (health check)
  - Output: JSON with status message

- `GET /health` - Health check endpoint
  - Output: JSON with status message

## 🔄 Microservices Communication

The system is designed with a clear separation of concerns:

1. **Frontend-Backend Communication**:
   - The Streamlit frontend communicates with the FastAPI backend via HTTP requests
   - API endpoints handle file uploads, queries, and return JSON responses

2. **Internal Service Communication**:
   - `ImageService`: Handles image processing, embedding generation, and database operations
   - `LLMService`: Communicates with Ollama to generate responses based on images and queries
   - `ModelManager`: Manages the ColQwen model for embedding generation
   - Database layer: Provides persistence for image embeddings

## 🚀 Running the Services

### Backend Service

```bash
# Navigate to backend directory
cd backend

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
python -m app.main
```

The backend service will start on http://localhost:8000 by default.

### Frontend Demo

```bash
# Navigate to demo directory
cd demo

# Install dependencies
pip install -r requirements.txt

# Start the Streamlit app
streamlit run app.py
```

The Streamlit demo will start on http://localhost:8501 by default.

### Using Docker

```bash
# Build and run the backend service
cd backend
docker build -t image-rag-backend .
docker run -p 8000:8000 image-rag-backend
```
