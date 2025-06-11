# System Summary: Image RAG (Colpali + LLaMA Vision)

## Overview
This project is a Retrieval-Augmented Generation (RAG) system for images and PDFs, combining ColQwen2 (via Colpali) for image embeddings and LLaMA Vision (via Ollama) for vision-based LLM responses. It is structured as a microservices architecture with a FastAPI backend and a Streamlit frontend demo.

---

## Main Folders & Their Roles
- **backend/**: FastAPI backend service, core logic, model management, API endpoints, and database.
- **demo/**: Streamlit UI for uploading/querying images and PDFs, interacts with backend via HTTP.
- **docs/**: Documentation (architecture, usage, etc).
- **tests/**: Unit tests for backend services.

---

## Key Backend Files (backend/app/)
- **main.py**: FastAPI app entry point. Sets up CORS, includes API routes, and provides health checks.
- **config.py**: Central config (model names, device, DB path, API host/port, LLM model).
- **api/routes.py**: Defines API endpoints:
  - `/index/image`: Index a single image
  - `/index/pdf`: Index a PDF (converts to images, indexes each page)
  - `/query`: Query the index with text, returns most similar image and LLM response
- **core/database.py**: SQLite DB connection, schema creation, embedding storage/retrieval.
- **core/memory.py**: Memory management (clears GPU/CPU cache, device selection).
- **models/model_loader.py**: Singleton for loading ColQwen2 model/processor, provides methods for image/query embedding and similarity computation.
- **services/image_service.py**: Handles image/PDF processing, embedding generation, duplicate detection, and DB storage. Also handles querying by text.
- **services/llm_service.py**: Calls Ollama LLM with image and query, returns vision-based response.

---

## Demo Frontend (demo/app.py)
- Streamlit app with two tabs:
  - **Add to Index**: Upload images or PDFs, sends to backend for indexing.
  - **Query Index**: Enter text query, gets most similar image and LLM response from backend.
- Uses requests to communicate with backend API endpoints.

---

## Data Flow
- **Indexing**: User uploads image/PDF → frontend sends to backend → backend generates embeddings, checks for duplicates, stores in SQLite.
- **Querying**: User enters text query → frontend sends to backend → backend computes query embedding, finds most similar image, gets LLM response, returns to frontend.

---

## Database Schema
- Table: `embeddings`
  - `id` (PK)
  - `image_base64` (image data)
  - `image_hash` (unique, for deduplication)
  - `embedding` (pickled numpy array)

---

## Other Notes
- **requirements.txt** files in root, backend, and demo specify dependencies.
- **Dockerfile** in backend for containerized deployment.
- **README.md** and **docs/architecture.md** provide detailed usage, setup, and architecture info.

---

## File Reference Table
| File/Folder | Purpose |
|-------------|---------|
| backend/app/main.py | FastAPI app entry |
| backend/app/api/routes.py | API endpoints |
| backend/app/services/image_service.py | Image/PDF processing, embedding, query |
| backend/app/services/llm_service.py | LLM (Ollama) response generation |
| backend/app/models/model_loader.py | Model/processor loading, embedding, similarity |
| backend/app/core/database.py | DB connection, schema, storage |
| backend/app/core/memory.py | Memory/device management |
| backend/app/config.py | Config (models, DB, API, device) |
| demo/app.py | Streamlit UI |
| tests/ | Unit tests |
| docs/ | Documentation |

---

This summary can be used as a quick reference to locate and understand the role of each main file in the system.

# System Summary: Text RAG (Doc Theme Bot)

## Overview
Doc Theme Bot is a full-stack, document-centric Retrieval-Augmented Generation (RAG) system for deep document research, theme extraction, and citation-based answers. It supports uploading, processing, and querying large sets of documents (PDFs, images) with advanced theme analysis and granular citation tracking. The backend is built with FastAPI, the frontend with Streamlit, and ChromaDB is used for vector storage.

---

## Main Folders & Their Roles
- **doc_theme_bot/backend/**: FastAPI backend, core logic, document parsing, vector store, RAG, theme/citation analysis, API endpoints.
- **doc_theme_bot/frontend/**: Streamlit UI for document upload, collection management, chat-based querying, and results display.
- **arxiv_pdfs/**, **cs.ai_pdfs/**: Example document storage for demo/testing.
- **data/**: Uploads, processed docs, and ChromaDB vector storage.

---

## Key Backend Files (doc_theme_bot/backend/app/)
- **main.py**: FastAPI app entry, sets up CORS, includes API routers for documents, chat, and collections.
- **core/config.py**: Central config (API keys, model names, DB paths, collection names, etc).
- **models/schemas.py**: Pydantic schemas for document processing, chat, RAG responses, themes, citations, etc.
- **services/doc_parser_fast.py**: Fast, rule-based document parser (PDF/image text extraction, chunking, embedding, vector store add).
- **services/vstore_svc.py**: Singleton for ChromaDB vector store, manages embedding storage/retrieval.
- **services/rag_svc.py**: Main RAG logic: retrieval, reranking, LLM calls, theme/citation synthesis.
- **services/theme_analyzer.py**: Theme extraction using clustering and LLM summarization.
- **services/cluster_analysis_svc.py**: Clusters document embeddings, generates summaries for each cluster.
- **services/citation_manager.py**: Verifies, deduplicates, and formats citations for RAG outputs.
- **api/docs_api.py**: Endpoints for document upload/processing.
- **api/chat_api.py**: Endpoints for chat-based querying (RAG).
- **api/collection_api.py**: Endpoints for collection creation, listing, and deletion.
- **api/v1/documents.py**: Upload multiple documents (VLM and fast parser), background processing.

---

## Demo Frontend (doc_theme_bot/frontend/ui.py)
- Streamlit app with sidebar for collection management (create, select, delete), document upload, and logs.
- Main chat interface for querying documents, viewing answers, themes, citations, and LLM thought process.
- Uses requests to communicate with backend API endpoints.

---

## Data Flow
- **Indexing**: User uploads documents → frontend sends to backend → backend parses, chunks, embeds, and stores in ChromaDB.
- **Querying**: User enters query → frontend sends to backend → backend retrieves relevant chunks, reranks, clusters, extracts themes, generates answer and citations, returns to frontend.

---

## Database/Vector Store
- Uses ChromaDB for vector storage of document chunks and metadata.
- Each collection is a separate semantic space in ChromaDB.

---

## API Endpoints (Key)
- `POST /api/v1/documents/upload-multiple` – Upload multiple documents (fast parser)
- `POST /api/v1/documents/upload-multiple-vlm` – Upload multiple documents (VLM parser)
- `POST /api/v1/chat/query` – Query documents and get answer, themes, citations
- `GET /api/v1/collections` – List all collections
- `POST /api/v1/collections` – Create a new collection
- `DELETE /api/v1/collections/{name}` – Delete a collection

---

## Other Notes
- **requirements.txt**: Lists all dependencies (FastAPI, Streamlit, LangChain, ChromaDB, OCR, etc).
- **scrap.py**: Utility to download PDFs from arXiv for demo/testing.
- **README.md** and **internship_rag_architecture.md**: Detailed usage, setup, and architecture info.
- **app.py**: Streamlit entry point, imports and runs frontend UI.

---

## File Reference Table
| File/Folder | Purpose |
|-------------|---------|
| doc_theme_bot/backend/app/main.py | FastAPI app entry |
| doc_theme_bot/backend/app/core/config.py | Config (API keys, models, DB, etc) |
| doc_theme_bot/backend/app/models/schemas.py | Pydantic schemas |
| doc_theme_bot/backend/app/services/doc_parser_fast.py | Fast document parser |
| doc_theme_bot/backend/app/services/vstore_svc.py | ChromaDB vector store |
| doc_theme_bot/backend/app/services/rag_svc.py | RAG logic (retrieval, rerank, LLM) |
| doc_theme_bot/backend/app/services/theme_analyzer.py | Theme extraction |
| doc_theme_bot/backend/app/services/cluster_analysis_svc.py | Clustering, summaries |
| doc_theme_bot/backend/app/services/citation_manager.py | Citation management |
| doc_theme_bot/backend/app/api/docs_api.py | Document upload API |
| doc_theme_bot/backend/app/api/chat_api.py | Chat/query API |
| doc_theme_bot/backend/app/api/collection_api.py | Collection management API |
| doc_theme_bot/backend/app/api/v1/documents.py | Multi-upload API |
| doc_theme_bot/frontend/ui.py | Streamlit UI |
| doc_theme_bot/app.py | Streamlit entry point |
| doc_theme_bot/scrap.py | PDF download utility |
| arxiv_pdfs/, cs.ai_pdfs/ | Example document storage |
| data/ | Uploads, processed docs, ChromaDB |
| README.md, internship_rag_architecture.md | Documentation |

---

This summary can be used as a quick reference to locate and understand the role of each main file in the Text RAG (Doc Theme Bot) system.
