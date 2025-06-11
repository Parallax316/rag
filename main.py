import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import os
import requests
import base64
import hashlib
from backend.app.services.image_service import ImageService
from doc_theme_bot.backend.app.services.vstore_svc import VectorStoreService
from backend.app.services.llm_service import LLMService
from doc_theme_bot.backend.app.services.doc_parser_fast import DocParserFastService

app = FastAPI()

logger = logging.getLogger(__name__)

# Initialize services
logger.info("Loading VLM (ColQwen2) model for image embeddings...")
image_service = ImageService()
logger.info("VLM (ColQwen2) model loaded.")

logger.info("Loading Ollama LLM service...")
llm_service = LLMService()
logger.info("Ollama LLM service ready.")

logger.info("Loading text embedding model (InstructorXL) for text RAG...")
text_service = VectorStoreService()
logger.info("Text embedding model loaded.")
logger.info("Loading DocParserFastService for document chunking and embedding...")
doc_parser_service = DocParserFastService(vector_store_service=text_service)
logger.info("DocParserFastService loaded.")

@app.get("/health")
def health():
    return {"status": "ok"}

# --- Collection Management ---
@app.post("/collections")
def create_collection(request: dict):
    name = request.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="Collection name required")
    text_service.create_collection(name)
    return {"status": "success", "message": f"Collection '{name}' created"}

@app.get("/collections")
def list_collections():
    return {"collections": text_service.list_collections()}

@app.delete("/collections/{name}")
def delete_collection(name: str):
    text_service.delete_collection(name)
    return {"status": "success", "message": f"Collection '{name}' deleted"}

# --- Upload Endpoints ---
@app.post("/upload/image")
def upload_image(file: UploadFile = File(...), collection: str = Form("default")):
    logger.info("Processing image upload and VLM embedding...")
    image_data = file.file.read()
    image_service.process_and_index_image(
        ImageService().model_manager.processor.process_images([ImageService().model_manager.processor.load_image(image_data)])[0],
        base64.b64encode(image_data).decode(),
        hashlib.sha256(image_data).hexdigest()
    )
    logger.info("Image uploaded and indexed.")
    return {"status": "success", "message": "Image uploaded and indexed"}

@app.post("/upload/text")
def upload_text(file: UploadFile = File(...), collection: str = Form("default")):
    logger.info(f"Received text file for collection '{collection}'. Using DocParserFastService for chunking and embedding...")
    # Save uploaded file to a temp path
    temp_path = f"temp_upload_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(file.file.read())
    # Use filename as source_doc_id for now
    source_doc_id = file.filename
    try:
        success = doc_parser_service.process_document(temp_path, source_doc_id, collection_name=collection)
        logger.info("Text document processed and indexed with DocParserFastService.")
        return {"status": "success", "message": "Text uploaded and indexed"}
    except Exception as e:
        logger.error(f"Error processing text document: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/upload/pdf")
def upload_pdf(file: UploadFile = File(...), collection: str = Form("default")):
    logger.info(f"Received PDF for collection '{collection}'. Using DocParserFastService for chunking and embedding...")
    temp_path = f"temp_upload_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(file.file.read())
    source_doc_id = file.filename
    try:
        success = doc_parser_service.process_document(temp_path, source_doc_id, collection_name=collection)
        logger.info("PDF processed and indexed with DocParserFastService.")
        return {"status": "success", "message": "PDF uploaded and indexed"}
    except Exception as e:
        logger.error(f"Error processing PDF document: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# --- Unified Upload & Process Endpoint ---
@app.post("/upload/document")
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    collection: str = Form("default")
):
    """
    Upload a document (PDF or image) and trigger both text and VLM processing in parallel.
    """
    logger.info(f"Received file '{file.filename}' for collection '{collection}'. Starting parallel processing...")
    temp_path = f"temp_upload_{file.filename}"
    with open(temp_path, "wb") as f_out:
        f_out.write(file.file.read())
    source_doc_id = file.filename

    def process_text():
        try:
            logger.info(f"[TextRAG] Processing '{temp_path}' with DocParserFastService...")
            doc_parser_service.process_document(temp_path, source_doc_id, collection_name=collection)
            logger.info(f"[TextRAG] Processing complete for '{temp_path}'.")
        except Exception as e:
            logger.error(f"[TextRAG] Error: {e}")

    def process_vlm():
        try:
            logger.info(f"[VLM] Processing '{temp_path}' with ImageService...")
            with open(temp_path, "rb") as f_img:
                image_data = f_img.read()
            image_service.process_image_file(image_data)
            logger.info(f"[VLM] Processing complete for '{temp_path}'.")
        except Exception as e:
            logger.error(f"[VLM] Error: {e}")

    # Start both processing tasks in the background
    background_tasks.add_task(process_text)
    background_tasks.add_task(process_vlm)

    logger.info(f"Background processing started for '{file.filename}' in collection '{collection}'.")
    return {"status": "processing_started", "message": f"File '{file.filename}' is being processed by both systems.", "collection": collection}

# --- Query Endpoint ---
@app.post("/query")
def query(query: dict):
    user_query = query.get("query")
    collection = query.get("collection", "default")
    logger.info("Running text RAG retrieval...")
    text_results = text_service.query_documents_with_scores(user_query, collection_name=collection)
    logger.info("Running image RAG retrieval (VLM)...")
    image_base64, image_score = image_service.query_images(user_query)
    logger.info("Running LLM (Ollama) for answer synthesis...")
    llm_response = None
    if image_base64:
        llm_response = llm_service.generate_response(user_query, image_base64)
    logger.info("Query complete.")
    return {
        "text": text_results,
        "image": {"image_base64": image_base64, "score": image_score},
        "llm_response": llm_response
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
