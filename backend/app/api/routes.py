from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import logging
from ..services.image_service import ImageService
from ..services.llm_service import LLMService
from ..utils.health_check import run_full_health_check

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
image_service = ImageService()
llm_service = LLMService()

@router.post("/index/image")
async def index_image(file: UploadFile = File(...), collection: str = Form("default")):
    """
    Index a single image file
    """
    try:
        # Read file content
        image_data = await file.read()
        
        # Process the image
        image_hash = image_service.process_image_file(image_data, collection)
        
        return JSONResponse(
            status_code=200,
            content={"status": "success", "message": "Image indexed successfully", "image_hash": image_hash}
        )
    except Exception as e:
        logger.error(f"Error indexing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/index/pdf")
async def index_pdf(file: UploadFile = File(...), collection: str = Form("default")):
    """
    Index a PDF file by converting it to images
    """
    try:
        # Check if file is PDF
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Read file content
        pdf_data = await file.read()
        
        # Process the PDF
        image_hashes = image_service.process_pdf_file(pdf_data, collection)
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success", 
                "message": f"PDF processed into {len(image_hashes)} images and indexed successfully",
                "image_hashes": image_hashes
            }
        )
    except Exception as e:
        logger.error(f"Error indexing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query")
async def query_index(query: str = Form(...), collection: str = Form("default")):
    """
    Query the image index with text and return image data for orchestrator processing
    """
    try:
        # Query the index
        images = image_service.query_images(query, collection)
        
        if not images:
            return JSONResponse(
                status_code=404,
                content={"status": "error", "message": "No images found in the index"}
            )
        
        # Return image data and query for orchestrator to process with local Ollama
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "images": images,
                "query": query,
                "collection": collection
            }
        )
    except Exception as e:
        logger.error(f"Error querying index: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health/detailed")
async def detailed_health_check():
    """
    Perform detailed health check including model, MongoDB, and GPU tests
    """
    try:
        results = run_full_health_check()
        return JSONResponse(
            status_code=200,
            content=results
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "failed", "error": str(e)}
        )
