from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import os
import sys

router = APIRouter()

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STORAGE_EMBEDDINGS_DIR = os.path.join(project_root, "storage", "embeddings")

@router.get("/embeddings/{file_id}")
async def get_embeddings(file_id: str):
    """
    Retrieve the generated embeddings parquet file.
    """
    file_path = os.path.join(STORAGE_EMBEDDINGS_DIR, f"{file_id}_chunks.parquet")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Embeddings not found")
    return FileResponse(file_path, media_type="application/octet-stream", filename=f"{file_id}.parquet")

@router.get("/embeddings/{file_id}/status")
async def get_embeddings_status(file_id: str):
    """
    Check if embeddings exist.
    """
    file_path = os.path.join(STORAGE_EMBEDDINGS_DIR, f"{file_id}_chunks.parquet")
    exists = os.path.exists(file_path)
    return {"file_id": file_id, "exists": exists}
