from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import os
import sys
from pathlib import Path

# Add the project root directory to sys.path to import dxf2masks
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

STORAGE_IMAGES_DIR = os.path.join(project_root, "storage", "images")
STORAGE_MASKS_DIR = os.path.join(project_root, "storage", "masks")
os.makedirs(STORAGE_MASKS_DIR, exist_ok=True)
CHUNK_MASKS_DIR = os.path.join(project_root, "chunk_masks")

router = APIRouter()

@router.get("/masks/{file_id}")
async def get_masks(file_id: str):
    if file_id != os.path.basename(file_id) or ".." in file_id or "/" in file_id or "\\" in file_id:
        raise HTTPException(status_code=400, detail="Invalid file_id.")

    stored_image_path = os.path.join(STORAGE_IMAGES_DIR, f"{file_id}.png")
    image_base = Path(stored_image_path).stem if os.path.exists(stored_image_path) else file_id

    candidate_paths = (
        os.path.join(CHUNK_MASKS_DIR, f"{image_base}_masks.json"),
        os.path.join(STORAGE_MASKS_DIR, f"{file_id}_masks.json"),
    )

    for file_path in candidate_paths:
        if os.path.exists(file_path):
            return FileResponse(path=file_path, media_type="application/json")

    raise HTTPException(status_code=404, detail="Masks file not found.")
