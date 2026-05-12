from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Dict, Any
import os
import sys

# Ensure src is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.search_cad import search_text, search_spatial

router = APIRouter(prefix="/search", tags=["search"])

class TextSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Text to search for")
    paths: List[str] = Field(default=["storage/json"], description="Files or folders to search in")

class BBoxSearchRequest(BaseModel):
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    paths: List[str] = Field(default=["storage/json"], description="Files or folders to search in")

class PointSearchRequest(BaseModel):
    x: float
    y: float
    paths: List[str] = Field(default=["storage/json"], description="Files or folders to search in")

@router.post("/text", response_model=List[Dict[str, Any]])
async def search_text_endpoint(request: TextSearchRequest):
    """
    Search for text in CAD files.
    """
    try:
        # Resolve paths relative to project root if needed
        # Assuming server is run from project root, relative paths like 'storage/json' are fine.
        results = search_text(request.query, request.paths)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/bbox", response_model=List[Dict[str, Any]])
async def search_bbox_endpoint(request: BBoxSearchRequest):
    """
    Search for entities within a bounding box.
    """
    try:
        bbox = (request.min_x, request.min_y, request.max_x, request.max_y)
        results = search_spatial(bbox, request.paths, mode='bbox')
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/point", response_model=List[Dict[str, Any]])
async def search_point_endpoint(request: PointSearchRequest):
    """
    Search for entities containing a point.
    """
    try:
        point = (request.x, request.y)
        results = search_spatial(point, request.paths, mode='point')
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files", response_model=List[str])
async def list_files():
    """
    List available JSON files in storage/json.
    """
    try:
        storage_dir = os.path.join(project_root, "storage", "json")
        if not os.path.exists(storage_dir):
            return []
        
        files = [f for f in os.listdir(storage_dir) if f.endswith('.json')]
        # Return full relative paths from project root or just filenames?
        # Frontend likely wants filenames to display, but backend search needs paths.
        # Let's return filenames, and frontend can prepend path or backend can handle it.
        # But search_text expects paths.
        # Let's return paths relative to project root.
        return [os.path.join("storage", "json", f) for f in files]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
