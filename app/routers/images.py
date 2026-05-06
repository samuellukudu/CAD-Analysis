from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from PIL import Image
import os
import json
import sys

# Increase PIL image size limit to handle large CAD exports
Image.MAX_IMAGE_PIXELS = None

router = APIRouter()

# Add project root to sys.path (optional if we just use relative paths for storage, but good for consistency)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STORAGE_DIR = os.path.join(project_root, "storage", "images")

@router.get("/images/{file_id}/content")
async def get_image_content(file_id: str, max_dim: int = Query(8192, ge=1024, le=16384)):
    """
    Retrieve the generated image (SVG preferred, fallback to PNG).
    For PNGs, automatically resizes images larger than max_dim (default 8192) to ensure they render in WebGL.
    """
    # Check for SVG first - Priority 1
    svg_path = os.path.join(STORAGE_DIR, f"{file_id}.svg")
    if os.path.exists(svg_path):
        return FileResponse(svg_path, media_type="image/svg+xml")

    # Check for cached resized PNG version - Priority 2
    resized_filename = f"{file_id}_web_{max_dim}.png"
    resized_path = os.path.join(STORAGE_DIR, resized_filename)
    
    if os.path.exists(resized_path):
        return FileResponse(resized_path, media_type="image/png")

    file_path = os.path.join(STORAGE_DIR, f"{file_id}.png")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
        
    # Check dimensions and resize if necessary
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            if width > max_dim or height > max_dim:
                # Calculate new size maintaining aspect ratio
                ratio = min(max_dim / width, max_dim / height)
                new_size = (int(width * ratio), int(height * ratio))
                
                # Resize and save cached version
                # Use conversion to RGB if RGBA to avoid issues with some formats, 
                # but keep RGBA if transparency is needed (CAD exports usually don't have transparency unless specified)
                # Inspecting the images showed RGB, so we can keep mode or convert.
                # Just resize.
                resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
                resized_img.save(resized_path, format="PNG", optimize=True)
                
                return FileResponse(resized_path, media_type="image/png")
            else:
                # Image is small enough, return original
                return FileResponse(file_path, media_type="image/png")
    except Exception as e:
        print(f"Error processing image {file_id}: {e}")
        # Fallback to original if processing fails
        return FileResponse(file_path, media_type="image/png")

@router.get("/images/{file_id}/transform")
async def get_image_transform(file_id: str):
    """
    Retrieve the transformation data for the image.
    """
    file_path = os.path.join(STORAGE_DIR, f"{file_id}_transform.json")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Transform not found")
    with open(file_path, "r") as f:
        return json.load(f)
