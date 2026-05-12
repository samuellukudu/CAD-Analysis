from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import base64
import json
import os
import shutil
import time

router = APIRouter(prefix="/api", tags=["files"])

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
STORAGE_DXF_DIR = os.path.join(project_root, "storage", "dxf")
STORAGE_JSON_DIR = os.path.join(project_root, "storage", "json")
STORAGE_IMAGES_DIR = os.path.join(project_root, "storage", "images")

os.makedirs(STORAGE_DXF_DIR, exist_ok=True)
os.makedirs(STORAGE_JSON_DIR, exist_ok=True)
os.makedirs(STORAGE_IMAGES_DIR, exist_ok=True)


def _safe_filename(name: str, fallback: str) -> str:
    base = os.path.basename(str(name or "")).strip()
    cleaned = "".join(ch for ch in base if ch >= " " and ch != "\x7f")
    cleaned = cleaned.replace("/", "_").replace("\\", "_")
    return cleaned or fallback


def ensure_dxf_filename(name: str) -> str:
    safe = _safe_filename(name, f"dxf-{int(time.time())}.dxf")
    if not safe.lower().endswith(".dxf"):
        safe = f"{safe}.dxf"
    return safe


def ensure_json_filename(name: str) -> str:
    safe = _safe_filename(name, f"dxf-parsed-data-{int(time.time())}.json")
    if safe.lower().endswith(".dxf"):
        safe = safe[:-4]
    if not safe.lower().endswith(".json"):
        safe = f"{safe}.json"
    return safe


class DxfDataPayload(BaseModel):
    filename: str
    dxfData: Dict[str, Any]


class DxfFileDataPayload(BaseModel):
    fileData: str
    filename: str


@router.get("/stored-dxf-files", response_model=List[Dict[str, Any]])
async def list_stored_dxf_files():
    if not os.path.exists(STORAGE_DXF_DIR):
        return []

    files = []
    try:
        for f in os.listdir(STORAGE_DXF_DIR):
            if f.lower().endswith(".dxf"):
                full_path = os.path.join(STORAGE_DXF_DIR, f)
                stat = os.stat(full_path)
                files.append({
                    "name": f,
                    "size": stat.st_size,
                    "modified": stat.st_mtime * 1000,
                    "path": full_path
                })
        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save-dxf-data")
async def save_dxf_data(payload: DxfDataPayload):
    try:
        filename = ensure_json_filename(payload.filename)
        file_path = os.path.join(STORAGE_JSON_DIR, filename)

        if not os.path.abspath(file_path).startswith(os.path.abspath(STORAGE_JSON_DIR)):
            raise HTTPException(status_code=403, detail="Access denied")

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload.dxfData, f, ensure_ascii=False)

        return {"status": "success", "filename": filename, "path": f"storage/json/{filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dxf-files", response_model=List[Dict[str, Any]])
async def list_parsed_dxf_files():
    try:
        files = []
        for f in os.listdir(STORAGE_JSON_DIR):
            if not f.lower().endswith(".json"):
                continue
            full_path = os.path.join(STORAGE_JSON_DIR, f)
            stat = os.stat(full_path)
            files.append({
                "name": f,
                "path": full_path,
                "size": stat.st_size,
                "modified": stat.st_mtime * 1000
            })
        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/latest-dxf")
async def get_latest_dxf():
    try:
        json_files = [f for f in os.listdir(STORAGE_JSON_DIR) if f.lower().endswith(".json")]
        if not json_files:
            raise HTTPException(status_code=404, detail="No DXF files found")

        latest_file = max(
            json_files,
            key=lambda f: os.path.getmtime(os.path.join(STORAGE_JSON_DIR, f))
        )
        latest_path = os.path.join(STORAGE_JSON_DIR, latest_file)

        with open(latest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {"filename": latest_file, "path": latest_path, "data": data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/check-cache/{filename}")
async def check_cache(filename: str):
    try:
        json_filename = ensure_json_filename(filename)
        json_path = os.path.join(STORAGE_JSON_DIR, json_filename)
        exists = os.path.exists(json_path)
        return {
            "exists": exists,
            "jsonFilename": json_filename,
            "jsonPath": json_path if exists else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save-dxf-file")
async def save_dxf_file(dxfFile: UploadFile = File(...)):
    try:
        filename = ensure_dxf_filename(dxfFile.filename)
        file_path = os.path.join(STORAGE_DXF_DIR, filename)

        if not os.path.abspath(file_path).startswith(os.path.abspath(STORAGE_DXF_DIR)):
            raise HTTPException(status_code=403, detail="Access denied")

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(dxfFile.file, buffer)

        return {"status": "success", "filename": filename, "message": "File saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save-dxf-file-data")
async def save_dxf_file_data(payload: DxfFileDataPayload):
    try:
        filename = ensure_dxf_filename(payload.filename)
        file_path = os.path.join(STORAGE_DXF_DIR, filename)

        if not os.path.abspath(file_path).startswith(os.path.abspath(STORAGE_DXF_DIR)):
            raise HTTPException(status_code=403, detail="Access denied")

        decoded = base64.b64decode(payload.fileData)
        with open(file_path, "wb") as f:
            f.write(decoded)

        return {"status": "success", "filename": filename, "path": f"storage/dxf/{filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dxf-file/{filename}")
async def get_dxf_file(filename: str):
    safe_name = ensure_dxf_filename(filename)
    file_path = os.path.join(STORAGE_DXF_DIR, safe_name)

    if not os.path.abspath(file_path).startswith(os.path.abspath(STORAGE_DXF_DIR)):
        raise HTTPException(status_code=403, detail="Access denied")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, media_type="application/dxf", filename=safe_name)


@router.delete("/dxf-file/{filename}")
async def delete_dxf_file(filename: str):
    safe_name = ensure_dxf_filename(filename)
    file_path = os.path.join(STORAGE_DXF_DIR, safe_name)

    if not os.path.abspath(file_path).startswith(os.path.abspath(STORAGE_DXF_DIR)):
        raise HTTPException(status_code=403, detail="Access denied")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        os.remove(file_path)
        json_name = ensure_json_filename(safe_name)
        json_path = os.path.join(STORAGE_JSON_DIR, json_name)
        had_json = False
        if os.path.exists(json_path):
            os.remove(json_path)
            had_json = True
        return {"status": "success", "deleted": {"dxf": safe_name, "json": json_name if had_json else None}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stored-svg-files", response_model=List[Dict[str, Any]])
async def list_stored_svg_files():
    if not os.path.exists(STORAGE_IMAGES_DIR):
        return []

    files = []
    try:
        for f in os.listdir(STORAGE_IMAGES_DIR):
            if f.lower().endswith(".svg"):
                full_path = os.path.join(STORAGE_IMAGES_DIR, f)
                stat = os.stat(full_path)
                files.append({
                    "name": f,
                    "size": stat.st_size,
                    "modified": stat.st_mtime * 1000,
                    "path": full_path
                })
        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/svg-file/{filename}")
async def get_svg_file(filename: str):
    file_path = os.path.join(STORAGE_IMAGES_DIR, filename)

    if not os.path.abspath(file_path).startswith(os.path.abspath(STORAGE_IMAGES_DIR)):
        raise HTTPException(status_code=403, detail="Access denied")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, media_type="image/svg+xml", filename=filename)


@router.delete("/svg-file/{filename}")
async def delete_svg_file(filename: str):
    svg_path = os.path.join(STORAGE_IMAGES_DIR, filename)

    if not os.path.abspath(svg_path).startswith(os.path.abspath(STORAGE_IMAGES_DIR)):
        raise HTTPException(status_code=403, detail="Access denied")

    if not os.path.exists(svg_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        os.remove(svg_path)
        png_filename = filename.rsplit(".", 1)[0] + ".png"
        png_path = os.path.join(STORAGE_IMAGES_DIR, png_filename)
        if os.path.exists(png_path):
            os.remove(png_path)
        return {"status": "success", "message": f"Deleted {filename} and associated files"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
