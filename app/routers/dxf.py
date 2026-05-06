import json
import asyncio
import sys
import multiprocessing as mp
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Request, Response
from fastapi.responses import FileResponse, JSONResponse
import shutil
import os
import tempfile
import uuid
import hashlib
from typing import List
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import threading
import glob
from typing import Optional
from pydantic import BaseModel
from PIL import Image
from src.utils import extract_dxf_metadata

# Add the project root directory to sys.path to import dxf2chunks and dxf2image
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

# Create storage directories if they don't exist
STORAGE_GEOJSON_DIR = os.path.join(project_root, "storage", "geojson")
STORAGE_IMAGES_DIR = os.path.join(project_root, "storage", "images")
STORAGE_MASKS_DIR = os.path.join(project_root, "storage", "masks")
STORAGE_EMBEDDINGS_DIR = os.path.join(project_root, "storage", "embeddings")
STORAGE_STATUS_DIR = os.path.join(project_root, "storage", "status")
STORAGE_DXF_DIR = os.path.join(project_root, "storage", "dxf")
STORAGE_METADATA_DIR = os.path.join(project_root, "storage", "metadata")
os.makedirs(STORAGE_GEOJSON_DIR, exist_ok=True)
os.makedirs(STORAGE_IMAGES_DIR, exist_ok=True)
os.makedirs(STORAGE_MASKS_DIR, exist_ok=True)
os.makedirs(STORAGE_EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(STORAGE_STATUS_DIR, exist_ok=True)
os.makedirs(STORAGE_DXF_DIR, exist_ok=True)
os.makedirs(STORAGE_METADATA_DIR, exist_ok=True)

def _load_dxf_modules():
    try:
        from src import dxf2chunks
        from src import dxf2image
        from src import dxf2masks
    except ImportError as e:
        print(f"Error importing modules: {e}")
        if os.getcwd() not in sys.path:
            sys.path.append(os.getcwd())
        from src import dxf2chunks
        from src import dxf2image
        from src import dxf2masks
    return dxf2chunks, dxf2image, dxf2masks

router = APIRouter()

_DXF_RUN_LOCK = threading.Lock()

class DxfListItem(BaseModel):
    dxf_id: str
    filename: str
    size_bytes: int
    modified_at: float
    source_filename: Optional[str] = None
    status: Optional[str] = None
    status_details: Optional[dict] = None

class DxfListResponse(BaseModel):
    count: int
    dxfs: list[DxfListItem]

def _get_dxf_active_runs(app) -> dict:
    runs = getattr(app.state, "dxf_active_runs", None)
    if runs is None:
        runs = {}
        app.state.dxf_active_runs = runs
    return runs

def _set_active_run(app, file_id: str, run_id: str) -> None:
    with _DXF_RUN_LOCK:
        _get_dxf_active_runs(app)[file_id] = run_id

def _get_active_run(app, file_id: str) -> str | None:
    with _DXF_RUN_LOCK:
        return _get_dxf_active_runs(app).get(file_id)

def _is_active_run(app, file_id: str, run_id: str) -> bool:
    return _get_active_run(app, file_id) == run_id

def _status_file_path(file_id: str) -> str:
    return os.path.join(STORAGE_STATUS_DIR, f"dxf_{file_id}.json")

def _metadata_file_path(file_id: str) -> str:
    return os.path.join(STORAGE_METADATA_DIR, f"{file_id}.json")

def _load_json_file(path: str) -> dict:
    try:
        with open(path, "r") as f:
            return json.load(f) or {}
    except Exception:
        return {}

def _collect_dxf_sources() -> dict[str, dict]:
    sources = {}
    if os.path.exists(STORAGE_STATUS_DIR):
        for status_file in glob.glob(os.path.join(STORAGE_STATUS_DIR, "dxf_*.json")):
            data = _load_json_file(status_file)
            file_id = data.get("id")
            details = data.get("details") or {}
            source_filename = details.get("source_filename")
            if file_id and source_filename:
                sources[file_id] = {"source_filename": source_filename}
    if os.path.exists(STORAGE_METADATA_DIR):
        for metadata_file in glob.glob(os.path.join(STORAGE_METADATA_DIR, "*.json")):
            data = _load_json_file(metadata_file)
            file_id = data.get("file_id") or data.get("id")
            if not file_id:
                file_id = os.path.splitext(os.path.basename(metadata_file))[0]
            source_filename = data.get("source_filename")
            if file_id and source_filename and file_id not in sources:
                sources[file_id] = {"source_filename": source_filename}
    return sources

def _ensure_storage_dirs() -> None:
    os.makedirs(STORAGE_GEOJSON_DIR, exist_ok=True)
    os.makedirs(STORAGE_IMAGES_DIR, exist_ok=True)
    os.makedirs(STORAGE_MASKS_DIR, exist_ok=True)
    os.makedirs(STORAGE_EMBEDDINGS_DIR, exist_ok=True)
    os.makedirs(STORAGE_STATUS_DIR, exist_ok=True)
    os.makedirs(STORAGE_DXF_DIR, exist_ok=True)
    os.makedirs(STORAGE_METADATA_DIR, exist_ok=True)

@router.get("/dxf", response_model=DxfListResponse)
async def list_dxf_files(include_status: bool = True):
    dxf_dir = Path(STORAGE_DXF_DIR)
    if not dxf_dir.exists():
        return {"count": 0, "dxfs": []}

    sources = _collect_dxf_sources()
    dxf_paths = sorted(
        (p for p in dxf_dir.iterdir() if p.is_file() and p.suffix.lower() == ".dxf"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    items: list[DxfListItem] = []
    for path in dxf_paths:
        stat = path.stat()
        dxf_id = path.stem
        source_filename = sources.get(dxf_id, {}).get("source_filename")
        status_value = None
        status_details = None
        if include_status:
            status = get_status(dxf_id)
            if status:
                status_value = status.get("status")
                status_details = status.get("details")
        items.append(
            DxfListItem(
                dxf_id=dxf_id,
                filename=path.name,
                size_bytes=int(stat.st_size),
                modified_at=float(stat.st_mtime),
                source_filename=source_filename,
                status=status_value,
                status_details=status_details,
            )
        )

    return {"count": len(items), "dxfs": items}

def _executor_ping():
    return True

def _server_worker_count() -> int:
    for key in ("UVICORN_WORKERS", "WEB_CONCURRENCY"):
        value = os.getenv(key)
        if value:
            try:
                parsed = int(value)
                if parsed >= 1:
                    return parsed
            except ValueError:
                continue
    return 1

def _executor_max_workers() -> int:
    configured = os.getenv("PROCESS_POOL_MAX_WORKERS")
    if configured:
        try:
            value = int(configured)
            if value >= 1:
                return value
        except ValueError:
            pass
    cpu_count = os.cpu_count() or 1
    base = max(1, cpu_count - 1)
    workers = _server_worker_count()
    if workers > 1:
        return max(1, base // workers)
    return base

def _executor_context():
    if sys.platform.startswith("linux"):
        return mp.get_context("spawn")
    return mp.get_context()

def _use_process_pool() -> bool:
    disabled = os.getenv("PROCESS_POOL_DISABLED", "").strip().lower() in {"1", "true", "yes"}
    if disabled:
        return False
    if "gunicorn" in (sys.argv[0] or "").lower():
        return False
    if os.getenv("GUNICORN_CMD_ARGS"):
        return False
    return True

def _ensure_executor(app):
    executor = getattr(app.state, "executor", None)
    if executor is None:
        app.state.executor = ProcessPoolExecutor(
            max_workers=_executor_max_workers(),
            mp_context=_executor_context(),
        )
        return app.state.executor

    try:
        future = executor.submit(_executor_ping)
        # Increase timeout to 30s to account for slow imports (torch, etc.) on fresh process startup
        future.result(timeout=30)
        return executor
    except Exception as e:
        print(f"Executor ping failed: {e}. Restarting executor...")
        if isinstance(e, BrokenProcessPool) or "process pool is not usable anymore" in str(e).lower() or isinstance(e, TimeoutError):
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            app.state.executor = ProcessPoolExecutor(
                max_workers=_executor_max_workers(),
                mp_context=_executor_context(),
            )
            return app.state.executor
        raise

def process_dxf_to_geojson(input_path: str, temp_dir: str):
    """
    Helper function to process a DXF file to GeoJSON.
    Returns the path to the generated GeoJSON file.
    """
    dxf2chunks, _, _ = _load_dxf_modules()
    output_geojson_path = os.path.join(temp_dir, f"{Path(input_path).stem}.geojson")
    output_viz_path = os.path.join(temp_dir, f"{Path(input_path).stem}_viz.png")
    
    try:
        print(f"Processing GeoJSON for {input_path}...")
        dxf2chunks.process_single_dxf(input_path, output_geojson_path, output_viz_path)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Error processing DXF to GeoJSON: {str(e)}")
        
    if not os.path.exists(output_geojson_path):
            raise RuntimeError("Processing failed to generate GeoJSON output.")
            
    return output_geojson_path

def process_dxf_to_image(input_path: str, temp_dir: str):
    """
    Helper function to process a DXF file to Image (PNG + Transform).
    Returns (png_path, transform_path, transform_data).
    """
    _, dxf2image, _ = _load_dxf_modules()
    try:
        print(f"Processing Image for {input_path}...")
        doc = dxf2image.extract_annotations(input_path)
        if not doc:
             raise RuntimeError("Failed to load DXF file content for image generation.")

        # output_png_path = os.path.join(temp_dir, f"{Path(input_path).stem}.png")
        
        # image_path, transform_path = dxf2image.export_to_png(
        #     doc, 
        #     output_path=output_png_path, 
        #     dpi=300, 
        #     use_text_bounds=False
        # )

        output_svg_path = os.path.join(temp_dir, f"{Path(input_path).stem}.svg")
        image_path, transform_path = dxf2image.export_to_svg(
            doc,
            output_path=output_svg_path,
            use_text_bounds=False,
            skip_hatches=False
        )
        
        if not image_path:
             raise RuntimeError("Image conversion failed.")
        
        transform_data = None
        if transform_path and os.path.exists(transform_path):
            with open(transform_path, 'r') as f:
                transform_data = json.load(f)
        else:
            # If no transform file was created, create a default one
            # For SVG with margins=0, the transform is simply the extents
            transform_path = os.path.splitext(output_svg_path)[0] + '_transform.json'
            # Try to get basic extents from the document
            extents = dxf2image.get_dxf_extents(doc)
            if extents:
                min_x, min_y, max_x, max_y = extents
                # For SVG, pixel dimensions are arbitrary, but we can use aspect ratio
                # Set a base dimension for reference (e.g. 1000px)
                dx, dy = max(max_x - min_x, 1e-6), max(max_y - min_y, 1e-6)
                
                # Use a high enough resolution base for any raster fallback logic
                base_dim = 2048 
                if dx > dy:
                    image_width = base_dim
                    image_height = int(base_dim * (dy / dx))
                else:
                    image_height = base_dim
                    image_width = int(base_dim * (dx / dy))
                    
                fit_scale = min(image_width / dx, image_height / dy)
                
                transform_data = {
                    'min_x': min_x, 'max_x': max_x,
                    'min_y': min_y, 'max_y': max_y,
                    'pixel_per_unit': fit_scale,
                    'image_width': image_width, 'image_height': image_height,
                    's_x': 1/fit_scale, 's_y': 1/fit_scale,
                    't_x': min_x, 't_y': min_y,
                    'H': dy
                }
                with open(transform_path, 'w') as f:
                    json.dump(transform_data, f, indent=4)
            
        return image_path, transform_path, transform_data

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Error processing DXF to Image: {str(e)}")

def process_masks_generation(geojson_path: str, transform_path: str, output_path: str):
    """
    Helper function to generate masks from GeoJSON and Transform.
    """
    _, _, dxf2masks = _load_dxf_modules()
    try:
        print(f"Generating masks for {geojson_path}...")
        
        # 1. Create DataFrame
        df = dxf2masks.create_layout_dataframe(geojson_path, transform_path)
        
        if df.empty:
            print("Warning: Empty dataframe created from layout.")
        
        # 2. Save to JSON
        dxf2masks.save_masks_to_json(df, output_path)
        
        if not os.path.exists(output_path):
             raise RuntimeError("Failed to generate masks file.")
             
        return output_path

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Error generating masks: {str(e)}")

def process_embeddings_generation(geojson_path: str, transform_path: str, image_path: str, output_path: str):
    """
    Helper function to generate embeddings from GeoJSON, Image and Transform.
    """
    try:
        from src.utils import create_layout_dataframe
        from src.image2embeddings import create_embedding_index
        
        print(f"Generating embeddings for {geojson_path}...")
        
        # 1. Create DataFrame
        df = create_layout_dataframe(geojson_path, transform_path)
        
        if df.empty:
            print("Warning: Empty dataframe created for embeddings.")
            return
            
        # 2. Generate Embeddings
        # output_path is the parquet file
        create_embedding_index(df, image_path, output_path)
        
        if not os.path.exists(output_path):
             raise RuntimeError("Failed to generate embeddings file.")
             
        return output_path

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Error generating embeddings: {str(e)}")

def update_status(file_id: str, status: str, details: dict = None):
    """
    Update the status of a file processing job.
    """
    _ensure_storage_dirs()
    status_file = _status_file_path(file_id)
    existing_details = {}
    if os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                existing = json.load(f) or {}
            existing_details = existing.get("details") or {}
        except Exception:
            existing_details = {}
    data = {
        "id": file_id,
        "status": status,
        "details": {**existing_details, **(details or {})}
    }
    dir_path = os.path.dirname(status_file) or "."
    tmp_file = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=dir_path, delete=False, prefix=".tmp_", suffix=".json") as f:
            tmp_file = f.name
            json.dump(data, f)
        os.replace(tmp_file, status_file)
    finally:
        if tmp_file and os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except OSError:
                pass

def get_status(file_id: str):
    """
    Get the status of a file processing job.
    """
    if not os.path.exists(STORAGE_STATUS_DIR):
        return None
    status_file = _status_file_path(file_id)
    if not os.path.exists(status_file):
        return None
    with open(status_file, "r") as f:
        return json.load(f)

def _get_source_filename(file_id: str) -> str | None:
    status = get_status(file_id)
    if status:
        details = status.get("details") or {}
        source_filename = details.get("source_filename")
        if source_filename:
            return source_filename
    metadata_path = _metadata_file_path(file_id)
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f) or {}
            source_filename = metadata.get("source_filename")
            if source_filename:
                return source_filename
        except Exception:
            return None
    return None

def run_processing_pipeline(app, file_id: str, run_id: str, input_path: str, temp_dir: str, executor, source_filename: str):
    """
    The main processing pipeline to be run in the background.
    """
    try:
        if not _is_active_run(app, file_id, run_id):
            return
        _ensure_storage_dirs()
        update_status(file_id, "processing", {"step": "geojson_and_image", "run_id": run_id})
        print(f"Starting background processing for {file_id}")

        if _use_process_pool() and executor is not None:
            try:
                future_geojson = executor.submit(process_dxf_to_geojson, input_path, temp_dir)
                future_image = executor.submit(process_dxf_to_image, input_path, temp_dir)
                temp_geojson_path = future_geojson.result()
                temp_png_path, temp_transform_path, transform_data = future_image.result()
            except Exception as e:
                if isinstance(e, BrokenProcessPool) or "terminated abruptly" in str(e).lower():
                    update_status(file_id, "processing", {"step": "geojson_and_image_fallback", "run_id": run_id})
                    temp_geojson_path = process_dxf_to_geojson(input_path, temp_dir)
                    temp_png_path, temp_transform_path, transform_data = process_dxf_to_image(input_path, temp_dir)
                else:
                    raise
        else:
            temp_geojson_path = process_dxf_to_geojson(input_path, temp_dir)
            temp_png_path, temp_transform_path, transform_data = process_dxf_to_image(input_path, temp_dir)
        if not _is_active_run(app, file_id, run_id):
            return
        update_status(file_id, "processing", {"step": "masks", "run_id": run_id})
        
        # 2. Masks
        temp_masks_path = os.path.join(temp_dir, f"{file_id}_masks.json")
        if _use_process_pool() and executor is not None:
            try:
                future_masks = executor.submit(process_masks_generation, temp_geojson_path, temp_transform_path, temp_masks_path)
                future_masks.result()
            except Exception as e:
                if isinstance(e, BrokenProcessPool) or "terminated abruptly" in str(e).lower():
                    update_status(file_id, "processing", {"step": "masks_fallback", "run_id": run_id})
                    process_masks_generation(temp_geojson_path, temp_transform_path, temp_masks_path)
                else:
                    raise
        else:
            process_masks_generation(temp_geojson_path, temp_transform_path, temp_masks_path)
        if not _is_active_run(app, file_id, run_id):
            return

        # 2.5 Embeddings
        # update_status(file_id, "processing", {"step": "embeddings", "run_id": run_id})
        # temp_embeddings_path = os.path.join(temp_dir, f"{file_id}_embeddings.parquet")
        
        # if _use_process_pool() and executor is not None:
        #     try:
        #         future_embeddings = executor.submit(process_embeddings_generation, temp_geojson_path, temp_transform_path, temp_png_path, temp_embeddings_path)
        #         future_embeddings.result()
        #     except Exception as e:
        #         if isinstance(e, BrokenProcessPool) or "terminated abruptly" in str(e).lower():
        #             update_status(file_id, "processing", {"step": "embeddings_fallback", "run_id": run_id})
        #             process_embeddings_generation(temp_geojson_path, temp_transform_path, temp_png_path, temp_embeddings_path)
        #         else:
        #             raise
        # else:
        #     process_embeddings_generation(temp_geojson_path, temp_transform_path, temp_png_path, temp_embeddings_path)

        if not _is_active_run(app, file_id, run_id):
            return
        
        # 3. Move files
        _ensure_storage_dirs()
        stored_geojson_path = os.path.join(STORAGE_GEOJSON_DIR, f"{file_id}.geojson")
        stored_png_path = os.path.join(STORAGE_IMAGES_DIR, f"{file_id}.png")
        stored_svg_path = os.path.join(STORAGE_IMAGES_DIR, f"{file_id}.svg")
        stored_transform_path = os.path.join(STORAGE_IMAGES_DIR, f"{file_id}_transform.json")
        stored_masks_path = os.path.join(STORAGE_MASKS_DIR, f"{file_id}_masks.json")
        stored_embeddings_path = os.path.join(STORAGE_EMBEDDINGS_DIR, f"{file_id}_chunks.parquet")
        stored_dxf_path = os.path.join(STORAGE_DXF_DIR, f"{file_id}.dxf")
        stored_metadata_path = _metadata_file_path(file_id)
        
        if os.path.exists(input_path):
            shutil.move(input_path, stored_dxf_path)
        shutil.move(temp_geojson_path, stored_geojson_path)
        if os.path.exists(temp_png_path):
            # Move SVG instead if it exists
            if temp_png_path.endswith('.svg'):
                shutil.move(temp_png_path, stored_svg_path)
            else:
                shutil.move(temp_png_path, stored_png_path)
        shutil.move(temp_transform_path, stored_transform_path)
        shutil.move(temp_masks_path, stored_masks_path)
        # if os.path.exists(temp_embeddings_path):
        #     shutil.move(temp_embeddings_path, stored_embeddings_path)
        if not _is_active_run(app, file_id, run_id):
            return
        update_status(file_id, "processing", {"step": "metadata", "run_id": run_id})
        try:
            metadata = extract_dxf_metadata(stored_dxf_path)
            metadata_payload = metadata.model_dump() if metadata else {}
            metadata_payload["source_filename"] = source_filename
            metadata_payload["file_id"] = file_id
            with open(stored_metadata_path, "w") as f:
                json.dump(metadata_payload, f)
        except Exception as metadata_error:
            update_status(
                file_id,
                "processing",
                {"step": "metadata_failed", "metadata_error": str(metadata_error), "run_id": run_id},
            )
        
        if not _is_active_run(app, file_id, run_id):
            return
        update_status(file_id, "completed", {"transform": transform_data, "run_id": run_id, "source_filename": source_filename})
        print(f"Processing completed for {file_id}")

    except Exception as e:
        print(f"Background processing failed for {file_id}: {e}")
        if _is_active_run(app, file_id, run_id):
            update_status(file_id, "failed", {"error": str(e), "run_id": run_id})
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

@router.post("/dxf")
async def process_dxf(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload a DXF file and start processing in the background.
    Returns 202 Accepted immediately.
    """
    if not file.filename.lower().endswith('.dxf'):
        raise HTTPException(status_code=400, detail="File must be a DXF file.")

    # Create a temporary directory for processing
    # Note: We need to keep this temp dir until background task is done.
    # But wait, if we return, the temp_dir variable is lost. 
    # We should create it and pass path to background task.
    temp_dir = tempfile.mkdtemp()
    
    try:
        source_filename = file.filename
        input_path = os.path.join(temp_dir, source_filename)
        
        # Calculate SHA256 hash while saving the file
        sha256_hash = hashlib.sha256()
        with open(input_path, "wb") as buffer:
            while True:
                chunk = await file.read(64 * 1024)
                if not chunk:
                    break
                sha256_hash.update(chunk)
                buffer.write(chunk)
        
        file_id = sha256_hash.hexdigest()
        run_id = uuid.uuid4().hex
        
        # Check cache
        status = get_status(file_id)
        if status and status["status"] == "completed":
             cached_source = status.get("details", {}).get("source_filename") or _get_source_filename(file_id)
             return JSONResponse(status_code=200, content={
                "id": file_id,
                "message": "File retrieved from cache.",
                "status": "completed",
                "status_url": f"/api/dxf/{file_id}/status",
                "geojson_url": f"/api/dxf/{file_id}",
                "geojson_download_url": f"/api/dxf/{file_id}/download",
                "dxf_url": f"/api/dxf/{file_id}/source",
                "dxf_download_url": f"/api/dxf/{file_id}/source/download",
                "dxf_metadata_url": f"/api/dxf/{file_id}/metadata",
                "image_url": f"/api/images/{file_id}/content",
                "transform_url": f"/api/images/{file_id}/transform",
                "masks_url": f"/api/masks/{file_id}",
                "transform": status["details"].get("transform"),
                "source_filename": cached_source
            })
        restarted = bool(status and status.get("status") == "processing")
        _set_active_run(request.app, file_id, run_id)
        update_status(file_id, "processing", {"step": "queued", "run_id": run_id, "source_filename": source_filename})

        # Start background task
        print(f"Initializing executor for {file_id}...")
        try:
            executor = _ensure_executor(request.app)
            print(f"Executor ready. Submitting background task for {file_id}...")
            background_tasks.add_task(run_processing_pipeline, request.app, file_id, run_id, input_path, temp_dir, executor, source_filename)
        except Exception as exec_err:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to initialize background processing: {exec_err}")
        
        return JSONResponse(status_code=202, content={
            "id": file_id, 
            "message": "Restarting processing." if restarted else "File accepted for processing.",
            "status": "processing",
            "status_url": f"/api/dxf/{file_id}/status",
            "run_id": run_id,
            "restarted": restarted,
            "source_filename": source_filename
        })
        
    except Exception as e:
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except OSError:
                pass
        
        import traceback
        traceback.print_exc()
        
        error_detail = str(e)
        if not error_detail:
            error_detail = f"{type(e).__name__}: An unexpected error occurred."
            
        print(f"Upload error: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)

@router.get("/dxf/{file_id}/status")
async def get_processing_status(request: Request, file_id: str):
    """
    Get the processing status of a file.
    """
    status = get_status(file_id)
    if not status:
        active_run = _get_active_run(request.app, file_id)
        if active_run:
            return {
                "id": file_id,
                "status": "processing",
                "details": {"step": "unknown", "run_id": active_run},
            }
        raise HTTPException(status_code=404, detail="Job not found")
    
    response = {
        "id": file_id,
        "status": status["status"],
        "details": status["details"]
    }
    
    if status["status"] == "completed":
        response.update({
            "geojson_url": f"/api/dxf/{file_id}",
            "geojson_download_url": f"/api/dxf/{file_id}/download",
            "dxf_url": f"/api/dxf/{file_id}/source",
            "dxf_download_url": f"/api/dxf/{file_id}/source/download",
            "dxf_metadata_url": f"/api/dxf/{file_id}/metadata",
            "image_url": f"/api/images/{file_id}/content",
            "transform_url": f"/api/images/{file_id}/transform",
            "masks_url": f"/api/masks/{file_id}",
            "transform": status["details"].get("transform")
        })
        
    return response

@router.get("/dxf/{file_id}")
async def get_dxf_geojson(file_id: str):
    """
    Retrieve the GeoJSON content for a processed DXF file.
    """
    file_path = os.path.join(STORAGE_GEOJSON_DIR, f"{file_id}.geojson")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Return the raw file content directly to avoid parsing overhead
    # and potential serialization issues.
    with open(file_path, 'rb') as f:
        content = f.read()
        
    return Response(content=content, media_type="application/json")

@router.get("/dxf/{file_id}/download", response_class=FileResponse)
async def download_dxf_geojson(file_id: str):
    """
    Download the GeoJSON file for a processed DXF.
    """
    file_path = os.path.join(STORAGE_GEOJSON_DIR, f"{file_id}.geojson")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    return FileResponse(
        path=file_path, 
        filename=f"processed_{file_id}.geojson",
        media_type="application/geo+json"
    )

@router.get("/dxf/{file_id}/source")
async def get_dxf_source(file_id: str):
    file_path = os.path.join(STORAGE_DXF_DIR, f"{file_id}.dxf")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    source_filename = _get_source_filename(file_id)
    filename = source_filename or f"{file_id}.dxf"
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/dxf"
    )

@router.get("/dxf/{file_id}/source/download", response_class=FileResponse)
async def download_dxf_source(file_id: str):
    file_path = os.path.join(STORAGE_DXF_DIR, f"{file_id}.dxf")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    source_filename = _get_source_filename(file_id)
    filename = source_filename or f"{file_id}.dxf"
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/dxf"
    )

@router.get("/dxf/{file_id}/metadata")
async def get_dxf_metadata(file_id: str):
    file_path = os.path.join(STORAGE_DXF_DIR, f"{file_id}.dxf")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    metadata_path = _metadata_file_path(file_id)
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        if "source_filename" not in metadata:
            source_filename = _get_source_filename(file_id)
            if source_filename:
                metadata["source_filename"] = source_filename
        return metadata
    metadata = extract_dxf_metadata(file_path)
    _ensure_storage_dirs()
    metadata_payload = metadata.model_dump()
    source_filename = _get_source_filename(file_id)
    if source_filename:
        metadata_payload["source_filename"] = source_filename
    with open(metadata_path, "w") as f:
        json.dump(metadata_payload, f)
    return metadata_payload

@router.delete("/dxf/{file_id}")
async def delete_dxf(file_id: str):
    if file_id != os.path.basename(file_id) or ".." in file_id or "/" in file_id or "\\" in file_id:
        raise HTTPException(status_code=400, detail="Invalid file id.")
    status = get_status(file_id)
    if status and status.get("status") == "processing":
        raise HTTPException(status_code=409, detail="Cannot delete while processing")
    paths: List[str] = []
    paths.append(os.path.join(STORAGE_DXF_DIR, f"{file_id}.dxf"))
    paths.append(os.path.join(STORAGE_GEOJSON_DIR, f"{file_id}.geojson"))
    paths.append(os.path.join(STORAGE_IMAGES_DIR, f"{file_id}.png"))
    paths.append(os.path.join(STORAGE_IMAGES_DIR, f"{file_id}_transform.json"))
    paths.append(os.path.join(STORAGE_MASKS_DIR, f"{file_id}_masks.json"))
    paths.append(os.path.join(STORAGE_EMBEDDINGS_DIR, f"{file_id}_chunks.parquet"))
    paths.append(_metadata_file_path(file_id))
    paths.append(_status_file_path(file_id))
    deleted = []
    for p in paths:
        if os.path.exists(p):
            try:
                os.remove(p)
                deleted.append(p)
            except OSError:
                pass
    return {"id": file_id, "deleted_count": len(deleted), "deleted": deleted}
