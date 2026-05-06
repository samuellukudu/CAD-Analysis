from __future__ import annotations

# Set environment variables FIRST, before any imports (macOS segfault prevention)
import os
import platform

if platform.system() == "Darwin":  # macOS
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import sys
import multiprocessing as mp
import tempfile
import hashlib
import shutil
import gc
import base64
import re
import threading
from urllib.parse import quote
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
if TYPE_CHECKING:
    import torch
    from PIL import Image

# Add the project root directory to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

# Create storage directories if they don't exist
STORAGE_PDFS_DIR = os.path.join(project_root, "storage", "pdfs")
STORAGE_EMBEDDINGS_DIR = os.path.join(project_root, "storage", "embeddings")
STORAGE_STATUS_DIR = os.path.join(project_root, "storage", "status")
GLOBAL_EMBEDDINGS_FILE = os.path.join(STORAGE_EMBEDDINGS_DIR, "all_embeddings.parquet")
GLOBAL_FAISS_INDEX = os.path.join(STORAGE_EMBEDDINGS_DIR, "faiss_index.faiss")

os.makedirs(STORAGE_PDFS_DIR, exist_ok=True)
os.makedirs(STORAGE_EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(STORAGE_STATUS_DIR, exist_ok=True)

try:
    from src.utils import (
        pdf_to_base64_pngs,
        pdf_pages_to_base64_pngs,
        base64_to_pillow,
        get_siglip_model_processor,
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())
    try:
        from src.utils import (
            pdf_to_base64_pngs,
            pdf_pages_to_base64_pngs,
            base64_to_pillow,
            get_siglip_model_processor,
        )
    except ImportError as e2:
        raise ImportError(f"Could not import required modules. Error: {e2}")

router = APIRouter()

class PdfListItem(BaseModel):
    pdf_id: str
    filename: str
    size_bytes: int
    modified_at: float
    pdf_source: Optional[str] = None
    status: Optional[str] = None
    status_url: Optional[str] = None
    view_url: str
    content_url: str

class PdfListResponse(BaseModel):
    count: int
    pdfs: list[PdfListItem]

def _content_disposition_inline(filename: str) -> str:
    cleaned = filename.replace("\\", "_").replace("/", "_").replace('"', "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        cleaned = "document.pdf"

    try:
        cleaned.encode("latin-1")
        return f'inline; filename="{cleaned}"'
    except UnicodeEncodeError:
        fallback = "".join(ch if (32 <= ord(ch) <= 126 and ch not in '\\"') else "_" for ch in cleaned)
        fallback = fallback.strip() or "document.pdf"
        encoded = quote(cleaned.encode("utf-8"))
        return f'inline; filename="{fallback}"; filename*=UTF-8\'\'{encoded}'

# Global model instance (lazy loaded) - only for main process
_model_config = None
_torch_configured = False
_DEVICE = None

# Caches
_FAISS_INDEX_CACHE = {}
_DF_CACHE = {}

def _np():
    import numpy as np
    return np

def _pd():
    import pandas as pd
    return pd

def _faiss():
    import faiss
    return faiss

def _ensure_torch_configured():
    global _torch_configured
    if _torch_configured:
        return
    
    import torch
    if platform.system() == "Darwin":  # macOS
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    _torch_configured = True

def get_device():
    global _DEVICE
    if _DEVICE is None:
        _ensure_torch_configured()
        import torch
        _DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    return _DEVICE

def get_model():
    """Lazy load the embedding model (for main process only)."""
    global _model_config
    if _model_config is None:
        _ensure_torch_configured()
        _model_config = get_siglip_model_processor()
    return _model_config


def _encode_query_text(query: str) -> np.ndarray:
    np = _np()
    faiss = _faiss()
    model_config = get_model()
    model = model_config["model"]
    processor = model_config["processor"]

    import torch
    device = next(model.parameters()).device
    inputs = processor(text=[query], padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        if not hasattr(model, "get_text_features"):
            raise RuntimeError("Loaded model does not support text embeddings (get_text_features missing).")
        query_emb = model.get_text_features(**inputs)

    query_emb = query_emb.detach().cpu().numpy().astype("float32")
    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)
    faiss.normalize_L2(query_emb)
    return query_emb

def encode_images_in_batches(images: List[Image.Image], batch_size: int = 16, progress_callback=None) -> "torch.Tensor":
    """Encode list of PIL images in batches for better GPU utilization."""
    import torch
    model_config = get_siglip_model_processor()
    model = model_config["model"]
    processor = model_config["processor"]
    device = next(model.parameters()).device

    embeddings = []
    total_images = len(images)
    
    for i in range(0, total_images, batch_size):
        batch = images[i:i + batch_size]
        with torch.no_grad():
            inputs = processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if not hasattr(model, "get_image_features"):
                raise RuntimeError("Loaded model does not support image embeddings (get_image_features missing).")
            emb = model.get_image_features(**inputs)
            
            # Handle cases where model returns a ModelOutput object instead of a Tensor
            if not hasattr(emb, "detach"):
                if hasattr(emb, "pooler_output") and emb.pooler_output is not None:
                    emb = emb.pooler_output
                elif hasattr(emb, "last_hidden_state"):
                    emb = emb.last_hidden_state
                    # If we fell back to last_hidden_state, it might be [B, L, D]. 
                    # We usually want [B, D]. 
                    # If it's SigLIP, usually the first token or pooled.
                    # But let's assume if we are here, we take what we can get or maybe it's already [B, D] if it was pooler-like?
                    # Actually, last_hidden_state is always [B, L, D].
                    if len(emb.shape) == 3:
                         # Simple mean pooling if we are forced to use last_hidden_state
                         emb = emb.mean(dim=1)
                elif isinstance(emb, (list, tuple)) and len(emb) > 0:
                     # Try to find a tensor in the tuple
                     for item in emb:
                         if hasattr(item, "detach"):
                             emb = item
                             break
            
            embeddings.append(emb.detach().cpu())
        
        # Report progress
        processed_count = min(i + batch_size, total_images)
        if progress_callback:
            progress_callback(processed_count, total_images)
            
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return torch.cat(embeddings, dim=0) if embeddings else torch.tensor([])

def process_pdf_to_embeddings(pdf_path: str, pdf_id: str, progress_callback=None) -> List[dict]:
    """
    Process a single PDF and return embedding records.
    Returns list of records with embeddings for each page.
    """
    try:
        print(f"Processing PDF: {pdf_path}")
        # Note: pdf_to_base64_pngs could also have progress, but it's usually faster than embedding.
        b64s = pdf_to_base64_pngs(pdf_path)
        
        images = []
        for b64 in b64s:
            img = base64_to_pillow(b64)
            if img:
                images.append(img)
        
        if not images:
            raise RuntimeError("No images extracted from PDF")
        
        embeddings = encode_images_in_batches(images, progress_callback=progress_callback)
        
        pdf_name = Path(pdf_path).name
        records = []
        for page_num, emb in enumerate(embeddings):
            records.append({
                "pdf_id": pdf_id,
                "pdf_source": pdf_name,
                "full_path": str(pdf_path),
                "page_number": page_num + 1,
                "embedding": emb.numpy()  # Store as numpy array for Parquet compatibility
            })
        
        print(f"Completed {pdf_name}: {len(images)} pages encoded.")
        return records
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Error processing PDF: {str(e)}")

def update_global_embeddings(new_records: List[dict], rebuild_index: bool = True):
    """Update the global embeddings parquet file with new records."""
    pd = _pd()
    try:
        # Load existing embeddings if they exist
        if os.path.exists(GLOBAL_EMBEDDINGS_FILE):
            existing_df = pd.read_parquet(GLOBAL_EMBEDDINGS_FILE)
            # Filter out any existing records for the same PDF IDs
            pdf_ids = {r['pdf_id'] for r in new_records}
            existing_df = existing_df[~existing_df['pdf_id'].isin(pdf_ids)]
            # Combine
            new_df = pd.DataFrame(new_records)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = pd.DataFrame(new_records)
        
        # Save updated embeddings
        combined_df.to_parquet(GLOBAL_EMBEDDINGS_FILE, compression="gzip")
        # Invalidate cache
        if GLOBAL_EMBEDDINGS_FILE in _DF_CACHE:
            del _DF_CACHE[GLOBAL_EMBEDDINGS_FILE]
            
        print(f"Updated embeddings file: {len(combined_df)} total records")
        
        if rebuild_index:
            rebuild_faiss_index()
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Error updating global embeddings: {str(e)}")

def rebuild_faiss_index():
    """Rebuild the FAISS index from the global embeddings file."""
    pd = _pd()
    np = _np()
    faiss = _faiss()
    try:
        if not os.path.exists(GLOBAL_EMBEDDINGS_FILE):
            print("No embeddings file found, skipping FAISS index rebuild")
            return
        
        df = pd.read_parquet(GLOBAL_EMBEDDINGS_FILE)
        if len(df) == 0:
            print("Empty embeddings file, skipping FAISS index rebuild")
            return
        
        embeddings = np.stack(df['embedding'].values).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create index
        index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner Product = cosine after L2 norm
        index.add(embeddings)
        
        # Save index
        faiss.write_index(index, GLOBAL_FAISS_INDEX)
        # Invalidate cache
        if GLOBAL_FAISS_INDEX in _FAISS_INDEX_CACHE:
            del _FAISS_INDEX_CACHE[GLOBAL_FAISS_INDEX]

        print(f"FAISS index rebuilt: {len(embeddings)} vectors")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Warning: Failed to rebuild FAISS index: {str(e)}")

def _rebuild_index_async(file_id: str, pdf_source: str):
    def worker():
        try:
            rebuild_faiss_index()
            update_status(file_id, "completed", {
                "pdf_id": file_id,
                "pdf_source": pdf_source,
                "index_status": "ready"
            })
        except Exception as e:
            update_status(file_id, "completed", {
                "pdf_id": file_id,
                "pdf_source": pdf_source,
                "index_status": "failed",
                "index_error": str(e)
            })
    threading.Thread(target=worker, daemon=True).start()

def update_status(file_id: str, status: str, details: dict = None):
    """Update the status of a file processing job."""
    status_file = os.path.join(STORAGE_STATUS_DIR, f"pdf_{file_id}.json")
    existing_details = {}
    if os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                existing = json.load(f) or {}
            existing_details = existing.get("details") or {}
        except Exception:
            existing_details = {}
    merged_details = {**existing_details, **(details or {})}
    data = {
        "id": file_id,
        "status": status,
        "details": merged_details
    }
    with open(status_file, "w") as f:
        json.dump(data, f)

def get_status(file_id: str):
    """Get the status of a file processing job."""
    status_file = os.path.join(STORAGE_STATUS_DIR, f"pdf_{file_id}.json")
    if not os.path.exists(status_file):
        return None
    with open(status_file, "r") as f:
        return json.load(f)

def process_pdf_with_progress(pdf_path: str, file_id: str, pdf_source: str) -> List[dict]:
    """Wrapper to run in worker process and report progress."""
    def progress_callback(current, total):
        update_status(file_id, "processing", {
            "pdf_id": file_id, 
            "pdf_source": pdf_source, 
            "step": "generating_embeddings", # More descriptive step name
            "progress": f"{current}/{total}"
        })
    return process_pdf_to_embeddings(pdf_path, file_id, progress_callback=progress_callback)

def run_processing_pipeline(file_id: str, input_path: str, temp_dir: str, executor):
    """The main processing pipeline to be run in the background."""
    try:
        pdf_source = Path(input_path).name
        update_status(file_id, "processing", {"pdf_id": file_id, "pdf_source": pdf_source, "step": "extracting_pages"})
        print(f"Starting background processing for PDF {file_id}")

        print(f"Starting background processing for PDF {file_id}")

        def update_progress(current, total):
            update_status(file_id, "processing", {
                "pdf_id": file_id, 
                "pdf_source": pdf_source, 
                "step": "embedding_pages",
                "progress": f"{current}/{total}"
            })

        # Process PDF to embeddings
        # Since we are using ProcessPoolExecutor, we cannot pass a callback function that updates state in the main process easily 
        # because the callback would need to be pickled and run in the worker process.
        # BUT 'update_status' writes to a file, so it IS safe to run in the worker process!
        # So we can pass 'update_progress' (which calls 'update_status') to the worker.
        # Wait, 'update_status' is defined in this module, so it should be picklable if top-level?
        # 'update_progress' is a closure here, which might fail pickling.
        # Better: Define a top-level helper or move logic.
        # Actually, let's just use 'process_pdf_and_report' wrapper or pass update_status and arguments.
        # But 'process_pdf_to_embeddings' is the target.
        # Let's verify if `update_status` is picklable. It is a top-level function.
        # We can pass `update_status` and `file_id` etc to `process_pdf_to_embeddings` and let it define the callback or call it directly.
        
        # Let's change the strategy slightly: `process_pdf_to_embeddings` will take `status_callback_args` 
        # and reconstruct the updater.
        # OR: Just pass `file_id` and have it call `update_status` directly.
        
        # Given `process_pdf_to_embeddings` is in the same file, let's modify it to take `file_id` and do update inside?
        # It already takes `pdf_id`.
        # Let's modify `process_pdf_to_embeddings` to optionally call `update_status` if imported. 
        # But we want to keep it somewhat pure.
        
        # Simpler: 'process_pdf_with_progress' wrapper function that is top-level.
        future_embeddings = executor.submit(process_pdf_with_progress, input_path, file_id, pdf_source)
        records = future_embeddings.result()
        
        if not records:
            raise RuntimeError("No embeddings generated from PDF")
        
        update_status(file_id, "processing", {"pdf_id": file_id, "pdf_source": pdf_source, "step": "writing_embeddings"})
        
        update_global_embeddings(records, rebuild_index=False)
        
        # Move PDF to storage
        stored_pdf_path = os.path.join(STORAGE_PDFS_DIR, f"{file_id}.pdf")
        shutil.move(input_path, stored_pdf_path)
        
        update_status(file_id, "completed", {
            "pages": len(records),
            "pdf_id": file_id,
            "pdf_source": records[0]["pdf_source"] if records else pdf_source,
            "index_status": "pending"
        })
        _rebuild_index_async(file_id, records[0]["pdf_source"] if records else pdf_source)
        print(f"Processing completed for {file_id}")

    except Exception as e:
        print(f"Background processing failed for {file_id}: {e}")
        update_status(file_id, "failed", {"pdf_id": file_id, "pdf_source": Path(input_path).name, "error": str(e)})
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

@router.post("/pdfs")
async def process_pdf(request: Request, background_tasks: BackgroundTasks, file: List[UploadFile] = File(...)):
    """
    Upload one or more PDF files and start processing in the background.
    Returns either a single object (when one file is uploaded) or a list wrapper.
    """
    if not file:
        raise HTTPException(status_code=400, detail="At least one PDF file is required.")

    executor = _ensure_executor(request.app)
    results = []
    any_processing = False

    for upload in file:
        if not upload.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="All uploaded files must be PDF files.")

        temp_dir = tempfile.mkdtemp()
        try:
            input_path = os.path.join(temp_dir, upload.filename)

            sha256_hash = hashlib.sha256()
            with open(input_path, "wb") as buffer:
                while True:
                    chunk = await upload.read(64 * 1024)  # 64KB chunks
                    if not chunk:
                        break
                    sha256_hash.update(chunk)
                    buffer.write(chunk)

            file_id = sha256_hash.hexdigest()
            pdf_id = file_id
            pdf_source = upload.filename

            status = get_status(file_id)
            if status and status["status"] == "completed":
                cached_source = (status.get("details") or {}).get("pdf_source") or (status.get("details") or {}).get("pdf_name") or pdf_source
                results.append({
                    "id": file_id,
                    "pdf_id": pdf_id,
                    "pdf_source": cached_source,
                    "message": "File retrieved from cache.",
                    "status": "completed",
                    "status_url": f"/api/pdfs/{file_id}/status",
                    "search_url": f"/api/pdfs/search"
                })
                shutil.rmtree(temp_dir)
                continue

            if status and status["status"] == "processing":
                cached_source = (status.get("details") or {}).get("pdf_source") or pdf_source
                results.append({
                    "id": file_id,
                    "pdf_id": pdf_id,
                    "pdf_source": cached_source,
                    "message": "File is already being processed.",
                    "status": "processing",
                    "status_url": f"/api/pdfs/{file_id}/status"
                })
                shutil.rmtree(temp_dir)
                continue

            background_tasks.add_task(run_processing_pipeline, file_id, input_path, temp_dir, executor)
            any_processing = True
            results.append({
                "id": file_id,
                "pdf_id": pdf_id,
                "pdf_source": pdf_source,
                "message": "File accepted for processing.",
                "status": "processing",
                "status_url": f"/api/pdfs/{file_id}/status"
            })

        except Exception as e:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            print(f"Upload error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    if len(results) == 1:
        single = results[0]
        code = 202 if single.get("status") == "processing" else 200
        return JSONResponse(status_code=code, content=single)

    return JSONResponse(
        status_code=202 if any_processing else 200,
        content={
            "total_files": len(results),
            "results": results
        }
    )

@router.get("/pdfs", response_model=PdfListResponse)
async def list_pdfs(include_metadata: bool = True, include_status: bool = True):
    pdf_dir = Path(STORAGE_PDFS_DIR)
    if not pdf_dir.exists():
        return {"count": 0, "pdfs": []}

    pdf_paths = sorted(
        (p for p in pdf_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    id_to_source: dict[str, str] = {}
    if include_metadata and os.path.exists(GLOBAL_EMBEDDINGS_FILE):
        try:
            pd = _pd()
            df = pd.read_parquet(GLOBAL_EMBEDDINGS_FILE, columns=["pdf_id", "pdf_source"])
            df = df.dropna(subset=["pdf_id"])
            for row in df.itertuples(index=False):
                pid = str(getattr(row, "pdf_id"))
                src = getattr(row, "pdf_source", None)
                if src is None:
                    continue
                if pid not in id_to_source:
                    id_to_source[pid] = str(src)
        except Exception:
            id_to_source = {}

    items: list[PdfListItem] = []
    for path in pdf_paths:
        stat = path.stat()
        pdf_id = path.stem
        pdf_source = id_to_source.get(pdf_id)

        status_value = None
        status_url = None
        if include_status:
            status = get_status(pdf_id)
            if status:
                status_value = status.get("status")
                status_url = f"/api/pdfs/{pdf_id}/status"

        items.append(
            PdfListItem(
                pdf_id=pdf_id,
                filename=path.name,
                size_bytes=int(stat.st_size),
                modified_at=float(stat.st_mtime),
                pdf_source=pdf_source,
                status=status_value,
                status_url=status_url,
                view_url=f"/api/pdfs/view?pdf_id={pdf_id}",
                content_url=f"/api/pdfs/{pdf_id}/content",
            )
        )

    return {"count": len(items), "pdfs": items}

@router.get("/pdfs/{pdf_id}/content")
async def get_pdf_content(pdf_id: str):
    if pdf_id != os.path.basename(pdf_id) or ".." in pdf_id or "/" in pdf_id or "\\" in pdf_id:
        raise HTTPException(status_code=400, detail="Invalid pdf_id.")

    file_path = os.path.join(STORAGE_PDFS_DIR, f"{pdf_id}.pdf")
    if not os.path.exists(file_path):
        resolved_id, file_path = _resolve_pdf_storage_path(pdf_id=pdf_id, pdf_source=None)
        pdf_id = resolved_id

    filename = f"{pdf_id}.pdf"
    if os.path.exists(GLOBAL_EMBEDDINGS_FILE):
        try:
            pd = _pd()
            df = pd.read_parquet(GLOBAL_EMBEDDINGS_FILE, columns=["pdf_id", "pdf_source"])
            match = df[df["pdf_id"] == pdf_id]
            if not match.empty:
                candidate = str(match.iloc[0]["pdf_source"])
                if candidate:
                    filename = os.path.basename(candidate).replace('"', "")
        except Exception:
            pass

    headers = {"Content-Disposition": _content_disposition_inline(filename)}
    return FileResponse(path=file_path, media_type="application/pdf", headers=headers)

@router.get("/pdfs/{file_id}/status")
async def get_processing_status(file_id: str):
    """Get the processing status of a PDF file."""
    status = get_status(file_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    details = status.get("details") or {}
    response = {
        "id": file_id,
        "pdf_id": details.get("pdf_id") or file_id,
        "pdf_source": details.get("pdf_source") or details.get("pdf_name"),
        "status": status["status"],
        "details": details
    }
    
    if status["status"] == "completed":
        response.update({
            "search_url": f"/api/pdfs/search"
        })
        
    return response

def _sha256_file(file_path: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

def _ensure_global_search_index(building_codes_parquet_path: str, building_codes_pdf_path: str):
    if os.path.exists(GLOBAL_EMBEDDINGS_FILE):
        if not os.path.exists(GLOBAL_FAISS_INDEX):
            rebuild_faiss_index()
        return

    if not os.path.exists(building_codes_parquet_path):
        raise HTTPException(status_code=404, detail="No global embeddings found and no building codes embeddings available.")

    pd = _pd()
    df = pd.read_parquet(building_codes_parquet_path)

    pdf_path = building_codes_pdf_path
    if "full_path" in df.columns and len(df) > 0:
        candidate = str(df.iloc[0]["full_path"])
        if os.path.exists(candidate):
            pdf_path = candidate

    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="Building codes PDF not found.")

    pdf_id = _sha256_file(pdf_path)
    df["pdf_id"] = pdf_id

    os.makedirs(STORAGE_EMBEDDINGS_DIR, exist_ok=True)
    os.makedirs(STORAGE_EMBEDDINGS_DIR, exist_ok=True)
    df.to_parquet(GLOBAL_EMBEDDINGS_FILE, compression="gzip")
    if GLOBAL_EMBEDDINGS_FILE in _DF_CACHE: del _DF_CACHE[GLOBAL_EMBEDDINGS_FILE]
    
    rebuild_faiss_index()

    storage_path = os.path.join(STORAGE_PDFS_DIR, f"{pdf_id}.pdf")
    if not os.path.exists(storage_path):
        shutil.copyfile(pdf_path, storage_path)

@router.get("/pdfs/search")
async def search_pdfs(
    query: str,
    top_k: int = 7,
    top_p: float = 0.95,
    include_images: bool = True,
    image_max_size: int = 1024
):
    """
    Search indexed PDFs using a text query.
    
    Args:
        query: Text query for searching building codes (required query parameter)
        top_k: Number of top results to return (default: 7)
        top_p: Top-p filtering threshold (default: 0.95)
        
    Returns:
        JSON with matching page indices and metadata
        
    Example:
        GET /api/pdfs/search?query=屋顶光伏安装的要求&top_k=7&top_p=0.95
    """
    building_codes_parquet_path = os.path.join(project_root, "building_codes_embeddings.parquet")
    building_codes_pdf_path = os.path.join(project_root, "BuildingCodes", "2021_International_Building_Code.pdf")
    
    if not query or not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query parameter is required and cannot be empty."
        )
    
    try:
        np = _np()
        pd = _pd()
        faiss = _faiss()
        _ensure_global_search_index(building_codes_parquet_path, building_codes_pdf_path)

        if not os.path.exists(GLOBAL_EMBEDDINGS_FILE) or not os.path.exists(GLOBAL_FAISS_INDEX):
            raise HTTPException(status_code=404, detail="Search index not found.")

        model_config = get_model()
        
        # Load FAISS index
        if GLOBAL_FAISS_INDEX not in _FAISS_INDEX_CACHE:
            _FAISS_INDEX_CACHE[GLOBAL_FAISS_INDEX] = faiss.read_index(GLOBAL_FAISS_INDEX)
        index = _FAISS_INDEX_CACHE[GLOBAL_FAISS_INDEX]
        
        if GLOBAL_EMBEDDINGS_FILE not in _DF_CACHE:
             _DF_CACHE[GLOBAL_EMBEDDINGS_FILE] = pd.read_parquet(GLOBAL_EMBEDDINGS_FILE)
             
        df = _DF_CACHE[GLOBAL_EMBEDDINGS_FILE]
        # We need specific columns but cached DF has all. That's fine.
        # df = pd.read_parquet(GLOBAL_EMBEDDINGS_FILE, columns=["pdf_id", "pdf_source", "full_path", "page_number"])
        
        query_emb = _encode_query_text(query)
        
        # Search with top-p filtering
        search_k = top_k * 4 if top_p < 1.0 else top_k
        scores, indices = index.search(query_emb, min(search_k, index.ntotal))
        scores = scores.flatten()
        indices = indices.flatten()
        
        # Optional: top-p filtering on scores
        if top_p < 1.0:
            # Softmax to probabilities
            probs = np.exp(scores - scores.max())
            probs /= probs.sum()
            sorted_idx = np.argsort(probs)[::-1]
            cum_probs = np.cumsum(probs[sorted_idx])
            keep = cum_probs <= top_p
            if not keep.any():
                keep[0] = True
            indices = indices[sorted_idx][keep]
            scores = scores[sorted_idx][keep]
        
        # Get top-k results
        final_indices = indices[:top_k]
        final_scores = scores[:top_k]
        
        # Build results with metadata
        results = []
        for idx, score in zip(final_indices, final_scores):
            if idx < len(df):
                row = df.iloc[int(idx)]
                results.append({
                    "page_index": int(idx),
                    "page_number": int(row['page_number']),
                    "pdf_source": str(row['pdf_source']),
                    "pdf_id": str(row["pdf_id"]) if not pd.isna(row["pdf_id"]) else None,
                    "similarity_score": float(score)
                })

        for item in results:
            if item.get("pdf_id") is None:
                item["pdf_view_url"] = None
                item["page_view_url"] = None
                continue
            item["pdf_view_url"] = f"/api/pdfs/view?pdf_id={item['pdf_id']}"
            item["page_view_url"] = f"/api/pdfs/view?pdf_id={item['pdf_id']}#page={int(item['page_number'])}"

        if include_images:
            grouped: dict[str, list[tuple[int, int]]] = {}
            for i, item in enumerate(results):
                pid = item.get("pdf_id")
                if not pid:
                    continue
                page_idx0 = int(item["page_number"]) - 1
                if page_idx0 < 0:
                    raise HTTPException(status_code=500, detail="Invalid page numbers found in search results.")
                grouped.setdefault(pid, []).append((i, page_idx0))

            for pid, items in grouped.items():
                storage_path = os.path.join(STORAGE_PDFS_DIR, f"{pid}.pdf")
                if not os.path.exists(storage_path):
                    match = df[df["pdf_id"] == pid]
                    if match.empty:
                        raise HTTPException(status_code=404, detail="PDF not found for one or more results.")
                    full_path = str(match.iloc[0]["full_path"])
                    if not os.path.exists(full_path):
                        raise HTTPException(status_code=404, detail="PDF not found for one or more results.")
                    shutil.copyfile(full_path, storage_path)

                page_indices = [page_idx0 for _, page_idx0 in items]
                b64s = pdf_pages_to_base64_pngs(
                    storage_path,
                    page_indices,
                    max_size=(image_max_size, image_max_size)
                )

                for (result_pos, _), img_b64 in zip(items, b64s):
                    img = base64_to_pillow(img_b64)
                    if img is None:
                        raise HTTPException(status_code=500, detail="Failed to convert one or more pages to images.")
                    results[result_pos]["image_base64"] = img_b64
                    results[result_pos]["width"] = img.width
                    results[result_pos]["height"] = img.height

        unique_pdf_ids = []
        for item in results:
            pid = item.get("pdf_id")
            if pid and pid not in unique_pdf_ids:
                unique_pdf_ids.append(pid)

        pdfs = []
        for pid in unique_pdf_ids:
            match = df[df["pdf_id"] == pid]
            if match.empty:
                continue
            pdfs.append({
                "pdf_id": pid,
                "pdf_source": str(match.iloc[0]["pdf_source"]),
                "pdf_storage_path": os.path.join(STORAGE_PDFS_DIR, f"{pid}.pdf")
            })

        pdf_id = unique_pdf_ids[0] if len(unique_pdf_ids) == 1 else None
        pdf_path = None
        stored_pdf_path = None
        if pdf_id:
            match = df[df["pdf_id"] == pdf_id]
            if not match.empty:
                pdf_path = str(match.iloc[0]["full_path"])
                stored_pdf_path = os.path.join(STORAGE_PDFS_DIR, f"{pdf_id}.pdf")
        
        return JSONResponse(content={
            "query": query,
            "total_results": len(results),
            "results": results,
            "pdf_id": pdf_id,
            "pdf_path": pdf_path if pdf_path and os.path.exists(pdf_path) else None,
            "pdf_storage_path": stored_pdf_path if stored_pdf_path and os.path.exists(stored_pdf_path) else None,
            "pdfs": pdfs
        })
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

def _resolve_pdf_storage_path(
    pdf_id: Optional[str],
    pdf_source: Optional[str]
) -> tuple[str, str]:
    if pdf_id:
        storage_path = os.path.join(STORAGE_PDFS_DIR, f"{pdf_id}.pdf")
        if os.path.exists(storage_path):
            return pdf_id, storage_path

        if os.path.exists(GLOBAL_EMBEDDINGS_FILE):
            pd = _pd()
            df = pd.read_parquet(GLOBAL_EMBEDDINGS_FILE, columns=["pdf_id", "full_path"])
            match = df[df["pdf_id"] == pdf_id]
            if not match.empty:
                full_path = str(match.iloc[0]["full_path"])
                if os.path.exists(full_path):
                    shutil.copyfile(full_path, storage_path)
                    return pdf_id, storage_path

        raise HTTPException(status_code=404, detail="PDF not found")

    if pdf_source:
        if os.path.exists(GLOBAL_EMBEDDINGS_FILE):
            pd = _pd()
            df = pd.read_parquet(GLOBAL_EMBEDDINGS_FILE, columns=["pdf_id", "pdf_source", "full_path"])
            match = df[df["pdf_source"] == pdf_source]
            if match.empty:
                raise HTTPException(status_code=404, detail="PDF not found")

            unique_ids = match["pdf_id"].dropna().unique().tolist()
            if len(unique_ids) != 1:
                raise HTTPException(status_code=409, detail="Multiple PDFs match pdf_source; use pdf_id instead")

            resolved_id = str(unique_ids[0])
            storage_path = os.path.join(STORAGE_PDFS_DIR, f"{resolved_id}.pdf")
            if os.path.exists(storage_path):
                return resolved_id, storage_path

            full_path = str(match.iloc[0]["full_path"])
            if os.path.exists(full_path):
                shutil.copyfile(full_path, storage_path)
                return resolved_id, storage_path

        raise HTTPException(status_code=404, detail="PDF not found")

    pdf_files = [
        f for f in os.listdir(STORAGE_PDFS_DIR)
        if f.lower().endswith(".pdf") and os.path.isfile(os.path.join(STORAGE_PDFS_DIR, f))
    ]
    if len(pdf_files) == 1:
        resolved_id = os.path.splitext(pdf_files[0])[0]
        storage_path = os.path.join(STORAGE_PDFS_DIR, pdf_files[0])
        return resolved_id, storage_path

    raise HTTPException(status_code=400, detail="Provide pdf_id or pdf_source")

def _remove_pdf_from_embeddings(pdf_id: str):
    try:
        pd = _pd()
        if not os.path.exists(GLOBAL_EMBEDDINGS_FILE):
            return False
        df = pd.read_parquet(GLOBAL_EMBEDDINGS_FILE)
        if "pdf_id" not in df.columns:
            return False
        new_df = df[df["pdf_id"] != pdf_id]
        if len(new_df) == 0:
            try:
                os.remove(GLOBAL_EMBEDDINGS_FILE)
            except OSError:
                pass
            if os.path.exists(GLOBAL_FAISS_INDEX):
                try:
                    os.remove(GLOBAL_FAISS_INDEX)
                except OSError:
                    pass
            if GLOBAL_EMBEDDINGS_FILE in _DF_CACHE:
                del _DF_CACHE[GLOBAL_EMBEDDINGS_FILE]
            if GLOBAL_FAISS_INDEX in _FAISS_INDEX_CACHE:
                del _FAISS_INDEX_CACHE[GLOBAL_FAISS_INDEX]
            return True
        new_df.to_parquet(GLOBAL_EMBEDDINGS_FILE, compression="gzip")
        if GLOBAL_EMBEDDINGS_FILE in _DF_CACHE:
            del _DF_CACHE[GLOBAL_EMBEDDINGS_FILE]
        rebuild_faiss_index()
        return True
    except Exception:
        return False

@router.delete("/pdfs/{pdf_id}")
async def delete_pdf(pdf_id: str):
    if pdf_id != os.path.basename(pdf_id) or ".." in pdf_id or "/" in pdf_id or "\\" in pdf_id:
        raise HTTPException(status_code=400, detail="Invalid pdf_id.")
    status = get_status(pdf_id)
    if status and status.get("status") == "processing":
        raise HTTPException(status_code=409, detail="Cannot delete while processing")
    file_path = os.path.join(STORAGE_PDFS_DIR, f"{pdf_id}.pdf")
    status_path = os.path.join(STORAGE_STATUS_DIR, f"pdf_{pdf_id}.json")
    deleted = []
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            deleted.append(file_path)
        except OSError:
            pass
    if os.path.exists(status_path):
        try:
            os.remove(status_path)
            deleted.append(status_path)
        except OSError:
            pass
    _remove_pdf_from_embeddings(pdf_id)
    return {"pdf_id": pdf_id, "deleted_count": len(deleted), "deleted": deleted}


@router.get("/pdfs/view")
async def view_pdf(
    pdf_id: Optional[str] = None,
    pdf_source: Optional[str] = None
):
    building_codes_parquet_path = os.path.join(project_root, "building_codes_embeddings.parquet")
    building_codes_pdf_path = os.path.join(project_root, "BuildingCodes", "2021_International_Building_Code.pdf")
    _ensure_global_search_index(building_codes_parquet_path, building_codes_pdf_path)

    resolved_id, file_path = _resolve_pdf_storage_path(pdf_id=pdf_id, pdf_source=pdf_source)

    with open(file_path, "rb") as f:
        pdf_bytes = f.read()

    pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")

    resolved_pdf_source = None
    if os.path.exists(GLOBAL_EMBEDDINGS_FILE):
        pd = _pd()
        df = pd.read_parquet(GLOBAL_EMBEDDINGS_FILE, columns=["pdf_id", "pdf_source"])
        match = df[df["pdf_id"] == resolved_id]
        if not match.empty:
            resolved_pdf_source = str(match.iloc[0]["pdf_source"])

    return JSONResponse(content={
        "pdf_id": resolved_id,
        "pdf_source": resolved_pdf_source,
        "mime_type": "application/pdf",
        "size_bytes": len(pdf_bytes),
        "pdf_base64": pdf_b64
    })

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
