# Set environment variables FIRST, before any imports
import os
import platform
from functools import lru_cache

if platform.system() == "Darwin":  # macOS
    # Disable threading to prevent segfaults - must be set before any imports
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Import order matters on macOS: PyMuPDF (fitz) should be imported before PyTorch
# Import utils first (which imports fitz) before importing torch
import faiss
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

# Now import model loading functions
# Don't import pdf_to_base64_pngs here - it imports fitz which can cause issues
# Import it only if needed in __main__
from src.utils import get_siglip_model_processor

# Lazy-load model to avoid segfaults at import time
_model_config = None
_torch_configured = False

def _ensure_torch_configured():
    global _torch_configured
    if _torch_configured:
        return
        
    import torch
    # Configure PyTorch threading BEFORE any model loading
    if platform.system() == "Darwin":  # macOS
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    _torch_configured = True

def get_model():
    global _model_config
    if _model_config is None:
        _ensure_torch_configured()
        _model_config = get_siglip_model_processor()
    return _model_config


@lru_cache(maxsize=512)
def _encode_query_text(query: str) -> np.ndarray:
    model_config = get_model()
    model = model_config["model"]
    processor = model_config["processor"]

    import torch
    device = next(model.parameters()).device
    inputs = processor(text=[query], padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        if hasattr(model, "get_text_features"):
            query_emb = model.get_text_features(**inputs)
        else:
            raise RuntimeError("Loaded model does not support text embeddings (get_text_features missing).")

        # Handle cases where model returns a ModelOutput object instead of a Tensor
        if not hasattr(query_emb, "detach"):
            if hasattr(query_emb, "pooler_output") and query_emb.pooler_output is not None:
                query_emb = query_emb.pooler_output
            elif hasattr(query_emb, "last_hidden_state"):
                query_emb = query_emb.last_hidden_state
                if len(query_emb.shape) == 3:
                     query_emb = query_emb.mean(dim=1)
            elif isinstance(query_emb, (list, tuple)) and len(query_emb) > 0:
                 if hasattr(query_emb[0], "detach"):
                     query_emb = query_emb[0]

    query_emb = query_emb.detach().cpu().numpy().astype("float32")

    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)

    faiss.normalize_L2(query_emb)
    return query_emb

# Define project root and absolute paths for embeddings
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FAISS_INDEX = os.path.join(project_root, "storage", "embeddings", "faiss_index.faiss")
DEFAULT_PARQUET_PATH = os.path.join(project_root, "storage", "embeddings", "all_embeddings.parquet")

# Cache for index and data
_FAISS_INDEX_CACHE = {}
_DF_CACHE = {}

def get_cached_index(path: str):
    if path not in _FAISS_INDEX_CACHE:
        if os.path.exists(path):
            _FAISS_INDEX_CACHE[path] = faiss.read_index(path)
        else:
            return None
    return _FAISS_INDEX_CACHE[path]

def get_cached_df(path: str, columns: Optional[List[str]] = None):
    # Tuple key for cache to handle different column requests? 
    # Or just cache full DF? Caching full DF is safer but memory heavy if we don't need it.
    # Given usage, we usually read same file. Let's just cache the full DF for simplicity/speed 
    # or rely on FS cache? No, parsing parquet is slow.
    # Let's map path -> df. If columns requested, we can slice.
    
    if path not in _DF_CACHE:
        if os.path.exists(path):
             _DF_CACHE[path] = pd.read_parquet(path)
        else:
            return None
    
    df = _DF_CACHE[path]
    if df is not None and columns:
        # Check if all columns exist
        existing_cols = [c for c in columns if c in df.columns]
        return df[existing_cols]
    return df

def retrieve_pages_with_metadata(
    query: str,
    faiss_index_path: str = DEFAULT_FAISS_INDEX,
    parquet_path: str = DEFAULT_PARQUET_PATH,
    top_k: int = 7,
    top_p: float = 0.95
) -> List[Dict[str, Any]]:
    """
    Retrieve top matching pages with full metadata (pdf_id, page_number, source).
    Defaults to the global storage index.
    """
    if not os.path.exists(faiss_index_path) or not os.path.exists(parquet_path):
        print(f"Warning: Index files not found at {faiss_index_path} or {parquet_path}")
        return []

    # Load index once or cache it
    index = get_cached_index(faiss_index_path)
    if index is None: return [] # should be caught by exists check ideally
    
    # Load metadata
    columns = ['pdf_id', 'page_number', 'pdf_source', 'full_path']
    df = get_cached_df(parquet_path, columns=columns)
    if df is None: return []
    
    query_emb = _encode_query_text(query)
    
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
    
    final_indices = indices[:top_k]
    final_scores = scores[:top_k]
    
    results = []
    for idx, score in zip(final_indices, final_scores):
        if idx < len(df):
            row = df.iloc[int(idx)]
            results.append({
                "index": int(idx),
                "pdf_id": str(row.get("pdf_id", "")),
                "page_number": int(row.get("page_number", 0)),
                "pdf_source": str(row.get("pdf_source", "")),
                "full_path": str(row.get("full_path", "")),
                "score": float(score)
            })
            
    return results

def retrieve_page_indices(
    query: str,
    faiss_index_path: str = "building_codes.faiss",
    parquet_path: str = "building_codes_embeddings.parquet",
    top_k: int = 7,
    top_p: float = 0.95
) -> list[int]:
    # Load index once or cache it
    # Load index once or cache it
    index = get_cached_index(faiss_index_path)
    if index is None: return []
    
    # Load only metadata (not embeddings)
    # df = pd.read_parquet(parquet_path, columns=['pdf_source', 'page_number'])
    # We don't actually use df here? Wait, we don't. 
    # Actually retrieve_page_indices doesn't use `df` for lookups, only for existence check?
    # No, the previous code read it but didn't use it except maybe implicitly?
    # Ah, the original code read it: `df = pd.read_parquet(...)` but never used `df` in `retrieve_page_indices`.
    # Wait, I see `results` construction uses `df` in `retrieve_pages_with_metadata`.
    # in `retrieve_page_indices`, it returns `final_indices` which are just integers.
    # So `df` read is superfluous? 
    # Let's keep it to be safe or remove if unused. 
    # It seems unused in the original `retrieve_page_indices` function body provided in context.
    # However, to be minimal change, I will just use cached check or skip.
    pass # df read removed as it was unused in original code snippet logic provided (indices only)
    
    query_emb = _encode_query_text(query)
    
    search_k = top_k * 4 if top_p < 1.0 else top_k
    scores, indices = index.search(query_emb, search_k)
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
    
    final_indices = sorted(indices[:top_k].tolist())
    
    return final_indices

if __name__ == "__main__":
    # Load model FIRST before any PDF processing to avoid threading conflicts on macOS
    # PyTorch/PyMuPDF threading can cause segfaults if initialized in wrong order
    # Lazy load torch here for main execution
    import torch
    print("Loading SigLIP model...")
    print(f"PyTorch threads: {torch.get_num_threads()}, interop: {torch.get_num_interop_threads()}")
    try:
        model_config = get_model()  # Pre-load model to initialize PyTorch before threading operations
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Process PDF AFTER model is loaded (if needed)
    # Uncomment the following lines if you need the PDF base64 data:
    # from src.utils import pdf_to_base64_pngs  # Import only when needed
    # print("Processing PDF pages...")
    # b64s = pdf_to_base64_pngs("BuildingCodes/2021_International_Building_Code.pdf")
    # df_embeds = pd.read_parquet("building_codes_embeddings.parquet")

    # Retrieve top matching page indices
    print("Retrieving page indices...")
    indices = retrieve_page_indices(
        query="屋顶光伏安装的要求",
        top_k=7,
        top_p=0.95
    )

    print(f"Retrieved indices: {indices}")
