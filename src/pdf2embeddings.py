import os
import argparse
import gc
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import List
import torch
import faiss
import numpy as np
from tqdm import tqdm
from PIL import Image
import pandas as pd
from src.utils import base64_to_pillow, pdf_to_base64_pngs
from src.utils import get_siglip_model_processor
# from src.utils import get_embedding_model

# # Offline mode (as in your original code)
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_HUB_OFFLINE"] = "1"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# REVISION_ID = "344d954da76eb8ad47a7aaff42d012e30c15b8fe"  # Your specific revision

# Load the model once globally
# model = get_embedding_model()
# model_config = get_siglip_model_processor()

# model = model_config['model']
# processor = model_config['processor']

def encode_images_in_batches(images: List[Image.Image], batch_size: int = 16) -> np.ndarray:
    """Encode list of PIL images in batches for better GPU utilization."""
    embeddings = []
    for i in tqdm(range(0, len(images), batch_size), desc="Encoding image batches"):
        batch = images[i:i + batch_size]
        batch_embeddings = []
        for image in batch:
            image_bytes = _image_to_bytes(image)
            emb = _encode_image_bytes(image_bytes)
            batch_embeddings.append(emb)
        if batch_embeddings:
            embeddings.append(np.stack(batch_embeddings))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not embeddings:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(embeddings, axis=0).astype("float32", copy=False)

def _image_to_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

@lru_cache(maxsize=512)
def _encode_image_bytes(image_bytes: bytes) -> np.ndarray:
    model_config = get_siglip_model_processor()
    model = model_config['model']
    processor = model_config['processor']
    model_device = next(model.parameters()).device

    image = Image.open(BytesIO(image_bytes)).convert("RGBA")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        
        if not hasattr(emb, "detach"):
            if hasattr(emb, "pooler_output") and emb.pooler_output is not None:
                emb = emb.pooler_output
            elif hasattr(emb, "last_hidden_state"):
                emb = emb.last_hidden_state
                if len(emb.shape) == 3:
                     emb = emb.mean(dim=1)
            elif isinstance(emb, (list, tuple)) and len(emb) > 0:
                 if hasattr(emb[0], "detach"):
                     emb = emb[0]

        emb = emb.detach().cpu().to(torch.float32)
    if emb.ndim == 2 and emb.shape[0] == 1:
        emb = emb[0]
    elif emb.ndim == 2 and emb.shape[0] > 1:
        emb = emb.mean(dim=0)
    return emb.numpy().astype("float32", copy=False)

# Build FAISS index once (after indexing all PDFs)
def build_faiss_index(parquet_path: str, index_path: str = "building_codes.faiss"):
    df = pd.read_parquet(parquet_path)
    embeddings = np.stack(df['embedding'].values).astype('float32')
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Choose index type:
    # For high accuracy + speed on <100k vectors:
    index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner Product = cosine after L2 norm
    
    # For millions of vectors (faster, approximate):
    # nlist = 100
    # quantizer = faiss.IndexFlatIP(dim)
    # index = faiss.IndexIVFFlat(quantizer, dim, nlist)
    # index.train(embeddings)
    
    index.add(embeddings)
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to {index_path}")

def process_pdf(pdf_path: str, output_records: List[dict], batch_size: int):
    """Process a single PDF and append records to the output list."""
    try:
        print(f"\nProcessing: {pdf_path}")
        b64s = pdf_to_base64_pngs(pdf_path)
        
        images = []
        for b64 in b64s:
            img = base64_to_pillow(b64)
            images.append(img)
        
        embeddings = encode_images_in_batches(images, batch_size=batch_size)
        
        pdf_name = Path(pdf_path).name
        for page_num, emb in enumerate(embeddings):
            output_records.append({
                "pdf_source": pdf_name,
                "full_path": str(pdf_path),
                "page_number": page_num + 1,
                "embedding": emb  # Store as numpy array for Parquet compatibility
            })
        
        print(f"Completed {pdf_name}: {len(images)} pages encoded.")
        
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate image embeddings for PDF pages using Jina CLIP v2.")
    parser.add_argument("inputs", nargs="+", help="PDF file paths or directories containing PDFs.")
    parser.add_argument("-o", "--output", default="building_codes_embeddings.parquet",
                        help="Output Parquet file (default: building_codes_embeddings.parquet)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for encoding (default: 16)")
    
    args = parser.parse_args()
    
    pdf_paths = []
    for inp in args.inputs:
        path = Path(inp)
        if path.is_dir():
            pdf_paths.extend(list(path.rglob("*.pdf")))
        elif path.is_file() and path.suffix.lower() == ".pdf":
            pdf_paths.append(path)
        else:
            print(f"Skipping invalid input: {inp}")
    
    if not pdf_paths:
        print("No PDF files found.")
        return
    
    print(f"Found {len(pdf_paths)} PDF files to process.")
    
    records = []
    
    # Process PDFs sequentially to avoid excessive memory usage
    for pdf_path in pdf_paths:
        process_pdf(str(pdf_path), records, batch_size=args.batch_size)
        gc.collect()
    
    # Save to Parquet
    print(f"\nSaving {len(records)} embeddings to {args.output}...")
    df = pd.DataFrame(records)
    df.to_parquet(args.output, compression="gzip")
    
    # Build FAISS index after indexing all PDFs
    build_faiss_index(args.output)
    # print(df.columns)
    print("Done!")

if __name__ == "__main__":
    main()
