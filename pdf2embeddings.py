import os
import argparse
import gc
import base64
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from PIL import Image
import fitz  # PyMuPDF
import pandas as pd
import concurrent.futures
import multiprocessing

# Offline mode (as in your original code)
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
REVISION_ID = "344d954da76eb8ad47a7aaff42d012e30c15b8fe"  # Your specific revision

# Load the model once globally
print("Loading Jina CLIP v2 model...")
model = SentenceTransformer(
    "LocalModels/jina-clip-v2-local",
    device=DEVICE,
    revision=REVISION_ID,
    trust_remote_code=True,
    truncate_dim=384,
    local_files_only=True
)
print("Model loaded.")

def pdf_to_base64_pngs(pdf_path: str, max_size: Tuple[int, int] = (1024, 1024)) -> List[str]:
    """Convert all pages of a PDF to list of base64-encoded PNG strings."""
    doc = fitz.open(pdf_path)
    
    def process_page(page_num: int) -> str:
        page = doc.load_page(page_num)
        rect = page.rect
        scale = min(max_size[0] / rect.width, max_size[1] / rect.height)
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        png_bytes = pix.tobytes("png")
        return base64.b64encode(png_bytes).decode('utf-8')
    
    base64_encoded_pngs = [None] * doc.page_count
    futures_to_page = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for page_num in range(doc.page_count):
            future = executor.submit(process_page, page_num)
            futures_to_page[future] = page_num
        
        for future in tqdm(concurrent.futures.as_completed(futures_to_page),
                           total=doc.page_count,
                           desc=f"Rendering pages [{Path(pdf_path).name}]"):
            page_num = futures_to_page[future]
            base64_encoded_pngs[page_num] = future.result()
    
    doc.close()
    return base64_encoded_pngs

def base64_to_pillow(base64_string: str) -> Image.Image:
    """Convert base64 PNG string to PIL Image."""
    if "base64," in base64_string:
        base64_string = base64_string.split("base64,")[1]
    image_bytes = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_bytes))

def encode_images_in_batches(images: List[Image.Image], batch_size: int = 16) -> torch.Tensor:
    """Encode list of PIL images in batches for better GPU utilization."""
    embeddings = []
    for i in tqdm(range(0, len(images), batch_size), desc="Encoding image batches"):
        batch = images[i:i + batch_size]
        with torch.no_grad():
            emb = model.encode(batch, show_progress_bar=False, convert_to_tensor=True, normalize_embeddings=True)
            embeddings.append(emb.cpu())
        gc.collect()
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
    return torch.cat(embeddings, dim=0)

def process_pdf(pdf_path: str, output_records: List[dict]):
    """Process a single PDF and append records to the output list."""
    try:
        print(f"\nProcessing: {pdf_path}")
        b64s = pdf_to_base64_pngs(pdf_path)
        
        images = []
        for b64 in b64s:
            img = base64_to_pillow(b64)
            images.append(img)
        
        embeddings = encode_images_in_batches(images)
        
        pdf_name = Path(pdf_path).name
        for page_num, emb in enumerate(embeddings):
            output_records.append({
                "pdf_source": pdf_name,
                "full_path": str(pdf_path),
                "page_number": page_num + 1,
                "embedding": emb.numpy()  # Store as numpy array for Parquet compatibility
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
        process_pdf(str(pdf_path), records)
        gc.collect()
    
    # Save to Parquet
    print(f"\nSaving {len(records)} embeddings to {args.output}...")
    df = pd.DataFrame(records)
    df.to_parquet(args.output, compression="gzip")
    print("Done!")

if __name__ == "__main__":
    main()