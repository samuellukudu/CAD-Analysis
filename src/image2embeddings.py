import os
from functools import lru_cache
from io import BytesIO
import torch
from tqdm import tqdm
from PIL import Image
from src.utils import get_segmentation_crops
from src.utils import get_siglip_model_processor
# from src.utils import get_embedding_model

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
revision_id = "344d954da76eb8ad47a7aaff42d012e30c15b8fe"

# model = get_embedding_model()
# model_config = get_siglip_model_processor()

# model = model_config['model']
# processor = model_config['processor']

def _image_to_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

@lru_cache(maxsize=512)
def _encode_image_bytes(image_bytes: bytes):
    model_config = get_siglip_model_processor()
    model = model_config['model']
    processor = model_config['processor']
    model_device = next(model.parameters()).device

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        output = model.get_image_features(**inputs)
        
        # Handle cases where model returns a ModelOutput object instead of a Tensor
        if not hasattr(output, "detach"):
            if hasattr(output, "pooler_output") and output.pooler_output is not None:
                output = output.pooler_output
            elif hasattr(output, "last_hidden_state"):
                output = output.last_hidden_state
                if len(output.shape) == 3:
                     output = output.mean(dim=1)
            elif isinstance(output, (list, tuple)) and len(output) > 0:
                 if hasattr(output[0], "detach"):
                     output = output[0]

        vec = output.detach().cpu()

        if vec.ndim == 2 and vec.shape[0] == 1:
            vec = vec[0]
        elif vec.ndim == 2 and vec.shape[0] > 1:
            vec = vec.mean(dim=0)
        vec = vec.numpy().astype("float32")
    return vec

def create_embedding_index(geometry_df, full_image_path, INDEX_FILE):
    print("--- 1. Identify Unique Chunks ---")
    # We only need one row per chunk_id to know it exists and which file it belongs to
    unique_chunks = geometry_df[['layout_id', 'chunk_id', 'dxf_file']].drop_duplicates().reset_index(drop=True)
    
    embeddings = []
    valid_indices = []
    
    print(f"--- 2. Processing {len(unique_chunks)} chunks ---")
    image_source = full_image_path
    if isinstance(full_image_path, str):
        try:
            with Image.open(full_image_path) as opened:
                image_source = opened.convert("RGBA").copy()
        except Exception:
            image_source = full_image_path

    for idx, row in tqdm(unique_chunks.iterrows(), total=unique_chunks.shape[0], desc="Encoding"):
        l_id = row['layout_id']
        c_id = row['chunk_id']
        # Get the clean crop
        raw_crop, _ = get_segmentation_crops(geometry_df, image_source, l_id, c_id)
        
        if raw_crop:
            image_bytes = _image_to_bytes(raw_crop)
            vec = _encode_image_bytes(image_bytes)
            embeddings.append(vec)
            valid_indices.append(idx)
        else:
            print(f"Skipping Layout {l_id} Chunk {c_id} (Image load fail)")

    # Filter the unique dataframe to only include successful ones
    final_index_df = unique_chunks.loc[valid_indices].copy()
    final_index_df['embedding'] = embeddings
    
    # SAVE TO PARQUET
    final_index_df.to_parquet(INDEX_FILE)
    print(f"--- Success! Saved {len(final_index_df)} vectors to {INDEX_FILE} ---")

# --- EXECUTE ---
# INDEX_FILE = 'chunk_embeddings.parquet'
# create_embedding_index(df, IMAGE_FILE)
