import math
from functools import lru_cache
import pandas as pd
import numpy as np
import json
import os
_mpl_cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".matplotlib"))
os.makedirs(_mpl_cache_dir, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpl_cache_dir)
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tqdm import tqdm
from io import BytesIO
import fitz
import base64
from PIL import Image
import concurrent.futures
import multiprocessing
import ezdxf
from shapely.geometry import LineString, Polygon, Point
from ezdxf.tools.text import plain_text
from shapely.ops import unary_union
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
from src.dxf_extraction import DXFExtractor, DXFEntity

def sanitize_surrogates(text: str) -> str:
    """Remove invalid Unicode surrogate characters that cause encoding errors."""
    if not isinstance(text, str):
        return str(text)
    return text.encode('utf-8', errors='ignore').decode('utf-8')

def coverage_ratio(roi_img, layout_img):
    """
    Computes area coverage of ROI relative to layout image.
    Assumes both are PIL Images.
    """
    roi_area = roi_img.width * roi_img.height
    layout_area = layout_img.width * layout_img.height

    return roi_area / layout_area

def load_transform(transform_path):
    with open(transform_path, 'r') as f:
        return json.load(f)

def transform_polygon_to_pixels(polygon_coords, transform):
    """
    Converts a list of [x, y] world coordinates to pixel coordinates.
    """
    coords = np.array(polygon_coords)
    
    # 1. Shift by Min X
    pixel_x = (coords[:, 0] - transform['min_x']) * transform['pixel_per_unit']
    
    # 2. Shift by Max Y and FLIP (because image Y=0 is top, DXF Y=0 is bottom)
    pixel_y = (transform['max_y'] - coords[:, 1]) * transform['pixel_per_unit']
    
    # 3. Stack and Round
    return np.column_stack((pixel_x, pixel_y)).astype(int)

# Alias for backward compatibility
DXFEntityMeta = DXFEntity

class DXFLayerMeta(BaseModel):
    layer: str = Field(...)
    color: Optional[int] = Field(None)
    linetype: Optional[str] = Field(None)
    lineweight: Optional[int] = Field(None)
    is_off: bool = Field(False)
    is_frozen: bool = Field(False)
    is_locked: bool = Field(False)
    entity_count: int = Field(0)
    entity_types: Dict[str, int] = Field(default_factory=dict)
    entities: List[DXFEntity] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

class DXFMetadata(BaseModel):
    dxf_path: str = Field(...)
    version: Optional[str] = Field(None)
    acad_release: Optional[str] = Field(None)
    units: Optional[int] = Field(None)
    total_layers: int = Field(0)
    total_entities: int = Field(0)
    entity_type_counts: Dict[str, int] = Field(default_factory=dict)
    entities: List[DXFEntity] = Field(default_factory=list)
    layers: List[DXFLayerMeta] = Field(default_factory=list)
    header: Dict[str, Any] = Field(default_factory=dict)
    docinfo: Dict[str, Any] = Field(default_factory=dict)
    blocks: List[Dict[str, Any]] = Field(default_factory=list)
    linetypes: List[Dict[str, Any]] = Field(default_factory=list)
    text_styles: List[Dict[str, Any]] = Field(default_factory=list)
    dimstyles: List[Dict[str, Any]] = Field(default_factory=list)
    layouts: List[Dict[str, Any]] = Field(default_factory=list)
    annotations: List[Dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

def _convert_to_metadata(result: Any, dxf_path: str) -> DXFMetadata:
    entities = result.entities
    total_entities = len(entities)
    
    entity_type_counts = {}
    layer_entity_counts = {}
    layer_type_counts = {}
    layer_entities_map = {}
    
    for e in entities:
        etype = e.type
        layer = e.layer
        
        entity_type_counts[etype] = entity_type_counts.get(etype, 0) + 1
        layer_entity_counts[layer] = layer_entity_counts.get(layer, 0) + 1
        
        if layer not in layer_type_counts:
            layer_type_counts[layer] = {}
        layer_type_counts[layer][etype] = layer_type_counts[layer].get(etype, 0) + 1
        
        if layer not in layer_entities_map:
            layer_entities_map[layer] = []
        layer_entities_map[layer].append(e)
        
    layers = []
    for l_info in result.layers:
        name = l_info.get("name")
        if not name: continue
        
        layers.append(DXFLayerMeta(
            layer=name,
            color=l_info.get("color"),
            linetype=l_info.get("linetype"),
            lineweight=l_info.get("lineweight"),
            is_off=l_info.get("is_off", False),
            is_frozen=l_info.get("is_frozen", False),
            is_locked=l_info.get("is_locked", False),
            entity_count=layer_entity_counts.get(name, 0),
            entity_types=layer_type_counts.get(name, {}),
            entities=layer_entities_map.get(name, [])
        ))
        
    annotations = []
    for e in entities:
        if e.type in ("TEXT", "MTEXT") and e.text:
            ann = {
                "layer": e.layer,
                "type": e.type,
                "text": e.text
            }
            if e.insert:
                ann["insert"] = e.insert
            annotations.append(ann)

    return DXFMetadata(
        dxf_path=dxf_path,
        version=result.metadata.get("version"),
        units=result.header.get("$INSUNITS"),
        total_layers=len(layers),
        total_entities=total_entities,
        entity_type_counts=dict(sorted(entity_type_counts.items(), key=lambda x: (-x[1], x[0]))),
        entities=entities,
        layers=sorted(layers, key=lambda l: l.layer),
        header=result.header,
        docinfo={}, 
        blocks=result.blocks,
        linetypes=result.linetypes,
        text_styles=result.styles,
        dimstyles=result.dimstyles,
        layouts=[],
        annotations=annotations
    )

def extract_dxf_metadata(dxf_path: str) -> DXFMetadata:
    extractor = DXFExtractor(dxf_path)
    result = extractor.extract_all()
    return _convert_to_metadata(result, dxf_path)

def build_dxf_entity_cache(dxf_path: str) -> DXFExtractor:
    return DXFExtractor(dxf_path)

def extract_dxf_metadata_for_polygons(entity_cache: Union[DXFExtractor, Dict[str, Any]], polygons: List[List[List[float]]]) -> DXFMetadata:
    if isinstance(entity_cache, DXFExtractor):
        extractor = entity_cache
    elif isinstance(entity_cache, dict) and "dxf_path" in entity_cache:
        extractor = DXFExtractor(entity_cache["dxf_path"])
    else:
        # Fallback empty
        return DXFMetadata(dxf_path="unknown")

    result = extractor.extract_within_polygons(polygons)
    return _convert_to_metadata(result, extractor.dxf_path)

# ==========================================
# 2. DATAFRAME CREATION LOGIC
# ==========================================

def create_layout_dataframe(geojson_path, transform_path):
    """
    Parses GeoJSON and Transform file to create a Pandas DataFrame.
    """
    # Load external files
    try:
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
        transform = load_transform(transform_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return pd.DataFrame()

    # Derive DXF filename from the geojson filename (assuming convention)
    # e.g., "chunked_data/Floor plan.geojson" -> "Floor plan.dxf"
    base_name = os.path.splitext(os.path.basename(geojson_path))[0]
    dxf_filename = f"{base_name}.dxf"
    metadata_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage", "metadata", f"{base_name}.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f) or {}
            source_filename = metadata.get("source_filename")
            if source_filename:
                dxf_filename = source_filename
        except Exception:
            pass

    rows = []

    print(f"[INFO] Processing {len(geojson_data['features'])} features...")

    for feature in geojson_data['features']:
        props = feature['properties']
        geom = feature['geometry']

        # Extract IDs
        layout_id = props.get('layout_id')
        chunk_id = props.get('chunk_id')
        try:
            if layout_id is not None:
                layout_id = int(layout_id)
        except Exception:
            pass
        try:
            if chunk_id is not None:
                chunk_id = int(chunk_id)
        except Exception:
            pass

        # Extract and Transform Geometry
        # Note: GeoJSON polygons are usually [[[x,y], [x,y]...]] (list of rings)
        # We take the first ring (exterior)
        if geom and 'coordinates' in geom:
            raw_coords = geom['coordinates'][0]
            
            # Apply the transformation logic to get pixel coordinates
            pixel_poly = transform_polygon_to_pixels(raw_coords, transform)

            rows.append({
                'dxf_file': dxf_filename,
                'layout_id': layout_id,
                'chunk_id': chunk_id,
                'chunks': pixel_poly,  # Storing the numpy array of pixels
                'chunks_dxf': raw_coords # Storing the raw world coordinates
            })

    # Create DataFrame
    df = pd.DataFrame(rows)
    return df

def get_segmentation_crops(
    df, 
    image_source, 
    target_layouts=None, 
    target_chunks=None, 
    padding=50
):
    """
    Queries the DataFrame and returns crops for specific layouts, specific chunks, 
    or a combination of both.
    
    Args:
        df (pd.DataFrame): Dataframe containing 'layout_id', 'chunk_id', and 'chunks'.
        image_source (str or PIL.Image): File path or existing PIL Image object.
        target_layouts (int/list, optional): Filter by Layout ID(s). If None, ignores layout filter.
        target_chunks (int/list, optional): Filter by Chunk ID(s). If None, includes ALL chunks in the target layout.
        padding (int): Padding around the bounding box.

    Returns:
        (PIL.Image, PIL.Image): Tuple (raw_crop, overlay_crop). 
                                Returns (None, None) if no data found.
    """
    
    # --- 1. NORMALIZE INPUTS ---
    # Helper to convert inputs to lists
    def to_list(val):
        if val is None: return None
        if isinstance(val, (list, tuple, np.ndarray)): return list(val)
        return [val]

    target_layouts = to_list(target_layouts)
    target_chunks = to_list(target_chunks)

    # --- 2. BUILD QUERY MASK ---
    # Start with a mask of all True
    mask = pd.Series([True] * len(df), index=df.index)

    if target_layouts is not None:
        mask &= df['layout_id'].isin(target_layouts)
    
    if target_chunks is not None:
        mask &= df['chunk_id'].isin(target_chunks)

    subset = df[mask].copy()

    if subset.empty:
        print(f"[WARN] No data found for Layouts={target_layouts}, Chunks={target_chunks}")
        return None, None

    # --- 3. LOAD IMAGE ---
    # Handle both string path and existing Image object to save IO time
    try:
        if isinstance(image_source, str):
            with Image.open(image_source) as opened:
                base_img = opened.convert("RGBA").copy()
        elif isinstance(image_source, Image.Image):
            base_img = image_source.convert("RGBA")
        else:
            raise ValueError("image_source must be a file path or PIL Image object")
    except Exception as e:
        print(f"[ERROR] Could not load image: {e}")
        return None, None

    # --- 4. CREATE OVERLAY ---
    overlay = Image.new("RGBA", base_img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Generate colors using a colormap
    # We create a unique key based on layout+chunk to assign distinct colors
    unique_ids = subset[['layout_id', 'chunk_id']].drop_duplicates()
    cmap = plt.get_cmap('tab20')
    
    color_map = {}
    for i, (idx, row) in enumerate(unique_ids.iterrows()):
        rgba_float = cmap(i % 20)
        rgb_int = tuple(int(c * 255) for c in rgba_float[:3])
        # Key: (layout_id, chunk_id), Value: (R, G, B, Alpha)
        color_map[(row['layout_id'], row['chunk_id'])] = rgb_int + (128,)

    # --- 5. DRAW & COLLECT COORDINATES ---
    all_points_for_crop = []

    for _, row in subset.iterrows():
        poly_arr = np.array(row['chunks']) # Ensure numpy array
        if poly_arr.size == 0: continue

        # Convert to list of tuples for Pillow
        poly_tuples = [tuple(pt) for pt in poly_arr]
        
        fill_color = color_map.get((row['layout_id'], row['chunk_id']), (0, 255, 0, 128))
        
        draw.polygon(poly_tuples, fill=fill_color, outline="white")
        all_points_for_crop.append(poly_arr)

    # Composite the overlay
    combined_img = Image.alpha_composite(base_img, overlay)

    # --- 6. CALCULATE CROP BOX ---
    if not all_points_for_crop:
        # Fallback if valid rows existed but had empty polygons
        return base_img.convert("RGB"), combined_img.convert("RGB")

    all_points = np.vstack(all_points_for_crop)
    
    min_x, min_y = all_points.min(axis=0)
    max_x, max_y = all_points.max(axis=0)

    width, height = base_img.size
    
    left = max(0, int(min_x) - padding)
    top = max(0, int(min_y) - padding)
    right = min(width, int(max_x) + padding)
    bottom = min(height, int(max_y) + padding)
    
    crop_box = (left, top, right, bottom)

    # --- 7. RETURN CROPS ---
    raw_crop = base_img.crop(crop_box).convert("RGB")
    overlay_crop = combined_img.crop(crop_box).convert("RGB")
    
    return raw_crop, overlay_crop

def pdf_to_base64_pngs(pdf_path, max_size=(1024, 1024)):
    """
    Convert PDF pages to base64 PNG strings.
    
    Note: PyMuPDF (fitz) is NOT thread-safe. On macOS, threading with PyMuPDF
    can cause segmentation faults when combined with PyTorch. This function
    uses single-threaded processing to avoid conflicts.
    """
    import platform
    
    # Open the PDF file
    doc = fitz.open(pdf_path)
    
    def process_page(page_num):
        # Load the page
        page = doc.load_page(page_num)
        
        # Calculate scale to fit max_size while preserving aspect ratio
        rect = page.rect
        scale = min(max_size[0] / rect.width, max_size[1] / rect.height)
        
        # Render the page as a PNG pixmap at the target scale
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        
        # Get PNG bytes directly and encode to base64
        png_bytes = pix.tobytes("png")
        return base64.b64encode(png_bytes).decode('utf-8')
    
    # PyMuPDF is NOT thread-safe - use single-threaded processing to avoid segfaults
    # Especially important on macOS when combined with PyTorch
    base64_encoded_pngs = []
    for page_num in tqdm(range(doc.page_count), desc="Processing pages"):
        base64_encoded_pngs.append(process_page(page_num))
    
    # Close the PDF document
    doc.close()
    
    return base64_encoded_pngs

def pdf_pages_to_base64_pngs(pdf_path, page_indices, max_size=(1024, 1024)):
    """
    Convert specific PDF pages to base64 PNG strings.
    
    Args:
        pdf_path: Path to the PDF file
        page_indices: List of 0-indexed page numbers to process
        max_size: Maximum size for the rendered images
    
    Returns:
        List of base64-encoded PNG strings in the same order as page_indices
    
    Note: Uses single-threaded processing to avoid PyMuPDF threading issues.
    """
    doc = fitz.open(pdf_path)
    
    def process_page(page_num):
        # Load the page
        page = doc.load_page(page_num)
        
        # Calculate scale to fit max_size while preserving aspect ratio
        rect = page.rect
        scale = min(max_size[0] / rect.width, max_size[1] / rect.height)
        
        # Render the page as a PNG pixmap at the target scale
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        
        # Get PNG bytes directly and encode to base64
        png_bytes = pix.tobytes("png")
        return base64.b64encode(png_bytes).decode('utf-8')
    
    # Validate page indices
    if not page_indices:
        doc.close()
        return []
    
    max_page = max(page_indices)
    if max_page >= doc.page_count:
        doc.close()
        raise ValueError(f"Page index {max_page} is out of range. PDF has {doc.page_count} pages (0-indexed).")
    
    # PyMuPDF is NOT thread-safe - use single-threaded processing
    # Process pages in requested order
    base64_encoded_pngs = []
    for idx in page_indices:
        base64_encoded_pngs.append(process_page(idx))
    
    # Close the PDF document
    doc.close()
    
    return base64_encoded_pngs

def base64_to_pillow(base64_string):
    """
    Decodes a Base64 string and converts it into a Pillow Image object.

    Args:
        base64_string (str): The Base64 encoded image string.

    Returns:
        PIL.Image.Image: A Pillow Image object, or None if an error occurs.
    """
    # Remove data URI prefix if it exists (e.g., "data:image/png;base64,")
    if "base64," in base64_string:
        base64_string = base64_string.split("base64,")[1]

    try:
        # Decode the Base64 string into bytes
        image_bytes = base64.b64decode(base64_string)

        # Create an in-memory binary stream from the bytes
        image_stream = BytesIO(image_bytes)

        # Open the image using Pillow
        image = Image.open(image_stream)
        return image

    except Exception as e:
        print(f"Error converting Base64 to Pillow Image: {e}")
        return None

# def get_embedding_model():
#     import torch
#     from sentence_transformers import SentenceTransformer

#     # Set environment variables BEFORE any torch operations
#     os.environ["TRANSFORMERS_OFFLINE"] = "1"
#     os.environ["HF_HUB_OFFLINE"] = "1"
    
#     # Note: Thread settings should be set by the caller before importing torch
#     # Don't set them here to avoid "already set" errors

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     revision_id = "344d954da76eb8ad47a7aaff42d012e30c15b8fe"

#     try:
#         model = SentenceTransformer(
#             "LocalModels/jina-clip-v2-local", 
#             device=device,
#             revision=revision_id,
#             trust_remote_code=True, 
#             truncate_dim=384,
#             local_files_only=True
#         )
#         return model
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         raise

@lru_cache(maxsize=1)
def get_siglip_model_processor():
    import os
    import torch
    from transformers import AutoProcessor, AutoModel

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set environment variables BEFORE any torch operations
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    model_name = "LocalModels/google-siglip-so400m-patch14-384-transformers-default-v1"
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    
    return {"model": model.eval(), "processor": processor}
