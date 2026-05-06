import ezdxf
import sys
import argparse
import os
import io
import json
import numpy as np
import cv2
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from ezdxf.addons.drawing import Frontend, RenderContext, pymupdf, layout, config
from ezdxf import bbox
from ezdxf.math import BoundingBox

# ============================================================
# Utility Functions
# ============================================================

def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def get_text_extents(doc):
    """Extract TEXT and MTEXT entity coordinates from DXF document."""
    msp = doc.modelspace()
    text_coords = []
    
    for entity in msp:
        if entity.dxftype() in ['TEXT', 'MTEXT']:
            try:
                if hasattr(entity, 'dxf') and hasattr(entity.dxf, 'insert'):
                    x = entity.dxf.insert.x
                    y = entity.dxf.insert.y
                    text_coords.append((x, y))
            except (AttributeError, ValueError):
                continue
    
    return text_coords if text_coords else None

def calculate_text_specific_transform(text_coords, image_width, image_height, padding_percent=0.05, min_size=10):
    """Calculate transform bounds based only on text coordinates with guards."""
    if not text_coords or len(text_coords) < 2:
        return None
    
    x_coords = [coord[0] for coord in text_coords]
    y_coords = [coord[1] for coord in text_coords]
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    x_range = max_x - min_x
    y_range = max_y - min_y
    
    # Enforce minimum size to prevent collapse
    if x_range < min_size:
        center_x = (min_x + max_x) / 2
        min_x = center_x - min_size / 2
        max_x = center_x + min_size / 2
        x_range = min_size
    if y_range < min_size:
        center_y = (min_y + max_y) / 2
        min_y = center_y - min_size / 2
        max_y = center_y + min_size / 2
        y_range = min_size
    
    if x_range == 0:
        x_range = 1.0
    if y_range == 0:
        y_range = 1.0
    
    x_padding = x_range * padding_percent
    y_padding = y_range * padding_percent
    
    min_x -= x_padding
    max_x += x_padding
    min_y -= y_padding
    max_y += y_padding
    
    x_range = max(max_x - min_x, 1e-6)
    y_range = max(max_y - min_y, 1e-6)
    
    pixel_per_unit_x = image_width / x_range
    pixel_per_unit_y = image_height / y_range
    pixel_per_unit = min(pixel_per_unit_x, pixel_per_unit_y)
    
    scaled_width = x_range * pixel_per_unit
    scaled_height = y_range * pixel_per_unit
    
    x_center_offset = (image_width - scaled_width) / (2 * pixel_per_unit)
    y_center_offset = (image_height - scaled_height) / (2 * pixel_per_unit)
    
    min_x -= x_center_offset
    max_x = min_x + (image_width / pixel_per_unit)
    min_y -= y_center_offset
    max_y = min_y + (image_height / pixel_per_unit)
    
    return {
        'min_x': min_x,
        'max_x': max_x,
        'min_y': min_y,
        'max_y': max_y,
        'pixel_per_unit': pixel_per_unit,
        'image_width': image_width,
        'image_height': image_height
    }

def get_dxf_extents(doc):
    """Compute DXF modelspace bounding box extents."""
    msp = doc.modelspace()
    cache = bbox.Cache()
    extents: BoundingBox = bbox.extents(msp, cache=cache)
    if extents.has_data:
        min_point = extents.extmin
        max_point = extents.extmax
        return min_point.x, min_point.y, max_point.x, max_point.y
    return None

def extract_annotations(dxf_file_path):
    """Load DXF file with fallback encoding."""
    for encoding in ['gb2312', 'utf-8']:
        try:
            return ezdxf.readfile(dxf_file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except ezdxf.DXFStructureError as e:
            print(f"❌ Invalid DXF structure: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None
    return None

def crop_image_whitespace(pil_image):
    """Crop out extra whitespace (RGB threshold)."""
    image = pil_image.convert("RGB")
    image_np = np.array(image)
    mask = np.any(image_np < 250, axis=-1)
    if not np.any(mask):
        return image, (0, 0, image.width, image.height)
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return image.crop((x0, y0, x1, y1)), (x0, y0, x1, y1)

def crop_image_magenta(pil_image, padding=50):
    """Crop an image by preserving magenta-framed regions and removing outside areas."""
    image = pil_image.convert("RGB")
    image_np = np.array(image)
    
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    lower_magenta = np.array([50, 50, 50])
    upper_magenta = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower_magenta, upper_magenta)
    
    if not np.any(mask):
        return image, (0, 0, image.width, image.height)
    
    coords = np.argwhere(mask > 0)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(image.width, x1 + padding)
    y1 = min(image.height, y1 + padding)
    
    return image.crop((x0, y0, x1, y1)), (x0, y0, x1, y1)

# ============================================================
# Coordinate Transformation Functions
# ============================================================

def image_to_dxf_coords(x_i, y_i, transform):
    """Convert image coordinates (x_I, y_I) to DXF coordinates (x_C, y_C)."""
    x_c = transform['s_x'] * x_i + transform['t_x']
    y_c = -transform['s_y'] * y_i + transform['t_y'] + transform['H']
    return x_c, y_c

def dxf_to_image_coords(x_c, y_c, transform):
    """Convert DXF coordinates (x_C, y_C) to image coordinates (x_I, y_I)."""
    x_i = (x_c - transform['t_x']) / transform['s_x']
    y_i = ((transform['t_y'] + transform['H'] - y_c)) / transform['s_y']
    return x_i, y_i

# ============================================================
# Core Conversion
# ============================================================

def export_to_png(doc, output_path="output.png", dpi=960, crop_type="magenta", use_text_bounds=False):
    """Render DXF → PNG and compute transform, adjusting for cropping."""
    msp = doc.modelspace()
    context = RenderContext(doc)
    backend = pymupdf.PyMuPdfBackend()
    cfg = config.Configuration(background_policy=config.BackgroundPolicy.DEFAULT)
    frontend = Frontend(context, backend, config=cfg)
    frontend.draw_layout(msp)

    page = layout.Page(210, 297, layout.Units.mm, margins=layout.Margins.all(0))
    settings = layout.Settings(
        fit_page=True,
        page_alignment=layout.PageAlignment.TOP_LEFT,
        crop_at_margins=False
    )

    # Ignore INSUNITS for unit-agnostic transformation (treat DXF units as arbitrary, no conversion to mm)
    settings.scale = 1.0

    png_bytes = backend.get_pixmap_bytes(page, fmt="png", dpi=dpi, settings=settings)
    pil_image = Image.open(io.BytesIO(png_bytes))

    transform = None

    if use_text_bounds:
        text_coords = get_text_extents(doc)
        if text_coords and len(text_coords) > 3:
            transform = calculate_text_specific_transform(
                text_coords, pil_image.width, pil_image.height
            )
        else:
            print(f"⚠️ Text bounds invalid or too few entities found, falling back to global bounds")
            use_text_bounds = False

    if not use_text_bounds:
        extents = get_dxf_extents(doc)
        if not extents:
            return None, None
        min_x, min_y, max_x, max_y = extents
        size_x = max(max_x - min_x, 1e-6)
        size_y = max(max_y - min_y, 1e-6)
        scaled_size_x = size_x * settings.scale
        scaled_size_y = size_y * settings.scale
        effective_width = page.width
        effective_height = page.height
        fit_scale = min(effective_width / scaled_size_x, effective_height / scaled_size_y)
        total_scale = settings.scale * fit_scale
        pixel_per_unit = total_scale * (dpi / 25.4)
        transform = {
            'min_x': min_x,
            'min_y': min_y,
            'max_x': max_x,
            'max_y': max_y,
            'pixel_per_unit': pixel_per_unit,
            'image_width': pil_image.width,
            'image_height': pil_image.height
        }

    # Apply cropping
    crop_bounds = None
    if crop_type == "magenta":
        pil_image, crop_bounds = crop_image_magenta(pil_image, padding=50)
    elif crop_type == "whitespace":
        pil_image, crop_bounds = crop_image_whitespace(pil_image)
    else:
        crop_bounds = (0, 0, pil_image.width, pil_image.height)

    # Adjust transform for cropped image
    if crop_bounds != (0, 0, pil_image.width, pil_image.height):
        x0, y0, x1, y1 = crop_bounds
        crop_width = x1 - x0
        crop_height = y1 - y0

        orig_pixel_per_unit = transform['pixel_per_unit']
        orig_min_x = transform['min_x']
        orig_max_y = transform['max_y']

        crop_min_x = orig_min_x + (x0 / orig_pixel_per_unit)
        crop_max_x = orig_min_x + (x1 / orig_pixel_per_unit)
        
        # Y mapping: image top (y=0) -> DXF top (high y), image bottom (y=height) -> DXF bottom (low y)
        crop_max_y = orig_max_y - (y0 / orig_pixel_per_unit)
        crop_min_y = orig_max_y - (y1 / orig_pixel_per_unit)

        # Enforce valid bounds
        crop_max_x = max(crop_max_x, crop_min_x + 1e-6)
        crop_max_y = max(crop_max_y, crop_min_y + 1e-6)

        transform = {
            'min_x': crop_min_x,
            'max_x': crop_max_x,
            'min_y': crop_min_y,
            'max_y': crop_max_y,
            'pixel_per_unit': orig_pixel_per_unit,  # ppu remains the same after cropping
            'image_width': crop_width,
            'image_height': crop_height
        }

    # Compute affine parameters: s_x, s_y = units per pixel = 1 / ppu
    ppu = transform['pixel_per_unit']
    s_x = 1.0 / ppu
    s_y = 1.0 / ppu  # Isotropic
    t_x = transform['min_x']
    t_y = transform['min_y']
    H = transform['max_y'] - transform['min_y']
    
    transform.update({
        's_x': float(s_x),
        's_y': float(s_y),
        't_x': float(t_x),
        't_y': float(t_y),
        'H': float(H)
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pil_image.save(output_path)

    dx = transform['max_x'] - transform['min_x']
    dy = transform['max_y'] - transform['min_y']
    if dx < 1e-3 or dy < 1e-3:
        print(f"WARNING: Collapsed bounds in {output_path}: dx={dx}, dy={dy}")

    transform_path = os.path.splitext(output_path)[0] + '_transform.json'
    with open(transform_path, 'w') as f:
        json.dump(convert_numpy_types(transform), f, indent=4)

    return output_path, transform_path

def process_single_file(dxf_path, output_dir, dpi=960, crop_type="magenta", use_text_bounds=False):
    """Convert one DXF file."""
    base_name = os.path.splitext(os.path.basename(dxf_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.png")
    try:
        doc = extract_annotations(dxf_path)
        if not doc:
            return dxf_path, "Load failed"
        return export_to_png(doc, output_path=output_path, dpi=dpi, crop_type=crop_type, use_text_bounds=use_text_bounds)
    except Exception as e:
        return dxf_path, str(e)

# ============================================================
# Batch Processing
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert DXF file(s) to PNG images with coordinate transforms."
    )
    parser.add_argument("input", help="Path to DXF file or directory")
    parser.add_argument("--dpi", type=int, default=960, help="Output DPI (default: 960)")
    parser.add_argument("--crop-type", choices=["magenta", "whitespace", "none"], default="magenta",
                        help="Crop method: magenta frame, whitespace, or none (default: magenta)")
    parser.add_argument("--text-bounds", action="store_true",
                        help="Use text-specific bounds for better text scaling (with fallback to global if invalid)")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel workers (default: 4)")
    args = parser.parse_args()

    input_path = args.input
    output_dir = os.path.join(os.path.dirname(__file__), "images")
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isdir(input_path):
        dxf_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith('.dxf')]
    else:
        dxf_files = [input_path]

    if not dxf_files:
        print("⚠️ No DXF files found.")
        return

    print(f"📂 Found {len(dxf_files)} DXF file(s). Converting...")

    results = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_single_file, path, output_dir, args.dpi, args.crop_type, args.text_bounds): path for path in dxf_files}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing DXFs"):
            dxf_path = futures[f]
            try:
                path, status = f.result()
                results.append((path, status))
            except Exception as e:
                results.append((dxf_path, f"Error: {e}"))

    print("\n✅ Conversion Summary:")
    for path, status in results:
        print(f" - {os.path.basename(path)}: {status}")

if __name__ == "__main__":
    main()