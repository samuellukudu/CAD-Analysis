import ezdxf
import sys
import argparse
import os
import io
import json
import re
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disable decompression bomb check for large CAD exports
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.font_manager as font_manager

# Tools for text cleaning
from ezdxf.tools.text import plain_text

# Imports for Rendering
from ezdxf.addons.drawing import Frontend, RenderContext, pymupdf, layout, config
from ezdxf.addons.drawing.config import BackgroundPolicy, ColorPolicy, LineweightPolicy

# Add this import near the top (with the other drawing imports)
from ezdxf.addons.drawing import svg          # ← important
from ezdxf.addons.drawing.svg import SVGBackend

from ezdxf import bbox

# --- MONKEY PATCH TO FIX PyMuPDF Layer Issue ---
# The pymupdf backend in ezdxf can crash if a layer name is None or not a string
# when calling the underlying MuPDF C functions. This patch ensures layer names are safe.
if hasattr(pymupdf, 'PyMuPdfRenderBackend'):
    original_get_ocg = pymupdf.PyMuPdfRenderBackend.get_optional_content_group

    def patched_get_optional_content_group(self, layer_name):
        if layer_name is None:
            layer_name = "Default"
        elif not isinstance(layer_name, str):
            layer_name = str(layer_name)
        
        try:
            return original_get_ocg(self, layer_name)
        except Exception:
            # Fallback for any other error (e.g. encoding issues)
            # Try to use a safe default layer
            if layer_name != "SafeLayer":
                try:
                    return original_get_ocg(self, "SafeLayer")
                except Exception:
                    pass
            raise

    pymupdf.PyMuPdfRenderBackend.get_optional_content_group = patched_get_optional_content_group
# -----------------------------------------------------

# ============================================================
# Advanced Text Cleaning & Formatting
# ============================================================

def decode_cad_unicode(text):
    r"""
    Decodes \U+XXXX and strips AutoCAD MTEXT formatting.
    """
    if not text:
        return ""
    
    # 1. Use ezdxf's built-in tool to strip MTEXT formatting (\A1;, \C7;, etc)
    text = plain_text(text)
    
    # 2. Manual Unicode Decoding for \U+XXXX
    def replace_unicode(match):
        hex_code = match.group(1)
        try:
            return chr(int(hex_code, 16))
        except:
            return match.group(0)

    # Regex to find \U+XXXX (Case insensitive)
    text = re.sub(r'\\U\+([0-9A-Fa-f]{4})', replace_unicode, text)
    
    # 3. Clean up newlines for MTEXT
    text = text.replace(r'\P', '\n')
    
    return text

def fix_dxf_entities(doc):
    """
    1. Decodes Unicode text in TEXT, MTEXT, and ATTRIB.
    2. Shrinks text size to prevent overlapping.
    """
    print(f"   --- Deep Cleaning & Resizing Text ---")
    
    count = 0
    if hasattr(doc, 'entitydb'):
        for entity in doc.entitydb.values():
            if entity.dxftype() in ['TEXT', 'MTEXT', 'ATTRIB', 'ATTDEF']:
                
                # --- FIX 1: DECODE TEXT ---
                if hasattr(entity, 'dxf') and hasattr(entity.dxf, 'text'):
                    original = entity.dxf.text
                    if original:
                        cleaned = decode_cad_unicode(original)
                        if cleaned != original:
                            entity.dxf.text = cleaned
                            count += 1
                
                # --- FIX 2: SHRINK TEXT (Prevent Overlap) ---
                # Reduce height by 25% and Width by 20%
                # if hasattr(entity.dxf, 'height'):
                #     entity.dxf.height *= 0.75 
                # if hasattr(entity.dxf, 'width'):
                #     entity.dxf.width *= 0.8 

    print(f"   ✅ Processed & Resized {count} text entities.")

def get_system_cjk_font():
    env_font = os.environ.get("CAD_CJK_FONT")
    if env_font and os.path.exists(env_font):
        return env_font
    
    # Prioritize Simplified Chinese Fonts for macOS
    candidates = [
        # Project Local Fonts (For Deployment)
        os.path.join(os.path.dirname(__file__), "fonts", "NotoSansCJKsc-Regular.otf"),
        os.path.join(os.path.dirname(__file__), "fonts", "NotoSansCJKsc-Medium.otf"),
        
        # User Installed Noto Sans CJK SC (Best open source option - Explicitly Simplified)
        os.path.expanduser("~/Library/Fonts/NotoSansCJKsc-Regular.otf"),
        os.path.expanduser("~/Library/Fonts/NotoSansCJKsc-Medium.otf"),
        "/Library/Fonts/NotoSansCJKsc-Regular.otf",

        # macOS System Fonts (Simplified)
        "/System/Library/Fonts/PingFang SC.ttc", # Specific SC version
        "/System/Library/Fonts/Heiti SC.ttc",
        "/System/Library/Fonts/PingFang.ttc", # Generic
        "/System/Library/Fonts/STHeiti Light.ttc", # Can be ambiguous
        
        # Linux / Server Fonts
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        
        # Fallbacks
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ]

    for path in candidates:
        if os.path.exists(path):
            return path
            
    try:
        fonts = set()
        for ext in ["ttf", "ttc", "otf"]:
            for fp in font_manager.findSystemFonts(fontext=ext):
                fonts.add(fp)
        
        # Keywords prioritized for Simplified Chinese
        keywords = [
            "sc", # Simplified Chinese
            "noto sans cjk sc",
            "notosanscjksc",
            "source han sans sc",
            "pingfang sc",
            "heiti sc",
            "simhei",
            "simsun",
            "msyh",
            "pingfang", # Generic fallback
            "heiti",
            "wenquanyi",
            "uming",
            "ukai",
            "ar pl",
        ]
        
        # Better search strategy: Iterate keywords (priority) and check all fonts
        for k in keywords:
            for path in fonts:
                name = os.path.basename(path).lower().replace("_", " ").replace("-", " ")
                if k in name:
                    return path

    except Exception:
        pass
    return None

def apply_font_fix(doc, font_path):
    """Force all styles to use the detected Chinese font."""
    if not font_path or not os.path.exists(font_path):
        return
    font_name = os.path.basename(font_path)
    if os.path.dirname(font_path) not in ezdxf.options.support_dirs:
        ezdxf.options.support_dirs.append(os.path.dirname(font_path))
    
    for style in doc.styles:
        style.dxf.font = font_name
        style.dxf.bigfont = ""

# ============================================================
# Standard Utilities
# ============================================================

def convert_numpy_types(obj):
    if isinstance(obj, dict): return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list): return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    else: return obj

def get_text_extents(doc):
    msp = doc.modelspace()
    text_coords = []
    for entity in msp:
        if entity.dxftype() in ['TEXT', 'MTEXT']:
            try:
                text_coords.append((entity.dxf.insert.x, entity.dxf.insert.y))
            except: continue
    return text_coords

def calculate_text_specific_transform(text_coords, image_width, image_height, padding_percent=0.05, min_size=10):
    if not text_coords or len(text_coords) < 2: return None
    x_coords, y_coords = zip(*text_coords)
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    x_range = max(max_x - min_x, min_size)
    y_range = max(max_y - min_y, min_size)
    
    min_x -= x_range * padding_percent
    max_x += x_range * padding_percent
    min_y -= y_range * padding_percent
    max_y += y_range * padding_percent
    
    pixel_per_unit = min(image_width / (max_x - min_x), image_height / (max_y - min_y))
    
    return {
        'min_x': min_x, 'max_x': max_x, 'min_y': min_y, 'max_y': max_y,
        'pixel_per_unit': pixel_per_unit, 'image_width': image_width, 'image_height': image_height
    }

def prune_dxf_outliers(doc):
    """
    Calculates robust extents and DELETES outlier entities from the modelspace.
    This ensures that subsequent rendering (which uses global extents) only sees valid geometry.
    """
    msp = doc.modelspace()
    
    # 1. Collect all entity bounding boxes
    cache = bbox.Cache()
    valid_boxes = []
    valid_entities = []
    
    # Get all entities
    entities = list(msp)
    
    if not entities:
        return None
        
    for entity in entities:
        try:
            ext = bbox.extents([entity], cache=cache)
            if ext.has_data:
                valid_boxes.append([ext.extmin.x, ext.extmin.y, ext.extmax.x, ext.extmax.y])
                valid_entities.append(entity)
        except Exception:
            continue
            
    if not valid_boxes:
        return None
        
    boxes = np.array(valid_boxes)
    
    # 2. If few entities, just use global bounds
    if len(boxes) < 10:
        min_x = np.min(boxes[:, 0])
        min_y = np.min(boxes[:, 1])
        max_x = np.max(boxes[:, 2])
        max_y = np.max(boxes[:, 3])
        return min_x, min_y, max_x, max_y
        
    # 3. Robust Filtering (IQR Method)
    centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
    centers_y = (boxes[:, 1] + boxes[:, 3]) / 2
    
    def get_bounds(values, factor=3.0):
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        if iqr == 0:
            std = np.std(values)
            if std == 0:
                return values[0], values[0]
            median = np.median(values)
            return median - 3 * std, median + 3 * std
        return q1 - factor * iqr, q3 + factor * iqr
        
    min_limit_x, max_limit_x = get_bounds(centers_x, factor=3.0)
    min_limit_y, max_limit_y = get_bounds(centers_y, factor=3.0)
    
    mask = (centers_x >= min_limit_x) & (centers_x <= max_limit_x) & \
           (centers_y >= min_limit_y) & (centers_y <= max_limit_y)
           
    if not np.any(mask):
        print("   ⚠️ All entities filtered by robust extent check. Fallback to full extents.")
        filtered_boxes = boxes
    else:
        outlier_count = len(boxes) - np.sum(mask)
        if outlier_count > 0:
            print(f"   ✅ Robust Pruning: Deleting {outlier_count} outlier entities.")
            outlier_indices = np.where(~mask)[0]
            for idx in outlier_indices:
                try:
                    msp.delete_entity(valid_entities[idx])
                except Exception:
                    pass
        filtered_boxes = boxes[mask]
        
    # 4. Calculate final extents
    min_x = np.min(filtered_boxes[:, 0])
    min_y = np.min(filtered_boxes[:, 1])
    max_x = np.max(filtered_boxes[:, 2])
    max_y = np.max(filtered_boxes[:, 3])
    
    return min_x, min_y, max_x, max_y

def get_dxf_extents(doc):
    """
    Calculates extents using ezdxf.bbox (assuming outliers are already pruned).
    """
    cache = bbox.Cache()
    try:
        extents = bbox.extents(doc.modelspace(), cache=cache)
        if extents.has_data:
            return extents.extmin.x, extents.extmin.y, extents.extmax.x, extents.extmax.y
    except Exception as e:
        pass
    return None

def extract_annotations(dxf_file_path):
    encodings = ['utf-8', 'cp936', 'gbk', 'gb18030'] # Prioritize UTF-8 as it is strict and common in modern exports
    for encoding in encodings:
        try:
            doc = ezdxf.readfile(dxf_file_path, encoding=encoding)
            return doc
        except (UnicodeDecodeError, ezdxf.DXFError):
            continue
    return None

def crop_image_whitespace(pil_image):
    image = pil_image.convert("RGB")
    image_np = np.array(image)
    # Use a high threshold for white background
    mask = np.any(image_np < 250, axis=-1)
    if not np.any(mask): return pil_image, (0,0,pil_image.width, pil_image.height)
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return image.crop((x0, y0, x1, y1)), (x0, y0, x1, y1)

# ============================================================
# Main Render Function
# ============================================================

def export_to_png(doc, output_path="output.png", dpi=300, use_text_bounds=False, skip_hatches=False):
    # 0. Prune Outliers
    prune_dxf_outliers(doc)

    # 1. Clean Text AND Shrink Size
    fix_dxf_entities(doc)
    
    if skip_hatches:
        print("   ⚠️ Skipping Hatch Entities for performance...")
        msp = doc.modelspace()
        # Remove all HATCH entities
        for entity in msp.query('HATCH'):
            msp.delete_entity(entity)

    # 2. Apply Font
    cjk_font_path = get_system_cjk_font()
    if cjk_font_path:
        print(f"   Applying Font: {cjk_font_path}")
        apply_font_fix(doc, cjk_font_path)
    else:
        print("   ⚠️ WARNING: No Chinese font found.")

    # 3. Config (COLOR + White Background)
    msp = doc.modelspace()
    cfg = config.Configuration(
        background_policy=BackgroundPolicy.WHITE, # White Paper
        color_policy=ColorPolicy.COLOR,           # <--- CHANGED TO COLOR (Red, Blue, etc.)
        lineweight_policy=LineweightPolicy.RELATIVE, 
        lineweight_scaling=1.0 
    )
    
    # 4. Render
    context = RenderContext(doc)
    backend = pymupdf.PyMuPdfBackend()
    frontend = Frontend(context, backend, config=cfg)
    frontend.draw_layout(msp)

    # 5. Page Size Logic
    extents = get_dxf_extents(doc)
    width_mm, height_mm = 297, 210
    if extents:
        min_x, min_y, max_x, max_y = extents
        dx, dy = abs(max_x - min_x), abs(max_y - min_y)
        if dx > 1 and dy > 1:
            aspect = dx / dy
            if aspect > (297/210): width_mm = 210 * aspect
            else: height_mm = 297 / aspect

    page = layout.Page(width_mm, height_mm, layout.Units.mm, margins=layout.Margins.all(1))
    settings = layout.Settings(fit_page=True, page_alignment=layout.PageAlignment.MIDDLE_CENTER)
    
    png_bytes = backend.get_pixmap_bytes(page, fmt="png", dpi=dpi, settings=settings)
    pil_image = Image.open(io.BytesIO(png_bytes))

    # 6. Transform Logic
    transform = None
    if use_text_bounds:
        text_coords = get_text_extents(doc)
        if text_coords:
            transform = calculate_text_specific_transform(text_coords, pil_image.width, pil_image.height)
        else:
            use_text_bounds = False

    if not use_text_bounds and extents:
        min_x, min_y, max_x, max_y = extents
        dx, dy = max(max_x - min_x, 1e-6), max(max_y - min_y, 1e-6)
        
        fit_scale = min(pil_image.width / dx, pil_image.height / dy)
        transform = {
            'min_x': min_x, 'max_x': max_x,
            'min_y': min_y, 'max_y': max_y,
            'pixel_per_unit': fit_scale,
            'image_width': pil_image.width, 'image_height': pil_image.height
        }

    # 7. Crop & Save
    pil_image, crop_bounds = crop_image_whitespace(pil_image)
    
    if transform and crop_bounds != (0, 0, transform['image_width'], transform['image_height']):
        x0, y0, x1, y1 = crop_bounds
        ppu = transform['pixel_per_unit']
        transform.update({
            'min_x': transform['min_x'] + (x0 / ppu),
            'max_x': transform['min_x'] + (x1 / ppu),
            'min_y': transform['max_y'] - (y1 / ppu),
            'max_y': transform['max_y'] - (y0 / ppu),
            'image_width': x1-x0, 'image_height': y1-y0
        })

    if transform:
        transform.update({
            's_x': 1/transform['pixel_per_unit'],
            's_y': 1/transform['pixel_per_unit'],
            't_x': transform['min_x'],
            't_y': transform['min_y'],
            'H': transform['max_y'] - transform['min_y']
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pil_image.save(output_path)
    
    transform_path = None
    if transform:
        transform_path = os.path.splitext(output_path)[0] + '_transform.json'
        with open(transform_path, 'w') as f:
            json.dump(convert_numpy_types(transform), f, indent=4)

    return output_path, transform_path

def export_to_svg(
    doc,
    output_path="output.svg",
    use_text_bounds=False,
    skip_hatches=False,
    xml_declaration=True
):
    """
    Export modelspace to SVG using the native ezdxf SVGBackend.
    
    Args:
        doc: ezdxf document
        output_path: where to save .svg file
        use_text_bounds: try to fit tightly around TEXT/MTEXT (less common for SVG)
        skip_hatches: skip HATCH entities for faster / lighter output
        xml_declaration: include <?xml ... ?> at the beginning
    
    Returns:
        (output_path, transform_path or None)
    """
    # 0. Prune Outliers (if not already done)
    prune_dxf_outliers(doc)
    
    # 1. Clean text & optionally shrink
    fix_dxf_entities(doc)
    
    if skip_hatches:
        print("   ⚠️ Skipping HATCH entities...")
        msp = doc.modelspace()
        for entity in msp.query('HATCH'):
            msp.delete_entity(entity)

    # 2. Apply CJK font fix (helps text appearance)
    cjk_font_path = get_system_cjk_font()
    if cjk_font_path:
        print(f"   Applying font: {cjk_font_path}")
        apply_font_fix(doc, cjk_font_path)
    else:
        print("   ⚠️ No suitable CJK font found – text may not render correctly.")

    # 3. Create drawing configuration
    #    You can still tune these – SVGBackend respects most of them
    cfg = config.Configuration(
        background_policy=BackgroundPolicy.WHITE,
        color_policy=ColorPolicy.COLOR,           # keeps entity colors
        lineweight_policy=LineweightPolicy.RELATIVE,
        lineweight_scaling=1.0,
        # hatch_policy = HatchPolicy.IGNORE,      # alternative to deleting hatches
        # text_policy = TextPolicy.OUTLINE,       # or PATH — but OUTLINE usually better
    )

    # 4. Render pipeline
    msp = doc.modelspace()
    context = RenderContext(doc)
    backend = SVGBackend()
    frontend = Frontend(context, backend, config=cfg)
    frontend.draw_layout(msp)

    # 5. Page / layout setup (SVGBackend supports cropping via page margins)
    extents = get_dxf_extents(doc)
    width_mm, height_mm = 297, 210   # A4 default

    if extents:
        min_x, min_y, max_x, max_y = extents
        dx = max(max_x - min_x, 1.0)
        dy = max(max_y - min_y, 1.0)
        aspect = dx / dy
        if aspect > (297 / 210):
            width_mm = 210 * aspect
        else:
            height_mm = 297 / aspect

    page = layout.Page(
        width_mm, height_mm,
        layout.Units.mm,
        margins=layout.Margins.all(0)   # Set to 0 for easier alignment
    )

    settings = layout.Settings(
        fit_page=True,
        page_alignment=layout.PageAlignment.MIDDLE_CENTER
    )

    # 6. Get final SVG string
    svg_content = backend.get_string(
        page,
        settings=settings,
        xml_declaration=xml_declaration
    )

    # Optional: tighter bounds around text content only
    transform = None
    if use_text_bounds:
        text_coords = get_text_extents(doc)
        if text_coords:
            # You would normally calculate viewBox here,
            # but most viewers ignore page size when viewBox is set.
            # For simplicity we reuse the same logic (you can refine it)
            img_w, img_h = 1200, 900   # dummy values – SVG doesn't need pixels
            transform = calculate_text_specific_transform(text_coords, img_w, img_h)
            # You could inject viewBox into SVG here if desired:
            # svg_content = inject_viewbox(svg_content, transform)

    # 7. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_content)

    transform_path = None
    if transform:
        transform_path = os.path.splitext(output_path)[0] + '_transform.json'
        with open(transform_path, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(transform), f, indent=4)

    print(f"   Saved SVG: {output_path}")
    if transform_path:
        print(f"   Transform : {transform_path}")

    return output_path, transform_path

def process_single_file(dxf_path, output_dir, dpi=1200, use_text_bounds=False, skip_hatches=False):
    base_name = os.path.splitext(os.path.basename(dxf_path))[0]
    # output_path = os.path.join(output_dir, f"{base_name}.png")
    # Decide output format by extension or argument
    output_png = os.path.join(output_dir, f"{base_name}.png")
    output_svg = os.path.join(output_dir, f"{base_name}.svg")

    try:
        doc = extract_annotations(dxf_path)
        if not doc: return dxf_path, "Load Failed"
        # return export_to_png(doc, output_path, dpi, use_text_bounds, skip_hatches)
        # Export both if you want (or choose one)
        export_to_png(doc, output_png, dpi=300, use_text_bounds=use_text_bounds, skip_hatches=skip_hatches)
        export_to_svg(doc, output_svg, use_text_bounds=use_text_bounds, skip_hatches=skip_hatches)
        
        return dxf_path, output_svg
    except Exception as e:
        return dxf_path, str(e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input DXF")
    parser.add_argument("--dpi", type=int, default=300) # High default for quality, use --skip-hatches for speed
    parser.add_argument("--text-bounds", action="store_true")
    parser.add_argument("--skip-hatches", action="store_true", help="Skip rendering HATCH entities for faster processing")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    input_path = args.input
    output_dir = os.path.join(os.path.dirname(__file__), "images_color")
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isdir(input_path):
        dxf_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith('.dxf')]
        
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(process_single_file, f, output_dir, args.dpi, args.text_bounds, args.skip_hatches): f for f in dxf_files}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {len(dxf_files)} files"):
                dxf_path, result = future.result()
                if isinstance(result, str) and not result.endswith('.png') and not result.endswith('.json'):
                    print(f"Error processing {dxf_path}: {result}")
    else:
        print(f"📂 Processing 1 files...")
        # Single file
        _, result_path = process_single_file(input_path, output_dir, args.dpi, args.text_bounds, args.skip_hatches)
        print(f"Finished: {result_path}")

if __name__ == "__main__":
    main()
