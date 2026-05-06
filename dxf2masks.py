import pandas as pd
import numpy as np
import json
import os
from datetime import date

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

    rows = []

    print(f"[INFO] Processing {len(geojson_data['features'])} features...")

    for feature in geojson_data['features']:
        props = feature['properties']
        geom = feature['geometry']

        # Extract IDs
        layout_id = props.get('layout_id')
        chunk_id = props.get('chunk_id')

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

def save_masks_to_json(df: pd.DataFrame, output_path: str):
    mask_data = []

    print(f"[INFO] Converting {len(df)} masks to JSON format...")

    for _, row in df.iterrows():
        poly_raw = row.get("chunks")
        dxf_poly_raw = row.get("chunks_dxf")

        if isinstance(poly_raw, np.ndarray):
            poly = poly_raw.tolist()
        elif isinstance(poly_raw, (list, tuple)):
            poly = [list(p) if isinstance(p, (np.ndarray, tuple)) else p for p in poly_raw]
        else:
            continue

        clean_poly = []
        for pt in poly:
            if isinstance(pt, (np.ndarray, list, tuple)):
                pt_list = [float(coord) for coord in pt]
                if len(pt_list) >= 2:
                    clean_poly.append(pt_list[:2])
        if len(clean_poly) < 3:
            continue

        # Process DXF coordinates if available
        clean_dxf_poly = []
        if isinstance(dxf_poly_raw, (list, tuple, np.ndarray)):
             if isinstance(dxf_poly_raw, np.ndarray):
                 dxf_poly = dxf_poly_raw.tolist()
             else:
                 dxf_poly = dxf_poly_raw
             
             for pt in dxf_poly:
                 if isinstance(pt, (np.ndarray, list, tuple)):
                     pt_list = [float(coord) for coord in pt]
                     if len(pt_list) >= 2:
                         clean_dxf_poly.append(pt_list[:2])

        points_np = np.array(clean_poly)
        min_x, min_y = points_np.min(axis=0)
        max_x, max_y = points_np.max(axis=0)

        bbox = [float(min_x), float(min_y), float(max_x - min_x), float(max_y - min_y)]

        layout_id = row.get("layout_id")
        chunk_id = row.get("chunk_id")
        try:
            layout_id = int(layout_id)
        except Exception:
            pass
        try:
            chunk_id = int(chunk_id)
        except Exception:
            pass

        mask_data.append(
            {
                "layout_id": layout_id,
                "chunk_id": chunk_id,
                "bbox": bbox,
                "segmentation": clean_poly,
                "segmentation_dxf": clean_dxf_poly if clean_dxf_poly else None,
            }
        )

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mask_data, f, indent=4, ensure_ascii=False)

    print(f"[SUCCESS] Masks saved to {output_path}")
    return mask_data

def generate_single_mask_json(
    df: pd.DataFrame,
    output_file: str = "all_chunk_masks.json",
    padding: int = 50,
    clip_to_image: bool = False,
    image_size: tuple = None  # (width, height) optional for clipping
):
    """
    Generates a single JSON with integer keys and robust polygon handling.
    """
    all_masks = {
        "layouts": {},
        "metadata": {
            "total_layouts": 0,
            "total_chunks": 0,
            "generated_on": str(date.today()),
            "padding_applied": padding,
            "source": "from DataFrame chunks column"
        }
    }

    unique_layouts = sorted(df['layout_id'].unique())
    all_masks["metadata"]["total_layouts"] = len(unique_layouts)

    total_chunks = 0

    for layout_id in unique_layouts:
        layout_df = df[df['layout_id'] == layout_id]
        unique_chunks = sorted(layout_df['chunk_id'].unique())

        # Use integer key!
        all_masks["layouts"][int(layout_id)] = {"chunks": {}}

        for chunk_id in unique_chunks:
            subset = layout_df[layout_df['chunk_id'] == chunk_id]

            polygons = []
            all_points = []

            for _, row in subset.iterrows():
                poly_raw = row['chunks']

                # Convert to list safely
                if isinstance(poly_raw, np.ndarray):
                    poly = poly_raw.tolist()
                elif isinstance(poly_raw, (list, tuple)):
                    poly = [list(p) if isinstance(p, (np.ndarray, tuple)) else p for p in poly_raw]
                else:
                    continue  # skip invalid

                # Ensure each point is [x, y] list of numbers
                clean_poly = []
                for pt in poly:
                    if isinstance(pt, (np.ndarray, list, tuple)):
                        pt_list = [float(coord) for coord in pt]
                        if len(pt_list) >= 2:
                            clean_poly.append(pt_list[:2])  # only x,y
                    elif isinstance(pt, (int, float)):
                        continue  # malformed
                if len(clean_poly) < 3:
                    continue  # skip degenerate polygons

                polygons.append(clean_poly)
                all_points.extend(clean_poly)

            if not polygons or not all_points:
                print(f"[WARN] Skipping empty/invalid chunk: layout {layout_id}, chunk {chunk_id}")
                continue

            points_np = np.array(all_points)
            min_x, min_y = points_np.min(axis=0)
            max_x, max_y = points_np.max(axis=0)

            left = min_x - padding
            top = min_y - padding
            right = max_x + padding
            bottom = max_y + padding

            # Optional: clip to image bounds
            if clip_to_image and image_size:
                w, h = image_size
                left = max(0, left)
                top = max(0, top)
                right = min(w, right)
                bottom = min(h, bottom)

            bbox = [float(left), float(top), float(right), float(bottom)]

            # Store with integer keys
            all_masks["layouts"][int(layout_id)]["chunks"][int(chunk_id)] = {
                "polygons": polygons,
                "bounding_box": bbox
            }
            total_chunks += 1

    all_masks["metadata"]["total_chunks"] = total_chunks
    print(f"Successfully generated {total_chunks} valid chunks across {len(unique_layouts)} layouts.")

    # Save
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_masks, f, indent=4, ensure_ascii=False)

    print(f"Saved to: {output_file}")
    return all_masks  # optionally return for immediate use
    
if __name__ == "__main__":
    GEOJSON = "chunked_data/AEC Plan Elev Sample.geojson"
    TRANSFORM = "images/AEC Plan Elev Sample_transform.json"
    IMAGE_FILE = "images/AEC Plan Elev Sample.png"
    if os.path.exists(GEOJSON) and os.path.exists(TRANSFORM) and os.path.exists(IMAGE_FILE):
        df = create_layout_dataframe(geojson_path=GEOJSON, transform_path=TRANSFORM)
        image_base = os.path.splitext(os.path.basename(IMAGE_FILE))[0]
        output_file = os.path.join("chunk_masks", f"{image_base}_masks.json")
        generate_single_mask_json(df, output_file=output_file, padding=50, clip_to_image=True, image_size=(1920, 1080))
    else:
        print("Sample files not found, skipping main execution.")
