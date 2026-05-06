import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import numpy as np
from matplotlib import colormaps

def visualize_geojson_layouts(geojson_path, max_layouts=None):
    # === 0. SAFETY CHECKS ===
    if not os.path.exists(geojson_path):
        print(f"[ERROR] File not found: {geojson_path}")
        return

    # Check if file is empty
    if os.path.getsize(geojson_path) == 0:
        print(f"[ERROR] The file '{geojson_path}' is empty (0 bytes).")
        print("       Check your pipeline generation step. Did the DXF extraction fail?")
        return

    # === 1. LOAD AND GROUP GEOJSON DATA ===
    print(f"Loading {geojson_path}...")
    temp_grouped = {} 

    try:
        with open(geojson_path, 'r') as f:
            data = json.load(f) # Load the whole FeatureCollection
    except json.JSONDecodeError as e:
        print(f"[ERROR] Could not decode JSON in '{geojson_path}'.")
        print(f"       Details: {e}")
        return

    # Check if 'features' key exists (in case it saved an empty list [])
    if 'features' not in data:
        print(f"[WARN] No 'features' key found in JSON. Is this a FeatureCollection?")
        return

    for feature in data['features']:
        props = feature['properties']
        geom = feature['geometry']
        
        lid = props['layout_id']
        cid = props['chunk_id']
        
        if not geom or 'coordinates' not in geom or not geom['coordinates']:
            continue
        
        # Extract the exterior ring coordinates
        poly_coords = geom['coordinates'][0]

        if lid not in temp_grouped:
            temp_grouped[lid] = {}
        if cid not in temp_grouped[lid]:
            temp_grouped[lid][cid] = []
        
        temp_grouped[lid][cid].append(poly_coords)

    # === 2. CONVERT TO VISUALIZATION FORMAT ===
    layout_ids = sorted(temp_grouped.keys())
    
    if not layout_ids:
        print("[WARN] No valid layouts found to visualize.")
        return

    if max_layouts:
        layout_ids = layout_ids[:max_layouts]

    for lid in layout_ids:
        chunk_dict = temp_grouped[lid]
        
        # Prepare data containers
        chunks_for_plot = []
        all_layout_coords = []

        # Convert dictionary {cid: [poly...]} to list [{'chunk_id': cid, 'coords_list': [...]}]
        sorted_cids = sorted(chunk_dict.keys())
        for cid in sorted_cids:
            polys = chunk_dict[cid]
            chunks_for_plot.append({
                'chunk_id': cid,
                'coords_list': polys
            })
            all_layout_coords.extend(polys)

        n_chunks = len(chunks_for_plot)

        # === 3. PLOTTING LOGIC ===
        
        # Calculate Global Bounds
        all_polys_flat = [pt for poly in all_layout_coords for pt in poly]
        if not all_polys_flat:
            continue
            
        all_points = np.array(all_polys_flat)
        min_x, min_y = all_points.min(axis=0)
        max_x, max_y = all_points.max(axis=0)
        
        # Add 5% padding
        pad_x = (max_x - min_x) * 0.05
        pad_y = (max_y - min_y) * 0.05
        global_xlim = (min_x - pad_x, max_x + pad_x)
        global_ylim = (min_y - pad_y, max_y + pad_y)

        # Dynamic grid setup
        cols = min(5, n_chunks + 1)
        rows = (n_chunks // cols) + 2

        fig = plt.figure(figsize=(5 * cols, 5 * rows))
        fig.suptitle(f"Layout {lid} → {len(all_layout_coords)} Regions, {n_chunks} Chunk{'s' if n_chunks != 1 else ''}",
                     fontsize=20, y=0.95)
        
        # Updated colormap retrieval
        cmap = colormaps.get_cmap("tab20")

        # --- A. Full Layout Overview ---
        ax_full = fig.add_subplot(rows, cols, 1)
        
        patches = []
        colors = []
        
        for chunk in chunks_for_plot:
            color = cmap(chunk['chunk_id'] % 20)
            for coords in chunk['coords_list']:
                if len(coords) >= 3:
                    patches.append(MplPolygon(np.array(coords), closed=True))
                    colors.append(color)

        if patches:
            collection = PatchCollection(patches, facecolor=colors, edgecolor='black', linewidth=0.5, alpha=0.9)
            ax_full.add_collection(collection)
        
        ax_full.set_xlim(global_xlim)
        ax_full.set_ylim(global_ylim)
        ax_full.set_aspect('equal')
        ax_full.set_title("Full Layout (Chunk Coloring)", fontsize=14, pad=20)
        ax_full.axis('off')

        # --- B. Individual Chunks ---
        for i, chunk in enumerate(chunks_for_plot):
            ax = fig.add_subplot(rows, cols, i + cols + 1)
            
            # Ghost background
            ghost_patches = []
            for other_chunk in chunks_for_plot:
                if other_chunk['chunk_id'] != chunk['chunk_id']:
                    for coords in other_chunk['coords_list']:
                        if len(coords) >= 3:
                            ghost_patches.append(MplPolygon(np.array(coords), closed=True))
            if ghost_patches:
                ghost_col = PatchCollection(ghost_patches, facecolor='#f0f0f0', edgecolor='#e0e0e0', linewidth=0.5)
                ax.add_collection(ghost_col)

            # Active Chunk
            chunk_patches = [MplPolygon(np.array(coords), closed=True) for coords in chunk['coords_list'] if len(coords) >= 3]

            if chunk_patches:
                color = cmap(chunk['chunk_id'] % 20)
                collection = PatchCollection(chunk_patches, facecolor=color, edgecolor='black', linewidth=0.7, alpha=0.9)
                ax.add_collection(collection)

            ax.set_xlim(global_xlim)
            ax.set_ylim(global_ylim)
            ax.set_aspect('equal')
            ax.set_title(f"Chunk {chunk['chunk_id']}", fontsize=12)
            ax.axis('off')

        # Hide unused subplots
        total_plots = 1 + n_chunks
        for j in range(total_plots, rows * cols):
            fig.add_subplot(rows, cols, j + 1).axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.1)
        # plt.savefig(f'demo_chunk_{lid}.png')
        plt.show()

if __name__ == "__main__":
    # Update this path to where your file is actually located
    visualize_geojson_layouts("chunked_data/AEC Plan Elev Sample.geojson", max_layouts=None)