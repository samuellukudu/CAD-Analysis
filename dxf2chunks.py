"""
dxf_to_chunks_strict.py

SOLUTIONS:
 1. NOISE PRUNING: Deletes tiny polygons to break "bridges".
 2. ACTIVE PROBING: Automatically hunts for natural walls/doors in any chunk > 4% area.
 3. PCA & ORIENTED BOUNDING BOXES (OBB): Replaces AABB with PCA. Calculates true 
    fill efficiency and aspect ratios regardless of how the CAD file is rotated.
 4. PCA-ALIGNED BISECTION: Slices failing rooms directly along their principal 
    component axes, guaranteeing cuts perfectly parallel to angled walls.
 5. WALL ANNIHILATION w/ OVERRIDE: Destroys long thin perimeter walls, but 
    safely protects massive open floor plans.
"""

import math
import os
import pathlib
import argparse
import json
from collections import defaultdict
from typing import List, Dict, Any

import ezdxf
import numpy as np
import networkx as nx
import networkx.algorithms.community as nx_comm
import torch
from torch_geometric.data import Data
from shapely.geometry import LineString, Point
from shapely.ops import unary_union, polygonize
from shapely.strtree import STRtree
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from matplotlib import colormaps
from tqdm import tqdm

# ==========================================
# 1. Geometry Extraction
# ==========================================
def sample_arc(center, radius, start_angle, end_angle, n_segments=64):
    if end_angle <= start_angle: end_angle += 360.0
    start, end = math.radians(start_angle), math.radians(end_angle)
    angles = np.linspace(start, end, max(3, n_segments))
    return [(center[0] + radius * math.cos(a), center[1] + radius * math.sin(a)) for a in angles]

def extract_lines_with_metadata(doc, curve_segments=32) -> List[Dict[str, Any]]:
    msp = doc.modelspace()
    results =[]
    for entity in msp:
        if entity.dxftype() in ('TEXT', 'MTEXT', 'INSERT', 'DIMENSION', 'HATCH'): continue
        geoms =[]
        try:
            if entity.dxftype() == "LINE":
                geoms.append(LineString([(entity.dxf.start.x, entity.dxf.start.y), (entity.dxf.end.x, entity.dxf.end.y)]))
            elif entity.dxftype() == "LWPOLYLINE":
                pts = list(entity.get_points('xy'))
                if len(pts) >= 2: geoms.append(LineString(pts))
            elif entity.dxftype() == "CIRCLE":
                pts = sample_arc((entity.dxf.center.x, entity.dxf.center.y), entity.dxf.radius, 0, 360, curve_segments)
                geoms.append(LineString(pts))
            elif entity.dxftype() == "ARC":
                pts = sample_arc((entity.dxf.center.x, entity.dxf.center.y), entity.dxf.radius, entity.dxf.start_angle, entity.dxf.end_angle, curve_segments)
                geoms.append(LineString(pts))
        except: pass
        
        for ln in geoms:
            if not ln.is_empty and ln.length > 0.1:
                results.append({'geometry': ln, 'layer': entity.dxf.layer})
    return results

# ==========================================
# 2. Region Building
# ==========================================
def build_regions(line_data, buffer_tol=1e-6):
    raw_lines = [d['geometry'] for d in line_data]
    union_geom = unary_union([ln.buffer(buffer_tol) for ln in raw_lines])
    
    if union_geom.geom_type == 'Polygon':
        boundary = union_geom.boundary
    elif union_geom.geom_type == 'MultiPolygon':
        boundary = union_geom.boundary
    else:
        boundary = union_geom
        
    polys = list(polygonize(boundary))
    source_tree = STRtree(raw_lines)
    
    regions =[]
    for poly in polys:
        if poly.is_empty or not poly.is_valid or poly.area < 0.1: continue
        
        query_result = source_tree.query(poly)
        relevant_indices = query_result.tolist() if hasattr(query_result, 'tolist') else [raw_lines.index(x) for x in query_result]

        detected_layers = [line_data[idx]['layer'] for idx in relevant_indices if raw_lines[idx].intersects(poly)]
        assigned_layer = max(set(detected_layers), key=detected_layers.count) if detected_layers else "0"

        c, b = poly.centroid, poly.bounds
        bounds_dict = {'minx': b[0], 'miny': b[1], 'maxx': b[2], 'maxy': b[3]}
        
        feat = np.array([poly.area, c.x, c.y, poly.length, len(poly.interiors), 
                         float(poly.convex_hull.area == poly.area), 
                         (b[2]-b[0])/(b[3]-b[1]) if (b[3]-b[1])>0 else 1.0], dtype=float)
        
        feat_norm = feat / (np.linalg.norm(feat) + 1e-9)
        regions.append({
            "geometry": poly, 
            "features": feat_norm, 
            "raw_features": feat, 
            "labels":[], 
            "layer": assigned_layer,
            "bounds": bounds_dict
        })
    return regions

def attach_labels(doc, regions):
    msp = doc.modelspace()
    texts =[]
    for e in msp:
        if e.dxftype() in ('TEXT', 'MTEXT'):
            txt = e.dxf.text if e.dxftype() == 'TEXT' else e.plain_text()
            texts.append({'pt': np.array([e.dxf.insert.x, e.dxf.insert.y]), 'text': txt})
        elif e.dxftype() == 'INSERT' and hasattr(e, 'attribs'):
            for a in e.attribs: texts.append({'pt': np.array([a.dxf.insert.x, a.dxf.insert.y]), 'text': a.dxf.text})

    if not texts or not regions: return regions
    
    reg_c = np.mean([r['raw_features'][1:3] for r in regions], axis=0)
    txt_c = np.mean([t['pt'] for t in texts], axis=0)
    offset = -(txt_c - reg_c) if np.linalg.norm(txt_c - reg_c) > 5000 else np.array([0., 0.])

    for r in regions:
        poly_loose = r['geometry'].buffer(0.5)
        matches = [t['text'] for t in texts if poly_loose.contains(Point(t['pt'] + offset))]
        r['labels'] = list(set(matches)) if matches else ["UNKNOWN"]
    return regions

# ==========================================
# 4. Graph & Layout Extraction
# ==========================================
def create_global_graph(regions):
    adj = defaultdict(list)
    buffered = [r['geometry'].buffer(5.0) for r in regions]
    n = len(regions)
    for i in range(n):
        for j in range(i+1, n):
            if buffered[i].intersects(buffered[j]):
                adj[i].append(j); adj[j].append(i)
                
    G = nx.Graph()
    for i, r in enumerate(regions):
        G.add_node(i, 
                   features=r['features'], 
                   raw_features=r['raw_features'], 
                   raw_xy=r['raw_features'][1:3], 
                   layer=r['layer'], 
                   bounds=r['bounds'])
        
    for u, nbrs in adj.items():
        for v in nbrs:
            weight = 10.0 if regions[u]['layer'] == regions[v]['layer'] else 1.0
            G.add_edge(u, v, weight=weight)
    return G

def get_layouts(global_graph, all_regions):
    components = list(nx.connected_components(global_graph))
    layouts =[]
    for comp in components:
        if len(comp) < 3: continue
        subgraph = global_graph.subgraph(comp).copy()
        
        nx.set_node_attributes(subgraph, {n: n for n in subgraph.nodes()}, 'global_node_id')
        
        node_list = list(subgraph.nodes())
        mapping = {global_id: i for i, global_id in enumerate(node_list)}
        subgraph = nx.relabel_nodes(subgraph, mapping)
        
        layout_regions = [all_regions[global_id] for global_id in node_list]
        layouts.append({"graph": subgraph, "regions": layout_regions})
        
    layouts.sort(key=lambda x: len(x['regions']), reverse=True)
    for i, l in enumerate(layouts): l['id'] = i
    return layouts

# ==========================================
# 5. CHUNK GENERATION (PCA / OBB AWARE)
# ==========================================
def create_chunks(layout, overlap=0):
    graph = layout['graph'].copy()
    regions = layout['regions']
    n_nodes = len(graph.nodes())
    if n_nodes == 0: return[]
    
    all_areas = [graph.nodes[n]['raw_features'][0] for n in graph.nodes()]
    median_area = np.median(all_areas) if all_areas else 0
    total_layout_area = sum(all_areas)
    
    noise_threshold = median_area * 0.2
    nodes_to_remove = [n for n in graph.nodes() if graph.nodes[n]['raw_features'][0] < noise_threshold]
    if len(nodes_to_remove) < n_nodes: 
        graph.remove_nodes_from(nodes_to_remove)
    
    if len(graph.nodes()) == 0: return[]

    initial_blocks = list(nx.connected_components(graph))
    blocks_to_process =[(set(b), 1.0, 0) for b in initial_blocks]
    final_communities =[]
    
    while blocks_to_process:
        block, current_res, depth = blocks_to_process.pop(0)
        
        if len(block) <= 1 or depth > 10:
            final_communities.append(block)
            continue
            
        subgraph = graph.subgraph(list(block))
        block_nodes = list(block)
        block_area = sum([graph.nodes[n]['raw_features'][0] for n in block_nodes])
        
        # ----------------------------------------------------
        # PCA & ORIENTED BOUNDING BOX (OBB)
        # ----------------------------------------------------
        # Extract all vertex coordinates from the constituent polygons
        coords = []
        for n in block_nodes:
            poly = regions[n]['geometry']
            coords.extend(list(poly.exterior.coords))
        coords = np.array(coords)
        
        pca = PCA(n_components=2)
        try:
            coords_pca = pca.fit_transform(coords)
            min_pca = coords_pca.min(axis=0)
            max_pca = coords_pca.max(axis=0)
            spans = max_pca - min_pca
        except:
            # Fallback for perfectly collinear or invalid data
            spans = np.array([1.0, 1.0])
            
        obb_area = (spans[0] * spans[1])
        obb_efficiency = block_area / (obb_area + 1e-9)
        aspect_ratio = max(spans[0], spans[1]) / (min(spans[0], spans[1]) + 1e-9)
        
        # ----------------------------------------------------
        # SPLIT CONDITIONS (OBB & Semantic Logic)
        # ----------------------------------------------------
        is_large_enough_to_split = block_area > 0.04 * total_layout_area
        is_stretched = (aspect_ratio > 1.9) and (block_area > 0.03 * total_layout_area)
        is_bad_obb = (obb_efficiency < 0.65) and (block_area > 0.03 * total_layout_area)
        
        valid_louvain_split = False
        next_res = current_res + 0.5
        
        # ATTEMPT 1: ACTIVE PROBING
        if is_large_enough_to_split and next_res <= 3.0:
            try:
                comms = list(nx_comm.louvain_communities(subgraph, weight='weight', resolution=next_res, seed=42))
            except:
                comms = [block]
                
            if len(comms) > 1:
                areas = [sum([graph.nodes[n]['raw_features'][0] for n in c]) for c in comms]
                areas.sort(reverse=True)
                
                # Check if it split into two meaningful chunks (> 1.5% layout area)
                if len(areas) >= 2 and areas[1] > 0.015 * total_layout_area:
                    valid_louvain_split = True
                    for comm in comms:
                        sub_sub = subgraph.subgraph(list(comm))
                        ccs = list(nx.connected_components(sub_sub))
                        for c in ccs:
                            if len(c) > 0: blocks_to_process.append((set(c), next_res, depth + 1))
                            
        if valid_louvain_split:
            continue
            
        # ATTEMPT 2: PCA-ALIGNED BISECTION
        # If Louvain couldn't find natural walls, but the OBB says the chunk is stretched or L-shaped, slice it!
        if is_stretched or is_bad_obb:
            centroids = np.array([graph.nodes[n]['raw_xy'] for n in block_nodes])
            try:
                # Transform node centroids into PCA space to find out which side of the room they are on
                centroids_pca = pca.transform(centroids)
                cut_axis = 0 if spans[0] > spans[1] else 1
                median_val = np.median(centroids_pca[:, cut_axis])
                
                left_nodes, right_nodes = [],[]
                for i, n in enumerate(block_nodes):
                    if centroids_pca[i, cut_axis] <= median_val:
                        left_nodes.append(n)
                    else:
                        right_nodes.append(n)
            except:
                left_nodes, right_nodes = [],[]
                
            if len(left_nodes) == 0 or len(right_nodes) == 0:
                half = len(block_nodes) // 2
                left_nodes = block_nodes[:half]
                right_nodes = block_nodes[half:]
                    
            for split_group in[left_nodes, right_nodes]:
                if len(split_group) == 0: continue
                sub_sub = graph.subgraph(split_group)
                ccs = list(nx.connected_components(sub_sub))
                for c in ccs:
                    if len(c) > 0:
                        blocks_to_process.append((set(c), 1.0, depth + 1)) 
            continue
            
        # ACCEPT CHUNK
        final_communities.append(block)

    # -----------------------------------------------------------------
    # FINALIZE CHUNKS (WALL ANNIHILATION w/ OBB OVERRIDE)
    # -----------------------------------------------------------------
    chunks =[]
    chunk_id_counter = 0  
    
    for core_nodes in final_communities:
        if not core_nodes: continue
        
        chunk_area = sum([graph.nodes[n]['raw_features'][0] for n in core_nodes])
        if chunk_area < (0.005 * total_layout_area): continue
        
        weighted_compactness = 0
        coords =[]
        
        for n in core_nodes:
            area = graph.nodes[n]['raw_features'][0]
            perimeter = graph.nodes[n]['raw_features'][3]
            
            compact = (4 * math.pi * area) / (perimeter ** 2 + 1e-9)
            weighted_compactness += compact * area
            coords.extend(list(regions[n]['geometry'].exterior.coords))
            
        avg_compactness = weighted_compactness / (chunk_area + 1e-9)
        coords = np.array(coords)
        
        try:
            pca = PCA(n_components=2)
            coords_pca = pca.fit_transform(coords)
            spans = coords_pca.max(axis=0) - coords_pca.min(axis=0)
            obb_area = spans[0] * spans[1] + 1e-9
        except:
            obb_area = chunk_area + 1e-9
            
        obb_efficiency = chunk_area / obb_area
        
        is_wall_junk = (avg_compactness < 0.10) or (obb_efficiency < 0.10)
        
        # Override to protect massive open areas/hallways that have weird perimeters
        if chunk_area > 0.035 * total_layout_area:
            is_wall_junk = False
            
        if is_wall_junk:
            continue 

        chunk_nodes = set(core_nodes)
        for _ in range(overlap):
            current_boundary = list(chunk_nodes)
            for n in current_boundary:
                if n in graph: chunk_nodes.update(graph.neighbors(n))
        
        subgraph = graph.subgraph(list(chunk_nodes)).copy()
        
        nx.set_node_attributes(subgraph, {n: n for n in subgraph.nodes()}, 'layout_node_id')
        nx.set_node_attributes(subgraph, {n: (n in core_nodes) for n in subgraph.nodes()}, 'is_core')
        
        subgraph_renamed = nx.convert_node_labels_to_integers(subgraph, first_label=0)
        subgraph_renamed.graph['chunk_id'] = chunk_id_counter
        chunks.append(subgraph_renamed)
        
        chunk_id_counter += 1
        
    return chunks

# ==========================================
# 6. PyG & Vis
# ==========================================
def graph_to_pyg(G):
    x = torch.tensor(np.array([G.nodes[i]['features'] for i in range(len(G))]), dtype=torch.float)
    edge_idx =[]
    for u, v in G.edges(): edge_idx.extend([[u, v], [v, u]])
    edge_index = torch.tensor(edge_idx, dtype=torch.long).t().contiguous() if edge_idx else torch.empty((2,0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

def visualize_chunks(layouts_with_chunks, save_path=None):
    if not layouts_with_chunks: return
    n_layouts = len(layouts_with_chunks)
    fig, axes = plt.subplots(n_layouts, 2, figsize=(20, 10 * n_layouts), squeeze=False)
    cmap = colormaps.get_cmap("tab20")
    
    for i, item in enumerate(layouts_with_chunks):
        chunk_graphs = item['chunk_graphs']
        regions = item['layout_data']['regions']
        region_colors = {n: None for n in range(len(regions))}
        
        for c_idx, cg in enumerate(chunk_graphs):
            color = cmap(c_idx % 20)
            for u, data in cg.nodes(data=True):
                orig_id = data['layout_node_id']
                if data.get('is_core', True): region_colors[orig_id] = color
                elif region_colors[orig_id] is None: region_colors[orig_id] = (0.9, 0.9, 0.9, 0.5)

        layout_graph = item['layout_data']['graph']
        layout_nodes = list(layout_graph.nodes())
        
        final_colors = [region_colors[n] if region_colors[n] is not None else (0.8, 0.8, 0.8, 0.3) for n in layout_nodes]

        ax_geo = axes[i, 0]
        patches = [MplPolygon(np.array(regions[n]['geometry'].exterior.coords), closed=True) for n in layout_nodes]
        p = PatchCollection(patches, facecolor=final_colors, edgecolor='black', lw=0.5, alpha=0.9)
        ax_geo.add_collection(p)
        ax_geo.autoscale_view()
        ax_geo.set_title(f"Layout {item['id']}: {len(chunk_graphs)} Chunks")
        ax_geo.set_aspect('equal')
        
        ax_graph = axes[i, 1]
        pos = {n: (regions[n]['raw_features'][1], regions[n]['raw_features'][2]) for n in layout_nodes}
        nx.draw(layout_graph, pos, ax=ax_graph, node_size=15, node_color=final_colors, edge_color='gray', width=0.3)
        ax_graph.set_title("Graph Connectivity")
        ax_graph.set_aspect('equal')

    plt.tight_layout()
    if save_path: plt.savefig(save_path)
    else: plt.show()

# ==========================================
# MAIN PIPELINE
# ==========================================
def save_as_geojson(final_output, output_path):
    features_list = []
    for item in final_output:
        regions = item['layout_data']['regions']
        for graph in item['chunk_graphs']:
            real_chunk_id = graph.graph['chunk_id'] 
            
            adj_map = {}
            for n in graph.nodes():
                layout_n = graph.nodes[n].get('layout_node_id')
                if layout_n is not None:
                    adj_map[int(layout_n)] =[
                        int(graph.nodes[v].get('layout_node_id')) 
                        for v in graph.neighbors(n) 
                        if graph.nodes[v].get('layout_node_id') is not None
                    ]

            for node_id in graph.nodes():
                layout_node_id = graph.nodes[node_id].get('layout_node_id')
                if layout_node_id is None: continue
                
                global_node_id = graph.nodes[node_id].get('global_node_id')
                
                poly = regions[layout_node_id]['geometry']
                poly_coords = [list(poly.exterior.coords)] +[list(h.coords) for h in poly.interiors]
                feats = graph.nodes[node_id]['features']
                
                feature = {
                    "type": "Feature",
                    "id": int(global_node_id), 
                    "geometry": {"type": "Polygon", "coordinates": poly_coords},
                    "properties": {
                        "layout_id": item['id'], 
                        "chunk_id": real_chunk_id,        
                        "node_id": int(layout_node_id), 
                        "global_node_id": int(global_node_id) if global_node_id is not None else -1,
                        "local_node_id": int(node_id),    
                        "features": feats.tolist() if hasattr(feats, 'tolist') else feats,
                        "neighbors": adj_map[int(layout_node_id)]
                    }
                }
                features_list.append(feature)
                
    with open(output_path, 'w') as f: 
        json.dump({"type": "FeatureCollection", "features": features_list}, f)

def process_single_dxf(dxf_path, output_geojson_path, output_viz_path):
    try:
        doc = ezdxf.readfile(dxf_path)
    except Exception as e:
        print(f"[ERROR] Read failed: {e}"); return

    lines = extract_lines_with_metadata(doc)
    regions = attach_labels(doc, build_regions(lines))
    global_graph = create_global_graph(regions)
    layouts = get_layouts(global_graph, regions)
    
    final_output =[] 
    
    for layout in layouts:
        chunk_graphs = create_chunks(layout, overlap=0)
        
        if not chunk_graphs:
            continue

        final_output.append({
            'id': layout['id'], 'layout_data': layout, 'chunk_graphs': chunk_graphs, 
            'chunks': [graph_to_pyg(cg) for cg in chunk_graphs]
        })
        
    save_as_geojson(final_output, output_geojson_path)
    visualize_chunks(final_output, output_viz_path)

def batch_pipeline(input_folder, output_folder="chunked_data"):
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    dxf_files =[f for f in os.listdir(input_folder) if f.lower().endswith('.dxf')]
    for filename in tqdm(dxf_files):
        stem = pathlib.Path(filename).stem
        process_single_dxf(os.path.join(input_folder, filename), 
                           os.path.join(output_folder, f"{stem}.geojson"), 
                           os.path.join(output_folder, f"{stem}_viz.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output_folder", "-o", default="chunked_data")
    args = parser.parse_args()
    if os.path.isfile(args.input):
        stem = pathlib.Path(args.input).stem
        process_single_dxf(args.input, os.path.join(args.output_folder, f"{stem}.geojson"), os.path.join(args.output_folder, f"{stem}_viz.png"))
    else:
        batch_pipeline(args.input, args.output_folder)