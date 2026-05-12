import json
import os
import argparse
import sys
import re
import math
import numpy as np
from shapely.geometry import Point, LineString, box
from shapely.strtree import STRtree


def decode_cad_unicode(text):
    if not text:
        return ""
    def _replace_unicode(match):
        try:
            return chr(int(match.group(1), 16))
        except Exception:
            return match.group(0)
    text = re.sub(r'\\U\+([0-9A-Fa-f]{4})', _replace_unicode, text)
    text = text.replace(r'\P', '\n')
    text = re.sub(r'\\[ACFHQTWf].*?;', '', text)
    text = re.sub(r'[{}]', '', text)
    return text.strip()


def _get_transform_matrix(tx, ty, sx, sy, rot):
    if abs(rot) > 2 * math.pi + 1e-6:
        rot = math.radians(rot)
    c = math.cos(rot)
    s = math.sin(rot)
    return np.array([[sx * c, -sy * s, tx],[sx * s,  sy * c, ty],[0,      0,      1],
    ], dtype=float)


def _get_text_center(entity):
    """
    Finds the exact mathematical center of the text to serve as our symmetric base.
    """
    start = entity.get("start") or entity.get("startPoint")
    end = entity.get("end") or entity.get("endPoint")
    
    if start and end and "x" in start and "y" in start and "x" in end and "y" in end:
        return {"x": (start["x"] + end["x"]) / 2.0, "y": (start["y"] + end["y"]) / 2.0}
    
    # Fallback to standard insertion points
    p = entity.get("position") or entity.get("insertPoint") or entity.get("center") or entity.get("startPoint")
    if p and "x" in p and "y" in p:
        return {"x": p["x"], "y": p["y"]}
        
    return None


def _collect_label_seeds(data, keyword):
    try:
        keyword_norm = re.sub(r"\s+", " ", keyword).strip()
        pattern = re.compile(keyword_norm)
    except re.error:
        pattern = None

    results =[]
    for ent in data.get("entities",[]):
        if ent.get("type") in ["TEXT", "MTEXT"]:
            raw_text = ent.get("text", "")
            text = decode_cad_unicode(raw_text)
            text_norm = re.sub(r"\s+", " ", text).strip()
            
            is_match = (pattern.search(text_norm) is not None) if pattern else (keyword in text_norm)

            if is_match:
                p = _get_text_center(ent)
                layer = ent.get("layer", "Unknown")
                if p:
                    x, y = p["x"], p["y"]
                    seed_str = f"{x:.2f},{y:.2f}"
                    results.append({"text": text, "x": float(x), "y": float(y), "layer": layer, "seed": seed_str})
                    
    results.sort(key=lambda r: (round(r["x"] / 10000), -r["y"]))
    return results


def _extract_all_segments(entities, blocks):
    """
    Recursively extract all visible line segments. Returns a list of Shapely LineStrings.
    """
    segments =[]

    def _walk(ent, parent_matrix, depth=0, max_depth=3):
        if depth > max_depth: return
        etype = ent.get("type")

        if etype == "INSERT":
            name = ent.get("name")
            ip = _get_text_center(ent)
            if not ip: return
            
            sx, sy = ent.get("xScale", 1) or 1, ent.get("yScale", 1) or 1
            rot = ent.get("rotation", 0) or 0
            local_m = _get_transform_matrix(ip["x"], ip["y"], sx, sy, rot)
            combined = parent_matrix @ local_m
            
            block_def = blocks.get(name)
            if block_def:
                children = block_def if isinstance(block_def, list) else block_def.get("entities",[])
                for child in children:
                    _walk(child, combined, depth=depth + 1, max_depth=max_depth)
            return

        elif etype == "LINE":
            s, e = ent.get("start"), ent.get("end")
            if s and e and "x" in s and "x" in e:
                aw = parent_matrix @ np.array([s["x"], s["y"], 1.0])
                bw = parent_matrix @ np.array([e["x"], e["y"], 1.0])
                segments.append(LineString([(aw[0], aw[1]), (bw[0], bw[1])]))
                
        elif etype in ("LWPOLYLINE", "POLYLINE"):
            verts = ent.get("vertices",[])
            coords =[]
            for v in verts:
                if isinstance(v, dict) and "x" in v and "y" in v:
                    vw = parent_matrix @ np.array([v["x"], v["y"], 1.0])
                    coords.append((vw[0], vw[1]))
            if len(coords) >= 2:
                for i in range(len(coords) - 1):
                    segments.append(LineString([coords[i], coords[i+1]]))
                if ent.get("closed", False) or ent.get("shape", False):
                    segments.append(LineString([coords[-1], coords[0]]))

    identity = np.eye(3, dtype=float)
    for ent in entities:
        _walk(ent, identity, depth=0, max_depth=3)
        
    return segments


def _get_isovist_polygon(seed_x, seed_y, local_segments, max_radius=30000.0, num_rays=360):
    """
    Casts 360 rays outwards to hit the nearest CAD lines. 
    Applies a median filter to seal gaps, open doors, and ignore noise.
    """
    seed = Point(seed_x, seed_y)
    tree = STRtree(local_segments)
    
    angles = np.linspace(0, 2 * math.pi, num_rays, endpoint=False)
    distances = np.full(num_rays, max_radius)
    
    # 1. Cast Rays
    for i, angle in enumerate(angles):
        end_x = seed_x + max_radius * math.cos(angle)
        end_y = seed_y + max_radius * math.sin(angle)
        ray = LineString([(seed_x, seed_y), (end_x, end_y)])
        
        # Fast spatial query to find lines in the path
        possible_matches = tree.query(ray)
        
        min_dist = max_radius
        for idx in possible_matches:
            seg = local_segments[idx]
            inter = ray.intersection(seg)
            if not inter.is_empty:
                # Shapely gracefully handles getting distance to Point, MultiPoint, or LineString
                d = seed.distance(inter)
                if d < min_dist:
                    min_dist = d
                    
        distances[i] = min_dist

    # 2. Filter Spikes (Fix CAD Drafting Gaps & Open Doors)
    # A 11-ray median filter effectively seals 10-degree gaps without distorting room shape.
    kernel_size = 11
    pad_size = kernel_size // 2
    
    # Pad array circularly to handle the 359-to-0 degree wrap around
    padded = np.concatenate((distances[-pad_size:], distances, distances[:pad_size]))
    filtered_distances = np.zeros_like(distances)
    
    for i in range(num_rays):
        window = padded[i : i + kernel_size]
        filtered_distances[i] = np.median(window)

    # 3. Convert back to absolute X, Y boundary points
    hit_points = []
    for i, angle in enumerate(angles):
        d = filtered_distances[i]
        hit_points.append((seed_x + d * math.cos(angle), seed_y + d * math.sin(angle)))
        
    return hit_points


def _select_candidate_bboxes_isovist(global_segments, segment_tree, seed_x, seed_y, topk: int):
    """
    Extracts symmetric bounding boxes using ray-casting.
    Returns a list of (min_x, min_y, max_x, max_y) tuples (at most topk).
    """
    out_bboxes = []
    
    # 1. Grab a local chunk of the building for fast ray-casting (30,000 unit radius ~ 30 meters)
    search_radius = 30000.0 
    search_window = box(seed_x - search_radius, seed_y - search_radius, seed_x + search_radius, seed_y + search_radius)
    
    indices = segment_tree.query(search_window)
    local_segments = [global_segments[i] for i in indices]

    if not local_segments:
        # Ultimate fallback if text is floating in deep space
        s = 2000.0
        out_bboxes.append((seed_x - s, seed_y - s, seed_x + s, seed_y + s))
    else:
        # 2. Generate the Isovist Ray Hits
        pts = _get_isovist_polygon(seed_x, seed_y, local_segments, max_radius=search_radius)
        
        # 3. Force Strict Symmetry based on the visual limits
        max_dx = max(abs(x - seed_x) for x, y in pts)
        max_dy = max(abs(y - seed_y) for x, y in pts)
        
        # Enforce reasonable minimums/maximums
        max_dx = min(max(max_dx, 1000.0), search_radius)
        max_dy = min(max(max_dy, 1000.0), search_radius)
        
        # 4. Generate Progressive Contexts (Top-K)
        # k=0: Strict Isovist (The Room itself)
        # k=1: 1.5x Multiplier (Room + Context)
        # k=2: 2.25x Multiplier (Macro View)
        for k in range(topk):
            scale = 1.0 * (1.5 ** k)
            dx = max_dx * scale
            dy = max_dy * scale
            
            out_bboxes.append((seed_x - dx, seed_y - dy, seed_x + dx, seed_y + dy))

    return out_bboxes[:topk]


def _count_geom_in_bbox(entities, blocks, target_bbox):
    import numpy as np
    t_min_x, t_min_y, t_max_x, t_max_y = target_bbox
    count = [0]
    
    def _walk(ent, parent_matrix, depth=0, max_depth=3):
        if depth > max_depth: return
        etype = ent.get("type")
        
        if etype == "INSERT":
            name = ent.get("name") or ent.get("blockName")
            if not name or name not in blocks:
                return
                
            p = ent.get("position") or ent.get("insertPoint") or {"x":0, "y":0}
            sx = ent.get("xScale", 1) or 1
            sy = ent.get("yScale", 1) or 1
            rot = ent.get("rotation", 0) or 0
            
            local_m = _get_transform_matrix(p.get("x",0), p.get("y",0), sx, sy, rot)
            combined = parent_matrix @ local_m
            
            block_def = blocks.get(name)
            if block_def:
                children = block_def if isinstance(block_def, list) else block_def.get("entities", [])
                for child in children:
                    _walk(child, combined, depth=depth + 1, max_depth=max_depth)
            return
            
        elif etype in ("LINE", "LWPOLYLINE", "POLYLINE", "ARC", "CIRCLE"):
            pts = []
            if etype == "LINE":
                s, e = ent.get("start"), ent.get("end")
                if s and e and "x" in s and "y" in s:
                    pts.append((s["x"], s["y"]))
                    pts.append((e["x"], e["y"]))
            elif etype in ("LWPOLYLINE", "POLYLINE"):
                for v in ent.get("vertices", []):
                    if isinstance(v, dict) and "x" in v and "y" in v:
                        pts.append((v["x"], v["y"]))
            elif etype in ("ARC", "CIRCLE"):
                c = ent.get("center")
                r = ent.get("radius", 0)
                if c and "x" in c and "y" in c:
                    pts.append((c["x"] - r, c["y"] - r))
                    pts.append((c["x"] + r, c["y"] + r))
            
            if not pts:
                p = ent.get("position") or ent.get("insertPoint") or ent.get("center")
                if p and "x" in p and "y" in p:
                    pts.append((p["x"], p["y"]))
                    
            if not pts:
                return
                
            pts_np = np.array(pts, dtype=float)
            ones = np.ones((pts_np.shape[0], 1), dtype=float)
            pts_h = np.hstack([pts_np, ones])
            transformed = (parent_matrix @ pts_h.T).T
            
            min_x = np.min(transformed[:, 0])
            min_y = np.min(transformed[:, 1])
            max_x = np.max(transformed[:, 0])
            max_y = np.max(transformed[:, 1])
            
            if not (max_x < t_min_x or min_x > t_max_x or max_y < t_min_y or min_y > t_max_y):
                count[0] += 1

    identity = np.eye(3, dtype=float)
    for ent in entities:
        _walk(ent, identity, depth=0, max_depth=3)
        
    return count[0]

def _filter_candidate_bboxes(data, bboxes, threshold=100, min_ar=0.35, max_ar=5.0):
    if not bboxes:
        return []
    valid_bboxes = []
    blocks = data.get("blocks", {})
    entities = data.get("entities", [])
    for bbox in bboxes:
        min_x, min_y, max_x, max_y = bbox
        width = max_x - min_x
        height = max_y - min_y
        if height == 0:
            continue
        ar = width / height
        if not (min_ar <= ar <= max_ar):
            continue
        geom_count = _count_geom_in_bbox(entities, blocks, bbox)
        if geom_count >= threshold:
            valid_bboxes.append(bbox)
    return valid_bboxes


def _match_header_line(index, seed, layer):
    return f"[match {index}] {seed} ({layer})"


def _format_bbox_string(bbox):
    return f"{bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f}"


def _bbox_area_tuple(bbox):
    return float(max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]))


def _safe_percent(part, whole):
    whole = float(whole)
    if whole <= 1e-12:
        return 0.0
    return float((float(part) / whole) * 100.0)


def _drawing_extent_and_area_from_segments(segments):
    if not segments:
        return (0.0, 0.0, 0.0, 0.0), 0.0
    min_x = min(seg.bounds[0] for seg in segments)
    min_y = min(seg.bounds[1] for seg in segments)
    max_x = max(seg.bounds[2] for seg in segments)
    max_y = max(seg.bounds[3] for seg in segments)
    bbox = (float(min_x), float(min_y), float(max_x), float(max_y))
    return bbox, _bbox_area_tuple(bbox)


def _bbox_tuple_to_dict(bbox):
    return {"min_x": float(bbox[0]), "min_y": float(bbox[1]), "max_x": float(bbox[2]), "max_y": float(bbox[3])}


def _union_bbox_tuples(boxes):
    """Axis-aligned union of rectangles (min_x, min_y, max_x, max_y)."""
    if not boxes:
        return (0.0, 0.0, 0.0, 0.0)
    min_x = min(b[0] for b in boxes)
    min_y = min(b[1] for b in boxes)
    max_x = max(b[2] for b in boxes)
    max_y = max(b[3] for b in boxes)
    return (min_x, min_y, max_x, max_y)


def _mean_bbox_xyxy_from_bboxes(boxes):
    """
    Mean of bbox coordinates across candidates: mean(min_x), mean(min_y), mean(max_x), mean(max_y).
    """
    if not boxes:
        return {"min_x": 0.0, "min_y": 0.0, "max_x": 0.0, "max_y": 0.0}
    sx0 = sy0 = sx1 = sy1 = 0.0
    n = len(boxes)
    for b in boxes:
        sx0 += b[0]
        sy0 += b[1]
        sx1 += b[2]
        sy1 += b[3]
    return {"min_x": sx0 / n, "min_y": sy0 / n, "max_x": sx1 / n, "max_y": sy1 / n}


def find_raycast_matches(
    keyword,
    *,
    file_path=None,
    data=None,
    bbox_direct=False,
    topk=3,
    max_matches_per_name=10,
    threshold=100,
    min_ar=0.35,
    max_ar=5.0,
):
    """
    Same shape as find_region.find_region_matches (no ``height`` field on seeds), plus
    ``method`` ``isovist``, and top-level ``name`` / ``path`` like find_region_matches.
    When bbox_direct is True, each match also includes bbox_union, bbox_union_xy,
    candidates_mean_bbox (CSV string), and candidates_mean_bbox_xyxy (dict), matching find_region.
    """
    if (file_path is None) == (data is None):
        raise ValueError("Provide exactly one of file_path or data")
    resolved_path = os.path.abspath(file_path) if file_path is not None else None
    if file_path is not None:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    seeds = _collect_label_seeds(data, keyword)
    if not seeds:
        return {
            "keyword": keyword,
            "name": keyword,
            "path": resolved_path,
            "method": "isovist",
            "matches": {},
        }

    global_segments = None
    segment_tree = None
    drawing_bbox_t = (0.0, 0.0, 0.0, 0.0)
    drawing_area = 0.0
    if bbox_direct:
        global_segments = _extract_all_segments(data.get("entities", []), data.get("blocks", {}))
        segment_tree = STRtree(global_segments)
        drawing_bbox_t, drawing_area = _drawing_extent_and_area_from_segments(global_segments)

    out = {}
    for i, lm in enumerate(seeds[: int(max_matches_per_name)], start=1):
        key = str(i)
        entry = {
            "index": i,
            "header": _match_header_line(i, lm["seed"], lm["layer"]),
            "text": lm["text"],
            "x": float(lm["x"]),
            "y": float(lm["y"]),
            "layer": lm["layer"],
            "seed": lm["seed"],
            "bboxes": [],
        }
        if bbox_direct:
            tuples = _select_candidate_bboxes_isovist(
                global_segments,
                segment_tree,
                lm["x"],
                lm["y"],
                topk=int(topk),
            )
            tuples = _filter_candidate_bboxes(data, tuples, threshold=threshold, min_ar=min_ar, max_ar=max_ar)
            entry["bboxes"] = [_format_bbox_string(b) for b in tuples]
            entry["bboxes_xy"] = [_bbox_tuple_to_dict(b) for b in tuples]
            union_t = _union_bbox_tuples(tuples)
            entry["bbox_union"] = _format_bbox_string(union_t)
            entry["bbox_union_xy"] = _bbox_tuple_to_dict(union_t)
            mean_rect = _mean_bbox_xyxy_from_bboxes(tuples)
            mean_tuple = (mean_rect["min_x"], mean_rect["min_y"], mean_rect["max_x"], mean_rect["max_y"])
            entry["candidates_mean_bbox"] = _format_bbox_string(mean_tuple)
            entry["candidates_mean_bbox_xyxy"] = {
                "min_x": float(mean_rect["min_x"]),
                "min_y": float(mean_rect["min_y"]),
                "max_x": float(mean_rect["max_x"]),
                "max_y": float(mean_rect["max_y"]),
            }
            best_area = _bbox_area_tuple(mean_tuple)
            union_area = _bbox_area_tuple(union_t)
            entry["best_area"] = float(best_area)
            entry["best_area_pct_of_drawing"] = _safe_percent(best_area, drawing_area)
            entry["union_area"] = float(union_area)
            entry["union_area_pct_of_drawing"] = _safe_percent(union_area, drawing_area)
            entry["drawing_bbox"] = _format_bbox_string(drawing_bbox_t)
            entry["drawing_bbox_xyxy"] = _bbox_tuple_to_dict(drawing_bbox_t)
            entry["drawing_area"] = float(drawing_area)
        out[key] = entry

    return {
        "keyword": keyword,
        "name": keyword,
        "path": resolved_path,
        "method": "isovist",
        "matches": out,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Find perfectly symmetric matching ROIs using Isovist Raycasting.")
    parser.add_argument("--file", required=True, help="Path to the JSON CAD file")
    parser.add_argument("--name", required=True, help="Name/keyword to search for")
    parser.add_argument("--bbox_direct", action="store_true", help="Output bboxes directly.")
    parser.add_argument("--topk", type=int, default=3, help="Top-K bboxes to output per label match")
    parser.add_argument("--max_matches_per_name", type=int, default=10, help="Limit label matches processed")
    parser.add_argument("--threshold", type=int, default=100, help="Minimum geometric entity count to keep a match")
    parser.add_argument("--min_ar", type=float, default=0.35, help="Minimum aspect ratio (width/height)")
    parser.add_argument("--max_ar", type=float, default=5.0, help="Maximum aspect ratio (width/height)")
    return parser.parse_args()


def main():
    args = _parse_args()

    print(f"Loading {args.file}...")
    try:
        result = find_raycast_matches(
            args.name,
            file_path=args.file,
            bbox_direct=args.bbox_direct,
            topk=args.topk,
            max_matches_per_name=args.max_matches_per_name,
            threshold=args.threshold,
            min_ar=args.min_ar,
            max_ar=args.max_ar,
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except OSError as e:
        print(f"Error loading file: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    matches_map = result["matches"]
    if not matches_map:
        print(f"\nNo text matching '{args.name}' found.")
        sys.exit(0)

    ordered_keys = sorted(matches_map.keys(), key=int)

    if not args.bbox_direct:
        print(f"\nFound {len(matches_map)} matches for '{args.name}':")
        print("=" * 100)
        for k in ordered_keys:
            r = matches_map[k]
            print(f"Text:  {r['text']}")
            print(f"Coord: X: {r['x']:10.2f}, Y: {r['y']:10.2f}  | Layer: {r['layer']}")
            print(f"-> Suggested --seed_point:  \"{r['seed']}\"")
            print("-" * 100)
        return

    print(
        f"\nFound {len(matches_map)} label matches for '{args.name}'. bbox_direct TopK={args.topk} (isovist)...",
        file=sys.stderr,
    )
    print("Extracting raw geometry for isovist...", file=sys.stderr)

    for k in ordered_keys:
        lm = matches_map[k]
        print(f"Text:  {lm['text']}")
        print(f"Coord: X: {lm['x']:10.2f}, Y: {lm['y']:10.2f}  | Layer: {lm['layer']}")
        print(lm["header"])
        for line in lm["bboxes"]:
            print(line)
        print(f"Union:  {lm['bbox_union']}")
        print(f"Mean:  {lm['candidates_mean_bbox']}")
        print(f"Area:  {lm.get('best_area', 0.0):.2f} ({lm.get('best_area_pct_of_drawing', 0.0):.2f}% of drawing)")
        print(
            f"Union Area:  {lm.get('union_area', 0.0):.2f} "
            f"({lm.get('union_area_pct_of_drawing', 0.0):.2f}% of drawing)"
        )
        print("-" * 100)


if __name__ == "__main__":
    main()