import json
import os
import argparse
import sys
import re

from src.occupancy_grids import (
    _collect_world_segments,
    _collect_geometry_points,
    _detect_drawing_slot,
    _refine_slot_cc,
    _tighten_slot_bbox,
    _drawing_extent_and_area,
    _filter_candidate_bboxes as _og_filter_candidate_bboxes,
    _rerank_bboxes_by_geometry,
    _bbox_iou,
)


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


def _get_text_properties(entity):
    props = {"x": None, "y": None, "height": 250.0}
    
    start = entity.get("start") or entity.get("startPoint")
    end = entity.get("end") or entity.get("endPoint")
    
    if start and end and "x" in start and "x" in end:
        props["x"] = (start["x"] + end["x"]) / 2.0
        props["y"] = (start["y"] + end["y"]) / 2.0
    else:
        p = entity.get("position") or entity.get("insertPoint") or entity.get("center") or entity.get("startPoint")
        if p and "x" in p:
            props["x"] = p["x"]
            props["y"] = p["y"]
            
    h = entity.get("height") or entity.get("textHeight") or entity.get("size")
    if h is not None:
        props["height"] = float(h)
    elif start and end and "y" in start and "y" in end:
        dy = abs(start["y"] - end["y"])
        if dy > 0: props["height"] = dy

    return props if props["x"] is not None else None


def _collect_label_seeds(data, keyword):
    from src.occupancy_grids import _traverse_entity_tree

    try:
        # If it's a regex with pipes, remove spaces around pipes
        keyword_norm = re.sub(r"\s*\|\s*", "|", keyword)
        keyword_norm = re.sub(r"\s+", " ", keyword_norm).strip()
        pattern = re.compile(keyword_norm)
    except re.error:
        pattern = None

    results = []
    blocks = data.get("blocks", {})
    
    def _apply_matrix_xy(m, x, y):
        return (
            float(m[0, 0] * x + m[0, 1] * y + m[0, 2]),
            float(m[1, 0] * x + m[1, 1] * y + m[1, 2]),
        )

    def on_leaf(ent, etype, m):
        if etype in ["TEXT", "MTEXT", "ATTRIB", "ATTDEF", "LEADER"]:
            raw_text = ent.get("text", "") or ent.get("value", "")
            if not raw_text:
                return
            text = decode_cad_unicode(raw_text)
            text_norm = re.sub(r"\s+", " ", text).strip()
            
            is_match = (pattern.search(text_norm) is not None) if pattern else (keyword in text_norm)

            if is_match:
                props = _get_text_properties(ent)
                if props:
                    x, y, h = props["x"], props["y"], props["height"]
                    xw, yw = _apply_matrix_xy(m, float(x), float(y))
                    
                    results.append({
                        "text": text, 
                        "x": float(xw), 
                        "y": float(yw), 
                        "height": float(h),
                        "layer": ent.get("layer", "Unknown"), 
                        "seed": f"{xw:.2f},{yw:.2f}"
                    })

    _traverse_entity_tree(data.get("entities", []), blocks, on_leaf, max_depth=5)
                    
    results.sort(key=lambda r: (round(r["x"] / 10000), -r["y"]))
    return results


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


def _bbox_tuple_to_dict(bbox):
    return {"min_x": float(bbox[0]), "min_y": float(bbox[1]), "max_x": float(bbox[2]), "max_y": float(bbox[3])}


def find_vector_matches(
    keyword,
    *,
    file_path=None,
    data=None,
    bbox_direct=False,
    topk=5,
    max_matches_per_name=10,
    threshold=100,
    min_ar=0.25,
    max_ar=5.0,
):
    """
    Same shape as occupancy_grids.find_region_matches: keyword, name, path, and matches
    dict keyed by "1","2",...

    Uses the 1D slot detection + CC refinement pipeline from occupancy_grids
    to find the drawing region containing each text seed.
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
            "matches": {},
        }

    drawing_bbox_t = (0.0, 0.0, 0.0, 0.0)
    drawing_area = 0.0
    geom_points = None
    segments = None
    extent = None

    if bbox_direct:
        drawing_bbox_t, drawing_area = _drawing_extent_and_area(data)
        geom_points = _collect_geometry_points(data)
        segments = _collect_world_segments(data, max_depth=3)
        if geom_points.shape[0] > 0:
            extent = (
                float(geom_points[:, 0].min()),
                float(geom_points[:, 1].min()),
                float(geom_points[:, 0].max()),
                float(geom_points[:, 1].max()),
            )

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
            "height": float(lm["height"]),
            "bboxes": [],
        }

        if bbox_direct:
            if segments and extent and geom_points is not None and geom_points.shape[0] > 0:
                coarse_slot = _detect_drawing_slot(segments, float(lm["x"]), float(lm["y"]), extent)
                slot = _refine_slot_cc(segments, float(lm["x"]), float(lm["y"]), coarse_slot)
                tight = _tighten_slot_bbox(geom_points, slot)
                b_tuples = [tight]
            else:
                s = 1000.0
                b_tuples = [(lm["x"] - s, lm["y"] - s, lm["x"] + s, lm["y"] + s)]

            b_tuples = _og_filter_candidate_bboxes(data, b_tuples, threshold=threshold, min_ar=min_ar, max_ar=max_ar)
            b_tuples = _rerank_bboxes_by_geometry(data, b_tuples)

            if not b_tuples:
                s = 1000.0
                b_tuples = [(lm["x"] - s, lm["y"] - s, lm["x"] + s, lm["y"] + s)]

            deduped = []
            for b in b_tuples:
                if any(_bbox_iou(b, prev) >= 0.95 for prev in deduped):
                    continue
                deduped.append(b)
            b_tuples = deduped[: max(1, int(topk))]

            entry["bboxes"] = [_format_bbox_string(b) for b in b_tuples]
            entry["bboxes_xy"] = [_bbox_tuple_to_dict(b) for b in b_tuples]

            union_t = (
                min(b[0] for b in b_tuples),
                min(b[1] for b in b_tuples),
                max(b[2] for b in b_tuples),
                max(b[3] for b in b_tuples),
            )
            best_bbox = b_tuples[0]

            entry["bbox_union"] = _format_bbox_string(union_t)
            entry["bbox_union_xy"] = _bbox_tuple_to_dict(union_t)
            entry["best_bbox"] = _format_bbox_string(best_bbox)
            entry["best_bbox_xyxy"] = _bbox_tuple_to_dict(best_bbox)
            entry["candidates_mean_bbox"] = _format_bbox_string(best_bbox)
            entry["candidates_mean_bbox_xyxy"] = _bbox_tuple_to_dict(best_bbox)
            entry["candidates_raw_mean_bbox"] = _format_bbox_string(best_bbox)
            entry["candidates_raw_mean_bbox_xyxy"] = _bbox_tuple_to_dict(best_bbox)

            best_area = _bbox_area_tuple(best_bbox)
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
        "matches": out,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Sub-region detection via slot + CC refinement.")
    parser.add_argument("--file", required=True, help="Path to the JSON CAD file")
    parser.add_argument("--name", required=True, help="Name/keyword to search for")
    parser.add_argument("--bbox_direct", action="store_true", help="Output bboxes directly.")
    parser.add_argument("--topk", type=int, default=5, help="Top-K bboxes to output per label match")
    parser.add_argument("--max_matches_per_name", type=int, default=10, help="Limit label matches processed")
    parser.add_argument("--threshold", type=int, default=100, help="Minimum geometric entity count to keep a match")
    parser.add_argument("--min_ar", type=float, default=0.25, help="Minimum aspect ratio (width/height)")
    parser.add_argument("--max_ar", type=float, default=5.0, help="Maximum aspect ratio (width/height)")
    return parser.parse_args()


def main():
    args = _parse_args()

    print(f"Loading {args.file}...")
    try:
        result = find_vector_matches(
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
        f"\nFound {len(matches_map)} label matches for '{args.name}'. Computing bbox_direct TopK={args.topk}...",
        file=sys.stderr,
    )

    for k in ordered_keys:
        lm = matches_map[k]
        print(f"Text:  {lm['text']}")
        print(f"Coord: X: {lm['x']:10.2f}, Y: {lm['y']:10.2f}  | Layer: {lm['layer']}")
        print(lm["header"])
        for line in lm["bboxes"]:
            print(line)
        print(f"Union:  {lm.get('bbox_union', '0.00,0.00,0.00,0.00')}")
        print(f"Mean:  {lm.get('candidates_mean_bbox', '0.00,0.00,0.00,0.00')}")
        print(f"Area:  {lm.get('best_area', 0.0):.2f} ({lm.get('best_area_pct_of_drawing', 0.0):.2f}% of drawing)")
        print(
            f"Union Area:  {lm.get('union_area', 0.0):.2f} "
            f"({lm.get('union_area_pct_of_drawing', 0.0):.2f}% of drawing)"
        )
        print("-" * 100)


if __name__ == "__main__":
    main()
