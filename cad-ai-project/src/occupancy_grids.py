from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata


def decode_cad_unicode(text):
    if not text:
        return ""

    def _replace_unicode(match):
        try:
            return chr(int(match.group(1), 16))
        except Exception:
            return match.group(0)

    text = re.sub(r"\\U\+([0-9A-Fa-f]{4})", _replace_unicode, text)
    text = text.replace(r"\P", "\n")
    text = re.sub(r"\\[ACFHQTWf].*?;", "", text)
    text = re.sub(r"[{}]", "", text)
    return text.strip()


def _get_transform_matrix(tx, ty, sx, sy, rot):
    import numpy as np

    if abs(rot) > 2 * 3.141592653589793 + 1e-6:
        rot = np.radians(rot)
    c = np.cos(rot)
    s = np.sin(rot)
    return np.array(
        [
            [sx * c, -sy * s, tx],
            [sx * s, sy * c, ty],
            [0, 0, 1],
        ],
        dtype=float,
    )


def _transform_points(points, matrix):
    import numpy as np

    pts = np.asarray(points, dtype=float)
    if pts.shape[0] == 0:
        return []
    ones = np.ones((pts.shape[0], 1), dtype=float)
    pts_h = np.hstack([pts, ones])
    transformed = (matrix @ pts_h.T).T
    return transformed[:, :2].tolist()


def _entity_insert_position(entity):
    p = entity.get("position") or entity.get("insertPoint") or entity.get("center") or entity.get("startPoint")
    if p and "x" in p and "y" in p:
        return {"x": p["x"], "y": p["y"]}
    return None


def _get_entity_points(entity, blocks=None, depth=0, max_depth=2):
    etype = entity.get("type")
    points = []

    if etype == "LINE":
        s = entity.get("start")
        e = entity.get("end")
        if s and e and "x" in s and "y" in s and "x" in e and "y" in e:
            points.append((s["x"], s["y"]))
            points.append((e["x"], e["y"]))

    elif etype in ("LWPOLYLINE", "POLYLINE"):
        verts = entity.get("vertices", []) or []
        for v in verts:
            if isinstance(v, dict) and "x" in v and "y" in v:
                points.append((v["x"], v["y"]))

    elif etype == "INSERT":
        if depth > max_depth:
            return []
        p = _entity_insert_position(entity)
        if not p:
            return []
        name = entity.get("name")
        if blocks and name in blocks:
            block_def = blocks.get(name)
            if isinstance(block_def, list):
                children = block_def
            elif isinstance(block_def, dict):
                children = block_def.get("entities", []) or []
            else:
                children = []
            child_points = []
            for ch in children:
                child_points.extend(_get_entity_points(ch, blocks=blocks, depth=depth + 1, max_depth=max_depth))
            if child_points:
                sx = entity.get("xScale", 1) or 1
                sy = entity.get("yScale", 1) or 1
                rot = entity.get("rotation", 0) or 0
                m = _get_transform_matrix(p["x"], p["y"], sx, sy, rot)
                return _transform_points(child_points, m)
        return [(p["x"], p["y"])]

    elif etype in ("CIRCLE", "ARC", "ELLIPSE", "POINT", "ATTRIB"):
        p = _entity_insert_position(entity)
        if p:
            points.append((p["x"], p["y"]))
        if etype == "CIRCLE":
            c = entity.get("center") or _entity_insert_position(entity)
            r = entity.get("radius")
            if c and r is not None and "x" in c and "y" in c:
                points.append((c["x"] + r, c["y"]))
                points.append((c["x"] - r, c["y"]))
                points.append((c["x"], c["y"] + r))
                points.append((c["x"], c["y"] - r))

    elif etype == "SOLID":
        for k in ("first", "second", "third", "fourth"):
            p = entity.get(k)
            if p and "x" in p and "y" in p:
                points.append((p["x"], p["y"]))

    elif etype == "HATCH":
        loops = entity.get("boundaryLoops", []) or []
        for loop in loops:
            poly = loop.get("polyline")
            if poly:
                for v in poly.get("vertices", []) or []:
                    if isinstance(v, dict) and "x" in v and "y" in v:
                        points.append((v["x"], v["y"]))
            for edge in loop.get("edges", []) or []:
                for key in ("start", "end", "center"):
                    p = edge.get(key)
                    if p and "x" in p and "y" in p:
                        points.append((p["x"], p["y"]))

    if not points:
        p = _entity_insert_position(entity)
        if p:
            points.append((p["x"], p["y"]))

    return points


def _points_extent(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _parse_args():
    parser = argparse.ArgumentParser(description="Find matching layout labels in CAD JSON.")
    parser.add_argument("--file", required=True, help="Path to the JSON CAD file")
    parser.add_argument("--name", required=True, help="Name/keyword to search for")
    parser.add_argument("--bbox_direct", action="store_true", help="Output bbox(es) directly")
    parser.add_argument("--topk", type=int, default=5, help="Top-K bboxes to output per label match")
    parser.add_argument("--levels", type=int, default=3, help="Kept for API compatibility")
    parser.add_argument("--max_matches_per_name", type=int, default=10, help="Limit label matches processed")
    parser.add_argument("--threshold", type=int, default=100, help="Minimum geometric entity count to keep a match")
    parser.add_argument("--min_ar", type=float, default=0.25, help="Minimum aspect ratio (width/height)")
    parser.add_argument("--max_ar", type=float, default=5.0, help="Maximum aspect ratio (width/height)")
    return parser.parse_args()


def _get_text_center(entity):
    start = entity.get("start") or entity.get("startPoint")
    end = entity.get("end") or entity.get("endPoint")
    if start and end and "x" in start and "y" in start and "x" in end and "y" in end:
        return {"x": (start["x"] + end["x"]) / 2.0, "y": (start["y"] + end["y"]) / 2.0}
    p = (
        entity.get("insertPoint")
        or entity.get("insertionPoint")
        or entity.get("position")
        or entity.get("alignmentPoint")
        or entity.get("startPoint")
        or entity.get("center")
    )
    if p and "x" in p and "y" in p:
        return {"x": p["x"], "y": p["y"]}
    if entity.get("type") == "LEADER":
        verts = entity.get("vertices", []) or []
        pts = [(v.get("x"), v.get("y")) for v in verts if isinstance(v, dict) and "x" in v and "y" in v]
        if pts:
            return {"x": sum(p[0] for p in pts) / len(pts), "y": sum(p[1] for p in pts) / len(pts)}
    return None


def _block_entity_children(blocks, name):
    if not name or not blocks or name not in blocks:
        return []
    block_def = blocks.get(name)
    if isinstance(block_def, list):
        return block_def
    if isinstance(block_def, dict):
        return block_def.get("entities", []) or []
    return []


def _traverse_entity_tree(entities, blocks, on_leaf, on_insert_without_block=None, parent_matrix=None, depth=0, max_depth=3):
    import numpy as np

    if parent_matrix is None:
        parent_matrix = np.eye(3, dtype=float)
    if depth > max_depth:
        return

    for ent in entities:
        etype = ent.get("type")
        if etype == "INSERT":
            name = ent.get("name")
            ip = _entity_insert_position(ent)
            if not ip:
                continue
            sx = ent.get("xScale", 1) or 1
            sy = ent.get("yScale", 1) or 1
            rot = ent.get("rotation", 0) or 0
            local_m = _get_transform_matrix(ip["x"], ip["y"], sx, sy, rot)
            combined = parent_matrix @ local_m
            children = _block_entity_children(blocks, name)
            if children:
                _traverse_entity_tree(
                    children,
                    blocks,
                    on_leaf,
                    on_insert_without_block=on_insert_without_block,
                    parent_matrix=combined,
                    depth=depth + 1,
                    max_depth=max_depth,
                )
            elif on_insert_without_block is not None:
                on_insert_without_block(ent, combined)
            continue
        on_leaf(ent, etype, parent_matrix)


def _collect_label_seeds(data, keyword):
    def _normalize_for_match(s):
        s = unicodedata.normalize("NFKC", str(s or ""))
        s = re.sub(r"\s+", " ", s).strip()
        return s.casefold()

    def _looks_like_regex(pattern_str):
        regex_meta = set(r".^$*+?{}[]\|()")
        return any((c in regex_meta) for c in pattern_str) or ("\\" in pattern_str)

    keyword_norm = _normalize_for_match(keyword)
    keyword_norm = re.sub(r"\s*\|\s*", "|", keyword_norm)
    
    pattern = None
    literal_fallback = None
    if keyword_norm:
        if not _looks_like_regex(keyword_norm):
            tokens = [t for t in keyword_norm.split(" ") if t]
            if len(tokens) >= 2:
                token_gap_re = r".*?".join(re.escape(t) for t in tokens)
                pattern = re.compile(token_gap_re)
            else:
                pattern = re.compile(re.escape(tokens[0])) if tokens else None
        else:
            try:
                pattern = re.compile(keyword_norm)
            except re.error:
                literal_fallback = keyword_norm

    blocks = data.get("blocks", {}) or {}
    text_types = {"TEXT", "MTEXT", "ATTRIB", "ATTDEF", "LEADER"}
    results = []

    def _entity_text_value(ent):
        for k in ("text", "value", "defaultValue", "tag", "name"):
            v = ent.get(k)
            if isinstance(v, str) and v.strip():
                return v
        return ""

    def _apply_matrix_xy(m, x, y):
        return (
            float(m[0, 0] * x + m[0, 1] * y + m[0, 2]),
            float(m[1, 0] * x + m[1, 1] * y + m[1, 2]),
        )

    def on_leaf(ent, etype, m):
        if etype not in text_types:
            return
        raw_text = _entity_text_value(ent)
        text = decode_cad_unicode(raw_text)
        text_norm = _normalize_for_match(text)
        if pattern is not None:
            is_match = pattern.search(text_norm) is not None
        else:
            is_match = bool(literal_fallback) and (literal_fallback in text_norm)
        if not is_match:
            return
        p = _get_text_center(ent)
        if p and "x" in p and "y" in p:
            xw, yw = _apply_matrix_xy(m, float(p["x"]), float(p["y"]))
            results.append(
                {
                    "text": text,
                    "x": float(xw),
                    "y": float(yw),
                    "layer": ent.get("layer", "Unknown"),
                    "seed": f"{xw:.2f},{yw:.2f}",
                }
            )

    _traverse_entity_tree(data.get("entities", []) or [], blocks, on_leaf=on_leaf, max_depth=5)
    if not results:
        return []

    dedup = {}
    for r in results:
        dedup[(r["seed"], r["text"], r["layer"])] = r
    results = list(dedup.values())
    results.sort(key=lambda r: (round(r["x"] / 10000), -r["y"]))
    return results


def _bbox_iou(b1, b2):
    min_x1, min_y1, max_x1, max_y1 = b1
    min_x2, min_y2, max_x2, max_y2 = b2
    inter_w = max(0.0, min(max_x1, max_x2) - max(min_x1, min_x2))
    inter_h = max(0.0, min(max_y1, max_y2) - max(min_y1, min_y2))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area1 = max(0.0, (max_x1 - min_x1)) * max(0.0, (max_y1 - min_y1))
    area2 = max(0.0, (max_x2 - min_x2)) * max(0.0, (max_y2 - min_y2))
    denom = area1 + area2 - inter
    if denom <= 0:
        return 0.0
    return float(inter / denom)


# ---------------------------------------------------------------------------
# Core layout detection: segment-rasterized iterative 1D slot detection
# ---------------------------------------------------------------------------

def _collect_geometry_points(data):
    import numpy as np

    blocks = data.get("blocks", {}) or {}
    points = []
    max_entity_dim = 100000
    for ent in data.get("entities", []) or []:
        pts = _get_entity_points(ent, blocks=blocks, depth=0, max_depth=2)
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if (max(xs) - min(xs) > max_entity_dim) or (max(ys) - min(ys) > max_entity_dim):
            continue
        points.extend(pts)
    if not points:
        return np.zeros((0, 2), dtype=float)
    points = list({(round(x, 3), round(y, 3)) for (x, y) in points})
    return np.asarray(points, dtype=float)


def _collect_world_segments(data, max_depth=3):
    """Collect all geometry segments as (x1, y1, x2, y2) in world coordinates.

    Traverses the entity tree up to *max_depth* INSERT levels, extracting every
    LINE edge, POLYLINE/LWPOLYLINE edge, and HATCH boundary edge.  Point-like
    entities (CIRCLE cardinal points, ARC/POINT centres) are stored as
    zero-length segments so they still contribute to 1-D histograms.
    """
    import numpy as np

    blocks = data.get("blocks", {}) or {}
    entities = data.get("entities", []) or []
    segments: list[tuple[float, float, float, float]] = []

    def _add_seg(x1, y1, x2, y2):
        segments.append((float(x1), float(y1), float(x2), float(y2)))

    def on_leaf(ent, etype, M):
        if etype == "LINE":
            s = ent.get("start")
            e = ent.get("end")
            if not s or not e or "x" not in s or "x" not in e:
                return
            a = M @ np.array([s["x"], s["y"], 1.0], dtype=float)
            b = M @ np.array([e["x"], e["y"], 1.0], dtype=float)
            _add_seg(a[0], a[1], b[0], b[1])
            return

        if etype in ("LWPOLYLINE", "POLYLINE"):
            verts = ent.get("vertices", []) or []
            coords = []
            for v in verts:
                if not isinstance(v, dict) or "x" not in v or "y" not in v:
                    continue
                w = M @ np.array([float(v["x"]), float(v["y"]), 1.0], dtype=float)
                coords.append((float(w[0]), float(w[1])))
            for j in range(len(coords) - 1):
                _add_seg(coords[j][0], coords[j][1], coords[j + 1][0], coords[j + 1][1])
            if len(coords) > 2 and (ent.get("closed") or ent.get("shape")):
                _add_seg(coords[-1][0], coords[-1][1], coords[0][0], coords[0][1])
            return

        if etype == "HATCH":
            for loop in ent.get("boundaryLoops", []) or []:
                poly = loop.get("polyline")
                if poly:
                    verts = poly.get("vertices", []) or []
                    coords = []
                    for v in verts:
                        if not isinstance(v, dict) or "x" not in v or "y" not in v:
                            continue
                        w = M @ np.array([float(v["x"]), float(v["y"]), 1.0], dtype=float)
                        coords.append((float(w[0]), float(w[1])))
                    for j in range(len(coords) - 1):
                        _add_seg(coords[j][0], coords[j][1], coords[j + 1][0], coords[j + 1][1])
                    if len(coords) > 2:
                        _add_seg(coords[-1][0], coords[-1][1], coords[0][0], coords[0][1])
                for edge in loop.get("edges", []) or []:
                    s = edge.get("start")
                    e = edge.get("end")
                    if s and e and "x" in s and "x" in e:
                        a = M @ np.array([s["x"], s["y"], 1.0], dtype=float)
                        b = M @ np.array([e["x"], e["y"], 1.0], dtype=float)
                        _add_seg(a[0], a[1], b[0], b[1])
            return

        if etype in ("CIRCLE", "ARC", "ELLIPSE"):
            c = ent.get("center") or _entity_insert_position(ent)
            if c and "x" in c and "y" in c:
                w = M @ np.array([float(c["x"]), float(c["y"]), 1.0], dtype=float)
                cx, cy = float(w[0]), float(w[1])
                r = float(ent.get("radius", 0) or 0)
                _add_seg(cx - r, cy, cx + r, cy)
                _add_seg(cx, cy - r, cx, cy + r)
            return

        p = _entity_insert_position(ent)
        if p:
            w = M @ np.array([float(p["x"]), float(p["y"]), 1.0], dtype=float)
            _add_seg(float(w[0]), float(w[1]), float(w[0]), float(w[1]))

    _traverse_entity_tree(entities, blocks, on_leaf, max_depth=max_depth)
    return segments


def _build_segment_hist_1d(segments, axis, gmin, gmax, n_bins):
    """Build a 1D histogram from segment endpoints on one axis.

    Each segment contributes its two endpoint coordinates (not the full span)
    so that the histogram reflects *point density* rather than segment extent.
    This preserves clean separation between real inter-drawing whitespace
    (near-zero count) and intra-drawing content (high count).

    Parameters
    ----------
    segments : list of (x1, y1, x2, y2)
    axis : int   0 for X, 1 for Y
    gmin, gmax : float   global extent on the chosen axis
    n_bins : int

    Returns
    -------
    (hist, bin_w) : (np.ndarray[int64], float)
    """
    import numpy as np

    span = gmax - gmin
    if span <= 1e-9:
        return np.zeros(n_bins, dtype=np.int64), 1.0
    bin_w = span / n_bins
    hist = np.zeros(n_bins, dtype=np.int64)
    for seg in segments:
        b1 = max(0, min(n_bins - 1, int((seg[axis] - gmin) / bin_w)))
        b2 = max(0, min(n_bins - 1, int((seg[axis + 2] - gmin) / bin_w)))
        hist[b1] += 1
        hist[b2] += 1
    return hist, bin_w


def _find_slot_1d(hist, seed_coord, gmin, bin_w, n_bins, min_gap_bins=5):
    """Find the contiguous occupied interval containing *seed_coord*.

    A "gap" requires *min_gap_bins* consecutive bins whose count is at or
    below a threshold of 1 % of the histogram peak.  The seed is snapped to
    the nearest occupied bin if it starts in a gap.

    Returns (slot_lo, slot_hi) in world coordinates.
    """
    import math

    nonzero = hist[hist > 0]
    if nonzero.size == 0:
        return gmin, gmin + bin_w * n_bins
    ref = int(sorted(nonzero)[nonzero.size * 3 // 4])
    thresh = max(1, math.ceil(ref * 0.02))

    seed_bin = int((seed_coord - gmin) / bin_w)
    seed_bin = max(0, min(n_bins - 1, seed_bin))

    if hist[seed_bin] <= thresh:
        for offset in range(1, n_bins):
            lo_try = seed_bin - offset
            hi_try = seed_bin + offset
            if 0 <= lo_try and hist[lo_try] > thresh:
                seed_bin = lo_try
                break
            if hi_try < n_bins and hist[hi_try] > thresh:
                seed_bin = hi_try
                break
        else:
            return gmin, gmin + bin_w * n_bins

    lo_bin = 0
    for b in range(seed_bin, -1, -1):
        if hist[b] <= thresh:
            run = 0
            for b2 in range(b, max(b - min_gap_bins, -1), -1):
                if hist[b2] <= thresh:
                    run += 1
                else:
                    break
            if run >= min_gap_bins:
                lo_bin = b + 1
                break

    hi_bin = n_bins - 1
    for b in range(seed_bin, n_bins):
        if hist[b] <= thresh:
            run = 0
            for b2 in range(b, min(b + min_gap_bins, n_bins)):
                if hist[b2] <= thresh:
                    run += 1
                else:
                    break
            if run >= min_gap_bins:
                hi_bin = b - 1
                break

    return gmin + lo_bin * bin_w, gmin + (hi_bin + 1) * bin_w


def _detect_drawing_slot(segments, seed_x, seed_y, extent, n_bins=400):
    """Iterative 1D gap detection to find the drawing slot containing the seed.

    Runs two 3-pass sweeps (Y->X->Y and X->Y->X) and keeps the result with
    the larger area.  Each pass builds a segment-rasterized 1D histogram,
    finds the slot boundaries around the seed, then filters segments to that
    slot before projecting onto the other axis.
    """
    gmin_x, gmin_y, gmax_x, gmax_y = extent
    min_gap = max(3, n_bins // 40)

    def _overlaps(seg, lo, hi, axis):
        s_lo = min(seg[axis], seg[axis + 2])
        s_hi = max(seg[axis], seg[axis + 2])
        return s_hi >= lo and s_lo <= hi

    def _run_yxy():
        h_y, bw_y = _build_segment_hist_1d(segments, 1, gmin_y, gmax_y, n_bins)
        yl, yh = _find_slot_1d(h_y, seed_y, gmin_y, bw_y, n_bins, min_gap)
        ys = [s for s in segments if _overlaps(s, yl, yh, 1)]
        h_x, bw_x = _build_segment_hist_1d(ys, 0, gmin_x, gmax_x, n_bins)
        xl, xh = _find_slot_1d(h_x, seed_x, gmin_x, bw_x, n_bins, min_gap)
        xs = [s for s in segments if _overlaps(s, xl, xh, 0)]
        h_y2, bw_y2 = _build_segment_hist_1d(xs, 1, gmin_y, gmax_y, n_bins)
        yl2, yh2 = _find_slot_1d(h_y2, seed_y, gmin_y, bw_y2, n_bins, min_gap)
        return (xl, yl2, xh, yh2)

    def _run_xyx():
        h_x, bw_x = _build_segment_hist_1d(segments, 0, gmin_x, gmax_x, n_bins)
        xl, xh = _find_slot_1d(h_x, seed_x, gmin_x, bw_x, n_bins, min_gap)
        xs = [s for s in segments if _overlaps(s, xl, xh, 0)]
        h_y, bw_y = _build_segment_hist_1d(xs, 1, gmin_y, gmax_y, n_bins)
        yl, yh = _find_slot_1d(h_y, seed_y, gmin_y, bw_y, n_bins, min_gap)
        ys = [s for s in segments if _overlaps(s, yl, yh, 1)]
        h_x2, bw_x2 = _build_segment_hist_1d(ys, 0, gmin_x, gmax_x, n_bins)
        xl2, xh2 = _find_slot_1d(h_x2, seed_x, gmin_x, bw_x2, n_bins, min_gap)
        return (xl2, yl, xh2, yh)

    slot_a = _run_yxy()
    slot_b = _run_xyx()
    area_a = max(0.0, slot_a[2] - slot_a[0]) * max(0.0, slot_a[3] - slot_a[1])
    area_b = max(0.0, slot_b[2] - slot_b[0]) * max(0.0, slot_b[3] - slot_b[1])
    return slot_b if area_b > area_a else slot_a


def _refine_slot_cc(segments, seed_x, seed_y, slot, cell_size=300, min_dim=5000):
    """Refine a coarse slot using 2D connected-component analysis.

    Builds an occupancy grid from segment rasterization within *slot*, labels
    connected components, and returns the bounding box of the smallest
    component that contains the seed and meets minimum dimension requirements.
    Falls back to the original *slot* when refinement is not beneficial.
    """
    import numpy as np
    from scipy.ndimage import label as ndimage_label

    sx0, sy0, sx1, sy1 = slot
    slot_w = sx1 - sx0
    slot_h = sy1 - sy0
    if slot_w <= 0 or slot_h <= 0:
        return slot
    slot_area = slot_w * slot_h

    nx = max(2, int(np.ceil(slot_w / cell_size)))
    ny = max(2, int(np.ceil(slot_h / cell_size)))
    if nx * ny > 4_000_000:
        return slot

    slot_segs = [
        s for s in segments
        if max(min(s[0], s[2]), sx0) <= min(max(s[0], s[2]), sx1)
        and max(min(s[1], s[3]), sy0) <= min(max(s[1], s[3]), sy1)
    ]
    if not slot_segs:
        return slot

    occ = np.zeros((ny, nx), dtype=bool)
    inv_cs = 1.0 / cell_size
    for seg in slot_segs:
        x1, y1, x2, y2 = seg
        dx = x2 - x1
        dy = y2 - y1
        length = max(abs(dx), abs(dy))
        n_steps = max(1, int(length * inv_cs))
        inv_n = 1.0 / n_steps
        for t in range(n_steps + 1):
            frac = t * inv_n
            ix = int((x1 + frac * dx - sx0) * inv_cs)
            iy = int((y1 + frac * dy - sy0) * inv_cs)
            if 0 <= ix < nx and 0 <= iy < ny:
                occ[iy, ix] = True

    labeled, num_features = ndimage_label(occ)
    if num_features == 0:
        return slot

    min_cells = 30
    containing = []
    for comp_id in range(1, num_features + 1):
        ys_c, xs_c = np.where(labeled == comp_id)
        if len(ys_c) < min_cells:
            continue
        bbox = (
            sx0 + float(xs_c.min()) * cell_size,
            sy0 + float(ys_c.min()) * cell_size,
            sx0 + float(xs_c.max() + 1) * cell_size,
            sy0 + float(ys_c.max() + 1) * cell_size,
        )
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        if bw < min_dim or bh < min_dim:
            continue
        if bbox[0] <= seed_x <= bbox[2] and bbox[1] <= seed_y <= bbox[3]:
            containing.append((bw * bh, bbox))

    if not containing:
        return slot

    containing.sort()
    refined = containing[0][1]
    if containing[0][0] < slot_area * 0.80:
        return refined
    return slot


def _tighten_slot_bbox(geom_points_xy, slot):
    """Tighten a coarse histogram slot to the actual geometry extents inside it."""
    import numpy as np

    mask = (
        (geom_points_xy[:, 0] >= slot[0])
        & (geom_points_xy[:, 0] <= slot[2])
        & (geom_points_xy[:, 1] >= slot[1])
        & (geom_points_xy[:, 1] <= slot[3])
    )
    pts = geom_points_xy[mask]
    if pts.shape[0] == 0:
        return slot
    return (
        float(pts[:, 0].min()),
        float(pts[:, 1].min()),
        float(pts[:, 0].max()),
        float(pts[:, 1].max()),
    )


# ---------------------------------------------------------------------------
# Helpers: formatting, area, filtering
# ---------------------------------------------------------------------------

def _bbox_tuple_to_dict(bbox):
    return {"min_x": float(bbox[0]), "min_y": float(bbox[1]), "max_x": float(bbox[2]), "max_y": float(bbox[3])}


def _format_bbox_string(bbox):
    return f"{bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f}"


def _bbox_area_tuple(bbox):
    return float(max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]))


def _safe_percent(part, whole):
    whole = float(whole)
    if whole <= 1e-12:
        return 0.0
    return float((float(part) / whole) * 100.0)


def _drawing_extent_and_area(data):
    blocks = data.get("blocks", {}) or {}
    points = []
    for ent in data.get("entities", []) or []:
        pts = _get_entity_points(ent, blocks=blocks, depth=0, max_depth=2)
        if pts:
            points.extend(pts)
    if not points:
        return (0.0, 0.0, 0.0, 0.0), 0.0
    bb = _points_extent(points)
    return bb, _bbox_area_tuple(bb)


def _count_geom_in_bbox(entities, blocks, target_bbox):
    import numpy as np

    t_min_x, t_min_y, t_max_x, t_max_y = target_bbox
    count = [0]

    def _walk(ent, parent_matrix, depth=0, max_depth=3):
        if depth > max_depth:
            return
        etype = ent.get("type")

        if etype == "INSERT":
            name = ent.get("name") or ent.get("blockName")
            if not name or name not in blocks:
                return
            p = ent.get("position") or ent.get("insertPoint") or {"x": 0, "y": 0}
            sx = ent.get("xScale", 1) or 1
            sy = ent.get("yScale", 1) or 1
            rot = ent.get("rotation", 0) or 0
            local_m = _get_transform_matrix(p.get("x", 0), p.get("y", 0), sx, sy, rot)
            combined = parent_matrix @ local_m
            block_def = blocks.get(name)
            children = block_def if isinstance(block_def, list) else block_def.get("entities", []) if block_def else []
            for child in children:
                _walk(child, combined, depth=depth + 1, max_depth=max_depth)
            return

        if etype in ("LINE", "LWPOLYLINE", "POLYLINE", "ARC", "CIRCLE"):
            pts = []
            if etype == "LINE":
                s = ent.get("start")
                e = ent.get("end")
                if s and e and "x" in s and "y" in s and "x" in e and "y" in e:
                    pts.append((s["x"], s["y"]))
                    pts.append((e["x"], e["y"]))
            elif etype in ("LWPOLYLINE", "POLYLINE"):
                for v in ent.get("vertices", []) or []:
                    if isinstance(v, dict) and "x" in v and "y" in v:
                        pts.append((v["x"], v["y"]))
            else:
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


def _filter_candidate_bboxes(data, bboxes, threshold=100, min_ar=0.25, max_ar=5.0):
    if not bboxes:
        return []
    blocks = data.get("blocks", {}) or {}
    entities = data.get("entities", []) or []
    valid = []
    for bbox in bboxes:
        min_x, min_y, max_x, max_y = bbox
        w = max_x - min_x
        h = max_y - min_y
        if h <= 1e-9:
            continue
        ar = w / h
        if not (min_ar <= ar <= max_ar):
            continue
        if _count_geom_in_bbox(entities, blocks, bbox) >= threshold:
            valid.append(bbox)
    return valid


def _rerank_bboxes_by_geometry(data, bboxes):
    if not bboxes:
        return []
    blocks = data.get("blocks", {}) or {}
    entities = data.get("entities", []) or []
    scored = []
    for b in bboxes:
        gc = _count_geom_in_bbox(entities, blocks, b)
        scored.append((gc, _bbox_area_tuple(b), b))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [t[2] for t in scored]


def _match_header_line(index, seed, layer):
    return f"[match {index}] {seed} ({layer})"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_region_matches(
    keyword,
    *,
    file_path=None,
    data=None,
    bbox_direct=False,
    topk=5,
    levels=3,
    max_matches_per_name=10,
    threshold=100,
    min_ar=0.25,
    max_ar=5.0,
):
    if (file_path is None) == (data is None):
        raise ValueError("Provide exactly one of file_path or data")

    resolved_path = os.path.abspath(file_path) if file_path is not None else None
    if file_path is not None:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    seeds = _collect_label_seeds(data, keyword)
    if not seeds:
        return {"keyword": keyword, "name": keyword, "path": resolved_path, "matches": {}}

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
            if segments and extent and geom_points is not None and geom_points.shape[0] > 0:
                coarse_slot = _detect_drawing_slot(segments, float(lm["x"]), float(lm["y"]), extent)
                slot = _refine_slot_cc(segments, float(lm["x"]), float(lm["y"]), coarse_slot)
                tight = _tighten_slot_bbox(geom_points, slot)
                b_tuples = [tight]
            else:
                s = 1000.0
                b_tuples = [(lm["x"] - s, lm["y"] - s, lm["x"] + s, lm["y"] + s)]

            b_tuples = _filter_candidate_bboxes(data, b_tuples, threshold=threshold, min_ar=min_ar, max_ar=max_ar)
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

        out[str(i)] = entry

    return {"keyword": keyword, "name": keyword, "path": resolved_path, "matches": out}


def main():
    args = _parse_args()
    print(f"Loading {args.file}...")
    try:
        result = find_region_matches(
            args.name,
            file_path=args.file,
            bbox_direct=args.bbox_direct,
            topk=args.topk,
            levels=args.levels,
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
