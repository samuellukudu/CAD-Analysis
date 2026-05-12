from __future__ import annotations

import json
import os
import argparse
import sys
import re
import unicodedata
import time


_DEBUG_LOG_PATH = "/Users/samuellukudu/QilaiCo/End2End/cad-ai-project/.cursor/debug-03d658.log"
_DEBUG_SESSION_ID = "03d658"


def _debug_log(hypothesis_id, location, message, data, run_id="initial"):
    payload = {
        "sessionId": _DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def decode_cad_unicode(text):
    if not text:
        return ""
    # Decode AutoCAD style \U+XXXX escapes.
    def _replace_unicode(match):
        try:
            return chr(int(match.group(1), 16))
        except Exception:
            return match.group(0)
    text = re.sub(r'\\U\+([0-9A-Fa-f]{4})', _replace_unicode, text)
    # Basic MTEXT cleanup.
    text = text.replace(r'\P', '\n')
    text = re.sub(r'\\[ACFHQTWf].*?;', '', text)
    text = re.sub(r'[{}]', '', text)
    return text.strip()


def _get_transform_matrix(tx, ty, sx, sy, rot):
    import numpy as np

    def _safe_float(value, default):
        try:
            v = float(value)
        except Exception:
            return float(default)
        if not np.isfinite(v):
            return float(default)
        return v

    tx = _safe_float(tx, 0.0)
    ty = _safe_float(ty, 0.0)
    sx = _safe_float(sx, 1.0)
    sy = _safe_float(sy, 1.0)
    rot = _safe_float(rot, 0.0)
    tx = float(np.clip(tx, -1.0e9, 1.0e9))
    ty = float(np.clip(ty, -1.0e9, 1.0e9))
    sx = float(np.clip(sx, -1.0e4, 1.0e4))
    sy = float(np.clip(sy, -1.0e4, 1.0e4))
    rot = float(np.clip(rot, -1.0e7, 1.0e7))

    # rot is often degrees; sometimes radians. Heuristic: if it's "too big", treat as degrees.
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
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.shape[0] == 0:
        return []
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        return []
    ones = np.ones((pts.shape[0], 1), dtype=float)
    pts_h = np.hstack([pts, ones])
    with np.errstate(all="ignore"):
        transformed = (matrix @ pts_h.T).T
    transformed = transformed[np.all(np.isfinite(transformed), axis=1)]
    if transformed.shape[0] == 0:
        return []
    return transformed[:, :2].tolist()


def _entity_insert_position(entity):
    # Try the common CAD fields used in this repo.
    p = entity.get("position") or entity.get("insertPoint") or entity.get("center") or entity.get("startPoint")
    if not p:
        return None
    if "x" in p and "y" in p:
        return {"x": p["x"], "y": p["y"]}
    return None


def _get_entity_points(entity, blocks=None, depth=0, max_depth=2):
    """
    Extract representative points for occupancy/distance-field evidence.
    This mirrors the spirit of `extract_roi_bbox.py`, but is reduced to what we need for bbox scoring.
    """
    etype = entity.get("type")
    points = []

    if etype == "LINE":
        s = entity.get("start")
        e = entity.get("end")
        if s and e and "x" in s and "y" in s and "x" in e and "y" in e:
            points.append((s["x"], s["y"]))
            points.append((e["x"], e["y"]))
            points.append(((s["x"] + e["x"]) / 2.0, (s["y"] + e["y"]) / 2.0))

    elif etype == "LWPOLYLINE":
        verts = entity.get("vertices", [])
        for v in verts:
            if isinstance(v, dict) and "x" in v and "y" in v:
                points.append((v["x"], v["y"]))
        if len(verts) > 1:
            for i in range(len(verts) - 1):
                a = verts[i]
                b = verts[i + 1]
                if isinstance(a, dict) and isinstance(b, dict) and "x" in a and "y" in a and "x" in b and "y" in b:
                    points.append(((a["x"] + b["x"]) / 2.0, (a["y"] + b["y"]) / 2.0))
            if entity.get("closed", False) or entity.get("shape", False):
                a = verts[-1]
                b = verts[0]
                if isinstance(a, dict) and isinstance(b, dict) and "x" in a and "y" in a and "x" in b and "y" in b:
                    points.append(((a["x"] + b["x"]) / 2.0, (a["y"] + b["y"]) / 2.0))

    elif etype == "POLYLINE":
        verts = entity.get("vertices", [])
        for v in verts:
            if isinstance(v, dict) and "x" in v and "y" in v:
                points.append((v["x"], v["y"]))
        if len(verts) > 1:
            for i in range(len(verts) - 1):
                a = verts[i]
                b = verts[i + 1]
                if isinstance(a, dict) and isinstance(b, dict) and "x" in a and "y" in a and "x" in b and "y" in b:
                    points.append(((a["x"] + b["x"]) / 2.0, (a["y"] + b["y"]) / 2.0))
            if entity.get("closed", False) or entity.get("shape", False):
                a = verts[-1]
                b = verts[0]
                if isinstance(a, dict) and isinstance(b, dict) and "x" in a and "y" in a and "x" in b and "y" in b:
                    points.append(((a["x"] + b["x"]) / 2.0, (a["y"] + b["y"]) / 2.0))

    elif etype == "INSERT":
        # Recursively extract points from the referenced block and apply the INSERT transform.
        if depth > max_depth:
            return []
        p = _entity_insert_position(entity)
        if not p:
            return []
        name = entity.get("name")
        if blocks and name in blocks:
            # block definition can be list or dict with "entities"
            block_def = blocks.get(name)
            if isinstance(block_def, list):
                block_entities = block_def
            elif isinstance(block_def, dict):
                block_entities = block_def.get("entities", [])
            else:
                block_entities = []

            local_points = []
            for child in block_entities:
                local_points.extend(_get_entity_points(child, blocks, depth=depth + 1, max_depth=max_depth))
            if local_points:
                sx = entity.get("xScale", 1) or 1
                sy = entity.get("yScale", 1) or 1
                rot = entity.get("rotation", 0) or 0
                matrix = _get_transform_matrix(p["x"], p["y"], sx, sy, rot)
                return _transform_points(local_points, matrix)

        # Fallback: use the insertion position only.
        return [(p["x"], p["y"])]

    elif etype in ["CIRCLE", "ARC", "ELLIPSE", "POINT", "ATTRIB"]:
        p = _entity_insert_position(entity)
        if p:
            points.append((p["x"], p["y"]))
        # Add a bit more for circles/ellipses if radius/major axis is present.
        if etype == "CIRCLE":
            c = entity.get("center") or _entity_insert_position(entity)
            r = entity.get("radius")
            if c and r is not None and "x" in c and "y" in c:
                points.append((c["x"] + r, c["y"]))
                points.append((c["x"] - r, c["y"]))
                points.append((c["x"], c["y"] + r))
                points.append((c["x"], c["y"] - r))
        elif etype in ["ELLIPSE"]:
            c = entity.get("center") or _entity_insert_position(entity)
            major = entity.get("majorAxis")
            if c and major and "x" in c and "y" in c and "x" in major and "y" in major:
                points.append((c["x"] + major["x"], c["y"] + major["y"]))
                points.append((c["x"] - major["x"], c["y"] - major["y"]))

    elif etype == "SOLID":
        for k in ["first", "second", "third", "fourth"]:
            p = entity.get(k)
            if p and "x" in p and "y" in p:
                points.append((p["x"], p["y"]))

    elif etype == "SPLINE":
        for p in entity.get("controlPoints", []) or []:
            if isinstance(p, dict) and "x" in p and "y" in p:
                points.append((p["x"], p["y"]))
        for p in entity.get("fitPoints", []) or []:
            if isinstance(p, dict) and "x" in p and "y" in p:
                points.append((p["x"], p["y"]))

    elif etype == "HATCH":
        loops = entity.get("boundaryLoops", []) or []
        for loop in loops:
            poly = loop.get("polyline")
            if poly:
                verts = poly.get("vertices", []) or []
                for v in verts:
                    if isinstance(v, dict) and "x" in v and "y" in v:
                        points.append((v["x"], v["y"]))
            # Edges fallback
            edges = loop.get("edges", []) or []
            for edge in edges:
                s = edge.get("start")
                e = edge.get("end")
                if s and "x" in s and "y" in s:
                    points.append((s["x"], s["y"]))
                if e and "x" in e and "y" in e:
                    points.append((e["x"], e["y"]))
                c = edge.get("center")
                if c and "x" in c and "y" in c:
                    points.append((c["x"], c["y"]))
                for p in edge.get("controlPoints", []) or []:
                    if isinstance(p, dict) and "x" in p and "y" in p:
                        points.append((p["x"], p["y"]))

    elif etype == "DIMENSION":
        for k in [
            "anchorPoint",
            "middleOfText",
            "linearOrAngularPoint1",
            "linearOrAngularPoint2",
            "textPoint",
            "defPoint",
            "defPoint1",
            "defPoint2",
        ]:
            p = entity.get(k)
            if p and "x" in p and "y" in p:
                points.append((p["x"], p["y"]))

    # Fallback: if entity has position-like fields, capture them.
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
    parser = argparse.ArgumentParser(description="Find matching text labels in CAD JSON.")
    parser.add_argument("--file", required=True, help="Path to the JSON CAD file")
    parser.add_argument(
        "--name",
        required=True,
        help="Name/keyword to search for (e.g., '一层平面图' or 'FJ-MC-10')",
    )

    parser.add_argument(
        "--bbox_direct",
        action="store_true",
        help="If set, output bbox(es) directly instead of seed points (SLAM-like occupancy-grid scoring).",
    )
    parser.add_argument("--topk", type=int, default=5, help="Top-K bboxes to output per label match")
    parser.add_argument("--levels", type=int, default=3, help="Occupancy pyramid levels (small fixed number)")
    parser.add_argument("--max_matches_per_name", type=int, default=10, help="Limit label matches processed")
    parser.add_argument("--threshold", type=int, default=100, help="Minimum geometric entity count to keep a match")
    parser.add_argument("--min_ar", type=float, default=0.35, help="Minimum aspect ratio (width/height)")
    parser.add_argument("--max_ar", type=float, default=5.0, help="Maximum aspect ratio (width/height)")

    return parser.parse_args()


def _get_text_center(entity):
    """Midpoint of TEXT/MTEXT extent when start/end exist; else insertion-style point."""
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
        xy = [(v.get("x"), v.get("y")) for v in verts if isinstance(v, dict) and "x" in v and "y" in v]
        if xy:
            return {
                "x": sum(float(x) for x, _y in xy) / float(len(xy)),
                "y": sum(float(y) for _x, y in xy) / float(len(xy)),
            }
    return None


def _collect_label_seeds(data, keyword):
    def _normalize_for_match(s: str) -> str:
        if s is None:
            return ""
        s = unicodedata.normalize("NFKC", str(s))
        s = re.sub(r"\s+", " ", s).strip()
        return s.casefold()

    def _normalize_loose(s: str) -> str:
        s = _normalize_for_match(s)
        # Unify common drawing separators/range marks and strip punctuation.
        s = (
            s.replace("～", "~")
            .replace("∼", "~")
            .replace("〜", "~")
            .replace("－", "-")
            .replace("—", "-")
            .replace("至", "~")
            .replace("到", "~")
        )
        s = re.sub(r"[\s\-_~]+", "", s)
        return s

    def _looks_like_regex(pattern_str: str) -> bool:
        # Heuristic: treat as explicit regex only if user included metacharacters.
        # This keeps plain multi-word queries behaving like "tokens in order" search.
        regex_meta = set(r".^$*+?{}[]\|()")
        return any((c in regex_meta) for c in pattern_str) or ("\\" in pattern_str)

    keyword_norm = _normalize_for_match(keyword)
    keyword_loose = _normalize_loose(keyword)
    # region agent log
    _debug_log(
        "H1",
        "find_region.py:_collect_label_seeds:keyword_norm",
        "Starting label seed collection",
        {"keyword": keyword, "keyword_norm": keyword_norm},
    )
    # endregion

    pattern = None
    literal_fallback = None

    if keyword_norm:
        if not _looks_like_regex(keyword_norm):
            # Plain-text query: allow the query to appear mid-annotation, with gaps between tokens.
            tokens = [t for t in keyword_norm.split(" ") if t]
            if len(tokens) >= 2:
                token_gap_re = r".*?".join(re.escape(t) for t in tokens)
                pattern = re.compile(token_gap_re)
            else:
                # Single token: exact substring match (escaped) is the least surprising behavior.
                pattern = re.compile(re.escape(tokens[0])) if tokens else None
        else:
            # Explicit regex: preserve existing behavior (auto-regex when compilable).
            try:
                pattern = re.compile(keyword_norm)
            except re.error:
                # If the user provides an invalid regex, fall back to literal substring match.
                pattern = None
                literal_fallback = keyword_norm

    results = []
    blocks = data.get("blocks", {}) or {}
    text_types = {"TEXT", "MTEXT", "ATTRIB", "ATTDEF", "LEADER"}
    top_level_text_count = 0
    top_level_match_count = 0
    block_text_count = 0
    block_match_count = 0
    block_match_samples = []
    all_type_counts = {"TEXT": 0, "MTEXT": 0, "ATTRIB": 0, "ATTDEF": 0, "LEADER": 0}
    loose_match_counts = {"TEXT": 0, "MTEXT": 0, "ATTRIB": 0, "ATTDEF": 0, "LEADER": 0}
    loose_match_samples = []
    generic_field_match_count = 0
    generic_field_match_samples = []
    fragment_char_hits = 0
    fragment_samples = []
    kw_chars = set(ch for ch in keyword_loose if ch)
    seen_match_samples = set()

    def _entity_text_value(ent):
        vals = []
        for k in ("text", "value", "defaultValue", "tag", "name"):
            v = ent.get(k)
            if isinstance(v, str) and v.strip():
                vals.append(v)
        if not vals:
            return ""
        # Prefer explicit text/value fields first.
        return vals[0]

    def _apply_matrix_xy(M, x, y):
        return (
            float(M[0, 0] * x + M[0, 1] * y + M[0, 2]),
            float(M[1, 0] * x + M[1, 1] * y + M[1, 2]),
        )

    # Top-level pass preserves old metrics and catches direct labels quickly.
    for ent in data.get("entities", []):
        et = ent.get("type")
        if et not in text_types:
            continue
        top_level_text_count += 1
        raw_text = _entity_text_value(ent)
        text = decode_cad_unicode(raw_text)
        text_norm = _normalize_for_match(text)
        text_loose = _normalize_loose(text)
        if text_loose and kw_chars:
            overlap = [ch for ch in text_loose if ch in kw_chars]
            if overlap and len(text_loose) <= 12:
                fragment_char_hits += 1
                if len(fragment_samples) < 12:
                    p0 = _get_text_center(ent) or {}
                    fragment_samples.append(
                        {
                            "text": text[:80],
                            "text_loose": text_loose[:80],
                            "overlap_chars": "".join(overlap[:12]),
                            "x": float(p0.get("x", 0.0)),
                            "y": float(p0.get("y", 0.0)),
                            "layer": ent.get("layer", "Unknown"),
                        }
                    )
        if pattern is not None:
            is_match = pattern.search(text_norm) is not None
        else:
            is_match = bool(literal_fallback) and (literal_fallback in text_norm)
        if is_match:
            top_level_match_count += 1

    # Recursive pass (all entities, with INSERT transforms) feeds actual results.
    def on_leaf(ent, etype, M):
        nonlocal block_text_count, block_match_count, generic_field_match_count
        if etype not in text_types:
            return
        block_text_count += 1
        all_type_counts[etype] = all_type_counts.get(etype, 0) + 1
        raw_text = _entity_text_value(ent)
        text = decode_cad_unicode(raw_text)
        text_norm = _normalize_for_match(text)
        text_loose = _normalize_loose(text)

        if keyword_loose and text_loose and (keyword_loose in text_loose):
            loose_match_counts[etype] = loose_match_counts.get(etype, 0) + 1
            if len(loose_match_samples) < 8:
                loose_match_samples.append(
                    {
                        "type": etype,
                        "text": text[:120],
                        "text_norm": text_norm[:120],
                        "text_loose": text_loose[:120],
                        "layer": ent.get("layer", "Unknown"),
                        "depth": -1,
                    }
                )

        if pattern is not None:
            is_match = pattern.search(text_norm) is not None
        else:
            is_match = bool(literal_fallback) and (literal_fallback in text_norm)
        if is_match:
            block_match_count += 1
            if len(block_match_samples) < 5:
                key = (text, ent.get("layer", "Unknown"))
                if key not in seen_match_samples:
                    seen_match_samples.add(key)
                    block_match_samples.append({"text": text, "layer": ent.get("layer", "Unknown"), "depth": -1})
            p = _get_text_center(ent)
            if p and "x" in p and "y" in p:
                xw, yw = _apply_matrix_xy(M, float(p["x"]), float(p["y"]))
                seed_str = f"{xw:.2f},{yw:.2f}"
                results.append(
                    {
                        "text": text,
                        "x": float(xw),
                        "y": float(yw),
                        "layer": ent.get("layer", "Unknown"),
                        "seed": seed_str,
                    }
                )

        # Generic string-field scan to detect labels stored outside the "text" key.
        for kf, vf in ent.items():
            if not isinstance(vf, str):
                continue
            vf_dec = decode_cad_unicode(vf)
            vf_loose = _normalize_loose(vf_dec)
            if keyword_loose and vf_loose and (keyword_loose in vf_loose):
                generic_field_match_count += 1
                if len(generic_field_match_samples) < 10:
                    generic_field_match_samples.append(
                        {
                            "type": etype or "",
                            "field": str(kf),
                            "value": vf_dec[:120],
                            "layer": ent.get("layer", "Unknown"),
                            "depth": -1,
                        }
                    )

    _traverse_entity_tree(
        data.get("entities", []),
        blocks,
        on_leaf=on_leaf,
        on_insert_without_block=None,
        max_depth=5,
    )
    if results:
        dedup = {}
        for r in results:
            dedup[(r["seed"], r["text"], r["layer"])] = r
        results = list(dedup.values())
    # region agent log
    _debug_log(
        "H2_H3_H4_H5",
        "find_region.py:_collect_label_seeds:scan_summary",
        "Top-level vs block-tree match counts",
        {
            "top_level_text_count": top_level_text_count,
            "top_level_match_count": top_level_match_count,
            "returned_result_count": len(results),
            "block_text_count": block_text_count,
            "block_match_count": block_match_count,
            "block_match_samples": block_match_samples,
            "all_type_counts": all_type_counts,
            "keyword_loose": keyword_loose,
            "loose_match_counts": loose_match_counts,
            "loose_match_samples": loose_match_samples,
            "generic_field_match_count": generic_field_match_count,
            "generic_field_match_samples": generic_field_match_samples,
            "fragment_char_hits": fragment_char_hits,
            "fragment_samples": fragment_samples,
            "uses_regex_pattern": pattern is not None,
            "literal_fallback": literal_fallback or "",
        },
    )
    # endregion
    if not results:
        # region agent log
        _debug_log(
            "H1_H2_H3",
            "find_region.py:_collect_label_seeds:no_results",
            "No top-level matching labels found",
            {"keyword_norm": keyword_norm},
        )
        # endregion
        return []
    # Sort primarily by X-column (grouped by ~10m to align stacks left-to-right)
    # Sort secondarily by Y-elevation descending (top floor to bottom floor)
    results.sort(key=lambda r: (round(r["x"] / 10000), -r["y"]))
    # region agent log
    _debug_log(
        "H4",
        "find_region.py:_collect_label_seeds:sorted_results",
        "Returning sorted top-level matches",
        {"result_count": len(results), "first_seed": results[0]["seed"] if results else ""},
    )
    # endregion
    return results


def _collect_layout_title_anchors(data):
    layout_title_re = re.compile(r"(平面图|剖面图|立面图|详图|大样图|节点图|机房图|布置图)")
    blocks = data.get("blocks", {}) or {}
    text_types = {"TEXT", "MTEXT", "ATTRIB", "ATTDEF", "LEADER"}
    results = []

    def _entity_text_value(ent):
        vals = []
        for k in ("text", "value", "defaultValue", "tag", "name"):
            v = ent.get(k)
            if isinstance(v, str) and v.strip():
                vals.append(v)
        return vals[0] if vals else ""

    def _apply_matrix_xy(M, x, y):
        return (
            float(M[0, 0] * x + M[0, 1] * y + M[0, 2]),
            float(M[1, 0] * x + M[1, 1] * y + M[1, 2]),
        )

    def on_leaf(ent, etype, M):
        if etype not in text_types:
            return
        text = decode_cad_unicode(_entity_text_value(ent))
        if not text or len(text) > 48 or layout_title_re.search(text) is None:
            return
        p = _get_text_center(ent)
        if not p or "x" not in p or "y" not in p:
            return
        xw, yw = _apply_matrix_xy(M, float(p["x"]), float(p["y"]))
        results.append({"text": text, "x": xw, "y": yw, "layer": ent.get("layer", "Unknown")})

    _traverse_entity_tree(
        data.get("entities", []),
        blocks,
        on_leaf=on_leaf,
        on_insert_without_block=None,
        max_depth=5,
    )
    dedup = {}
    for r in results:
        dedup[(round(r["x"], 2), round(r["y"], 2), r["text"])] = r
    out = list(dedup.values())
    out.sort(key=lambda r: (round(r["x"] / 10000), -r["y"]))
    return out


def _bbox_iou(b1, b2) -> float:
    """
    IoU for axis-aligned bboxes in (min_x, min_y, max_x, max_y) world coords.
    """
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


def _build_occupancy_bool_from_points(points_xy, extent, cell_size):
    """
    Build boolean occupancy grid and its origin for WORLD->grid mapping.
    extent = (min_x, min_y, max_x, max_y)
    """
    import numpy as np

    min_x, min_y, max_x, max_y = extent
    width = max_x - min_x
    height = max_y - min_y

    nx = int(np.ceil(width / cell_size)) + 1
    ny = int(np.ceil(height / cell_size)) + 1
    nx = max(nx, 2)
    ny = max(ny, 2)

    occ = np.zeros((ny, nx), dtype=bool)
    if not points_xy:
        return occ, min_x, min_y

    for x, y in points_xy:
        ix = int((x - min_x) / cell_size)
        iy = int((y - min_y) / cell_size)
        if 0 <= ix < nx and 0 <= iy < ny:
            occ[iy, ix] = True

    return occ, float(min_x), float(min_y)


def _localize_seed_to_occupied_cell(occ_bool, seed_ix, seed_iy, max_radius_cells=4):
    """
    Find nearest occupied cell to (seed_ix, seed_iy) within max_radius_cells.
    Returns (ix, iy) or None.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    ny, nx = occ_bool.shape
    if not (0 <= seed_ix < nx and 0 <= seed_iy < ny):
        return None

    if occ_bool[seed_iy, seed_ix]:
        return int(seed_ix), int(seed_iy)

    occ_idx = np.argwhere(occ_bool)  # (iy, ix)
    if occ_idx.shape[0] == 0:
        return None

    tree = cKDTree(occ_idx.astype(float))
    dist, i = tree.query([[seed_iy, seed_ix]], k=1)
    if float(dist[0]) > float(max_radius_cells):
        return None

    iy, ix = occ_idx[int(i[0])]
    return int(ix), int(iy)


def _connected_component_bbox(occ_bool, start_ix, start_iy):
    """
    8-connected component bbox in grid indices for an occupancy boolean grid.
    Returns (min_ix, min_iy, max_ix, max_iy) or None.
    """
    import numpy as np

    ny, nx = occ_bool.shape
    if not (0 <= start_ix < nx and 0 <= start_iy < ny):
        return None
    if not occ_bool[start_iy, start_ix]:
        return None

    visited = np.zeros_like(occ_bool, dtype=bool)
    stack = [(start_iy, start_ix)]
    visited[start_iy, start_ix] = True

    min_ix = max_ix = start_ix
    min_iy = max_iy = start_iy

    # BFS/DFS
    while stack:
        y, x = stack.pop()
        min_ix = min(min_ix, x)
        max_ix = max(max_ix, x)
        min_iy = min(min_iy, y)
        max_iy = max(max_iy, y)

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                ny2 = y + dy
                nx2 = x + dx
                if 0 <= ny2 < ny and 0 <= nx2 < nx:
                    if occ_bool[ny2, nx2] and not visited[ny2, nx2]:
                        visited[ny2, nx2] = True
                        stack.append((ny2, nx2))

    return int(min_ix), int(min_iy), int(max_ix), int(max_iy)


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
    """
    Depth-first walk with INSERT expanded via block definitions.
    on_leaf(ent, etype, parent_matrix) receives world-from-local matrix for non-INSERT entities.
    on_insert_without_block(ent) when INSERT has no resolvable block (optional).
    """
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
                for child in children:
                    _traverse_entity_tree(
                        [child],
                        blocks,
                        on_leaf,
                        on_insert_without_block,
                        combined,
                        depth + 1,
                        max_depth,
                    )
            elif on_insert_without_block is not None and (not name or not blocks or name not in blocks):
                on_insert_without_block(ent)
            continue
        on_leaf(ent, etype, parent_matrix)


def _cc_bbox_fill_from_occ(geom_points_xy, seed_x, seed_y, extent, cell_size, dilate_structure, seed_radius_cells=6):
    """
    One pyramid level: dilated occupancy from geom_points_xy in extent, flood from seed.
    Returns (bbox_world_4tuple, fill_ratio) or (None, None).
    """
    import numpy as np
    from scipy.ndimage import binary_dilation

    ex0, ey0, ex1, ey1 = extent
    mask = (
        (geom_points_xy[:, 0] >= ex0)
        & (geom_points_xy[:, 0] <= ex1)
        & (geom_points_xy[:, 1] >= ey0)
        & (geom_points_xy[:, 1] <= ey1)
    )
    pts_sub = geom_points_xy[mask].tolist()
    if not pts_sub:
        return None, None

    occ_bool, origin_x, origin_y = _build_occupancy_bool_from_points(pts_sub, extent, cell_size)
    if not np.any(occ_bool):
        return None, None

    occ_dil = binary_dilation(occ_bool, structure=dilate_structure)
    seed_ix = int((seed_x - origin_x) / cell_size)
    seed_iy = int((seed_y - origin_y) / cell_size)
    start = _localize_seed_to_occupied_cell(occ_dil, seed_ix, seed_iy, max_radius_cells=seed_radius_cells)
    if start is None:
        return None, None

    bbox_idx = _connected_component_bbox(occ_dil, start[0], start[1])
    if bbox_idx is None:
        return None, None

    min_ix, min_iy, max_ix, max_iy = bbox_idx
    bbox = (
        origin_x + min_ix * cell_size,
        origin_y + min_iy * cell_size,
        origin_x + (max_ix + 1) * cell_size,
        origin_y + (max_iy + 1) * cell_size,
    )
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None, None

    total_cells = float((max_ix - min_ix + 1) * (max_iy - min_iy + 1))
    occ_cells = float(np.sum(occ_dil[min_iy : max_iy + 1, min_ix : max_ix + 1]))
    fill_ratio = occ_cells / (total_cells + 1e-9)
    return bbox, fill_ratio


def _collect_structural_world_points(entities, blocks, max_depth=3):
    import numpy as np

    acc = []

    def add_local(M, pts):
        for x, y in pts:
            v = M @ np.array([float(x), float(y), 1.0], dtype=float)
            acc.append((float(v[0]), float(v[1])))

    def on_leaf(ent, etype, M):
        if etype == "SOLID":
            pts = []
            for k in ["first", "second", "third", "fourth"]:
                p = ent.get(k)
                if p and "x" in p and "y" in p:
                    pts.append((p["x"], p["y"]))
            if pts:
                add_local(M, pts)
        elif etype in ("LWPOLYLINE", "POLYLINE"):
            if not (ent.get("closed", False) or ent.get("shape", False)):
                return
            verts = ent.get("vertices", []) or []
            pts = []
            for v in verts:
                if isinstance(v, dict) and "x" in v and "y" in v:
                    pts.append((v["x"], v["y"]))
            if pts:
                add_local(M, pts)
        elif etype == "HATCH":
            loops = ent.get("boundaryLoops", []) or []
            for loop in loops:
                poly = loop.get("polyline")
                if poly:
                    verts = poly.get("vertices", []) or []
                    pts = []
                    for v in verts:
                        if isinstance(v, dict) and "x" in v and "y" in v:
                            pts.append((v["x"], v["y"]))
                    if pts:
                        add_local(M, pts)
                for edge in loop.get("edges", []) or []:
                    s = edge.get("start")
                    e = edge.get("end")
                    if s and "x" in s and "y" in s:
                        add_local(M, [(s["x"], s["y"])])
                    if e and "x" in e and "y" in e:
                        add_local(M, [(e["x"], e["y"])])
                    c = edge.get("center")
                    if c and "x" in c and "y" in c:
                        add_local(M, [(c["x"], c["y"])])

    _traverse_entity_tree(entities, blocks, on_leaf, on_insert_without_block=None, max_depth=max_depth)
    return acc


def _dense_sample_points_in_slot(entities, blocks, sample_step, slot_rect, max_points, max_depth=3):
    """Sample points along LINE / open+closed polylines / HATCH / position fallbacks, clipped to slot."""
    import math

    import numpy as np

    dense_ex0, dense_ey0, dense_ex1, dense_ey1 = slot_rect
    bucket_w = max(float(sample_step), 1e-9)
    seen_buckets = set()

    def stamp(xw, yw, acc):
        if len(acc) >= max_points:
            return
        if dense_ex0 <= xw <= dense_ex1 and dense_ey0 <= yw <= dense_ey1:
            bx = int((float(xw) - dense_ex0) / bucket_w)
            by = int((float(yw) - dense_ey0) / bucket_w)
            key = (bx, by)
            if key in seen_buckets:
                return
            seen_buckets.add(key)
            acc.append((float(xw), float(yw)))

    def sample_segment(ax, ay, bx, by, step, acc):
        length = float(np.hypot(bx - ax, by - ay))
        if length <= 1e-9:
            stamp(ax, ay, acc)
            return
        n = int(math.ceil(length / max(step, 1e-9)))
        n = min(max(n, 1), 800)
        ts = np.linspace(0.0, 1.0, n + 1, dtype=float)
        xs = ax + ts * (bx - ax)
        ys = ay + ts * (by - ay)
        for xw, yw in zip(xs.tolist(), ys.tolist()):
            stamp(xw, yw, acc)
            if len(acc) >= max_points:
                return

    def sample_polyline(coords, step, acc, closed=False):
        if len(coords) < 2:
            return
        for i in range(len(coords) - 1):
            ax, ay = coords[i]
            bx, by = coords[i + 1]
            sample_segment(ax, ay, bx, by, step, acc)
            if len(acc) >= max_points:
                return
        if closed and len(coords) > 2:
            sample_segment(coords[-1][0], coords[-1][1], coords[0][0], coords[0][1], step, acc)

    def _as_radians(angle):
        a = float(angle)
        if abs(a) > 2 * math.pi + 1e-6:
            return float(np.radians(a))
        return a

    acc = []

    def on_insert_no_block(ent):
        p = _entity_insert_position(ent)
        if p:
            stamp(p["x"], p["y"], acc)

    def on_leaf(ent, etype, M):
        if len(acc) >= max_points:
            return
        if etype == "LINE":
            s = ent.get("start")
            e = ent.get("end")
            if not s or not e or "x" not in s or "y" not in s or "x" not in e or "y" not in e:
                return
            aw = (M @ np.array([s["x"], s["y"], 1.0], dtype=float))[:2]
            bw = (M @ np.array([e["x"], e["y"], 1.0], dtype=float))[:2]
            sample_segment(aw[0], aw[1], bw[0], bw[1], sample_step, acc)
            return
        if etype in ("ARC", "CIRCLE"):
            c = ent.get("center") or _entity_insert_position(ent)
            r = ent.get("radius")
            if not c or r is None or "x" not in c or "y" not in c:
                return
            cw = (M @ np.array([float(c["x"]), float(c["y"]), 1.0], dtype=float))[:2]
            radius = float(r)
            if radius <= 1e-9:
                stamp(cw[0], cw[1], acc)
                return
            if etype == "ARC":
                start_angle = _as_radians(ent.get("startAngle", 0.0) or 0.0)
                end_angle = _as_radians(ent.get("endAngle", 0.0) or 0.0)
                span = end_angle - start_angle
                if span <= 0:
                    span += 2.0 * math.pi
                n = int(min(64, max(12, math.ceil(span / (math.pi / 16.0)))))
                ts = np.linspace(start_angle, start_angle + span, n, dtype=float)
                closed = False
            else:
                ts = np.linspace(0.0, 2.0 * math.pi, 24, dtype=float)
                closed = True
            coords = [(float(cw[0] + radius * math.cos(t)), float(cw[1] + radius * math.sin(t))) for t in ts.tolist()]
            sample_polyline(coords, sample_step, acc, closed=closed)
            return
        if etype == "ELLIPSE":
            c = ent.get("center") or _entity_insert_position(ent)
            major = ent.get("majorAxis")
            ratio = ent.get("axisRatio", 1.0)
            if not c or not major or "x" not in c or "y" not in c or "x" not in major or "y" not in major:
                return
            ratio = float(ratio) if ratio is not None else 1.0
            major_vec = np.array([float(major["x"]), float(major["y"])], dtype=float)
            a = float(np.hypot(major_vec[0], major_vec[1]))
            if a <= 1e-9:
                return
            b = a * max(0.05, min(20.0, ratio))
            cw = (M @ np.array([float(c["x"]), float(c["y"]), 1.0], dtype=float))[:2]
            ang = float(math.atan2(major_vec[1], major_vec[0]))
            ca = math.cos(ang)
            sa = math.sin(ang)
            ts = np.linspace(0.0, 2.0 * math.pi, 28, dtype=float)
            coords = []
            for t in ts.tolist():
                ct = math.cos(t)
                st = math.sin(t)
                x0 = a * ct
                y0 = b * st
                xr = x0 * ca - y0 * sa
                yr = x0 * sa + y0 * ca
                coords.append((float(cw[0] + xr), float(cw[1] + yr)))
            sample_polyline(coords, sample_step, acc, closed=True)
            return
        if etype == "SPLINE":
            pts = ent.get("fitPoints", []) or ent.get("controlPoints", []) or []
            coords = []
            for p in pts:
                if not isinstance(p, dict) or "x" not in p or "y" not in p:
                    continue
                v = (M @ np.array([float(p["x"]), float(p["y"]), 1.0], dtype=float))[:2]
                coords.append((float(v[0]), float(v[1])))
            if len(coords) >= 2:
                sample_polyline(coords, sample_step, acc, closed=False)
            return
        if etype in ("LWPOLYLINE", "POLYLINE"):
            verts = ent.get("vertices", []) or []
            if len(verts) < 2:
                return
            coords = []
            for v in verts:
                if not isinstance(v, dict) or "x" not in v or "y" not in v:
                    continue
                v_w = M @ np.array([float(v["x"]), float(v["y"]), 1.0], dtype=float)
                coords.append((float(v_w[0]), float(v_w[1])))
            if len(coords) < 2:
                return
            closed = bool(ent.get("closed", False) or ent.get("shape", False))
            sample_polyline(coords, sample_step, acc, closed=closed)
            return
        if etype == "HATCH":
            loops = ent.get("boundaryLoops", []) or []
            for loop in loops:
                poly = loop.get("polyline")
                if poly:
                    verts = poly.get("vertices", []) or []
                    if len(verts) < 2:
                        continue
                    coords = []
                    for v in verts:
                        if not isinstance(v, dict) or "x" not in v or "y" not in v:
                            continue
                        v_w = M @ np.array([float(v["x"]), float(v["y"]), 1.0], dtype=float)
                        coords.append((float(v_w[0]), float(v_w[1])))
                    if len(coords) < 2:
                        continue
                    sample_polyline(coords, sample_step, acc, closed=len(coords) > 2)
                    if len(acc) >= max_points:
                        return
                for edge in loop.get("edges", []) or []:
                    s = edge.get("start")
                    e = edge.get("end")
                    if s and e and "x" in s and "y" in s and "x" in e and "y" in e:
                        aw = (M @ np.array([s["x"], s["y"], 1.0], dtype=float))[:2]
                        bw = (M @ np.array([e["x"], e["y"], 1.0], dtype=float))[:2]
                        sample_segment(aw[0], aw[1], bw[0], bw[1], sample_step, acc)
                        if len(acc) >= max_points:
                            return
            return
        p = _entity_insert_position(ent)
        if p:
            v_w = M @ np.array([p["x"], p["y"], 1.0], dtype=float)
            stamp(float(v_w[0]), float(v_w[1]), acc)

    _traverse_entity_tree(entities, blocks, on_leaf, on_insert_no_block, max_depth=max_depth)
    return acc


def _segment_aabb_intersects_roi(roi, ax, ay, bx, by):
    """Axis-aligned bbox vs segment AABB overlap (cheap filter)."""
    r0, r1, r2, r3 = roi
    sminx, smaxx = (ax, bx) if ax <= bx else (bx, ax)
    sminy, smaxy = (ay, by) if ay <= by else (by, ay)
    if smaxx < r0 or sminx > r2 or smaxy < r1 or sminy > r3:
        return False
    return True


def _segment_angle_length(ax, ay, bx, by):
    import math

    dx, dy = bx - ax, by - ay
    L = math.hypot(dx, dy)
    if L < 1e-12:
        return None, 0.0
    th = abs(math.atan2(dy, dx))
    th = min(th, math.pi - th)  # undirected angle in [0, pi/2]
    return th, float(L)


def _is_orthogonal_direction(th, deg_tol=12.0):
    import math

    d = math.radians(deg_tol)
    return th < d or abs(th - math.pi / 2) < d


def _collect_roi_segment_features(entities, blocks, roi, local_s, max_segments=8000):
    """
    Cheap vector-only statistics in a world ROI (no text).
    Supports soft plan vs strip priors (Phase 2, strategy 1).
    """
    import math

    import numpy as np

    r = tuple(roi)
    total_line_len = 0.0
    ortho_line_len = 0.0
    hatch_boundary_len = 0.0
    lengths_for_q = []
    n_dimlike = 0
    n_bins = 9
    hist = np.zeros(n_bins, dtype=float)
    bin_w = (math.pi / 2) / n_bins
    n_seg = 0
    ls = max(float(local_s), 1e-9)

    def _bin_index(th):
        return int(min(n_bins - 1, max(0, th / bin_w)))

    def _emit_segment(ax, ay, bx, by, is_hatch):
        nonlocal n_seg, total_line_len, ortho_line_len, hatch_boundary_len, lengths_for_q
        if n_seg >= max_segments:
            return
        if not _segment_aabb_intersects_roi(r, ax, ay, bx, by):
            return
        th, L = _segment_angle_length(ax, ay, bx, by)
        if L <= 0 or th is None:
            return
        n_seg += 1
        hist[_bin_index(th)] += L
        if is_hatch:
            hatch_boundary_len += L
        else:
            total_line_len += L
            if _is_orthogonal_direction(th):
                ortho_line_len += L
            lengths_for_q.append(L / ls)

    def _emit_polyline(coords, is_hatch, closed=False):
        if len(coords) < 2:
            return
        for i in range(len(coords) - 1):
            ax, ay = coords[i]
            bx, by = coords[i + 1]
            _emit_segment(ax, ay, bx, by, is_hatch)
            if n_seg >= max_segments:
                return
        if closed and len(coords) > 2 and n_seg < max_segments:
            _emit_segment(coords[-1][0], coords[-1][1], coords[0][0], coords[0][1], is_hatch)

    def _as_radians(angle):
        a = float(angle)
        if abs(a) > 2 * math.pi + 1e-6:
            return float(np.radians(a))
        return a

    def _walk(ent, parent_matrix, depth=0, max_depth=3):
        nonlocal n_dimlike
        if depth > max_depth or n_seg >= max_segments:
            return
        etype = ent.get("type")

        if etype == "INSERT":
            name = ent.get("name")
            ip = _entity_insert_position(ent)
            if not ip:
                return
            sx = ent.get("xScale", 1) or 1
            sy = ent.get("yScale", 1) or 1
            rot = ent.get("rotation", 0) or 0
            local_m = _get_transform_matrix(ip["x"], ip["y"], sx, sy, rot)
            combined = parent_matrix @ local_m
            if name and blocks and name in blocks:
                block_def = blocks.get(name)
                if isinstance(block_def, list):
                    children = block_def
                elif isinstance(block_def, dict):
                    children = block_def.get("entities", [])
                else:
                    children = []
                for child in children:
                    _walk(child, combined, depth=depth + 1, max_depth=max_depth)
            return

        if etype == "DIMENSION":
            r0, r1, r2, r3 = r
            for k in (
                "anchorPoint",
                "middleOfText",
                "linearOrAngularPoint1",
                "linearOrAngularPoint2",
                "textPoint",
                "defPoint",
                "defPoint1",
                "defPoint2",
            ):
                p = ent.get(k)
                if p and "x" in p and "y" in p:
                    v = parent_matrix @ np.array([float(p["x"]), float(p["y"]), 1.0], dtype=float)
                    xw, yw = float(v[0]), float(v[1])
                    if r0 <= xw <= r2 and r1 <= yw <= r3:
                        n_dimlike += 1
                        break
            return

        if etype == "LEADER":
            for p in ent.get("vertices", []) or []:
                if isinstance(p, dict) and "x" in p and "y" in p:
                    v = parent_matrix @ np.array([float(p["x"]), float(p["y"]), 1.0], dtype=float)
                    if r[0] <= v[0] <= r[2] and r[1] <= v[1] <= r[3]:
                        n_dimlike += 1
                        break
            return

        if etype == "LINE":
            s = ent.get("start")
            e = ent.get("end")
            if not s or not e or "x" not in s or "y" not in s or "x" not in e or "y" not in e:
                return
            aw = parent_matrix @ np.array([s["x"], s["y"], 1.0], dtype=float)
            bw = parent_matrix @ np.array([e["x"], e["y"], 1.0], dtype=float)
            _emit_segment(float(aw[0]), float(aw[1]), float(bw[0]), float(bw[1]), False)
            return
        if etype in ("ARC", "CIRCLE"):
            c = ent.get("center") or _entity_insert_position(ent)
            radius_value = ent.get("radius")
            if not c or radius_value is None or "x" not in c or "y" not in c:
                return
            cw = parent_matrix @ np.array([float(c["x"]), float(c["y"]), 1.0], dtype=float)
            radius = float(radius_value)
            if radius <= 1e-9:
                return
            if etype == "ARC":
                start_angle = _as_radians(ent.get("startAngle", 0.0) or 0.0)
                end_angle = _as_radians(ent.get("endAngle", 0.0) or 0.0)
                span = end_angle - start_angle
                if span <= 0:
                    span += 2.0 * math.pi
                n = int(min(48, max(10, math.ceil(span / (math.pi / 18.0)))))
                ts = np.linspace(start_angle, start_angle + span, n, dtype=float)
                closed = False
            else:
                ts = np.linspace(0.0, 2.0 * math.pi, 20, dtype=float)
                closed = True
            coords = [(float(cw[0] + radius * math.cos(t)), float(cw[1] + radius * math.sin(t))) for t in ts.tolist()]
            _emit_polyline(coords, False, closed=closed)
            return
        if etype == "ELLIPSE":
            c = ent.get("center") or _entity_insert_position(ent)
            major = ent.get("majorAxis")
            ratio = ent.get("axisRatio", 1.0)
            if not c or not major or "x" not in c or "y" not in c or "x" not in major or "y" not in major:
                return
            ratio = float(ratio) if ratio is not None else 1.0
            major_vec = np.array([float(major["x"]), float(major["y"])], dtype=float)
            a = float(np.hypot(major_vec[0], major_vec[1]))
            if a <= 1e-9:
                return
            b = a * max(0.05, min(20.0, ratio))
            cw = parent_matrix @ np.array([float(c["x"]), float(c["y"]), 1.0], dtype=float)
            ang = float(math.atan2(major_vec[1], major_vec[0]))
            ca = math.cos(ang)
            sa = math.sin(ang)
            ts = np.linspace(0.0, 2.0 * math.pi, 22, dtype=float)
            coords = []
            for t in ts.tolist():
                ct = math.cos(t)
                st = math.sin(t)
                x0 = a * ct
                y0 = b * st
                xr = x0 * ca - y0 * sa
                yr = x0 * sa + y0 * ca
                coords.append((float(cw[0] + xr), float(cw[1] + yr)))
            _emit_polyline(coords, False, closed=True)
            return
        if etype == "SPLINE":
            pts = ent.get("fitPoints", []) or ent.get("controlPoints", []) or []
            coords = []
            for p in pts:
                if not isinstance(p, dict) or "x" not in p or "y" not in p:
                    continue
                v = parent_matrix @ np.array([float(p["x"]), float(p["y"]), 1.0], dtype=float)
                coords.append((float(v[0]), float(v[1])))
            if len(coords) >= 2:
                _emit_polyline(coords, False, closed=False)
            return

        if etype in ("LWPOLYLINE", "POLYLINE"):
            verts = ent.get("vertices", []) or []
            if len(verts) < 2:
                return
            coords = []
            for v in verts:
                if not isinstance(v, dict) or "x" not in v or "y" not in v:
                    continue
                v_w = parent_matrix @ np.array([float(v["x"]), float(v["y"]), 1.0], dtype=float)
                coords.append((float(v_w[0]), float(v_w[1])))
            if len(coords) < 2:
                return
            closed = bool(ent.get("closed", False) or ent.get("shape", False))
            _emit_polyline(coords, False, closed=closed)
            return

        if etype == "HATCH":
            loops = ent.get("boundaryLoops", []) or []
            for loop in loops:
                poly = loop.get("polyline")
                if poly:
                    verts = poly.get("vertices", []) or []
                    if len(verts) < 2:
                        continue
                    coords = []
                    for v in verts:
                        if not isinstance(v, dict) or "x" not in v or "y" not in v:
                            continue
                        v_w = parent_matrix @ np.array([float(v["x"]), float(v["y"]), 1.0], dtype=float)
                        coords.append((float(v_w[0]), float(v_w[1])))
                    if len(coords) < 2:
                        continue
                    _emit_polyline(coords, True, closed=len(coords) > 2)
                for edge in loop.get("edges", []) or []:
                    s = edge.get("start")
                    ept = edge.get("end")
                    if s and ept and "x" in s and "y" in s and "x" in ept and "y" in ept:
                        aw = parent_matrix @ np.array([s["x"], s["y"], 1.0], dtype=float)
                        bw = parent_matrix @ np.array([ept["x"], ept["y"], 1.0], dtype=float)
                        _emit_segment(float(aw[0]), float(aw[1]), float(bw[0]), float(bw[1]), True)
            return

    identity = np.eye(3, dtype=float)
    for ent in entities:
        _walk(ent, identity, depth=0, max_depth=3)

    total_stroke = total_line_len + hatch_boundary_len
    if total_line_len > 1e-9:
        ortho_frac = ortho_line_len / total_line_len
    else:
        ortho_frac = 0.35

    hatch_ratio = hatch_boundary_len / (total_stroke + 1e-9)

    H = 0.0
    if hist.sum() > 1e-9:
        p = hist / hist.sum()
        p = p[p > 1e-12]
        H = float(-np.sum(p * np.log(p + 1e-15)))
    h_max = math.log(n_bins)
    norm_entropy = H / h_max if h_max > 1e-9 else 1.0

    if lengths_for_q:
        median_len_ratio = float(np.median(np.asarray(lengths_for_q, dtype=float)))
    else:
        median_len_ratio = 1.0

    return {
        "ortho_frac": ortho_frac,
        "hatch_ratio": hatch_ratio,
        "norm_entropy": norm_entropy,
        "n_dimlike": n_dimlike,
        "total_line_len": total_line_len,
        "hatch_boundary_len": hatch_boundary_len,
        "median_len_ratio": median_len_ratio,
    }


def _soft_geometry_priors(feats, span_x, span_y):
    """
    Map handcrafted ROI features + local kNN elongation to soft plan-like / strip-like cues in [0, 1].
    Fixed transparent weights (no training).
    """
    elong = max(span_x, span_y) / max(min(span_x, span_y), 1e-9)
    ortho_frac = float(feats.get("ortho_frac", 0.35))
    hatch_ratio = float(feats.get("hatch_ratio", 0.0))
    norm_entropy = float(feats.get("norm_entropy", 1.0))
    n_dim = int(feats.get("n_dimlike", 0))
    med_lr = float(feats.get("median_len_ratio", 1.0))

    elong_plan_term = max(0.0, 1.0 - min(1.0, (elong - 1.0) / 7.0))
    elong_strip_term = min(1.0, max(0.0, (elong - 1.0) / 11.0))

    grid_plan_term = ortho_frac * (0.55 + 0.45 * (1.0 - norm_entropy))
    hatch_term = min(1.0, hatch_ratio * 2.0)
    dim_term = min(1.0, n_dim / 25.0)
    len_balance = max(0.0, 1.0 - abs(med_lr - 1.0) / 3.0)

    geo_plan = (
        0.38 * elong_plan_term
        + 0.28 * grid_plan_term
        + 0.18 * hatch_term
        + 0.10 * dim_term
        + 0.06 * len_balance
    )

    geo_strip = 0.72 * elong_strip_term + 0.18 * (1.0 - ortho_frac) + 0.10 * min(1.0, norm_entropy)

    geo_plan = max(0.0, min(1.0, geo_plan))
    geo_strip = max(0.0, min(1.0, geo_strip))
    return geo_plan, geo_strip


def _geom_aspect_bonus(bbox, geo_plan, geo_strip):
    """Small bounded bonus so candidate aspect matches inferred layout (plan vs strip)."""
    w = max(bbox[2] - bbox[0], 1e-9)
    h = max(bbox[3] - bbox[1], 1e-9)
    asp = max(w / h, h / w)

    strength = 0.32
    if geo_strip >= geo_plan:
        t = min(1.0, max(0.0, (asp - 1.0) / 14.0))
        bonus = strength * geo_strip * t
        if asp < 2.0:
            bonus -= 0.12 * strength * geo_strip * (2.0 - asp) / 2.0
        return bonus
    excess = max(0.0, asp - 5.0) / 15.0
    return strength * geo_plan * max(0.0, 1.0 - excess)


def _bbox_contains_point(bbox, x, y, margin=0.0):
    return (bbox[0] - margin) <= x <= (bbox[2] + margin) and (bbox[1] - margin) <= y <= (bbox[3] + margin)


def _bbox_distance_to_point(bbox, x, y):
    dx = max(bbox[0] - x, 0.0, x - bbox[2])
    dy = max(bbox[1] - y, 0.0, y - bbox[3])
    return float((dx * dx + dy * dy) ** 0.5)


def _poly_area(coords):
    if len(coords) < 3:
        return 0.0
    area = 0.0
    for i in range(len(coords)):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % len(coords)]
        area += x1 * y2 - x2 * y1
    return float(abs(area) * 0.5)


def _poly_ortho_frac(coords):
    import math

    if len(coords) < 2:
        return 0.0
    total = 0.0
    ortho = 0.0
    closed_coords = list(coords)
    if coords[0] != coords[-1]:
        closed_coords = list(coords) + [coords[0]]
    for i in range(len(closed_coords) - 1):
        ax, ay = closed_coords[i]
        bx, by = closed_coords[i + 1]
        dx = bx - ax
        dy = by - ay
        seg_len = float((dx * dx + dy * dy) ** 0.5)
        if seg_len <= 1e-9:
            continue
        total += seg_len
        th = abs(math.atan2(dy, dx))
        th = min(th, math.pi - th)
        if _is_orthogonal_direction(th, deg_tol=10.0):
            ortho += seg_len
    if total <= 1e-9:
        return 0.0
    return float(ortho / total)


def _collect_layout_frame_candidates(entities, blocks, slot_rect=None, max_depth=3):
    import numpy as np

    candidates = []

    def _intersects_slot(bbox):
        if slot_rect is None:
            return True
        return _bbox_iou(bbox, slot_rect) > 0.0 or not (
            bbox[2] < slot_rect[0] or bbox[0] > slot_rect[2] or bbox[3] < slot_rect[1] or bbox[1] > slot_rect[3]
        )

    def _append_candidate(coords):
        if len(coords) < 4:
            return
        bbox = _points_extent(coords)
        area_bbox = _bbox_area_tuple(bbox)
        if area_bbox <= 1e-6 or not _intersects_slot(bbox):
            return
        poly_area = _poly_area(coords)
        rectangularity = float(poly_area / (area_bbox + 1e-9))
        ortho_frac = _poly_ortho_frac(coords)
        candidates.append(
            {
                "bbox": bbox,
                "rectangularity": rectangularity,
                "ortho_frac": ortho_frac,
                "area": area_bbox,
            }
        )

    def on_leaf(ent, etype, M):
        if etype in ("LWPOLYLINE", "POLYLINE"):
            if not bool(ent.get("closed", False) or ent.get("shape", False)):
                return
            verts = ent.get("vertices", []) or []
            coords = []
            for v in verts:
                if not isinstance(v, dict) or "x" not in v or "y" not in v:
                    continue
                vw = M @ np.array([float(v["x"]), float(v["y"]), 1.0], dtype=float)
                if np.all(np.isfinite(vw[:2])):
                    coords.append((float(vw[0]), float(vw[1])))
            _append_candidate(coords)
            return
        if etype == "HATCH":
            loops = ent.get("boundaryLoops", []) or []
            for loop in loops:
                poly = loop.get("polyline")
                if not poly:
                    continue
                verts = poly.get("vertices", []) or []
                coords = []
                for v in verts:
                    if not isinstance(v, dict) or "x" not in v or "y" not in v:
                        continue
                    vw = M @ np.array([float(v["x"]), float(v["y"]), 1.0], dtype=float)
                    if np.all(np.isfinite(vw[:2])):
                        coords.append((float(vw[0]), float(vw[1])))
                _append_candidate(coords)

    _traverse_entity_tree(entities, blocks, on_leaf=on_leaf, on_insert_without_block=None, max_depth=max_depth)
    candidates.sort(key=lambda c: (c["rectangularity"], c["ortho_frac"], c["area"]), reverse=True)

    out = []
    for cand in candidates:
        bb = cand["bbox"]
        if any(_bbox_iou(bb, prev["bbox"]) >= 0.85 for prev in out):
            continue
        out.append(cand)
        if len(out) >= 120:
            break
    return out


def _anchor_partition_bbox(points_xy, base_bbox, anchors_xy, target_anchor, local_s):
    import numpy as np

    if points_xy.shape[0] == 0:
        return None
    mask = (
        (points_xy[:, 0] >= base_bbox[0])
        & (points_xy[:, 0] <= base_bbox[2])
        & (points_xy[:, 1] >= base_bbox[1])
        & (points_xy[:, 1] <= base_bbox[3])
    )
    pts = points_xy[mask]
    if pts.shape[0] < 20:
        return None
    anchors = np.asarray(list(anchors_xy), dtype=float).reshape(-1, 2)
    if anchors.shape[0] == 0:
        return None
    target = np.asarray(target_anchor, dtype=float).reshape(1, 2)
    target_idx = int(np.argmin(np.sum((anchors - target) ** 2, axis=1)))
    if anchors.shape[0] > 1:
        d2 = np.sum((pts[:, None, :] - anchors[None, :, :]) ** 2, axis=2)
        nearest_idx = np.argmin(d2, axis=1)
        target_pts = pts[nearest_idx == target_idx]
        if target_pts.shape[0] < 20:
            target_d = np.sqrt(d2[:, target_idx])
            second_d = np.partition(d2, 1, axis=1)[:, 1] ** 0.5
            dominance = target_d <= np.maximum(0.82 * second_d, 4.0 * float(local_s))
            target_pts = pts[dominance]
    else:
        target_pts = pts
    if target_pts.shape[0] < 20:
        return None
    px = np.percentile(target_pts[:, 0], [1.0, 99.0])
    py = np.percentile(target_pts[:, 1], [1.0, 99.0])
    pad = max(2.0 * float(local_s), 0.015 * max(base_bbox[2] - base_bbox[0], base_bbox[3] - base_bbox[1]))
    refined = (float(px[0] - pad), float(py[0] - pad), float(px[1] + pad), float(py[1] + pad))
    if refined[2] <= refined[0] or refined[3] <= refined[1]:
        return None
    base_area = _bbox_area_tuple(base_bbox)
    refined_area = _bbox_area_tuple(refined)
    if refined_area < 0.12 * base_area or refined_area > 1.05 * base_area:
        return None
    return refined


def _detect_drawing_slot(geom_points_xy, seed_x, seed_y, local_s):
    """
    Determine the axis-aligned bounding box of the drawing slot that contains
    (seed_x, seed_y) by detecting zero-density whitespace corridors in 1D
    projections of all geometry points.

    Algorithm (O(N) + O(B) where B is the number of histogram bins, a small
    constant):
      1. Project all points onto X and Y axes.
      2. Bin counts into a coarse histogram whose bin width is proportional to
         the local geometry scale `local_s`.  Bins with zero count represent
         whitespace corridors that physically separate distinct drawings.
      3. Walking outward from the seed in each axis direction, the first
         zero-count bin we cross marks the boundary of the drawing slot.
      4. If no zero bin is found in a direction the full extent is used
         (meaning the entire point cloud in that direction belongs to one
         drawing — correct behaviour for a single-drawing file).

    Parameters
    ----------
    geom_points_xy : np.ndarray, shape (N, 2)
    seed_x, seed_y : float  – world coordinates of the matched label anchor
    local_s        : float  – median nearest-neighbour distance near seed
                              (sets the histogram resolution)

    Returns
    -------
    (slot_min_x, slot_min_y, slot_max_x, slot_max_y) : tuple of float
    """
    import numpy as np

    if geom_points_xy.shape[0] == 0:
        half = max(local_s * 4.0, 1000.0)
        return (seed_x - half, seed_y - half, seed_x + half, seed_y + half)

    xs = geom_points_xy[:, 0]
    ys = geom_points_xy[:, 1]

    global_min_x, global_max_x = float(xs.min()), float(xs.max())
    global_min_y, global_max_y = float(ys.min()), float(ys.max())

    def _find_slot_1d(coords, seed_coord, g_min, g_max, local_s_axis):
        """Return (slot_lo, slot_hi) along a single axis."""
        span = g_max - g_min
        if span <= 1e-9:
            return g_min, g_max

        s0 = max(float(local_s_axis), 1e-6)
        desired_bin_w = max(s0 * 8.0, span / 320.0)
        n_bins = int(np.ceil(span / desired_bin_w))
        n_bins = int(max(80, min(420, n_bins)))
        bin_w = span / float(n_bins)

        # Count points per bin.
        hist = np.zeros(n_bins, dtype=np.int64)
        idxs = np.floor((coords - g_min) / bin_w).astype(np.int64)
        idxs = np.clip(idxs, 0, n_bins - 1)
        np.add.at(hist, idxs, 1)

        nonzero = hist[hist > 0]
        if nonzero.size == 0:
            return g_min, g_max
        med = float(np.median(nonzero))
        p95 = float(np.percentile(nonzero, 95.0))
        gap_thresh = int(max(0.0, min(med * 0.10, p95 * 0.02)))
        content_thresh = int(max(1.0, med * 0.30))
        gap_run = int(max(2, min(6, round(0.015 * n_bins))))

        def _nearest_content_bin(seed_bin):
            if hist[seed_bin] >= content_thresh:
                return seed_bin
            for offset in range(1, n_bins):
                lo_try = seed_bin - offset
                hi_try = seed_bin + offset
                if 0 <= lo_try and hist[lo_try] >= content_thresh:
                    return lo_try
                if hi_try < n_bins and hist[hi_try] >= content_thresh:
                    return hi_try
            if hist[seed_bin] > gap_thresh:
                return seed_bin
            for offset in range(1, n_bins):
                lo_try = seed_bin - offset
                hi_try = seed_bin + offset
                if 0 <= lo_try and hist[lo_try] > gap_thresh:
                    return lo_try
                if hi_try < n_bins and hist[hi_try] > gap_thresh:
                    return hi_try
            return seed_bin

        def _find_boundary(seed_bin, direction):
            consec = 0
            b = seed_bin
            while 0 <= b < n_bins:
                if hist[b] <= gap_thresh:
                    consec += 1
                    if consec >= gap_run:
                        if direction < 0:
                            return b + 1
                        return b - gap_run
                else:
                    consec = 0
                b += direction
            return 0 if direction < 0 else (n_bins - 1)

        seed_bin = int((seed_coord - g_min) / bin_w)
        seed_bin = max(0, min(n_bins - 1, seed_bin))
        seed_bin = _nearest_content_bin(seed_bin)
        lo_bin = _find_boundary(seed_bin, -1)
        hi_bin = _find_boundary(seed_bin, +1)
        lo_bin = int(max(0, min(n_bins - 1, lo_bin)))
        hi_bin = int(max(0, min(n_bins - 1, hi_bin)))
        if hi_bin < lo_bin:
            lo_bin, hi_bin = hi_bin, lo_bin

        slot_lo = g_min + lo_bin * bin_w
        slot_hi = g_min + (hi_bin + 1) * bin_w
        return float(slot_lo), float(slot_hi)

    slot_min_x, slot_max_x = _find_slot_1d(xs, seed_x, global_min_x, global_max_x, local_s)
    slot_min_y, slot_max_y = _find_slot_1d(ys, seed_y, global_min_y, global_max_y, local_s)

    return (slot_min_x, slot_min_y, slot_max_x, slot_max_y)


def _select_bbox_connected_component(data, keyword, label_match, topk: int, levels: int):
    """
    Connected-component ROI from a multi-scale dilated occupancy grid (floor-plan friendly).
    Returns a list of (min_x, min_y, max_x, max_y) tuples in world units.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    blocks = data.get("blocks", {})
    entities = data.get("entities", [])
    seed_x, seed_y = float(label_match["x"]), float(label_match["y"])

    struct_pts = _collect_structural_world_points(entities, blocks, max_depth=3)
    if struct_pts:
        struct_pts = list({(round(x, 3), round(y, 3)) for (x, y) in struct_pts})
        struct_points_xy = np.asarray(struct_pts, dtype=float)
    else:
        struct_points_xy = np.zeros((0, 2), dtype=float)

    def _structural_present(bbox):
        if struct_points_xy.shape[0] == 0:
            return False
        min_x, min_y, max_x, max_y = bbox
        x = struct_points_xy[:, 0]
        y = struct_points_xy[:, 1]
        mask = (x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)
        return bool(np.any(mask))

    geom_points = []
    max_entity_dim = 100000
    for ent in entities:
        pts = _get_entity_points(ent, blocks=blocks, depth=0, max_depth=3)
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if (max(xs) - min(xs) > max_entity_dim) or (max(ys) - min(ys) > max_entity_dim):
            continue
        geom_points.extend(pts)

    if not geom_points:
        return []

    geom_points = list({(round(x, 3), round(y, 3)) for (x, y) in geom_points})
    geom_points_xy = np.asarray(geom_points, dtype=float)
    geom_points_xy_unclipped = geom_points_xy.copy()
    tree = cKDTree(geom_points_xy)

    k = min(25, len(geom_points))
    dists, idxs = tree.query([[seed_x, seed_y]], k=k)
    dists = np.asarray(dists[0], dtype=float)
    dists = dists[dists > 1e-9]
    if len(dists) == 0:
        s = 1000.0
    else:
        s = float(np.median(dists[: min(len(dists), 15)]))
        if s <= 0:
            s = 1000.0
    nearest_dist = float(dists[0]) if len(dists) > 0 else None

    idxs0 = np.asarray(idxs[0], dtype=int)
    idxs0 = np.atleast_1d(idxs0).reshape(-1)
    idxs0 = idxs0[(idxs0 >= 0) & (idxs0 < geom_points_xy.shape[0])]
    if idxs0.size == 0:
        idxs0 = np.asarray(
            [int(np.argmin(np.sum((geom_points_xy - np.asarray([[seed_x, seed_y]], dtype=float)) ** 2, axis=1)))],
            dtype=int,
        )
    near_pts = geom_points_xy[idxs0].reshape(-1, 2)
    near_xs = near_pts[:, 0]
    near_ys = near_pts[:, 1]
    span_x = float(np.max(near_xs) - np.min(near_xs))
    span_y = float(np.max(near_ys) - np.min(near_ys))
    span_x = span_x if span_x > 1e-6 else s
    span_y = span_y if span_y > 1e-6 else s

    level_count_cc2 = max(1, int(levels))
    cell_finest = s / (2.0 ** max(0, level_count_cc2 - 1))
    cell_finest = max(cell_finest, s / 16.0)
    sample_step = cell_finest / 2.0

    slot_rect = _detect_drawing_slot(geom_points_xy, seed_x, seed_y, s)
    dense_ex0, dense_ey0, dense_ex1, dense_ey1 = slot_rect
    seed_inside_slot = bool(dense_ex0 <= seed_x <= dense_ex1 and dense_ey0 <= seed_y <= dense_ey1)
    prefer_frames_only = False

    gx0, gy0, gx1, gy1 = _points_extent(geom_points_xy.tolist())
    global_w = max(1.0, gx1 - gx0)
    global_h = max(1.0, gy1 - gy0)

    def clamp_extent(ext):
        ex0, ey0, ex1, ey1 = ext
        pad_x = 0.01 * global_w
        pad_y = 0.01 * global_h
        ex0 = max(gx0 - pad_x, ex0)
        ey0 = max(gy0 - pad_y, ey0)
        ex1 = min(gx1 + pad_x, ex1)
        ey1 = min(gy1 + pad_y, ey1)
        return (ex0, ey0, ex1, ey1)

    if not seed_inside_slot:
        rescue_half_w = max(span_x * 16.0, 120.0 * s, 0.18 * global_w)
        rescue_half_h = max(span_y * 16.0, 120.0 * s, 0.18 * global_h)
        rescue_extent = clamp_extent((seed_x - rescue_half_w, seed_y - rescue_half_h, seed_x + rescue_half_w, seed_y + rescue_half_h))
        rx0, ry0, rx1, ry1 = rescue_extent
        x0 = geom_points_xy_unclipped[:, 0]
        y0 = geom_points_xy_unclipped[:, 1]
        local_mask = (x0 >= rx0) & (x0 <= rx1) & (y0 >= ry0) & (y0 <= ry1)
        local_points = geom_points_xy_unclipped[local_mask]
        if local_points.shape[0] >= 50:
            sx0, sy0, sx1, sy1 = _detect_drawing_slot(local_points, seed_x, seed_y, s)
            slot_rect = (max(rx0, sx0), max(ry0, sy0), min(rx1, sx1), min(ry1, sy1))
        else:
            slot_rect = rescue_extent
        dense_ex0, dense_ey0, dense_ex1, dense_ey1 = slot_rect
        seed_inside_slot = bool(dense_ex0 <= seed_x <= dense_ex1 and dense_ey0 <= seed_y <= dense_ey1)
        if not seed_inside_slot:
            slot_rect = rescue_extent
            dense_ex0, dense_ey0, dense_ex1, dense_ey1 = slot_rect
            seed_inside_slot = True
            prefer_frames_only = True

    slot_span_x = max(1.0, dense_ex1 - dense_ex0)
    slot_span_y = max(1.0, dense_ey1 - dense_ey0)
    span_x = max(span_x, 0.08 * slot_span_x, 8.0 * s)
    span_y = max(span_y, 0.08 * slot_span_y, 8.0 * s)

    if struct_points_xy.shape[0] > 0 and seed_inside_slot:
        sx_arr = struct_points_xy[:, 0]
        sy_arr = struct_points_xy[:, 1]
        slot_mask = (
            (sx_arr >= dense_ex0) & (sx_arr <= dense_ex1) & (sy_arr >= dense_ey0) & (sy_arr <= dense_ey1)
        )
        struct_points_xy = struct_points_xy[slot_mask]

    if seed_inside_slot:
        dense_points = _dense_sample_points_in_slot(
            entities, blocks, sample_step, slot_rect, max_points=250000, max_depth=3
        )
        if dense_points:
            dense_points = list({(round(x, 3), round(y, 3)) for (x, y) in dense_points})
            geom_points_xy = np.asarray(dense_points, dtype=float)
    else:
        geom_points_xy = geom_points_xy_unclipped

    label_matches = _collect_label_seeds(data, keyword)
    if not label_matches:
        label_matches = [label_match]
    target_anchors = [(float(lm["x"]), float(lm["y"])) for lm in label_matches]
    target_anchor = (seed_x, seed_y)
    layout_title_matches = _collect_layout_title_anchors(data)
    competitive_anchors = []
    for lm in layout_title_matches:
        ax = float(lm["x"])
        ay = float(lm["y"])
        if dense_ex0 - 0.10 * slot_span_x <= ax <= dense_ex1 + 0.10 * slot_span_x and dense_ey0 - 0.10 * slot_span_y <= ay <= dense_ey1 + 0.10 * slot_span_y:
            competitive_anchors.append((ax, ay))
    for ax, ay in target_anchors:
        competitive_anchors.append((float(ax), float(ay)))
    competitive_anchors = list({(round(ax, 2), round(ay, 2)) for ax, ay in competitive_anchors})
    partition_anchors = target_anchors if len(target_anchors) > 1 else competitive_anchors

    def count_anchors_inside(bbox):
        min_x, min_y, max_x, max_y = bbox
        cnt = 0
        for ax, ay in competitive_anchors:
            if min_x <= ax <= max_x and min_y <= ay <= max_y:
                cnt += 1
        return cnt

    def extra_anchor_count(bbox):
        return max(0, count_anchors_inside(bbox) - 1)

    gx0, gy0, gx1, gy1 = _points_extent(geom_points_xy.tolist())
    global_w = max(1.0, gx1 - gx0)
    global_h = max(1.0, gy1 - gy0)

    feat_pad_x = max(span_x * 6.0, s * 8.0)
    feat_pad_y = max(span_y * 6.0, s * 8.0)
    feat_roi = clamp_extent((seed_x - feat_pad_x, seed_y - feat_pad_y, seed_x + feat_pad_x, seed_y + feat_pad_y))
    geo_feats = _collect_roi_segment_features(entities, blocks, feat_roi, local_s=s, max_segments=8000)
    geo_plan, geo_strip = _soft_geometry_priors(geo_feats, span_x, span_y)

    slot_area = max(1.0, slot_span_x * slot_span_y)
    frame_candidates = _collect_layout_frame_candidates(entities, blocks, slot_rect=slot_rect, max_depth=3)
    scored_frames = []
    for cand in frame_candidates:
        bb = cand["bbox"]
        area = max(1.0, cand["area"])
        area_frac = area / slot_area
        if area_frac < 0.002 or area_frac > 0.80:
            continue
        dist = _bbox_distance_to_point(bb, seed_x, seed_y)
        proximity = max(0.0, 1.0 - dist / max(slot_span_x, slot_span_y, 1.0))
        near_seed = 1.0 if _bbox_contains_point(bb, seed_x, seed_y, margin=6.0 * s) else 0.0
        struct_present = _structural_present(bb)
        anchor_penalty = 0.18 * float(extra_anchor_count(bb))
        size_term = max(0.0, 1.0 - abs(area_frac - 0.06) / 0.18)
        large_penalty = max(0.0, area_frac - 0.22) * 1.5
        score = (
            0.30 * near_seed
            + 0.26 * proximity
            + 0.22 * float(cand["rectangularity"])
            + 0.12 * float(cand["ortho_frac"])
            + 0.10 * (1.0 if struct_present else 0.0)
            + 0.08 * size_term
            - anchor_penalty
            - large_penalty
        )
        scored_frames.append((score + 0.25, bb, -1, struct_present, cand["rectangularity"]))

    view_window_mult = 16.0
    level_count = max(1, int(levels))
    stage_levels = list(range(level_count))
    dilate_radius_cells = 1
    structure = np.ones((2 * dilate_radius_cells + 1, 2 * dilate_radius_cells + 1), dtype=bool)

    view_candidates = []
    prev_bbox = None
    for level in stage_levels:
        cell_size = max(s / (2.0 ** level), s / 16.0)
        half_w = max(span_x * view_window_mult, 0.22 * slot_span_x, 22.0 * s)
        half_h = max(span_y * view_window_mult, 0.22 * slot_span_y, 22.0 * s)
        extent = clamp_extent((seed_x - half_w, seed_y - half_h, seed_x + half_w, seed_y + half_h))
        bbox, fill_ratio = _cc_bbox_fill_from_occ(
            geom_points_xy, seed_x, seed_y, extent, cell_size, structure, seed_radius_cells=6
        )
        if bbox is None:
            continue

        text_cnt = 1 if _bbox_contains_point(bbox, seed_x, seed_y, margin=2.0 * s) else 0
        extra_anchors = extra_anchor_count(bbox)
        stability = _bbox_iou(bbox, prev_bbox) if prev_bbox is not None else 0.0
        struct_present = _structural_present(bbox)
        area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        size_penalty = -area / float(max(1.0, global_w * global_h))
        geom_b = _geom_aspect_bonus(bbox, geo_plan, geo_strip)
        struct_f = 1.0 if struct_present else 0.0
        score = (
            0.12 * float(text_cnt)
            + 0.30 * float(stability)
            + 0.10 * float(size_penalty)
            + 0.22 * struct_f
            + 0.26 * float(fill_ratio)
            + geom_b
            - 0.32 * float(extra_anchors)
        )
        view_candidates.append((score, bbox, level, struct_present, fill_ratio))
        prev_bbox = bbox

    # region agent log
    _debug_log(
        "H8",
        "find_region.py:_select_bbox_connected_component:slot_seed_relation",
        "Seed relation to detected slot and unclipped baseline",
        {
            "keyword": keyword,
            "seed_x": seed_x,
            "seed_y": seed_y,
            "slot_rect": [dense_ex0, dense_ey0, dense_ex1, dense_ey1],
            "seed_inside_slot": seed_inside_slot,
            "slot_clipping_enabled": seed_inside_slot,
            "clipped_point_count": int(geom_points_xy.shape[0]),
            "unclipped_point_count": int(geom_points_xy_unclipped.shape[0]),
        },
    )
    # endregion

    # Compare with a coarse unclipped baseline run (instrumentation only).
    try:
        test_cell = max(s / (2.0 ** max(0, level_count - 1)), s / 16.0)
        test_half_w = span_x * view_window_mult
        test_half_h = span_y * view_window_mult
        test_extent = clamp_extent((seed_x - test_half_w, seed_y - test_half_h, seed_x + test_half_w, seed_y + test_half_h))
        test_bbox, test_fill = _cc_bbox_fill_from_occ(
            geom_points_xy_unclipped, seed_x, seed_y, test_extent, test_cell, structure, seed_radius_cells=6
        )
        _debug_log(
            "H8",
            "find_region.py:_select_bbox_connected_component:unclipped_probe",
            "Unclipped occupancy probe around seed",
            {
                "test_extent": [float(v) for v in test_extent],
                "test_cell": float(test_cell),
                "test_fill": None if test_fill is None else float(test_fill),
                "test_bbox": None if test_bbox is None else [float(v) for v in test_bbox],
            },
        )
    except Exception as e:
        _debug_log(
            "H8",
            "find_region.py:_select_bbox_connected_component:unclipped_probe_error",
            "Unclipped occupancy probe failed",
            {"error": str(e)[:200]},
        )

    def _rank_key(r):
        score, bbox, lvl, struct_present, fill_ratio = r
        return (float(score), 1.0 if struct_present else 0.0, float(fill_ratio))

    view_candidates.sort(key=_rank_key, reverse=True)
    top_view = view_candidates[: min(3, len(view_candidates))]

    inner_candidates = []
    inner_levels = stage_levels[max(0, level_count - 2) :]
    for _, view_bbox, _, _, _ in top_view:
        pad_world = 0.02 * (view_bbox[2] - view_bbox[0] + view_bbox[3] - view_bbox[1]) + 2.0 * s
        inner_extent_base = (
            max(gx0, view_bbox[0] - pad_world),
            max(gy0, view_bbox[1] - pad_world),
            min(gx1, view_bbox[2] + pad_world),
            min(gy1, view_bbox[3] + pad_world),
        )
        prev_inner_bbox = None
        for level in inner_levels:
            cell_size = max(s / (2.0 ** level), s / 16.0)
            bbox, fill_ratio = _cc_bbox_fill_from_occ(
                geom_points_xy, seed_x, seed_y, inner_extent_base, cell_size, structure, seed_radius_cells=6
            )
            if bbox is None:
                continue

            text_cnt = 1 if _bbox_contains_point(bbox, seed_x, seed_y, margin=2.0 * s) else 0
            extra_anchors = extra_anchor_count(bbox)
            stability = _bbox_iou(bbox, prev_inner_bbox) if prev_inner_bbox is not None else 0.0
            struct_present = _structural_present(bbox)
            area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            size_penalty = -area / float(
                max(1.0, (view_bbox[2] - view_bbox[0]) * (view_bbox[3] - view_bbox[1]) + 1e-9)
            )
            geom_b = _geom_aspect_bonus(bbox, geo_plan, geo_strip)
            struct_f = 1.0 if struct_present else 0.0
            score = (
                0.10 * float(text_cnt)
                + 0.28 * float(stability)
                + 0.10 * float(size_penalty)
                + 0.24 * struct_f
                + 0.28 * float(fill_ratio)
                + geom_b
                - 0.34 * float(extra_anchors)
            )
            inner_candidates.append((score, bbox, level, struct_present, fill_ratio))
            prev_inner_bbox = bbox

    # Merge candidates.
    if prefer_frames_only and scored_frames:
        merged = [(sc, bb, lvl, sp, fr) for (sc, bb, lvl, sp, fr) in scored_frames]
    else:
        merged = (
            [(sc, bb, lvl, sp, fr) for (sc, bb, lvl, sp, fr) in scored_frames]
            + [(sc, bb, lvl, sp, fr) for (sc, bb, lvl, sp, fr) in view_candidates]
            + [(sc, bb, lvl, sp, fr) for (sc, bb, lvl, sp, fr) in inner_candidates]
        )
    if not merged:
        return []

    # Dedupe by IoU overlap.
    merged.sort(key=lambda r: (float(r[0]), 1.0 if r[3] else 0.0, float(r[4])), reverse=True)
    out_bboxes = []
    for sc, bb, _lvl, _sp, _fr in merged:
        refined_bb = _anchor_partition_bbox(geom_points_xy, bb, partition_anchors, target_anchor, s) or bb
        if any(_bbox_iou(refined_bb, prev) >= 0.6 for prev in out_bboxes):
            continue
        out_bboxes.append(refined_bb)
        if len(out_bboxes) >= topk:
            break

    # If dedupe reduced too much, pad with remaining high-score bboxes.
    if len(out_bboxes) < topk:
        for sc, bb, _lvl, _sp, _fr in merged:
            refined_bb = _anchor_partition_bbox(geom_points_xy, bb, partition_anchors, target_anchor, s) or bb
            if any(_bbox_iou(refined_bb, prev) >= 0.9 for prev in out_bboxes):
                continue
            out_bboxes.append(refined_bb)
            if len(out_bboxes) >= topk:
                break

    if not out_bboxes:
        return []

    # region agent log
    # H11: top candidate may be loose and include title/legend strips.
    # Compute geometry-only tight extents inside top1 for trim diagnostics.
    try:
        top1 = out_bboxes[0]
        x = geom_points_xy[:, 0]
        y = geom_points_xy[:, 1]
        m = (x >= top1[0]) & (x <= top1[2]) & (y >= top1[1]) & (y <= top1[3])
        pts = geom_points_xy[m]
        tight_box = None
        tight_p1_p99 = None
        if pts.shape[0] >= 20:
            px = np.percentile(pts[:, 0], [1.0, 99.0])
            py = np.percentile(pts[:, 1], [1.0, 99.0])
            tight_p1_p99 = [float(px[0]), float(py[0]), float(px[1]), float(py[1])]
            ex = _points_extent(pts.tolist())
            tight_box = [float(ex[0]), float(ex[1]), float(ex[2]), float(ex[3])]
        _debug_log(
            "H11",
            "find_region.py:_select_bbox_connected_component:tighten_probe",
            "Geometry-only tightening probe for top1 bbox",
            {
                "keyword": keyword,
                "top1_bbox": [float(v) for v in top1],
                "points_in_top1": int(pts.shape[0]),
                "tight_bbox_minmax": tight_box,
                "tight_bbox_p1_p99": tight_p1_p99,
            },
        )
    except Exception as e:
        _debug_log(
            "H11",
            "find_region.py:_select_bbox_connected_component:tighten_probe_error",
            "Tightening probe failed",
            {"error": str(e)[:200]},
        )
    # endregion

    # region agent log
    _debug_log(
        "H6",
        "find_region.py:_select_bbox_connected_component:selection_summary",
        "BBox selection diagnostics for seed",
        {
            "keyword": keyword,
            "seed_x": seed_x,
            "seed_y": seed_y,
            "local_scale_s": s,
            "nearest_geom_dist": nearest_dist,
            "slot_rect": [dense_ex0, dense_ey0, dense_ex1, dense_ey1],
            "view_candidate_count": len(view_candidates),
            "inner_candidate_count": len(inner_candidates),
            "output_bbox_count": len(out_bboxes),
            "output_bboxes": [[float(b[0]), float(b[1]), float(b[2]), float(b[3])] for b in out_bboxes[:5]],
            "output_areas": [float((b[2] - b[0]) * (b[3] - b[1])) for b in out_bboxes[:5]],
        },
    )
    # endregion

    return out_bboxes


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
    Mean of bbox coordinates across candidates.

    For each candidate bbox (min_x, min_y, max_x, max_y), we average each coordinate
    independently across ``boxes`` to form a "mean rectangle".
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
    bbox = _points_extent(points)
    return bbox, _bbox_area_tuple(bbox)


def _select_candidate_bboxes(data, keyword, label_match, topk: int, levels: int):
    """
    Connected-component occupancy ROI for one label match.
    Returns up to topk bbox strings (min_x,min_y,max_x,max_y).
    """
    raw = _select_bbox_connected_component(data, keyword, label_match, topk=topk, levels=levels)
    return [_format_bbox_string(b) for b in raw]


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
            pts_np = pts_np[np.all(np.isfinite(pts_np), axis=1)]
            if pts_np.shape[0] == 0:
                return
            ones = np.ones((pts_np.shape[0], 1), dtype=float)
            pts_h = np.hstack([pts_np, ones])
            with np.errstate(all="ignore"):
                transformed = (parent_matrix @ pts_h.T).T
            transformed = transformed[np.all(np.isfinite(transformed), axis=1)]
            if transformed.shape[0] == 0:
                return
            
            min_x = np.min(transformed[:, 0])
            min_y = np.min(transformed[:, 1])
            max_x = np.max(transformed[:, 0])
            max_y = np.max(transformed[:, 1])
            
            if not (max_x < t_min_x or min_x > t_max_x or max_y < t_min_y or min_y > t_max_y):
                count[0] += 1

    import numpy as np
    identity = np.eye(3, dtype=float)
    for ent in entities:
        _walk(ent, identity, depth=0, max_depth=3)
        
    return count[0]

def _filter_candidate_bboxes(data, bboxes, threshold=100, min_ar=0.35, max_ar=5.0):
    if not bboxes:
        return []
    blocks = data.get("blocks", {})
    entities = data.get("entities", [])

    def _run_filter(thr, ar_min, ar_max):
        valid = []
        for bbox in bboxes:
            min_x, min_y, max_x, max_y = bbox
            width = max_x - min_x
            height = max_y - min_y
            if height == 0:
                continue
            ar = width / height
            if not (ar_min <= ar <= ar_max):
                continue
            geom_count = _count_geom_in_bbox(entities, blocks, bbox)
            if geom_count >= thr:
                valid.append(bbox)
        return valid

    valid_bboxes = _run_filter(int(threshold), float(min_ar), float(max_ar))
    if valid_bboxes:
        return valid_bboxes

    relaxed_threshold = int(max(20, int(threshold) // 3))
    relaxed_min_ar = float(min(0.10, float(min_ar)))
    relaxed_max_ar = float(max(12.0, float(max_ar)))
    relaxed_bboxes = _run_filter(relaxed_threshold, relaxed_min_ar, relaxed_max_ar)
    if relaxed_bboxes:
        return relaxed_bboxes

    stats = _bbox_debug_stats(data, bboxes)
    scored = []
    for st, bb in zip(stats, bboxes):
        scored.append((st["geom_count"], st["geom_density"], -abs(st["ar"] - 1.5), -st["area"], bb))
    scored.sort(reverse=True)
    return [t[-1] for t in scored]


def _bbox_debug_stats(data, bboxes):
    blocks = data.get("blocks", {})
    entities = data.get("entities", [])
    out = []
    for bb in bboxes:
        min_x, min_y, max_x, max_y = bb
        w = float(max_x - min_x)
        h = float(max_y - min_y)
        ar = float(w / h) if h > 1e-9 else 0.0
        area = float(max(0.0, w) * max(0.0, h))
        gc = int(_count_geom_in_bbox(entities, blocks, bb))
        out.append(
            {
                "bbox": [float(min_x), float(min_y), float(max_x), float(max_y)],
                "w": w,
                "h": h,
                "area": area,
                "ar": ar,
                "geom_count": gc,
                "geom_density": float(gc / (area + 1e-9)),
            }
        )
    return out


def _rerank_bboxes_by_geometry(data, bboxes):
    if not bboxes:
        return []
    stats = _bbox_debug_stats(data, bboxes)
    scored = []
    for st, bb in zip(stats, bboxes):
        compactness = st["geom_density"]
        richness = st["geom_count"]
        area_penalty = -st["area"]
        aspect_bonus = -abs(st["ar"] - 1.6)
        scored.append((compactness, richness, aspect_bonus, area_penalty, bb))
    scored.sort(reverse=True)
    return [t[4] for t in scored]


def _match_header_line(index, seed, layer):
    """Same text as CLI ``[match {i}] seed (layer)`` (no trailing newline)."""
    return f"[match {index}] {seed} ({layer})"


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
    min_ar=0.35,
    max_ar=5.0,
):
    """
    Load CAD JSON (or use pre-parsed `data`), find TEXT/MTEXT labels matching `keyword`
    (regex if valid, else substring), and return structured results for reuse in other code.

    Provide exactly one of ``file_path`` or ``data``.

    Returns
    -------
    dict
        ``{"keyword": str, "name": str, "path": str | None, "matches": {"1": {...}, ...}}`` —
        ``name`` duplicates ``keyword`` for callers; ``path`` is the absolute JSON path when
        ``file_path`` was given, else ``None``. Keys under ``matches`` are 1-based indices as
        strings (mirrors ``[match 6]`` in CLI output).

        Each match value has:

        - ``index`` (int): same as the CLI match number
        - ``header`` (str): e.g. ``"[match 6] 1191410.04,590271.45 (PUB_TEXT)"``
        - ``seed``, ``layer``, ``text``, ``x``, ``y``
        - ``bboxes`` (list[str]): bbox lines as ``"min_x,min_y,max_x,max_y"`` (two decimals),
          same as printed under the header when ``bbox_direct`` is True; otherwise ``[]``
        - ``bboxes_xy`` (list[dict]): parsed corners ``{"min_x","min_y","max_x","max_y"}``;
          present only when ``bbox_direct`` is True
        - ``bbox_union`` (str), ``bbox_union_xy`` (dict): axis-aligned union of all Top-K
          candidate bboxes; present only when ``bbox_direct`` is True
        - ``candidates_mean_bbox`` (str): mean rectangle as bbox CSV
          ``\"mean_min_x,mean_min_y,mean_max_x,mean_max_y\"``; present only when ``bbox_direct`` is True
        - ``candidates_mean_bbox_xyxy`` (dict): mean of each candidate bbox's coordinates
          ``mean(min_x), mean(min_y), mean(max_x), mean(max_y)``; present only when ``bbox_direct`` is True
    """
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
    if bbox_direct:
        drawing_bbox_t, drawing_area = _drawing_extent_and_area(data)

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
            raw_b_tuples = _select_bbox_connected_component(data, keyword, lm, topk=int(topk), levels=int(levels))
            b_tuples = _filter_candidate_bboxes(
                data, raw_b_tuples, threshold=threshold, min_ar=min_ar, max_ar=max_ar
            )
            if not b_tuples and raw_b_tuples:
                b_tuples = list(raw_b_tuples)
            b_tuples = _rerank_bboxes_by_geometry(data, b_tuples)
            # region agent log
            _debug_log(
                "H9",
                "find_region.py:find_region_matches:candidate_shape_geom_stats",
                "Candidate bbox geometry stats for ranking analysis",
                {
                    "keyword": keyword,
                    "match_index": i,
                    "seed_x": float(lm["x"]),
                    "seed_y": float(lm["y"]),
                    "raw_candidate_count": len(raw_b_tuples),
                    "candidate_stats": _bbox_debug_stats(data, b_tuples),
                },
            )
            _debug_log(
                "H10",
                "find_region.py:find_region_matches:geometry_rerank_top1",
                "Top1 after geometry rerank",
                {
                    "keyword": keyword,
                    "match_index": i,
                    "top1_after_rerank": [float(v) for v in b_tuples[0]] if b_tuples else [],
                },
            )
            # endregion
            entry["bboxes"] = [_format_bbox_string(b) for b in b_tuples]
            entry["bboxes_xy"] = [_bbox_tuple_to_dict(b) for b in b_tuples]
            union_t = _union_bbox_tuples(b_tuples)
            entry["bbox_union"] = _format_bbox_string(union_t)
            entry["bbox_union_xy"] = _bbox_tuple_to_dict(union_t)
            mean_rect = _mean_bbox_xyxy_from_bboxes(b_tuples)
            mean_tuple = (mean_rect["min_x"], mean_rect["min_y"], mean_rect["max_x"], mean_rect["max_y"])
            # Use top-ranked candidate as representative bbox for compatibility.
            # Coordinate-mean bbox is kept under explicit raw_mean keys for diagnostics.
            best_bbox = b_tuples[0] if b_tuples else mean_tuple
            entry["best_bbox"] = _format_bbox_string(best_bbox)
            entry["best_bbox_xyxy"] = _bbox_tuple_to_dict(best_bbox)
            entry["candidates_mean_bbox"] = _format_bbox_string(best_bbox)
            entry["candidates_mean_bbox_xyxy"] = _bbox_tuple_to_dict(best_bbox)
            best_area = _bbox_area_tuple(best_bbox)
            union_area = _bbox_area_tuple(union_t)
            entry["best_area"] = float(best_area)
            entry["best_area_pct_of_drawing"] = _safe_percent(best_area, drawing_area)
            entry["union_area"] = float(union_area)
            entry["union_area_pct_of_drawing"] = _safe_percent(union_area, drawing_area)
            entry["drawing_bbox"] = _format_bbox_string(drawing_bbox_t)
            entry["drawing_bbox_xyxy"] = _bbox_tuple_to_dict(drawing_bbox_t)
            entry["drawing_area"] = float(drawing_area)
            entry["candidates_raw_mean_bbox"] = _format_bbox_string(mean_tuple)
            entry["candidates_raw_mean_bbox_xyxy"] = {
                "min_x": float(mean_rect["min_x"]),
                "min_y": float(mean_rect["min_y"]),
                "max_x": float(mean_rect["max_x"]),
                "max_y": float(mean_rect["max_y"]),
            }
            # region agent log
            def _bbox_area(bb):
                return float(max(0.0, bb[2] - bb[0]) * max(0.0, bb[3] - bb[1]))

            def _bbox_center(bb):
                return float((bb[0] + bb[2]) * 0.5), float((bb[1] + bb[3]) * 0.5)

            top1 = b_tuples[0] if b_tuples else (0.0, 0.0, 0.0, 0.0)
            t1cx, t1cy = _bbox_center(top1)
            mcx, mcy = _bbox_center(mean_tuple)
            ucx, ucy = _bbox_center(union_t)
            _debug_log(
                "H7",
                "find_region.py:find_region_matches:output_bbox_summary",
                "Compare top1 vs mean vs union bboxes",
                {
                    "keyword": keyword,
                    "match_index": i,
                    "seed_x": float(lm["x"]),
                    "seed_y": float(lm["y"]),
                    "candidate_count": len(b_tuples),
                    "top1_bbox": [float(v) for v in top1],
                    "mean_bbox": [float(v) for v in mean_tuple],
                    "union_bbox": [float(v) for v in union_t],
                    "top1_area": _bbox_area(top1),
                    "mean_area": _bbox_area(mean_tuple),
                    "union_area": _bbox_area(union_t),
                    "d_seed_top1_center": float(((t1cx - lm["x"]) ** 2 + (t1cy - lm["y"]) ** 2) ** 0.5),
                    "d_seed_mean_center": float(((mcx - lm["x"]) ** 2 + (mcy - lm["y"]) ** 2) ** 0.5),
                    "d_seed_union_center": float(((ucx - lm["x"]) ** 2 + (ucy - lm["y"]) ** 2) ** 0.5),
                },
            )
            # endregion
        out[key] = entry

    return {
        "keyword": keyword,
        "name": keyword,
        "path": resolved_path,
        "matches": out,
    }


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
