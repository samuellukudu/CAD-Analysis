import os
import math
import json
import logging
from collections import defaultdict
from typing import List, Optional, Dict, Any, Union

import pandas as pd
from PIL import Image
from tqdm import tqdm
import dspy

from src.retrieve_pages import retrieve_pages_with_metadata
from src.utils import pdf_pages_to_base64_pngs, base64_to_pillow


def search_building_codes(query: str, top_k: int = 5, top_p: float = 0.95) -> List[dspy.Image]:
    """
    Retrieve the most relevant building code pages as images based on the query.
    Uses the global index to find pages from uploaded PDFs and the standard building code.
    """
    results = retrieve_pages_with_metadata(query=query, top_k=top_k, top_p=top_p)

    images = []
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    storage_pdfs_dir = os.path.join(project_root, "storage", "pdfs")

    for res in results:
        pdf_id = res.get("pdf_id")
        full_path = res.get("full_path")
        page_idx = res.get("page_number", 1) - 1

        pdf_path = None

        if pdf_id:
            candidate = os.path.join(storage_pdfs_dir, f"{pdf_id}.pdf")
            if os.path.exists(candidate):
                pdf_path = candidate

        if not pdf_path and full_path and os.path.exists(full_path):
            pdf_path = full_path

        if pdf_path:
            try:
                b64s = pdf_pages_to_base64_pngs(pdf_path, [page_idx])
                if b64s:
                    images.append(dspy.Image.from_PIL(base64_to_pillow(b64s[0])))
            except Exception:
                continue

    return images


class _TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            msg = msg.encode("utf-8", "backslashreplace").decode("utf-8")
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("src.dspy_agent")
    level_name = (os.getenv("LOG_LEVEL") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        stream_handler = _TqdmLoggingHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(stream_handler)

        log_file = os.getenv("LOG_FILE")
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            logger.addHandler(file_handler)

    return logger


def _summarize_defects(defects: List[Any]) -> Dict[str, Any]:
    by_severity: Dict[str, int] = defaultdict(int)
    by_type: Dict[str, int] = defaultdict(int)
    for d in defects:
        by_severity[d.severity.value if hasattr(d.severity, "value") else str(d.severity)] += 1
        by_type[d.type] += 1
    return {
        "count": len(defects),
        "by_severity": dict(sorted(by_severity.items(), key=lambda x: x[0])),
        "by_type": dict(sorted(by_type.items(), key=lambda x: (-x[1], x[0]))),
    }


def _format_duration_s(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    return f"{minutes}m{rem:04.1f}s"


def _env_or_raise(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _safe_int(value: Any) -> Any:
    try:
        return int(value)
    except Exception:
        return value


def _clean_polygon(poly_raw: Any) -> List[List[float]]:
    if poly_raw is None:
        return []
    if hasattr(poly_raw, "tolist"):
        poly_raw = poly_raw.tolist()
    if not isinstance(poly_raw, (list, tuple)):
        return []
    cleaned: List[List[float]] = []
    for pt in poly_raw:
        if hasattr(pt, "tolist"):
            pt = pt.tolist()
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            try:
                x = float(pt[0])
                y = float(pt[1])
            except Exception:
                continue
            if math.isnan(x) or math.isnan(y):
                continue
            cleaned.append([round(x, 2), round(y, 2)])
    return cleaned


def _polygon_area(points: List[List[float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _sum_polygon_area(polygons: List[List[List[float]]]) -> float:
    return float(sum(_polygon_area(poly) for poly in polygons))


def _bbox_area(bbox: Optional[List[float]]) -> Optional[float]:
    if not bbox or len(bbox) < 4:
        return None
    min_x, min_y, max_x, max_y = bbox
    return float(max(0.0, max_x - min_x) * max(0.0, max_y - min_y))


def _aggregate_bbox(polygons: List[List[List[float]]]) -> Optional[List[float]]:
    points = [pt for poly in polygons for pt in poly]
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def _geometry_context_for_chunk(df: pd.DataFrame, layout_id: Any, chunk_id: Any) -> Dict[str, Any]:
    if chunk_id is not None:
        subset = df[(df["layout_id"] == layout_id) & (df["chunk_id"] == chunk_id)]
        if subset.empty:
            l_match = df["layout_id"].astype(str) == str(layout_id) if "layout_id" in df.columns else False
            c_match = df["chunk_id"].astype(str) == str(chunk_id) if "chunk_id" in df.columns else False
            if "layout_id" in df.columns and "chunk_id" in df.columns:
                subset = df[l_match & c_match]
    else:
        # Layout-wide context: all chunks in layout
        subset = df[df["layout_id"] == layout_id]
        if subset.empty and "layout_id" in df.columns:
            subset = df[df["layout_id"].astype(str) == str(layout_id)]

    if subset.empty:
        return {}
    pixel_polys: List[List[List[float]]] = []
    dxf_polys: List[List[List[float]]] = []
    for _, row in subset.iterrows():
        pixel_poly = _clean_polygon(row.get("chunks"))
        if pixel_poly:
            pixel_polys.append(pixel_poly)
        dxf_poly = _clean_polygon(row.get("chunks_dxf"))
        if dxf_poly:
            dxf_polys.append(dxf_poly)
    if not pixel_polys and not dxf_polys:
        return {}
    pixel_bbox = _aggregate_bbox(pixel_polys)
    dxf_bbox = _aggregate_bbox(dxf_polys)
    pixel_area = _sum_polygon_area(pixel_polys) if pixel_polys else 0.0
    dxf_area = _sum_polygon_area(dxf_polys) if dxf_polys else 0.0
    pixel_bbox_area = _bbox_area(pixel_bbox)
    dxf_bbox_area = _bbox_area(dxf_bbox)
    return {
        "layout_id": _safe_int(layout_id),
        "chunk_id": _safe_int(chunk_id),
        "pixel_bbox": pixel_bbox,
        "dxf_bbox": dxf_bbox,
        "pixel_area": pixel_area,
        "dxf_area": dxf_area,
        "pixel_polygons": pixel_polys,
        "dxf_polygons": dxf_polys,
        "pixel_polygon_count": len(pixel_polys),
        "dxf_polygon_count": len(dxf_polys),
        "pixel_coverage_ratio": (pixel_area / pixel_bbox_area) if pixel_bbox_area else 0.0,
        "dxf_coverage_ratio": (dxf_area / dxf_bbox_area) if dxf_bbox_area else 0.0,
    }


def _trim_count_map(counts: Optional[Dict[str, int]], top_n: int) -> Dict[str, int]:
    if not counts:
        return {}
    items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    if top_n > 0:
        items = items[:top_n]
    return dict(items)


def _semantic_layer_keywords() -> List[str]:
    raw = os.getenv("DXF_LAYER_SEMANTIC_KEYWORDS")
    if raw:
        return [kw.strip().upper() for kw in raw.split(",") if kw.strip()]
    return [
        "WALL",
        "DOOR",
        "WINDOW",
        "BEAM",
        "COLUMN",
        "GRID",
        "ROOM",
        "DIM",
        "DIMS",
        "EQUIP",
        "EQUIPMENT",
        "PIPE",
        "DUCT",
        "HVAC",
        "ELECT",
        "PLUMB",
        "FIRE",
        "STRUCT",
        "FOUND",
        "SLAB",
        "STAIR",
        "RISER",
        "OPENING",
        "HEADER",
        "FOOTPRINT",
        "GARAGE",
        "TEXT",
    ]


def _is_semantic_layer(name: Optional[str], keywords: List[str]) -> bool:
    if not name:
        return False
    name_upper = name.upper()
    return any(kw in name_upper for kw in keywords)


def _trim_dxf_metadata(roi_metadata) -> Dict[str, Any]:
    layer_top_n = int(os.getenv("DXF_LAYER_TOP_N") or "8")
    layer_type_top_n = int(os.getenv("DXF_LAYER_TYPE_TOP_N") or "5")
    text_top_n = int(os.getenv("DXF_TEXT_TOP_N") or "25")
    example_per_type = int(os.getenv("DXF_ENTITY_EXAMPLES_PER_TYPE") or "3")
    keywords = _semantic_layer_keywords()
    layers = list(getattr(roi_metadata, "layers", []) or [])
    layers_sorted = sorted(layers, key=lambda l: (-getattr(l, "entity_count", 0), getattr(l, "layer", "")))
    semantic_layers = [l for l in layers_sorted if _is_semantic_layer(getattr(l, "layer", ""), keywords)]
    selected_layers = semantic_layers[:layer_top_n]
    if len(selected_layers) < layer_top_n:
        remaining = [l for l in layers_sorted if l not in selected_layers]
        selected_layers.extend(remaining[: max(0, layer_top_n - len(selected_layers))])
    annotations = list(getattr(roi_metadata, "annotations", []) or [])
    trimmed_annotations = []
    for annotation in annotations[: max(0, text_top_n)]:
        if not isinstance(annotation, dict):
            continue
        text_value = annotation.get("text")
        if not isinstance(text_value, str) or not text_value.strip():
            continue
        trimmed = {
            "layer": annotation.get("layer", ""),
            "type": annotation.get("type", ""),
            "text": text_value.strip(),
        }
        insert_point = annotation.get("insert")
        if insert_point:
            trimmed["insert"] = insert_point
        trimmed_annotations.append(trimmed)

    # Determine which entity types are globally significant (top N types)
    # We want to ensure we have examples for ALL these types.
    global_type_counts = getattr(roi_metadata, "entity_type_counts", None) or {}
    # Use layer_type_top_n as a heuristic for "significant types" globally as well, 
    # or just ensure we cover everything that appears in the summary.
    # The summary uses layer_type_top_n.
    entity_type_counts_top = _trim_count_map(global_type_counts, layer_type_top_n)
    target_types = set(entity_type_counts_top.keys())

    entity_groups: Dict[str, List[Dict[str, Any]]] = {}
    
    # Iterate ALL layers to find examples, prioritizing dense/semantic ones (layers_sorted)
    # This ensures we find rare entities (like CIRCLE) even if they are on minor layers.
    for layer in layers_sorted:
        layer_name = getattr(layer, "layer", "") or ""
        layer_entities = getattr(layer, "entities", [])
        
        for entity in layer_entities:
            entity_type = getattr(entity, "type", "UNKNOWN")
            
            # Skip if we already have enough examples for this type
            if len(entity_groups.get(entity_type, [])) >= example_per_type:
                continue

            # Skip text types (handled by annotations)
            if entity_type in ("TEXT", "MTEXT"):
                continue
            
            # Optionally: Only collect examples for types that are in the "top" list?
            # Or collect everything? 
            # If the user complains about "missing entities shown in summary", we MUST cover target_types.
            # If there are other types not in summary, maybe we skip them to save tokens?
            # Let's collect examples for target_types primarily. 
            # If we want to be safe and show everything, we can remove the filter.
            # Given the prompt "missing entities as shown in the summary", let's strictly ensure target_types coverage.
            # But what if summary trims something interesting?
            # Let's stick to target_types to be consistent with the summary.
            if entity_type not in target_types:
                continue

            entity_payload = {
                "layer": layer_name,
                "type": entity_type,
                "geometry": getattr(entity, "geometry", {})
            }
            entity_groups.setdefault(entity_type, []).append(entity_payload)
            
    for annotation in trimmed_annotations:
        entity_type = annotation.get("type") or "TEXT"
        entity_groups.setdefault(entity_type, []).append(annotation)

    return {
        "version": getattr(roi_metadata, "version", None),
        "units": getattr(roi_metadata, "units", None),
        "total_layers": int(getattr(roi_metadata, "total_layers", 0) or 0),
        "total_entities": int(getattr(roi_metadata, "total_entities", 0) or 0),
        # Compact summaries that are generally most useful to the vision model:
        # - Which entity types dominate per layer
        # - A small set of text/annotation snippets
        # Avoid dumping full geometry coordinates (can be very large and not necessary for corroboration).
        "entity_type_counts_top": entity_type_counts_top,
        "annotations": trimmed_annotations,
        # Keep a tiny set of examples (type + layer + geometry type only).
        "entity_examples": {
            etype: examples[: max(0, example_per_type)]
            for etype, examples in {
                etype: [
                    {
                        "layer": e.get("layer", ""),
                        "type": e.get("type", ""),
                        "geometry": e.get("geometry"),
                    }
                    for e in items
                    if isinstance(e, dict)
                ]
                for etype, items in entity_groups.items()
                if etype not in ("TEXT", "MTEXT")
            }.items()
        },
    }


def _load_resized_image(image_path: str, max_dimension: int = 4096) -> Image.Image:
    """
    Load an image from disk and resize it if it exceeds max_dimension.
    Helps prevent base64 encoding issues with massive images.
    """
    Image.MAX_IMAGE_PIXELS = None

    img = Image.open(image_path)

    if max(img.width, img.height) > max_dimension:
        ratio = max_dimension / max(img.width, img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    return img


def clean_for_serialization(obj: Any) -> Any:
    """
    Recursively clean objects for JSON serialization.
    Handles dspy.Image and other non-serializable types.
    """
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {k: clean_for_serialization(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_serialization(v) for v in obj]
    if isinstance(obj, tuple):
        return [clean_for_serialization(v) for v in obj]

    if isinstance(obj, dspy.Image):
        return f"<dspy.Image>"

    if hasattr(obj, "model_dump"):
        return clean_for_serialization(obj.model_dump())
    if hasattr(obj, "to_dict"):
        return clean_for_serialization(obj.to_dict())

    return str(obj)


def sanitize_for_openai(obj: Any) -> Any:
    if isinstance(obj, str):
        return "".join(ch for ch in obj if not (0xD800 <= ord(ch) <= 0xDFFF))
    if isinstance(obj, (int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_openai(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_openai(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_openai(v) for v in obj]
    if isinstance(obj, dspy.Image):
        return obj
    if hasattr(obj, "model_dump"):
        return sanitize_for_openai(obj.model_dump())
    if hasattr(obj, "to_dict"):
        return sanitize_for_openai(obj.to_dict())
    return sanitize_for_openai(str(obj))


def format_trajectory(trajectory: Union[List[Any], Dict[str, Any]]) -> Dict[str, Any]:
    """
    Format the DSPy trajectory into a flattened dictionary structure 
    as requested by the user.
    """
    if isinstance(trajectory, dict):
        return clean_for_serialization(trajectory)

    formatted = {}
    for i, step in enumerate(trajectory):
        if isinstance(step, str):
            formatted[f"step_{i}"] = step
            continue

        if hasattr(step, "thought"):
            formatted[f"thought_{i}"] = clean_for_serialization(step.thought)
        elif isinstance(step, dict) and "thought" in step:
            formatted[f"thought_{i}"] = clean_for_serialization(step["thought"])

        if hasattr(step, "action"):
            action = step.action
            if hasattr(action, "name"):
                formatted[f"tool_name_{i}"] = clean_for_serialization(action.name)
                formatted[f"tool_args_{i}"] = clean_for_serialization(action.args if hasattr(action, "args") else {})
            elif isinstance(action, dict):
                formatted[f"tool_name_{i}"] = clean_for_serialization(action.get("name"))
                formatted[f"tool_args_{i}"] = clean_for_serialization(action.get("args"))
            else:
                formatted[f"action_{i}"] = clean_for_serialization(str(action))

        elif isinstance(step, dict) and "action" in step:
            formatted[f"action_{i}"] = clean_for_serialization(step["action"])

        if hasattr(step, "observation"):
            formatted[f"observation_{i}"] = clean_for_serialization(step.observation)
        elif isinstance(step, dict) and "observation" in step:
            formatted[f"observation_{i}"] = clean_for_serialization(step["observation"])

    return formatted


def _is_ascii(text: str) -> bool:
    return all(ord(ch) < 128 for ch in text)


def _contains_cjk(text: str) -> bool:
    for ch in text:
        code = ord(ch)
        if (
            0x4E00 <= code <= 0x9FFF
            or 0x3400 <= code <= 0x4DBF
            or 0x20000 <= code <= 0x2A6DF
            or 0x2A700 <= code <= 0x2B73F
            or 0x2B740 <= code <= 0x2B81F
            or 0x2B820 <= code <= 0x2CEAF
            or 0xF900 <= code <= 0xFAFF
            or 0x2F800 <= code <= 0x2FA1F
        ):
            return True
    return False


def _defects_have_cjk(defects_list: List[Any]) -> bool:
    for d in defects_list:
        for field_name in ("type", "description", "location", "how_to_fix"):
            value = getattr(d, field_name, None)
            if isinstance(value, str) and _contains_cjk(value):
                return True
    return False
