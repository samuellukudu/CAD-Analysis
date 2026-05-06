import json
import os
import io
import contextlib
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, ValidationError

from src.agent_utils import _geometry_context_for_chunk, _trim_dxf_metadata, clean_for_serialization
from src.utils import (
    build_dxf_entity_cache,
    create_layout_dataframe,
    extract_dxf_metadata_for_polygons,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DATA_ROOT = os.path.join(PROJECT_ROOT, "storage")

_DF_CACHE: dict[tuple[str, str], tuple[float, float, Any]] = {}
_DXF_CACHE: dict[str, tuple[float, Any]] = {}


class GeometryContextRequest(BaseModel):
    geojson_path: str = Field(..., min_length=1)
    transform_path: str = Field(..., min_length=1)
    layout_id: int
    chunk_id: int
    include_polygons: bool = False
    max_polygons: int = 5
    max_points_per_polygon: int = 200


class DxfMetadataRequest(BaseModel):
    dxf_path: str = Field(..., min_length=1)
    geojson_path: str = Field(..., min_length=1)
    transform_path: str = Field(..., min_length=1)
    layout_id: int
    chunk_id: int
    raw: bool = False


def _resolve_input_path(path: str) -> str:
    if not path or not isinstance(path, str):
        return path
    normalized = os.path.normpath(path)
    if os.path.isabs(normalized):
        try:
            if os.path.commonpath([normalized, INPUT_DATA_ROOT]) == INPUT_DATA_ROOT:
                return normalized
        except Exception:
            pass
        parts = normalized.split(os.sep)
        if "storage" in parts:
            storage_index = parts.index("storage")
            rel_parts = parts[storage_index + 1 :]
            if rel_parts:
                return os.path.join(INPUT_DATA_ROOT, *rel_parts)
        return normalized
    return os.path.abspath(os.path.join(INPUT_DATA_ROOT, normalized))


def _abs_path(path: str) -> str:
    return _resolve_input_path(path)


def _validate_file(path: str, label: str) -> str | None:
    if not path:
        return f"Error: {label} path is required."
    if not os.path.exists(path):
        return f"Error: {label} file not found: {path}"
    return None


def _build_geometry_context(
    geojson_path: str,
    transform_path: str,
    layout_id: int,
    chunk_id: Optional[int] = None,
) -> Dict[str, Any]:
    geojson_path = _abs_path(geojson_path)
    transform_path = _abs_path(transform_path)
    missing = _validate_file(geojson_path, "GeoJSON") or _validate_file(transform_path, "Transform")
    if missing:
        return {"status": "error", "data": {}, "error": missing}
    # Cache the expensive GeoJSON->DataFrame transform (keyed by mtimes)
    key = (geojson_path, transform_path)
    try:
        gj_mtime = os.path.getmtime(geojson_path)
        tf_mtime = os.path.getmtime(transform_path)
    except Exception:
        gj_mtime = -1.0
        tf_mtime = -1.0
    cached = _DF_CACHE.get(key)
    if cached and cached[0] == gj_mtime and cached[1] == tf_mtime:
        df = cached[2]
    else:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            df = create_layout_dataframe(geojson_path, transform_path)
        _DF_CACHE[key] = (gj_mtime, tf_mtime, df)
    if df.empty:
        return {"status": "empty", "data": {}}
    context = _geometry_context_for_chunk(df, layout_id, chunk_id)
    if not context:
        return {"status": "not_found", "data": {}}
    # Default to compact context to avoid oversized tool payloads.
    # The vision model usually needs bbox/area/coverage + polygon counts, not every vertex.
    return {"status": "ok", "data": context}


def query_geometry_context(payload: Dict[str, Any]) -> str:
    try:
        request = GeometryContextRequest.model_validate(payload)
    except ValidationError as exc:
        return json.dumps({"status": "error", "data": {}, "error": str(exc)}, ensure_ascii=True)
    result = _build_geometry_context(
        geojson_path=request.geojson_path,
        transform_path=request.transform_path,
        layout_id=request.layout_id,
        chunk_id=request.chunk_id,
    )
    if result.get("status") == "ok":
        data = result.get("data") or {}
        if not request.include_polygons:
            # Remove potentially huge arrays
            data = dict(data)
            data.pop("pixel_polygons", None)
            data.pop("dxf_polygons", None)
            result["data"] = data
        else:
            # Truncate polygons/points to keep payload bounded
            max_polys = max(0, int(request.max_polygons))
            max_pts = max(0, int(request.max_points_per_polygon))
            data = dict(data)
            for key in ("pixel_polygons", "dxf_polygons"):
                polys = data.get(key) or []
                if isinstance(polys, list):
                    polys = polys[:max_polys] if max_polys else []
                    truncated = []
                    for poly in polys:
                        if isinstance(poly, list):
                            truncated.append(poly[:max_pts] if max_pts else [])
                        else:
                            truncated.append(poly)
                    data[key] = truncated
            result["data"] = data
    return json.dumps(result, ensure_ascii=True)


def query_dxf_metadata(payload: Dict[str, Any]) -> str:
    try:
        request = DxfMetadataRequest.model_validate(payload)
    except ValidationError as exc:
        return json.dumps({"status": "error", "data": {}, "error": str(exc)}, ensure_ascii=True)
    dxf_path = _abs_path(request.dxf_path)
    missing = _validate_file(dxf_path, "DXF")
    if missing:
        return json.dumps({"status": "error", "data": {}, "error": missing}, ensure_ascii=True)
    geometry_context = _build_geometry_context(
        geojson_path=request.geojson_path,
        transform_path=request.transform_path,
        layout_id=request.layout_id,
        chunk_id=request.chunk_id,
    )
    if geometry_context.get("status") != "ok":
        return json.dumps(
            {"status": geometry_context.get("status"), "data": {}},
            ensure_ascii=True,
        )
    dxf_polys = geometry_context.get("data", {}).get("dxf_polygons") or []
    if not dxf_polys:
        return json.dumps({"status": "roi_empty", "data": {}}, ensure_ascii=True)
    # Cache full DXF entity extraction (expensive)
    try:
        dxf_mtime = os.path.getmtime(dxf_path)
    except Exception:
        dxf_mtime = -1.0
    cached_dxf = _DXF_CACHE.get(dxf_path)
    if cached_dxf and cached_dxf[0] == dxf_mtime:
        dxf_entity_cache = cached_dxf[1]
    else:
        dxf_entity_cache = build_dxf_entity_cache(dxf_path)
        _DXF_CACHE[dxf_path] = (dxf_mtime, dxf_entity_cache)
    if not dxf_entity_cache:
        return json.dumps({"status": "cache_missing", "data": {}}, ensure_ascii=True)
    try:
        roi_metadata = extract_dxf_metadata_for_polygons(dxf_entity_cache, dxf_polys)
        if request.raw:
            # Return full metadata without trimming, but ensure JSON serializability
            # roi_metadata is a Pydantic model, so model_dump() gives us a dict
            return json.dumps({"status": "ok", "data": clean_for_serialization(roi_metadata)}, ensure_ascii=True)
        trimmed_metadata = _trim_dxf_metadata(roi_metadata)
    except Exception:
        return json.dumps({"status": "error", "data": {}, "error": "Failed to extract ROI DXF metadata"}, ensure_ascii=True)
    return json.dumps({"status": "ok", "data": trimmed_metadata}, ensure_ascii=True)
