import json
import os

from src.chat_system import (
    list_layouts,
    run_cad_analyzer,
    run_design_defect_analysis,
    run_layout_analyzer,
)


def _pick_fixture() -> dict[str, str]:
    project_root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(project_root)
    storage_images = os.path.join(project_root, "storage", "images")
    storage_geojson = os.path.join(project_root, "storage", "geojson")
    storage_dxf = os.path.join(project_root, "storage", "dxf")

    candidates = sorted(
        f for f in os.listdir(storage_images) if f.endswith(".png") and not f.endswith("_viz.png")
    )
    if not candidates:
        raise RuntimeError(f"No PNG fixtures found under {storage_images}")

    stem = os.path.splitext(candidates[0])[0]
    full_image_path = os.path.join(storage_images, f"{stem}.png")
    transform_path = os.path.join(storage_images, f"{stem}_transform.json")
    geojson_path = os.path.join(storage_geojson, f"{stem}.geojson")
    dxf_path = os.path.join(storage_dxf, f"{stem}.dxf")

    return {
        "stem": stem,
        "full_image_path": full_image_path,
        "transform_path": transform_path,
        "geojson_path": geojson_path,
        "dxf_path": dxf_path,
    }


def _select_layout(layouts: list[dict[str, object]]) -> dict[str, object]:
    if not layouts:
        raise RuntimeError("No layouts returned from list_layouts")
    return max(layouts, key=lambda item: int(item.get("chunk_count") or 0))


def main() -> int:
    fixture = _pick_fixture()
    layouts_raw = list_layouts(
        {
            "geojson_path": fixture["geojson_path"],
            "transform_path": fixture["transform_path"],
            "include_chunk_ids": True,
            "max_chunks_per_layout": 500,
        }
    )
    layouts_result = json.loads(layouts_raw)
    if layouts_result.get("status") != "ok":
        raise RuntimeError(f"Unexpected list_layouts result: {layouts_raw}")

    layout = _select_layout(layouts_result.get("data", {}).get("layouts") or [])
    layout_id = int(layout.get("layout_id"))
    chunk_ids = layout.get("chunk_ids") or []
    if not chunk_ids:
        raise RuntimeError(f"No chunk_ids returned for layout {layout_id}")
    if layout.get("chunk_ids_truncated"):
        raise RuntimeError(f"Chunk ids truncated for layout {layout_id}")
    chunk_id = int(chunk_ids[0])

    cad_raw = run_cad_analyzer(
        {
            "image_path": fixture["full_image_path"],
            "user_query": "Summarize the drawing for defect analysis.",
        }
    )
    cad_result = json.loads(cad_raw)
    cad_description = cad_result.get("response")
    if not cad_description:
        raise RuntimeError(f"Unexpected run_cad_analyzer result: {cad_raw}")

    layout_raw = run_layout_analyzer(
        {
            "full_image_path": fixture["full_image_path"],
            "geojson_path": fixture["geojson_path"],
            "transform_path": fixture["transform_path"],
            "layout_id": layout_id,
            "padding": 50,
            "description": cad_description,
            "user_query": "Describe layout details for defect detection.",
        }
    )
    layout_result = json.loads(layout_raw)
    layout_description = layout_result.get("response")
    if not layout_description:
        raise RuntimeError(f"Unexpected run_layout_analyzer result: {layout_raw}")

    defect_dry_run = os.getenv("DEFECT_DRY_RUN", "1").lower() in ("1", "true", "yes")
    defect_raw = run_design_defect_analysis(
        {
            "full_image_path": fixture["full_image_path"],
            "geojson_path": fixture["geojson_path"],
            "transform_path": fixture["transform_path"],
            "layout_id": layout_id,
            "chunk_ids": chunk_ids,
            "padding": 50,
            "description": layout_description,
            "user_query": "Identify design defects and compliance issues.",
            "dry_run": defect_dry_run,
        }
    )
    defect_result = json.loads(defect_raw)
    if defect_dry_run:
        if defect_result.get("status") != "ok" or not defect_result.get("dry_run"):
            raise RuntimeError(f"Unexpected defect dry-run result: {defect_raw}")
        work_items = defect_result.get("work_items") or []
        if not any(
            item.get("layout_id") == layout_id and item.get("chunk_id") == chunk_id
            for item in work_items
        ):
            raise RuntimeError(f"Unexpected defect work_items: {work_items}")

    print(
        json.dumps(
            {
                "stem": fixture["stem"],
                "layout_id": layout_id,
                "chunk_id": chunk_id,
                "chunk_count": len(chunk_ids),
                "defect_dry_run": defect_dry_run,
                "defect_keys": sorted(defect_result.keys()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
