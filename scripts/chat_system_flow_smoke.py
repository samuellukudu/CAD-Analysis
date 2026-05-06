import json
import os

from src.chat_system import run_layout_analyzer
from src.chat_tools_metadata import query_geometry_context, query_dxf_metadata
from src.utils import create_layout_dataframe


def _pick_fixture() -> dict[str, str]:
    project_root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(project_root)
    storage_images = os.path.join(project_root, "storage", "images")
    storage_geojson = os.path.join(project_root, "storage", "geojson")
    storage_dxf = os.path.join(project_root, "storage", "dxf")

    candidates = sorted(f for f in os.listdir(storage_images) if f.endswith(".png"))
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


def main() -> int:
    fixture = _pick_fixture()
    df = create_layout_dataframe(fixture["geojson_path"], fixture["transform_path"])
    if df.empty:
        raise RuntimeError("Fixture GeoJSON/transform produced an empty dataframe")

    layout_id = int(df["layout_id"].dropna().astype(int).iloc[0])
    chunk_id = int(df[df["layout_id"] == layout_id]["chunk_id"].dropna().astype(int).iloc[0])

    geometry = query_geometry_context(
        {
            "geojson_path": fixture["geojson_path"],
            "transform_path": fixture["transform_path"],
            "layout_id": layout_id,
            "chunk_id": chunk_id,
        }
    )
    dxf_meta = query_dxf_metadata(
        {
            "dxf_path": fixture["dxf_path"],
            "geojson_path": fixture["geojson_path"],
            "transform_path": fixture["transform_path"],
            "layout_id": layout_id,
            "chunk_id": chunk_id,
        }
    )

    payload = {
        "full_image_path": fixture["full_image_path"],
        "geojson_path": fixture["geojson_path"],
        "transform_path": fixture["transform_path"],
        "layout_id": layout_id,
        "padding": 50,
        "description": "fixture",
        "user_query": "dry run",
        "dry_run": True,
    }
    result_raw = run_layout_analyzer(payload)
    result = json.loads(result_raw)
    if result.get("status") != "ok" or not result.get("dry_run"):
        raise RuntimeError(f"Unexpected result: {result_raw}")

    print(
        json.dumps(
            {
                "stem": fixture["stem"],
                "layout_id": layout_id,
                "chunk_id": chunk_id,
                "layout_image_size": result["debug"]["image_sizes"]["layout_image"],
                "geometry_status": json.loads(geometry).get("status", "unknown") if isinstance(geometry, str) else "unknown",
                "dxf_metadata_prefix": dxf_meta[:80] if isinstance(dxf_meta, str) else str(type(dxf_meta)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

