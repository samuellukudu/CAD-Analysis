import json
import os
import sys
import shutil
import asyncio
import threading
from collections import defaultdict
from typing import Optional, List, Dict, Any, AsyncGenerator, Set
import uuid
from pathlib import Path
import tempfile
import zipfile
import pandas as pd
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel
from starlette.concurrency import iterate_in_threadpool

# Add project root to sys.path if needed
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

def _load_agent_pipeline():
    from src.dspy_agent import run_agent_pipeline
    return run_agent_pipeline

router = APIRouter()

STORAGE_STATUS_DIR = os.path.join(project_root, "storage", "status")
STORAGE_RESULTS_DIR = os.path.join(project_root, "storage", "results")
STORAGE_GEOJSON_DIR = os.path.join(project_root, "storage", "geojson")

_AGENT_RUN_LOCK = threading.Lock()
_ACTIVE_AGENT_RUNS: Dict[str, str] = {}

def _set_active_agent_run(file_id: str, run_id: str) -> None:
    with _AGENT_RUN_LOCK:
        _ACTIVE_AGENT_RUNS[file_id] = run_id

def _is_active_agent_run(file_id: str, run_id: str) -> bool:
    with _AGENT_RUN_LOCK:
        return _ACTIVE_AGENT_RUNS.get(file_id) == run_id

def _agent_status_file_path(file_id: str) -> str:
    return os.path.join(STORAGE_STATUS_DIR, f"agent_{file_id}.json")

def update_status(file_id: str, status: str, details: dict = None):
    """Update the status of a file processing job."""
    os.makedirs(STORAGE_STATUS_DIR, exist_ok=True)
    status_file = _agent_status_file_path(file_id)
    existing_details = {}
    if os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                existing = json.load(f) or {}
            existing_details = existing.get("details") or {}
        except Exception:
            existing_details = {}
    merged_details = {**existing_details, **(details or {})}
    data = {
        "id": file_id,
        "status": status,
        "details": merged_details
    }
    with open(status_file, "w") as f:
        json.dump(data, f)

def get_status(file_id: str):
    """Get the status of a file processing job."""
    status_file = _agent_status_file_path(file_id)
    if not os.path.exists(status_file):
        return None
    with open(status_file, "r") as f:
        return json.load(f)

# --- Stream Manager ---
class StreamManager:
    def __init__(self):
        # file_id -> Set[asyncio.Queue]
        self.listeners: Dict[str, Set[asyncio.Queue]] = defaultdict(set)

    async def subscribe(self, file_id: str) -> AsyncGenerator[str, None]:
        queue = asyncio.Queue()
        self.listeners[file_id].add(queue)
        try:
            while True:
                data = await queue.get()
                if data is None:  # Signal to stop
                    break
                yield data
        finally:
            self.listeners[file_id].remove(queue)

    async def publish(self, file_id: str, data: str):
        if file_id in self.listeners:
            for queue in self.listeners[file_id]:
                await queue.put(data)

    async def close(self, file_id: str):
        """Signal all listeners to close."""
        if file_id in self.listeners:
            for queue in self.listeners[file_id]:
                await queue.put(None)
            # We don't remove keys here to allow late subscribers to potentially see "done" state if we implemented history
            # But for now, we just close active connections.

stream_manager = StreamManager()
# ----------------------

class AgentQuery(BaseModel):
    query: Optional[str] = "detect design errors"
    file_id: Optional[str] = None

class ResultsExportRequest(BaseModel):
    file_ids: Optional[List[str]] = None

def get_latest_completed_file_id() -> Optional[str]:
    if not os.path.exists(STORAGE_GEOJSON_DIR):
        return None
    geojson_files = [
        os.path.join(STORAGE_GEOJSON_DIR, f)
        for f in os.listdir(STORAGE_GEOJSON_DIR)
        if f.endswith(".geojson")
    ]
    if not geojson_files:
        return None
    geojson_files.sort(key=os.path.getmtime, reverse=True)
    latest = geojson_files[0]
    return Path(latest).stem

def _normalize_json_to_dataframe(data: Any) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame()
    if isinstance(data, list):
        if len(data) == 0:
            return pd.DataFrame()
        if all(isinstance(item, dict) for item in data):
            return pd.json_normalize(data)
        return pd.DataFrame({"value": data})
    if isinstance(data, dict):
        return pd.json_normalize(data)
    return pd.DataFrame({"value": [data]})

def _write_results_xlsx(results_dir: str, output_path: str) -> bool:
    json_files = [
        ("Defects", "defects.json"),
        ("Report", "report.json"),
        ("Layout Analysis", "layout_analysis.json"),
        ("CAD Analysis", "cad_analysis.json"),
    ]
    written = False
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, filename in json_files:
            file_path = os.path.join(results_dir, filename)
            if not os.path.exists(file_path):
                continue
            with open(file_path, "r") as f:
                data = json.load(f)
            df = _normalize_json_to_dataframe(data)
            
            if sheet_name == "Defects" and not df.empty and "severity" in df.columns:
                def highlight_severity(row):
                    severity = str(row.get('severity', '')).lower()
                    color = ''
                    if severity == 'high':
                        color = 'background-color: #ff9999'  # Red
                    elif severity == 'medium':
                        color = 'background-color: #ffff99'  # Yellow
                    elif severity == 'low':
                        color = 'background-color: #99ccff'  # Blue
                    return [color] * len(row)

                try:
                    styler = df.style.apply(highlight_severity, axis=1)
                    styler.to_excel(writer, sheet_name=sheet_name, index=False)
                except Exception as e:
                    # Fallback if styling fails
                    print(f"Styling failed for {sheet_name}: {e}")
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            written = True
    return written

def _cleanup_path(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
        return
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass

async def stream_generator(func, *args, **kwargs):
    """
    Run a generator in a separate thread and stream results to an async queue.
    Useful for generators that rely on thread-local state (like dspy context) which
    breaks when using iterate_in_threadpool (which switches threads per yield).
    """
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    
    def worker():
        try:
            for item in func(*args, **kwargs):
                loop.call_soon_threadsafe(queue.put_nowait, (None, item))
            loop.call_soon_threadsafe(queue.put_nowait, (None, None))
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, (e, None))
            
    threading.Thread(target=worker, daemon=True).start()
    
    while True:
        error, item = await queue.get()
        if error:
            raise error
        if item is None:
            break
        yield item

async def run_analysis_background(file_id: str, run_id: str, query: Optional[str]):
    """
    Background task that runs the agent pipeline, persists results to disk,
    and broadcasts events to SSE subscribers.
    """
    if not _is_active_agent_run(file_id, run_id):
        return
    update_status(file_id, "processing", {"query": query, "run_id": run_id, "message": "Initializing analysis"})
    results_dir = os.path.join(STORAGE_RESULTS_DIR, file_id)
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize files with empty lists/objects if they don't exist
    defects_file = os.path.join(results_dir, "defects.json")
    if not os.path.exists(defects_file):
        with open(defects_file, "w") as f:
            json.dump([], f)
            
    layout_analysis_file = os.path.join(results_dir, "layout_analysis.json")
    if not os.path.exists(layout_analysis_file):
        with open(layout_analysis_file, "w") as f:
            json.dump([], f)

    # Init progress tracking
    progress = {
        "cad_analysis": False,
        "layouts_analyzed": [],
        "defects_count": 0,
        "defects_by_layout": {},
        "report_generated": False
    }

    try:
        # Run generator in a dedicated thread to preserve dspy context
        run_agent_pipeline = _load_agent_pipeline()
        pipeline_iterator = stream_generator(run_agent_pipeline, file_id=file_id, user_query=query)
        
        async for event in pipeline_iterator:
            if not _is_active_agent_run(file_id, run_id):
                return
            event_type = event.get("type")
            data = event.get("data")
            
            should_update_status = False
            status_update = {}

            if event_type == "status":
                status_update["message"] = data.get("message")
                should_update_status = True
            
            # Persist to disk
            elif event_type == "cad_analysis":
                progress["cad_analysis"] = True
                status_update["progress"] = progress
                should_update_status = True
                with open(os.path.join(results_dir, "cad_analysis.json"), "w") as f:
                    json.dump(data, f, indent=2)
                    
            elif event_type == "layout_analysis":
                l_id = data.get("layout_id")
                if l_id not in progress["layouts_analyzed"]:
                    progress["layouts_analyzed"].append(l_id)
                status_update["progress"] = progress
                should_update_status = True
                current_layouts = []
                if os.path.exists(layout_analysis_file):
                    try:
                        with open(layout_analysis_file, "r") as f:
                            current_layouts = json.load(f)
                    except json.JSONDecodeError:
                        current_layouts = []
                
                current_layouts.append(data)
                with open(layout_analysis_file, "w") as f:
                    json.dump(current_layouts, f, indent=2)
                    
            elif event_type == "defect_found":
                new_defects = data.get("defects", [])
                layout_id = data.get("layout_id")
                if new_defects:
                    progress["defects_count"] += len(new_defects)
                    
                    if layout_id:
                        current_layout_count = progress["defects_by_layout"].get(layout_id, 0)
                        progress["defects_by_layout"][layout_id] = current_layout_count + len(new_defects)
                    
                    status_update["progress"] = progress
                    should_update_status = True
                    current_defects = []
                    if os.path.exists(defects_file):
                        try:
                            with open(defects_file, "r") as f:
                                current_defects = json.load(f)
                        except json.JSONDecodeError:
                            current_defects = []
                    
                    current_defects.extend(new_defects)
                    with open(defects_file, "w") as f:
                        json.dump(current_defects, f, indent=2)
                        
            elif event_type == "report":
                progress["report_generated"] = True
                status_update["progress"] = progress
                should_update_status = True
                report_path = os.path.join(results_dir, "report.json")
                existing_reports = []
                if os.path.exists(report_path):
                    try:
                        with open(report_path, "r") as f:
                            current = json.load(f)
                        if isinstance(current, list):
                            existing_reports = current
                        elif isinstance(current, dict) and isinstance(current.get("layouts"), list):
                            existing_reports = current["layouts"]
                    except json.JSONDecodeError:
                        existing_reports = []
                if isinstance(data, dict):
                    layout_id = data.get("layout_id")
                    if layout_id is not None:
                        existing_reports = [r for r in existing_reports if r.get("layout_id") != layout_id]
                existing_reports.append(data)
                with open(report_path, "w") as f:
                    json.dump(existing_reports, f, indent=2)

            if should_update_status:
                update_status(file_id, "processing", {**status_update, "run_id": run_id})
            
            # Broadcast SSE event
            # Format: data: <json_string>\n\n
            await stream_manager.publish(file_id, f"data: {json.dumps(event)}\n\n")

        if _is_active_agent_run(file_id, run_id):
            update_status(file_id, "completed", {"run_id": run_id})

    except Exception as e:
        # Log error to a status file
        with open(os.path.join(results_dir, "error.json"), "w") as f:
            json.dump({"error": str(e)}, f)
        print(f"Analysis failed for {file_id}: {e}")
        
        error_event = {"type": "error", "data": {"message": str(e)}}
        if _is_active_agent_run(file_id, run_id):
            await stream_manager.publish(file_id, f"data: {json.dumps(error_event)}\n\n")
            update_status(file_id, "failed", {"error": str(e), "run_id": run_id})
    
    finally:
        # Close the stream when done
        await stream_manager.close(file_id)

@router.post("/agent/analyze")
async def analyze_agent(
    body: AgentQuery,
    background_tasks: BackgroundTasks
):
    """
    Trigger the AI agent analysis pipeline in the background.
    
    - If `file_id` is not provided, uses the latest completed upload.
    - Returns 202 Accepted immediately.
    - Connect to `/api/agent/{file_id}/stream` for real-time SSE updates.
    """
    file_id = body.file_id
    if not file_id:
        file_id = get_latest_completed_file_id()
        
    if not file_id:
        raise HTTPException(status_code=404, detail="No processed files found. Please upload a DXF file first.")
    
    run_id = uuid.uuid4().hex
    _set_active_agent_run(file_id, run_id)
    await stream_manager.close(file_id)

    # Clean up previous results if restarting analysis
    results_dir = os.path.join(STORAGE_RESULTS_DIR, file_id)
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    
    # Start background task
    background_tasks.add_task(run_analysis_background, file_id, run_id, body.query)
    
    return JSONResponse(
        status_code=202,
        content={
            "message": "Analysis started in background.",
            "file_id": file_id,
            "run_id": run_id,
            "stream_url": f"/api/agent/{file_id}/stream",
            "endpoints": {
                "cad_analysis": f"/api/agent/{file_id}/cad-analysis",
                "layout_analysis": f"/api/agent/{file_id}/layout-analysis",
                "defects": f"/api/agent/{file_id}/defects",
                    "report": f"/api/agent/{file_id}/report"
            }
        }
    )

@router.get("/agent/{file_id}/status")
async def get_agent_status(file_id: str):
    """Get the processing status of an agent analysis job."""
    status = get_status(file_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status

@router.get("/agent/{file_id}/stream")
async def stream_agent_analysis(file_id: str):
    """
    Stream real-time analysis events for a specific file via SSE.
    """
    return StreamingResponse(
        stream_manager.subscribe(file_id),
        media_type="text/event-stream"
    )

@router.get("/agent/{file_id}/cad-analysis")
async def get_cad_analysis(file_id: str):
    """Retrieve the CAD analysis result."""
    file_path = os.path.join(STORAGE_RESULTS_DIR, file_id, "cad_analysis.json")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="CAD analysis not found or not yet generated.")
    
    with open(file_path, "r") as f:
        return json.load(f)

@router.get("/agent/{file_id}/layout-analysis")
async def get_layout_analysis(file_id: str):
    """Retrieve the list of layout analyses."""
    file_path = os.path.join(STORAGE_RESULTS_DIR, file_id, "layout_analysis.json")
    if not os.path.exists(file_path):
        # Return empty list if not yet generated, or 404? 
        # User wants "sent as soon as content is generated", so empty list is better than 404 if job started
        return []
    
    with open(file_path, "r") as f:
        return json.load(f)

@router.get("/agent/{file_id}/defects")
async def get_defects(file_id: str):
    """Retrieve the list of detected defects."""
    file_path = os.path.join(STORAGE_RESULTS_DIR, file_id, "defects.json")
    if not os.path.exists(file_path):
        return []
    
    with open(file_path, "r") as f:
        return json.load(f)

@router.get("/agent/{file_id}/report")
async def get_report(file_id: str):
    """Retrieve the final aggregated report."""
    file_path = os.path.join(STORAGE_RESULTS_DIR, file_id, "report.json")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Report not yet generated.")
    
    with open(file_path, "r") as f:
        return json.load(f)

def _get_source_filename(file_id: str) -> str:
    status_path = os.path.join(STORAGE_STATUS_DIR, f"dxf_{file_id}.json")
    if os.path.exists(status_path):
        try:
            with open(status_path, "r") as f:
                data = json.load(f)
                filename = data.get("details", {}).get("source_filename")
                if filename:
                    # Remove extension if present
                    root, _ = os.path.splitext(filename)
                    return root
        except Exception:
            pass
    return file_id

@router.get("/agent/{file_id}/export-xlsx")
async def export_results_xlsx(file_id: str, background_tasks: BackgroundTasks):
    results_dir = os.path.join(STORAGE_RESULTS_DIR, file_id)
    if not os.path.isdir(results_dir):
        raise HTTPException(status_code=404, detail="Results not found.")
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, f"{file_id}_results.xlsx")
    written = _write_results_xlsx(results_dir, output_path)
    if not written:
        _cleanup_path(temp_dir)
        raise HTTPException(status_code=404, detail="No results files available to export.")
    background_tasks.add_task(_cleanup_path, temp_dir)
    
    source_name = _get_source_filename(file_id)
    return FileResponse(
        path=output_path,
        filename=f"{source_name}_results.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

@router.post("/agent/results/export-xlsx")
async def export_all_results_xlsx(body: ResultsExportRequest, background_tasks: BackgroundTasks):
    if not os.path.isdir(STORAGE_RESULTS_DIR):
        raise HTTPException(status_code=404, detail="Results directory not found.")
    all_dirs = [
        d for d in os.listdir(STORAGE_RESULTS_DIR)
        if os.path.isdir(os.path.join(STORAGE_RESULTS_DIR, d))
    ]
    file_ids = body.file_ids or all_dirs
    if not file_ids:
        raise HTTPException(status_code=404, detail="No result folders found.")
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "results_exports.zip")
    added = False
    
    # Track used names to avoid collisions in zip
    used_names = {}
    
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_id in file_ids:
            if file_id != os.path.basename(file_id) or ".." in file_id or "/" in file_id or "\\" in file_id:
                continue
            results_dir = os.path.join(STORAGE_RESULTS_DIR, file_id)
            if not os.path.isdir(results_dir):
                continue
            output_path = os.path.join(temp_dir, f"{file_id}_results.xlsx")
            written = _write_results_xlsx(results_dir, output_path)
            if not written:
                _cleanup_path(output_path)
                continue
            
            source_name = _get_source_filename(file_id)
            base_filename = f"{source_name}_results.xlsx"
            
            # Handle duplicates
            if base_filename in used_names:
                used_names[base_filename] += 1
                name_root, name_ext = os.path.splitext(base_filename)
                final_filename = f"{name_root}_{used_names[base_filename]}{name_ext}"
            else:
                used_names[base_filename] = 0
                final_filename = base_filename
                
            zf.write(output_path, arcname=final_filename)
            added = True
            _cleanup_path(output_path)
    if not added:
        _cleanup_path(temp_dir)
        raise HTTPException(status_code=404, detail="No results files available to export.")
    background_tasks.add_task(_cleanup_path, temp_dir)
    return FileResponse(
        path=zip_path,
        filename="results_exports.zip",
        media_type="application/zip"
    )
