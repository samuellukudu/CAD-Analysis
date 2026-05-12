from __future__ import annotations

import os
import re
import sys
import shlex
import subprocess
from openai import OpenAI
import json
import math
import datetime
import random
import itertools
import collections
import numpy as np
import pandas as pd
import scipy as sp
from typing import Any
from dotenv import load_dotenv
import glob
import io
import contextlib
import tqdm
import threading
import queue
import time
import uuid
import concurrent.futures

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

MODEL_NAME = os.getenv("TEXT_MODEL")
MAX_OUTPUT_LENGTH = 5000
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import cad_utils

STORAGE_DIR = os.path.join(PROJECT_ROOT, "storage")
STORAGE_JSON_DIR = os.path.join(STORAGE_DIR, "json")
AGENTS_DIR = os.path.join(PROJECT_ROOT, "agents")
SRC_CHAT_FILE = os.path.join(PROJECT_ROOT, "src", "chat.py")
DEFAULT_MAX_TOOL_DEPTH = 0
SUBPROCESS_TIMEOUT_SECONDS = None
MODEL_STREAM_IDLE_TIMEOUT_SECONDS = None
MODEL_REQUEST_TIMEOUT_SECONDS = None

# #region agent log
_AGENT_DEBUG_LOG = "/Users/samuellukudu/QilaiCo/End2End/cad-ai-project/.cursor/debug-815a61.log"
_debug_request_tls = threading.local()


def _debug_request_enter(trace_id: str | None = None) -> None:
    rid = trace_id or str(uuid.uuid4())
    stack = getattr(_debug_request_tls, "stack", None)
    if stack is None:
        stack = []
        _debug_request_tls.stack = stack
    stack.append(rid)


def _debug_request_exit() -> None:
    stack = getattr(_debug_request_tls, "stack", None)
    if stack:
        stack.pop()


def _debug_request_id() -> str | None:
    stack = getattr(_debug_request_tls, "stack", None)
    return stack[-1] if stack else None


def _agent_debug_log(location: str, message: str, data: dict[str, Any], hypothesis_id: str) -> None:
    try:
        rid = _debug_request_id()
        payload: dict[str, Any] = {
            "sessionId": "815a61",
            "timestamp": int(time.time() * 1000),
            "location": location,
            "message": message,
            "data": data,
            "hypothesisId": hypothesis_id,
        }
        if rid:
            payload["requestId"] = rid
        with open(_AGENT_DEBUG_LOG, "a", encoding="utf-8") as _df:
            _df.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass


# #endregion

TOOL_OUTPUT_SUMMARY_THRESHOLD = int(os.getenv("TOOL_OUTPUT_SUMMARY_THRESHOLD", "2500"))
MAX_SUMMARIZE_LENGTH = 50000

def truncate_output(output: str, max_length: int = MAX_OUTPUT_LENGTH, tool_name: str = "unknown_tool") -> str:
    output_len = len(output)
    if output_len <= TOOL_OUTPUT_SUMMARY_THRESHOLD:
        return output
        
    if output_len > MAX_SUMMARIZE_LENGTH:
        print(f"Output from {tool_name} is absurdly long ({output_len} chars). Auto-saving to disk to prevent context pollution...")
        
        tmp_dir = os.path.join(STORAGE_DIR, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        file_id = str(uuid.uuid4())[:8]
        tmp_path = os.path.join(tmp_dir, f"{tool_name}_output_{file_id}.txt")
        
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(output)
            
        preview_head = output[:1000]
        preview_tail = output[-1000:] if output_len > 2000 else ""
        
        return (
            f"System Note: Output from '{tool_name}' was extremely large ({output_len} chars) and would exceed the context window.\n"
            f"Instead of failing, the full raw output has been automatically saved to:\n"
            f"  {tmp_path}\n\n"
            f"You can now use `terminal` (e.g., `grep`, `head`, `tail`) or `python` to analyze this specific file without re-running the heavy command.\n\n"
            f"--- PREVIEW (First & Last 1000 chars) ---\n"
            f"{preview_head}\n\n"
            f"... [ {output_len - 2000} characters omitted ] ...\n\n"
            f"{preview_tail}"
        )
    
    print(f"Output from {tool_name} is long ({output_len} chars). Summarizing to prevent context pollution...")
    try:
        summary_prompt = f"Summarize the following raw tool output from '{tool_name}' concisely. Keep ALL important facts, file paths, coordinate numbers, and code snippets intact. Do not add commentary. \n\nRAW OUTPUT:\n{output[:15000]}"
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.0,
            max_tokens=1000
        )
        return "--- AUTO-SUMMARIZED BY SYSTEM ---\n" + res.choices[0].message.content
    except Exception as e:
        print(f"Summarizer failed: {e}")
        if output_len <= max_length:
            return output
        return output[:max_length] + f"\n... (Output truncated, total length: {output_len})"

def _format_timeout_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "unbounded"
    if seconds.is_integer():
        return str(int(seconds))
    return f"{seconds:.1f}"


def _subprocess_timeout_message(tool_name: str, timeout_seconds: float | None, command: str) -> str:
    timeout_str = _format_timeout_seconds(timeout_seconds)
    return (
        f"{tool_name} timed out after {timeout_str}s. "
        f"Command: {command}. Try a narrower query, smaller region, or fewer files."
    )


def _iter_stream_with_idle_timeout(stream, idle_timeout_seconds: float | None):
    """
    Consume provider stream in a worker thread so the main loop can enforce idle timeout.
    """
    events: queue.Queue[tuple[str, Any]] = queue.Queue()
    sentinel_done = object()

    def _reader():
        try:
            for chunk in stream:
                events.put(("chunk", chunk))
            events.put(("done", sentinel_done))
        except Exception as exc:
            events.put(("error", exc))

    threading.Thread(target=_reader, daemon=True).start()

    while True:
        try:
            if idle_timeout_seconds is None:
                kind, payload = events.get()
            else:
                kind, payload = events.get(timeout=idle_timeout_seconds)
        except queue.Empty as exc:
            # #region agent log
            _agent_debug_log(
                "chat.py:_iter_stream_with_idle_timeout",
                "model stream idle timeout (no chunks)",
                {"idle_timeout_seconds": idle_timeout_seconds},
                "H-C",
            )
            # #endregion
            raise TimeoutError(
                f"Model stream idle timeout after {_format_timeout_seconds(idle_timeout_seconds)}s."
            ) from exc

        if kind == "chunk":
            yield payload
            continue
        if kind == "error":
            raise payload
        if kind == "done":
            return


def _command_within_scope(command: str) -> tuple[bool, str]:
    # Block only literal root (ls /, find /) - not project paths under allowed_prefixes
    if re.search(r"\b(find|ls|cat|head|tail|grep)\s+/\s*$", command):
        return False, "Access denied: absolute root paths are not allowed."
    if ".dxf" in command.lower() or "storage/dxf" in command.lower():
        return False, "Access denied: use only JSON files under storage/json."

    absolute_paths = re.findall(r"(?<![\w.-])(/[^\s\"'`|;&]+)", command)
    allowed_prefixes = (STORAGE_DIR, AGENTS_DIR, SRC_CHAT_FILE)
    allowed_redirects = ("/dev/null", "/dev/stderr", "/dev/stdout")
    for raw_path in absolute_paths:
        normalized = os.path.abspath(raw_path)
        if normalized == SRC_CHAT_FILE:
            continue
        if normalized in allowed_redirects:
            continue  # Standard redirect targets, safe
        if not any(
            normalized == prefix or normalized.startswith(prefix + os.sep)
            for prefix in allowed_prefixes
            if prefix != SRC_CHAT_FILE
        ):
            return False, f"Access denied: '{raw_path}' is outside allowed scope."
        if normalized.startswith(STORAGE_DIR + os.sep):
            if os.path.splitext(normalized)[1].lower() == ".dxf":
                return False, "Access denied: use only JSON files under storage/json."
    return True, ""

def terminal(command: str) -> str:
    is_allowed, message = _command_within_scope(command)
    if not is_allowed:
        print(message)
        return message

    try:
        tokens = shlex.split(command)
    except ValueError as e:
        error_msg = f"Command parsing error: {e}"
        print(error_msg)
        return error_msg

    blocked = {"rm", "sudo", "dd", "chmod"}
    if tokens and tokens[0].lower() in blocked:
        msg = "Cannot execute 'rm, sudo, dd, chmod' commands since they are dangerous"
        print(msg)
        return msg

    print(f"Executing terminal command `{command}`")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            shell=True,
            check=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        output = result.stdout if result.stdout else "(No output returned)"
        return truncate_output(output, tool_name="terminal")
    except subprocess.CalledProcessError as e:
        return f"Command failed: {e.stderr}"
    except subprocess.TimeoutExpired:
        return _subprocess_timeout_message("terminal", SUBPROCESS_TIMEOUT_SECONDS, command)

# Removed Tools: grep, glob, read_file (consolidated into terminal)

# New Tool: read_agent_instructions
def read_agent_instructions(agent_name: str) -> str:
    """
    Read the instructions for a specific agent from the agents directory.
    """
    # sanitize agent_name
    agent_name = os.path.basename(agent_name)
    if not agent_name.endswith(".md"):
        agent_name += ".md"
        
    path = os.path.join(PROJECT_ROOT, "agents", agent_name)
    if not os.path.exists(path):
        return f"Agent '{agent_name}' not found."
        
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Failed to read agent instructions: {e}"

def get_python_executable():
    """Helper to find the project's python executable."""
    venv_python = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")
    if os.path.exists(venv_python):
        return venv_python
    return "python3"

# New Tool: search_cad_file
def search_cad_file(command: str, **kwargs) -> str:
    """
    Executes the src/search_cad.py script with the provided arguments.
    """
    # Construct the command
    script_path = os.path.join(PROJECT_ROOT, "src", "search_cad.py")
    if not os.path.exists(script_path):
        return f"Error: {script_path} not found."

    cmd = [get_python_executable(), script_path, command]
    
    # Handle paths argument - ensure it's a list
    paths = kwargs.get("paths", [STORAGE_JSON_DIR])
    if isinstance(paths, str):
        paths = [paths]
    
    # Add arguments based on command
    if command == "text":
        if "query" not in kwargs:
            return "Error: 'query' argument is required for text search."
        
        queries = kwargs["query"]
        if isinstance(queries, str):
            # Handle potential stringified JSON list from LLM
            q_str = queries.strip()
            if q_str.startswith("[") and q_str.endswith("]"):
                try:
                    parsed = json.loads(q_str)
                    if isinstance(parsed, list):
                        queries = [str(x) for x in parsed]
                    else:
                        queries = [queries]
                except json.JSONDecodeError:
                    queries = [queries]
            else:
                queries = [queries]
        
        for q in queries:
            cmd.extend(["-q", str(q)])
            
        cmd.extend(paths)
        
    elif command == "bbox":
        for arg in ["min_x", "min_y", "max_x", "max_y"]:
            if arg not in kwargs:
                return f"Error: '{arg}' argument is required for bbox search."
            cmd.append(str(kwargs[arg]))
        cmd.extend(paths)
        
    elif command == "point":
        for arg in ["x", "y"]:
            if arg not in kwargs:
                return f"Error: '{arg}' argument is required for point search."
            cmd.append(str(kwargs[arg]))
        cmd.extend(paths)
        
    else:
        return f"Error: Unknown command '{command}'. Available: text, bbox, point."

    print(f"Executing search_cad_file: {' '.join(cmd)}")
    try:
        env = os.environ.copy()
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = PROJECT_ROOT

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
            env=env
        )
        stdout = result.stdout or ""
        if command != "text":
            return truncate_output(stdout, tool_name="search_cad_file")

        # For text search, return a file-level summary for discovery:
        # [{file, total_hits, hits_by_keyword:{...}}]
        try:
            parsed = json.loads(stdout)
        except Exception:
            # If stdout isn't valid JSON, fall back to the raw output.
            return truncate_output(stdout, tool_name="search_cad_file")

        if not isinstance(parsed, list):
            return truncate_output(stdout, tool_name="search_cad_file")

        hits_by_file: dict[str, dict[str, int]] = {}
        for rec in parsed:
            if not isinstance(rec, dict):
                continue
            file_path = rec.get("file")
            if not file_path:
                continue
            matched = rec.get("matched_queries") or []
            if not isinstance(matched, list):
                continue
            bucket = hits_by_file.setdefault(str(file_path), {})
            for kw in matched:
                if kw is None:
                    continue
                k = str(kw)
                bucket[k] = bucket.get(k, 0) + 1

        summary = [
            {
                "file": file_path,
                "total_hits": sum(hits.values()),
                "hits_by_keyword": dict(sorted(hits.items(), key=lambda kv: (-kv[1], kv[0]))),
            }
            for file_path, hits in hits_by_file.items()
        ]
        summary.sort(key=lambda r: (-int(r.get("total_hits", 0)), str(r.get("file", ""))))

        return truncate_output(json.dumps(summary, indent=2, ensure_ascii=False), tool_name="search_cad_file")
    except subprocess.CalledProcessError as e:
        return f"Search failed: {e.stderr}"
    except subprocess.TimeoutExpired:
        return _subprocess_timeout_message("search_cad_file", SUBPROCESS_TIMEOUT_SECONDS, " ".join(cmd))


# New Tool: find_regions
def find_regions(
    file: str,
    target_keyword: str,
    layout: str = None,
    vlm: bool = False,
) -> str:
    """
    Finds regions of interest using the geometric bbox pipeline.
    Returns bounding boxes only — use EXCLUSIVELY when the file and target label are already explicit.
    layout is only for sub-region/room/section lookups constrained within a parent layout.
    Set vlm=true to enable VLM visual refinement (boundary padding / cropping) when the bbox will be visualized.
    If no visualization is needed, keep vlm=false.
    DO NOT use this as a general file search tool. For exploration, use search_cad_file, terminal, or custom python instead.
    """
    script_path = os.path.join(PROJECT_ROOT, "src", "refine_bbox_vlm.py")
    if not os.path.exists(script_path):
        return f"Error: {script_path} not found."
    if not os.path.exists(file):
        return f"Error: File '{file}' not found."
    if not target_keyword or not str(target_keyword).strip():
        return "Error: 'target_keyword' is required."

    cmd = [get_python_executable(), script_path, "--file", file, "--target_keyword", str(target_keyword)]
    if layout:
        cmd.extend(["--layout", str(layout)])
    if vlm:
        cmd.append("--vlm")

    print(f"Executing find_regions: {' '.join(cmd)}")
    try:
        timeout_seconds = None
        
        env = os.environ.copy()
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = PROJECT_ROOT

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout_seconds,
            env=env
        )
        
        stdout = result.stdout
        marker = "--- Final Refined Bounding Boxes ---"
        if marker in stdout:
            parts = stdout.split(marker)
            json_str = parts[-1].strip()
            try:
                start_idx = json_str.find('[')
                end_idx = json_str.rfind(']')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_str = json_str[start_idx:end_idx+1]
                    parsed = json.loads(json_str)
                    return json.dumps(parsed)
                else:
                    return f"Error: Could not find valid JSON array in output. Raw output:\n{truncate_output(stdout, tool_name='find_regions')}"
            except json.JSONDecodeError as e:
                return f"Error parsing JSON output from find_regions: {e}. Raw output:\n{truncate_output(stdout, tool_name='find_regions')}"
        else:
            return f"Error: Marker '{marker}' not found in output. Raw output:\n{truncate_output(stdout, tool_name='find_regions')}"

    except subprocess.CalledProcessError as e:
        return f"find_regions failed with exit code {e.returncode}.\nSTDERR:\n{truncate_output(e.stderr, tool_name='find_regions')}\nSTDOUT:\n{truncate_output(e.stdout, tool_name='find_regions')}"
    except subprocess.TimeoutExpired:
        return _subprocess_timeout_message("find_regions", timeout_seconds, " ".join(cmd))


# New Tool: visualize_cad
def visualize_cad(file_path: str, bbox: str, output: str) -> str:
    """
    Generates an SVG visualization of a specific CAD region.
    Wraps visualize_cad.py.
    """
    script_path = os.path.join(PROJECT_ROOT, "src", "visualize_cad.py")
    if not os.path.exists(script_path):
        return f"Error: {script_path} not found."

    # Validate file_path
    if not os.path.exists(file_path):
         return f"Error: File '{file_path}' not found."

    # Ensure output directory exists (storage/images)
    images_dir = os.path.join(STORAGE_DIR, "images")
    if not os.path.exists(images_dir):
        try:
            os.makedirs(images_dir)
            print(f"Created directory: {images_dir}")
        except OSError as e:
            return f"Error creating images directory: {e}"

    # Force output to be in storage/images
    # Strip any directory path provided in 'output' and just take the filename
    filename = os.path.basename(output)
    if not filename.lower().endswith(".svg"):
        filename += ".svg"
        
    output_path = os.path.join(images_dir, filename)

    cmd = [get_python_executable(), script_path, file_path, "--bbox", bbox, "--output", output_path]
    
    print(f"Executing visualize_cad: {' '.join(cmd)}")
    try:
        env = os.environ.copy()
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = PROJECT_ROOT

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
            env=env
        )
        return truncate_output(result.stdout, tool_name="visualize_cad") + f"\nVisualization successfully saved to: {output_path}"
    except subprocess.CalledProcessError as e:
        return f"Visualization failed: {e.stderr}"
    except subprocess.TimeoutExpired:
        return _subprocess_timeout_message("visualize_cad", SUBPROCESS_TIMEOUT_SECONDS, " ".join(cmd))


# New Tool: extract_region_data
def extract_region_data(file_path: str, bbox: str, text_only: bool = False) -> str:
    """
    Extracts entity data within a specific bounding box from a CAD JSON file.
    Wraps src/extract_roi_data.py.
    """
    script_path = os.path.join(PROJECT_ROOT, "src", "extract_roi_data.py")
    if not os.path.exists(script_path):
        return f"Error: {script_path} not found."

    # Validate file_path
    if not os.path.exists(file_path):
         return f"Error: File '{file_path}' not found."

    cmd = [get_python_executable(), script_path, "--file", file_path, "--bbox", bbox, "--llm-mode"]
    if text_only:
        cmd.append("--text-only")
    
    print(f"Executing extract_region_data: {' '.join(cmd)}")
    try:
        env = os.environ.copy()
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = PROJECT_ROOT

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
            env=env
        )
        return truncate_output(result.stdout, tool_name="extract_region_data")
    except subprocess.CalledProcessError as e:
        return f"Extraction failed with exit code {e.returncode}.\nSTDERR:\n{truncate_output(e.stderr, tool_name='extract_region_data')}\nSTDOUT:\n{truncate_output(e.stdout, tool_name='extract_region_data')}"
    except subprocess.TimeoutExpired:
        return _subprocess_timeout_message("extract_region_data", SUBPROCESS_TIMEOUT_SECONDS, " ".join(cmd))

# New Visual Analysis Tools
from src.image_analysis import VisualAnalyzer

_visual_analyzer_instance = None

def get_visual_analyzer():
    global _visual_analyzer_instance
    if _visual_analyzer_instance is None:
        _visual_analyzer_instance = VisualAnalyzer()
    return _visual_analyzer_instance

def _ensure_image_exists(image_path: str, cad_file_path: str = None, bbox: str = None) -> str:
    """Helper to ensure image exists, creating it if necessary."""
    if os.path.exists(image_path):
        return image_path
        
    if not cad_file_path or not bbox:
        raise ValueError(f"Image {image_path} does not exist, and cad_file_path/bbox were not provided to create it.")
        
    print(f"Image {image_path} not found. Creating it using visualize_cad...")
    res = visualize_cad(cad_file_path, bbox, image_path)
    if "Error" in res:
        raise ValueError(f"Failed to create image: {res}")
    
    # visualize_cad saves it to storage/images, so let's get the exact path
    filename = os.path.basename(image_path)
    if not filename.lower().endswith(".svg"):
        filename += ".svg"
    actual_path = os.path.join(STORAGE_DIR, "images", filename)
    if os.path.exists(actual_path):
        return actual_path
    
    raise ValueError(f"Failed to create image at {actual_path}")

def _get_text_annotations(cad_file_path: str, bbox: str) -> str:
    """Helper to extract text annotations for a given bbox."""
    try:
        script_path = os.path.join(PROJECT_ROOT, "src", "extract_roi_data.py")
        cmd = [get_python_executable(), script_path, "--file", cad_file_path, "--bbox", bbox, "--text-only", "--llm-mode"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except Exception as e:
        return f"Failed to extract text annotations: {e}"

def analyze_image_description(image_path: str, context: str = "", query: str = "", cad_file_path: str = None, bbox: str = None) -> str:
    """
    Provide an exhaustively detailed description of a CAD drawing image (SVG or PNG).
    """
    try:
        actual_path = _ensure_image_exists(image_path, cad_file_path, bbox)
        if not query:
            query = "Provide an exhaustively detailed description of this CAD drawing. Include all text labels, room names, grid axes, dimensions, structural elements (columns, walls, doors, windows), and their exact spatial relationships. Organize the response logically with clear markdown headings."
        analyzer = get_visual_analyzer()
        result = analyzer.describe(image_path=actual_path, context=context, query=query)
        return str(result)
    except Exception as e:
        return f"Error in analyze_image_description: {e}"

def analyze_image_design(image_path: str, context: str = "", query: str = "", cad_file_path: str = None, bbox: str = None, include_description: bool = False) -> str:
    """
    Identifies design defects, spatial conflicts, and structural inconsistencies visually.
    """
    try:
        actual_path = _ensure_image_exists(image_path, cad_file_path, bbox)
        
        enhanced_context = context
        analyzer = get_visual_analyzer()
        
        if include_description:
            desc = analyzer.describe(image_path=actual_path, context=context)
            enhanced_context += f"\n\nImage Description:\n{desc}"
            
        if cad_file_path and bbox:
            text_annotations = _get_text_annotations(cad_file_path, bbox)
            enhanced_context += f"\n\nExtracted Text Annotations:\n{text_annotations}"

        if not query:
            query = "Identify any potential design errors, inconsistencies, or omissions in this floor plan."
            
        result = analyzer.analyze_design(image_path=actual_path, context=enhanced_context, query=query)
        return json.dumps([d.model_dump() for d in result], indent=2)
    except Exception as e:
        return f"Error in analyze_image_design: {e}"

def analyze_image_compliance(image_path: str, context: str = "", query: str = "", cad_file_path: str = None, bbox: str = None, include_description: bool = False) -> str:
    """
    Checks for safety, accessibility, and standard building code violations visually.
    """
    try:
        actual_path = _ensure_image_exists(image_path, cad_file_path, bbox)
        
        enhanced_context = context
        analyzer = get_visual_analyzer()
        
        if include_description:
            desc = analyzer.describe(image_path=actual_path, context=context)
            enhanced_context += f"\n\nImage Description:\n{desc}"
            
        if cad_file_path and bbox:
            text_annotations = _get_text_annotations(cad_file_path, bbox)
            enhanced_context += f"\n\nExtracted Text Annotations:\n{text_annotations}"

        if not query:
            query = "Check this floor plan for compliance issues regarding safety, accessibility, and standard building codes."
            
        result = analyzer.analyze_compliance(image_path=actual_path, context=enhanced_context, query=query)
        return json.dumps([c.model_dump() for c in result], indent=2)
    except Exception as e:
        return f"Error in analyze_image_compliance: {e}"

# 1. Define the safe_import logic
def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """
    A wrapper around __import__ that only allows specific libraries.
    """
    allowed_modules = {
        "math", "datetime", "random", "itertools", "collections",
        "numpy", "pandas", "scipy", "json", "re", "glob", "tqdm", "os", "csv", "io",
        "src",
    }
    
    # Check the root module name (e.g. 'numpy' from 'numpy.random')
    root_name = name.split('.')[0]
    
    if root_name in allowed_modules:
        return __import__(name, globals, locals, fromlist, level)
    
    raise ImportError(f"Importing '{name}' is restricted. Allowed: {list(allowed_modules)}")

# 2. Define the tool function
def python(code: str) -> str:
    """
    Execute Python code in a safe environment.
    """
    
    # Define what built-in functions the agent can use
    allowed_builtins = {
        "print": print,
        "range": range,
        "len": len,
        "int": int,
        "float": float,
        "str": str,
        "sum": sum,
        "min": min,
        "max": max,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "sorted": sorted,
        "reversed": reversed,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "abs": abs,
        "round": round,
        "pow": pow,
        "divmod": divmod,
        "all": all,
        "any": any,
        "bool": bool,
        "chr": chr,
        "ord": ord,
        "slice": slice,
        "type": type,
        "isinstance": isinstance,
        "hasattr": hasattr,
        "getattr": getattr,
        "setattr": setattr,
        "os": os,
        "open": open,
        "re": re,
        # CRITICAL FIX: Allow imports via our safe wrapper
        "__import__": safe_import, 
    }

    # Define the environment variables (pre-loaded libraries)
    # Pre-inject cad_utils so agent can use extract_entities, decode_cad_unicode, etc.
    # without defining or importing them (avoids "name 'extract_entities' is not defined")
    safe_globals = {
        "__builtins__": allowed_builtins,
        "math": math,
        "datetime": datetime,
        "random": random,
        "itertools": itertools,
        "collections": collections,
        "np": np,
        "numpy": np,
        "pd": pd,
        "pandas": pd,
        "sp": sp,
        "scipy": sp,
        "glob": glob,
        "tqdm": tqdm,
        # CAD entity extraction (from cad_utils) - always available
        "extract_entities": cad_utils.extract_entities,
        "extract_entities_recursive": cad_utils.extract_entities,
        "extract_text_from_entity": cad_utils.extract_text_from_entity,
        "iter_text_entities": cad_utils.iter_text_entities,
        "extract_text_annotations": cad_utils.extract_text_annotations,
        "decode_cad_unicode": cad_utils.decode_cad_unicode,
        "get_entity_position": cad_utils.get_entity_position,
        "get_block_entities": cad_utils.get_block_entities,
        "normalize_text": cad_utils.normalize_text,
        "cad_utils": cad_utils,
    }

    local_vars = {}
    output_buffer = io.StringIO()

    try:
        # Capture stdout to prevent printing to real stdout and to capture for the agent
        with contextlib.redirect_stdout(output_buffer):
            # Execute the code
            exec(code, safe_globals, local_vars)
        
        # Clean up internal variables before returning
        if "__builtins__" in local_vars:
            del local_vars["__builtins__"]
            
        captured_output = output_buffer.getvalue()
        
        # Format the output more cleanly
        result_parts = []
        if captured_output.strip():
            result_parts.append(f"STDOUT:\n{captured_output}")
            
        # Only show local variables that are relevant (not modules or internal stuff)
        relevant_vars = {k: v for k, v in local_vars.items() 
                        if not k.startswith('_') and not hasattr(v, '__module__')}
        
        if relevant_vars:
            result_parts.append(f"Variables:\n{str(relevant_vars)}")
            
        if not result_parts:
            return "(No output or variables returned)"
            
        return truncate_output("\n\n".join(result_parts), tool_name="python")
        
    except Exception as e:
        return f"Python execution error: {e}"

# New Memory Tools
_agent_memory = {}

def save_to_memory(key: str, value: str) -> str:
    """Saves a key-value pair to the agent's scratchpad memory."""
    _agent_memory[key] = value
    return f"Saved '{key}' to memory."

def read_memory(key: str = None) -> str:
    """Reads a key from memory, or returns all memory if key is None."""
    if key:
        if key not in _agent_memory:
            return f"Key '{key}' not found in memory."
        return str(_agent_memory[key])
    return json.dumps(_agent_memory, indent=2)

def delegate_to_agent(agent_name: str, task: str) -> str:
    """Delegates a sub-task to a specialized agent and returns its output."""
    instructions = read_agent_instructions(agent_name)
    if "not found" in instructions:
        return instructions
        
    sub_messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": task}
    ]
    
    # Run the agent in a bounded context
    gen = run_agent(sub_messages, max_tool_depth=10, agent_name=agent_name)
    final_content = ""
    try:
        while True:
            event = next(gen)
            if event.get("type") == "content":
                final_content += event.get("content", "")
    except StopIteration as e:
        pass
        
    return final_content if final_content else "Agent completed task but returned no output."

MAP_FN = {
    "terminal": terminal,
    "python": python,
    "read_agent_instructions": read_agent_instructions,
    "delegate_to_agent": delegate_to_agent,
    "save_to_memory": save_to_memory,
    "read_memory": read_memory,
    "search_cad_file": search_cad_file,
    "find_regions": find_regions,
    "visualize_cad": visualize_cad,
    "extract_region_data": extract_region_data,
    "analyze_image_description": analyze_image_description,
    "analyze_image_design": analyze_image_design,
    "analyze_image_compliance": analyze_image_compliance,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": (
                "USE: Shell on allowed paths only—inventory (ls on primary JSON dir), grep/head/tail/cat on known files, "
                "small CLIs. "
                "When file paths are unknown, call search_cad_file first and only then use terminal on returned paths. "
                "NOT: Bulk structured entity extraction (use python with explicit JSON paths and helper-first flow). "
                "NOT: Use search_cad_file for simple file listing. "
                "NOT: Repeated brute-force probing loops across many terminal calls without consolidating via python."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command, e.g. ls, grep, head, cat, or python script path with args.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_cad_file",
            "description": (
                "USE: Primary corpus discovery tool—which JSON files contain given text, overlap a bbox, or contain a point. "
                "This is the required first discovery tool when the target files are unknown. "
                "For text discovery, prefer one bundled query list in a single call to reduce retries. "
                "Requires real criteria (query for text; coordinates for bbox/point). "
                "NOT: Inventory ('what files exist', 'list CAD files')—use terminal ls on the JSON directory once; "
                "NOT: Repeated identical calls—after paths are known, use terminal/python on those paths."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["text", "bbox", "point"],
                        "description": "Search mode.",
                    },
                    "query": {
                        "description": "Text(s) to search for (required for 'text' command). Single string or list of strings.",
                        "anyOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ],
                    },
                    "min_x": {"type": "number"},
                    "min_y": {"type": "number"},
                    "max_x": {"type": "number"},
                    "max_y": {"type": "number"},
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Files or directories to search. Defaults to storage/json.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_regions",
            "description": (
                "USE EXCLUSIVELY to extract bounding boxes for a specific region prior to visualization or data extraction. "
                "REQUIRES: file path and target_keyword must already be known from prior search_cad_file/terminal/python discovery. "
                "Returns bounding boxes only, not search evidence. "
                "Use layout only when target_keyword is a sub-region, room, or section inside a parent layout. "
                "vlm=false (default): fast geometric bbox detection — use for data extraction or when visualization is not needed. "
                "vlm=true: REQUIRED when the bbox will be used for visualize_cad; this applies VLM padding/cropping for cleaner rendering. "
                "PRODUCES: bbox arrays consumed by extract_region_data and visualize_cad. "
                "NOT: general file search or discovery—use search_cad_file, terminal, or custom python for that."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "Path to the JSON CAD file.",
                    },
                    "target_keyword": {
                        "type": "string",
                        "description": "Primary keyword to look for (e.g., '10th floor', 'conference room').",
                    },
                    "layout": {
                        "type": "string",
                        "description": "Optional parent layout used only to scope a sub-region/room/section search (e.g., '10th floor plan' when looking for 'conference room').",
                    },
                    "vlm": {
                        "type": "boolean",
                        "description": "Enable VLM visual refinement (boundary padding/cropping). Set true when planning to call visualize_cad; keep false for non-visual extraction workflows. Default false.",
                    },
                },
                "required": ["file", "target_keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "visualize_cad",
            "description": (
                "USE: Render SVG for a known JSON file and bbox string. "
                "REQUIRES: bbox from a prior find_regions call or user-provided coordinates. Do not guess bbox values. "
                "If this bbox came from find_regions in this turn, find_regions must have used vlm=true. "
                "NOT: Discovering bbox or target—call find_regions first to obtain the bbox."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the JSON CAD file.",
                    },
                    "bbox": {
                        "type": "string",
                        "description": "Region 'min_x,min_y,max_x,max_y'.",
                    },
                    "output": {
                        "type": "string",
                        "description": "Output path for the SVG file.",
                    },
                },
                "required": ["file_path", "bbox", "output"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_region_data",
            "description": (
                "USE: Extract entity data (in minified JSON format) from a specific bounding box within a CAD JSON file. "
                "REQUIRES: bbox from a prior find_regions call or user-provided coordinates. Do not guess bbox values. "
                "Always uses LLM mode to save tokens. Optionally filter to text entities only. "
                "NOT: Discovery or search—call find_regions first to obtain the bbox."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the JSON CAD file.",
                    },
                    "bbox": {
                        "type": "string",
                        "description": "Bounding box in format 'min_x,min_y,max_x,max_y'.",
                    },
                    "text_only": {
                        "type": "boolean",
                        "description": "Set to true to extract only text entities (TEXT, MTEXT, ATTRIB).",
                        "default": False
                    },
                },
                "required": ["file_path", "bbox"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python",
            "description": (
                "USE: Structured extraction, aggregation, math, or custom search on explicit JSON path(s) after terminal grep/head scoped the work, or when search_cad_file cannot express the needed search. "
                "Preferred for cross-file annotation/entity analysis after shortlist discovery. "
                "Paths should come from search_cad_file discovery (or explicit user-provided paths); then terminal can sample and python can aggregate deeply. "
                "extract_entities, iter_text_entities, extract_text_from_entity, extract_text_annotations, decode_cad_unicode, "
                "get_entity_position, get_block_entities, normalize_text are PRE-LOADED; may import src.cad_utils. "
                "Prefer these helpers over writing custom traversal/decoder scripts; only add custom logic when helper coverage is insufficient. "
                "NOT: First pass over huge minified JSON (grep in terminal first); "
                "NOT: Blind corpus-wide search when search_cad_file or terminal can answer faster; "
                "NOT: Reimplement CAD decode/traversal—follow cad_entity_extraction template on top of helpers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python source to execute in the sandbox.",
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_agent_instructions",
            "description": (
                "USE: Rare inspection of what a named agent is for."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "Agent name (e.g. planner, detail_expert).",
                    },
                },
                "required": ["agent_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delegate_to_agent",
            "description": (
                "USE: Delegate a complex sub-task to a specialized agent (e.g. planner, file_triage_agent). "
                "This runs the agent in a separate context and returns its final textual answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "Agent name (e.g. planner, file_triage_agent).",
                    },
                    "task": {
                        "type": "string",
                        "description": "The specific task instruction or prompt to give the sub-agent.",
                    },
                },
                "required": ["agent_name", "task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_to_memory",
            "description": (
                "USE: Save important intermediate results (file paths, bounding boxes, summaries) "
                "to a scratchpad so you don't forget them across multiple tool calls."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_memory",
            "description": (
                "USE: Read a previously saved value from the scratchpad memory. "
                "Leave key empty to get all memory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Optional key to read."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_image_description",
            "description": (
                "Provide an exhaustively detailed description of a CAD drawing image. "
                "REQUIRES: image_path from a prior visualize_cad call. If no image exists yet, call visualize_cad first with a bbox from find_regions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Path to the image (SVG or PNG)."},
                    "context": {"type": "string", "description": "Supplementary context.", "default": ""},
                    "query": {"type": "string", "description": "User query.", "default": ""},
                    "cad_file_path": {"type": "string", "description": "Original CAD JSON file path (if image creation is needed)."},
                    "bbox": {"type": "string", "description": "Bounding box (if image creation is needed)."}
                },
                "required": ["image_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_image_design",
            "description": (
                "Identifies design defects, spatial conflicts, and structural inconsistencies visually. "
                "REQUIRES: image_path from a prior visualize_cad call. If no image exists yet, call visualize_cad first with a bbox from find_regions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Path to the image (SVG or PNG)."},
                    "context": {"type": "string", "description": "Supplementary context.", "default": ""},
                    "query": {"type": "string", "description": "User query.", "default": ""},
                    "cad_file_path": {"type": "string", "description": "Original CAD JSON file path (for text annotations extraction and image creation)."},
                    "bbox": {"type": "string", "description": "Bounding box (for text annotations extraction and image creation)."},
                    "include_description": {"type": "boolean", "description": "If true, generates an image description and appends it to the context.", "default": False}
                },
                "required": ["image_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_image_compliance",
            "description": (
                "Checks for safety, accessibility, and standard building code violations visually. "
                "REQUIRES: image_path from a prior visualize_cad call. If no image exists yet, call visualize_cad first with a bbox from find_regions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Path to the image (SVG or PNG)."},
                    "context": {"type": "string", "description": "Supplementary context.", "default": ""},
                    "query": {"type": "string", "description": "User query.", "default": ""},
                    "cad_file_path": {"type": "string", "description": "Original CAD JSON file path (for text annotations extraction and image creation)."},
                    "bbox": {"type": "string", "description": "Bounding box (for text annotations extraction and image creation)."},
                    "include_description": {"type": "boolean", "description": "If true, generates an image description and appends it to the context.", "default": False}
                },
                "required": ["image_path"]
            }
        }
    },
]

def run_agent(
    messages,
    max_tool_depth=DEFAULT_MAX_TOOL_DEPTH,
    agent_name="Assistant",
    *,
    agent_tools: list[dict] | None = None,
    _stream_trace_id: str | None = None,
):
    """
    Runs one full assistant turn including tool chaining.
    Streams events (tokens, tool calls, results).
    Returns the updated messages list via StopIteration.
    """
    _debug_request_enter(_stream_trace_id)
    try:
        updated = yield from _run_agent_core(
            messages, max_tool_depth=max_tool_depth, agent_name=agent_name,
            agent_tools=agent_tools,
        )
        return updated
    finally:
        _debug_request_exit()


def _run_agent_core(messages, max_tool_depth=DEFAULT_MAX_TOOL_DEPTH, agent_name="Assistant", agent_tools=None):
    tool_depth = 0
    last_find_regions_used_vlm: bool | None = None

    while True:
        # #region agent log
        _agent_debug_log(
            "chat.py:run_agent:loop",
            "run_agent iteration",
            {"tool_depth": tool_depth, "max_tool_depth": max_tool_depth},
            "H-D",
        )
        # #endregion
        # Add a mid-flight reminder if the agent is struggling
        if tool_depth == 8 or tool_depth == 15:
             messages.append({
                 "role": "system",
                 "content": "SYSTEM REMINDER: You have made several tool calls without returning a final answer. If you are stuck retrying the same approach (e.g. failing ROI verification on the same point), you MUST immediately change your strategy, abandon your current seed point/file, and try a completely different approach."
             })

        effective_tools = agent_tools if agent_tools is not None else tools
        request_kwargs = {
            "model": MODEL_NAME,
            "messages": messages,
            "tools": effective_tools,
            "tool_choice": "auto",
            "stream": True,
            "temperature": 0.2,
            "top_p": 1.0,
            "extra_body": {
                "top_k": 3,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
            },
        }
        if MODEL_REQUEST_TIMEOUT_SECONDS is not None:
            request_kwargs["timeout"] = MODEL_REQUEST_TIMEOUT_SECONDS

        try:
            stream = client.chat.completions.create(**request_kwargs)
        except Exception as e:
            print(f"[run_agent] model request failed: {e}")
            yield {"type": "error", "content": f"\n[Model request failed: {e}]"}
            return messages

        assistant_message = {"role": "assistant", "content": ""}
        tool_calls_buffer = {}

        # Notify start of response
        yield {"type": "turn_start", "agent_name": agent_name}

        # --- STREAM LOOP ---
        stream_finish_reason: str | None = None
        try:
            for chunk in _iter_stream_with_idle_timeout(stream, MODEL_STREAM_IDLE_TIMEOUT_SECONDS):
                ch0 = chunk.choices[0] if chunk.choices else None
                if ch0 is not None and getattr(ch0, "finish_reason", None):
                    stream_finish_reason = str(ch0.finish_reason)
                delta = ch0.delta if ch0 is not None else None
                if delta is None:
                    continue

                # Stream normal content
                if delta.content:
                    yield {"type": "content", "content": delta.content}
                    assistant_message["content"] += delta.content

                # Collect tool calls incrementally
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        tool_calls_buffer.setdefault(idx, {
                            "id": "",
                            "function": {"name": "", "arguments": ""}
                        })

                        if tc.id:
                            tool_calls_buffer[idx]["id"] = tc.id
                        if tc.function.name:
                            tool_calls_buffer[idx]["function"]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls_buffer[idx]["function"]["arguments"] += tc.function.arguments
        except TimeoutError as e:
            # #region agent log
            _agent_debug_log(
                "chat.py:run_agent:stream",
                "stream TimeoutError",
                {"error": str(e), "tool_depth": tool_depth},
                "H-C",
            )
            # #endregion
            print(f"[run_agent] stream timeout: {e}")
            yield {"type": "error", "content": f"\n[{e}]"}
            messages.append(assistant_message)
            return messages
        except Exception as e:
            # #region agent log
            _agent_debug_log(
                "chat.py:run_agent:stream",
                "stream Exception",
                {"exc_type": type(e).__name__, "error": str(e), "tool_depth": tool_depth},
                "H-B",
            )
            # #endregion
            print(f"[run_agent] stream failed: {e}")
            yield {"type": "error", "content": f"\n[Model stream failed: {e}]"}
            messages.append(assistant_message)
            return messages

        # Convert dict -> ordered list
        tool_calls = list(tool_calls_buffer.values())

        if not tool_calls:
            fallback_calls = []
            for match in re.finditer(r"<tool_code>\s*(\{.*?\})\s*</tool_(?:call|code)>", assistant_message["content"], re.DOTALL):
                try:
                    payload = json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
                name = payload.get("name")
                args = payload.get("arguments", {})
                if not name:
                    continue
                fallback_calls.append({
                    "id": f"fallback_{len(fallback_calls)}",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args),
                    },
                })
            tool_calls = fallback_calls

        # #region agent log
        _agent_debug_log(
            "chat.py:run_agent:after_stream",
            "stream pass complete (after fallback)",
            {
                "tool_depth": tool_depth,
                "content_len": len(assistant_message.get("content") or ""),
                "tool_calls_n": len(tool_calls),
                "finish_reason": stream_finish_reason,
            },
            "H-A",
        )
        # #endregion

        # If no tool calls → we're done
        if not tool_calls:
            # #region agent log
            _agent_debug_log(
                "chat.py:run_agent:final_no_tools",
                "ending turn without tools",
                {
                    "tool_depth": tool_depth,
                    "content_len": len(assistant_message.get("content") or ""),
                },
                "H-A",
            )
            # #endregion
            messages.append(assistant_message)
            return messages

        if max_tool_depth > 0 and tool_depth >= max_tool_depth:
            err_msg = f"Error: Reached maximum tool depth of {max_tool_depth}. Terminating reasoning loop."
            yield {"type": "error", "content": f"\n[{err_msg}]"}
            messages.append({"role": "system", "content": err_msg})
            break

        tool_depth += 1

        # Attach tool calls to assistant message
        assistant_message["tool_calls"] = tool_calls
        messages.append(assistant_message)
        
        # Yield detected tool calls
        yield {"type": "tool_calls_detected", "tool_calls": tool_calls}

        # Execute tools in parallel
        def execute_tool(tc):
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            
            if not name:
                return tc, name, args, "Error: Tool name is missing."
            if name not in MAP_FN:
                return tc, name, args, f"Error: Tool '{name}' not found."
            
            nonlocal last_find_regions_used_vlm
            try:
                if name == "find_regions":
                    last_find_regions_used_vlm = bool(args.get("vlm", False))
                    result = MAP_FN[name](**args)
                elif name == "visualize_cad":
                    if last_find_regions_used_vlm is False:
                        result = (
                            "Policy violation: visualization requires bbox from find_regions with vlm=true "
                            "to apply VLM padding/cropping. Re-run find_regions with vlm=true, then call visualize_cad."
                        )
                    else:
                        result = MAP_FN[name](**args)
                else:
                    result = MAP_FN[name](**args)
            except Exception as e:
                result = f"Tool '{name}' failed with error: {e}"
                
            return tc, name, args, result

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for tc in tool_calls:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                yield {"type": "tool_start", "tool": name, "args": args, "tool_call_id": tc["id"]}
                futures.append(executor.submit(execute_tool, tc))
                
            # Need to collect results in order
            for future in futures:
                tc, name, args, result = future.result()
                
                # Check for policy violation nudge
                if name == "visualize_cad" and "Policy violation:" in str(result):
                    messages.append({
                        "role": "system",
                        "content": (
                            "SYSTEM NUDGE: For visualization flows, call find_regions with vlm=true before visualize_cad. "
                            "For non-visual extraction flows, keep vlm=false."
                        ),
                    })

                yield {"type": "tool_result", "tool": name, "result": result, "tool_call_id": tc["id"]}

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": name,
                    "content": str(result),
                })

        # Loop continues → model sees tool output and responds again


def get_available_agents():
    """Scans the agents directory and returns a summary of available agents."""
    agents_dir = os.path.join(PROJECT_ROOT, "agents")
    if not os.path.exists(agents_dir):
        return "No agents found."
    
    agent_info = []
    for f in sorted(os.listdir(agents_dir)):
        if f.endswith(".md"):
            try:
                path = os.path.join(agents_dir, f)
                agent_name = f[:-3]
                description = ""
                tools = None

                # Parse only YAML front-matter header (between the first pair of '---')
                with open(path, "r", encoding="utf-8") as file:
                    in_front_matter = False
                    for line in file:
                        stripped = line.strip()
                        if not in_front_matter:
                            if stripped == "---":
                                in_front_matter = True
                            continue

                        # front-matter ends
                        if stripped == "---":
                            break

                        m_name = re.match(r"^name:\s*(.*)\s*$", stripped)
                        if m_name:
                            agent_name = m_name.group(1).strip().strip('"').strip("'")
                            continue

                        m_desc = re.match(r"^description:\s*(.*)\s*$", stripped)
                        if m_desc:
                            description = m_desc.group(1).strip()
                            continue

                        m_tools = re.match(r"^tools:\s*(.*)\s*$", stripped)
                        if m_tools:
                            tools_raw = m_tools.group(1).strip()
                            try:
                                tools = json.loads(tools_raw)  # tools are expressed as a JSON array
                            except Exception:
                                tools = None

                if not description:
                    description = "No description available."
                tools_list = tools if isinstance(tools, list) else []
                tools_text = ", ".join(str(t) for t in tools_list)
                if tools_text:
                    agent_info.append(f"- {agent_name}: {description} (tools: {tools_text})")
                else:
                    agent_info.append(f"- {agent_name}: {description}")
            except Exception:
                agent_info.append(f"- {f[:-3]}")
                
    return "\n".join(agent_info)

def get_available_contexts():
    """Scans the context directory and returns a summary with descriptions."""
    context_dir = os.path.join(PROJECT_ROOT, "context")
    if not os.path.exists(context_dir):
        return "No contexts found."

    CONTEXT_DESCRIPTIONS = {
        "cad_entity_extraction": "MANDATORY Python template for CAD entity extraction — call before writing any extraction code",
        "find_plan_seeds": "Operational workflow for ROI: use find_regions tool with target_keyword and optional layout to extract bounding boxes directly, then extract_region_data",
        "dev": "Implementation mindset: geometric algorithms, JSON parsing, CAD entity transformations, precision",
        "research": "Compliance/standards mindset: keyword extraction, regulatory alignment, bilingual text analysis",
        "review": "QA/validation mindset: design errors, dimensional mismatches, structural consistency checks",
    }
    
    context_info = []
    for f in sorted(os.listdir(context_dir)):
        if f.endswith(".md"):
            name = f[:-3]
            desc = CONTEXT_DESCRIPTIONS.get(name, "")
            if desc:
                context_info.append(f"- {name}: {desc}")
            else:
                context_info.append(f"- {name}")
            
    return "\n".join(context_info)


def get_cli_system_prompt(available_agents: str, available_contexts: str) -> str:
    return f"""
    You are a CLI-based CAD analysis agent. Your goal is to produce precise, evidence-backed answers using available tools.

You MUST:
- Base conclusions only on tool outputs (paths, snippets, counts, coordinates)
- Avoid speculation
- Explicitly state when evidence is insufficient

---

# CORE EXECUTION MODEL (STRICT)

You MUST follow this decision pipeline:

## STEP 1 — Specialized agent check
If the task matches a specialized agent:
→ call `delegate_to_agent` with a clear task
→ STOP (do not execute tools yourself)

---

## STEP 2 — Determine task type

Classify the request into ONE of:

1. DISCOVERY (unknown files / search problem)
2. EXTRACTION (known files, structured data needed)
3. REGION-BASED (requires bounding boxes)
4. VISUAL (rendering or image reasoning)
5. DEVELOPMENT / COMPLIANCE (requires reasoning playbooks)

---

## STEP 3 — Select FIRST tool (MANDATORY)

| Task type | First tool |
|----------|-----------|
| DISCOVERY | `search_cad_file` |
| FILE LIST / DISK | `terminal` (`ls`) |
| KNOWN FILE INSPECTION | `terminal` |
| EXTRACTION | `read_context('cad_entity_extraction')` → `python` |
| REGION-BASED | `find_regions` |
| VISUAL | `visualize_cad` (ONLY if bbox already exists) |
| DEV / QA / COMPLIANCE | `read_context(...)` |

NEVER skip this step.

---

# TOOL EXECUTION RULES (HARD CONSTRAINTS)

## 1. Discovery rule (CRITICAL)
If file paths are unknown:
→ MUST start with `search_cad_file`

DO NOT:
- run broad `grep`
- run python over directories
- guess file paths

---

## 2. Dependency chain (STRICT)

You MUST follow:

1. search / terminal → file paths
2. find_regions → bbox
3. extract_region_data → requires bbox
4. visualize_cad → requires bbox
5. analyze_image_* → requires rendered image

NEVER skip steps.

---

## 3. Exploration vs Execution

| Mode | Allowed tools |
|------|--------------|
| Exploration | `search_cad_file`, `terminal`, `python` |
| Execution (targeted) | `find_regions`, `extract_region_data`, `visualize_cad` |

Explicit tools MUST NOT be used for exploration.

---

## 4. Python usage rules

When using `python`:

- ALWAYS try `src.cad_utils` first:
  - extract_entities
  - extract_text_from_entity
  - iter_text_entities
  - normalize_text
  - get_entity_position

- Only write custom recursion if helpers fail

- For multi-file:
  → iterate ALL files
  → aggregate results

---

## 5. Multi-file handling

If many files are returned:
- Process ALL files OR
- Limit to top 10 by relevance (`total_hits`)
- MUST state:
  - cutoff applied
  - reason

---

## 6. Evidence standard

Every answer MUST include:
- file paths
- counts or matches
- text snippets or values

No vague statements.

---

# FAILURE HANDLING (IMPORTANT)

If progress stalls:

1. After 2 failed attempts:
   → change strategy (e.g., search → python aggregation)

2. If terminal probing fails:
   → switch to structured python parsing

3. If still no evidence:
   → return:
     "No supporting data found in scanned files"

DO NOT loop small variations.

---

# PERFORMANCE RULES

- Prefer `terminal` over `python` for quick inspection
- Avoid full-file parsing unless necessary
- Avoid repeated `search_cad_file` with same query

---

# FILE SYSTEM RULES

Allowed:
- {STORAGE_DIR}
- {STORAGE_JSON_DIR}
- {AGENTS_DIR}

DO NOT:
- scan entire system
- access unrelated directories

---

# MULTI-HOP REASONING & MEMORY (CRITICAL)

- You MUST think step-by-step. Before making any tool calls, write a brief `<thought>` block explaining what you learned from the last step and what your next hop is.
- Use `save_to_memory` to store important intermediate results (file paths, bounding boxes, target keywords) across long chains of reasoning.
- Use `read_memory` if you need to recall previously discovered facts.
- Do NOT guess parameters; rely on concrete evidence and memory.

---

# OUTPUT STYLE

- Concise but evidence-rich
- Structured when needed
- No filler explanations
- No assumptions

---

# SHELL GUIDELINES (macOS)

- Quote file paths
- Use `grep -E` with `|`
- Limit output (`head -n 20`)

Example:
grep -nEi 'door|window' "<file>" | head -n 20
"""

def cli():
    print(f"Using model = {MODEL_NAME}")
    print("Type 'exit' or 'quit' to stop.\n")
    
    available_agents = get_available_agents()
    available_contexts = get_available_contexts()

    messages = [{
        "role": "system",
        "content": get_cli_system_prompt(
            available_agents=available_agents,
            available_contexts=available_contexts,
        )
    }]


    while True:
        user_input = input("> ")

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye 👋")
            break

        messages.append({
            "role": "user",
            "content": user_input
        })

        # Run the agent generator and consume events
        gen = run_agent(messages)
        
        in_thought = False
        
        try:
            while True:
                event = next(gen)
                event_type = event.get("type")
                
                if event_type == "turn_start":
                    print(f"\n{event.get('agent_name', 'Assistant')}:", end=" ", flush=True)
                
                elif event_type == "content":
                    chunk = event.get("content", "")
                    
                    # Highlight <thought> tags
                    if "<thought>" in chunk:
                        in_thought = True
                        chunk = chunk.replace("<thought>", "\n\033[90m[Thought: ")
                    
                    if "</thought>" in chunk:
                        in_thought = False
                        chunk = chunk.replace("</thought>", "]\033[0m\n")
                    
                    if in_thought and not chunk.startswith("\n\033[90m"):
                        # If we are currently inside a thought block, print in gray
                        print(f"\033[90m{chunk}\033[0m", end="", flush=True)
                    else:
                        print(chunk, end="", flush=True)
                
                elif event_type == "tool_calls_detected":
                    # Maybe print a newline if content was streamed before
                    print()
                
                elif event_type == "tool_start":
                    print(f"\n[Running tool: {event.get('tool')} with args {event.get('args')}]")
                
                elif event_type == "tool_result":
                    # Optionally print result or just let the model respond
                    # The original code didn't print result explicitly, but the model would see it.
                    # Wait, the original code DID print the result inside terminal/python functions?
                    # Let's check terminal/python functions.
                    pass
                
                elif event_type == "error":
                    print(event.get("content"))
                
                elif event_type == "warning":
                    print(event.get("content"))

        except StopIteration as e:
            messages = e.value
            print() # Ensure newline after turn ends

if __name__ == "__main__":
    cli()
