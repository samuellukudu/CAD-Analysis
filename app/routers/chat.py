import asyncio
import json
import os
import re
import threading
from typing import Any, AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.chat_system import (
    SYSTEM_PROMPT,
    _stream_chat_completion_events,
    _tool_result_to_content,
    execute_tool,
    extract_file_references,
    get_available_dxf_files,
    resolve_file_references,
    tools,
)
from src.planning_agent import run_planning_agent_stream

router = APIRouter()


class ChatMessage(BaseModel):
    role: str = Field(..., min_length=1)
    content: str = Field(default="")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)
    planning_mode: Optional[bool] = None


def _planning_mode_default() -> bool:
    return os.getenv("CHAT_PLANNING_MODE", "true").lower() in ("true", "1", "yes")


def _make_event(event_type: str, content: str = "", payload: Optional[dict] = None) -> dict[str, Any]:
    return {"type": event_type, "content": content, "payload": payload or {}}


def _summarize_result_text(result: str, max_chars: int = 800) -> str:
    text = str(result or "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[truncated]"


def _format_sse(event: dict[str, Any]) -> str:
    # Use ensure_ascii=True to prevent surrogate characters from breaking UTF-8 encoding
    payload = json.dumps(event, ensure_ascii=True)
    return f"data: {payload}\n\n"


def _storage_dir() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(project_root, "storage")


def _format_files_list() -> str:
    files = get_available_dxf_files()
    if not files:
        return "No DXF files found in storage."
    lines = [f"Found {len(files)} files:"]
    lines.extend(f" - {f['filename']}" for f in files)
    return "\n".join(lines)


def _format_folders_list() -> str:
    storage_dir = _storage_dir()
    if not os.path.exists(storage_dir):
        return f"Storage directory not found: {storage_dir}"
    folders = [
        d
        for d in os.listdir(storage_dir)
        if os.path.isdir(os.path.join(storage_dir, d)) and not d.startswith(".")
    ]
    folders.sort()
    if not folders:
        return f"No folders found in {storage_dir}."
    lines = [f"Found {len(folders)} folders in storage:"]
    lines.extend(f" - {name}" for name in folders)
    return "\n".join(lines)


def _merge_file_references(base: dict[str, dict], extra: dict[str, dict]) -> dict[str, dict]:
    merged = dict(base or {})
    for key, value in (extra or {}).items():
        if key not in merged or merged[key] is None:
            merged[key] = value
            continue
        if value is None:
            continue
        merged[key] = {**merged[key], **{k: v for k, v in value.items() if v is not None}}
    return merged


def _extract_history_file_references(history: list[ChatMessage]) -> dict[str, dict]:
    refs: dict[str, dict] = {}
    for msg in history:
        content = (msg.content or "").strip()
        if not content:
            continue
        refs = _merge_file_references(refs, extract_file_references(content))
    return refs


def _find_active_filename(history: list[ChatMessage]) -> str | None:
    for msg in reversed(history):
        content = (msg.content or "").strip()
        if not content:
            continue
        match = re.match(r"active file:\s*(.+)", content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _build_file_reference(file_entry: dict[str, Any]) -> dict[str, dict]:
    file_id = file_entry.get("id")
    display_filename = file_entry.get("source_filename") or file_entry.get("filename")
    if not file_id or not display_filename:
        return {}
    storage_dir = _storage_dir()
    storage_dxf = os.path.join(storage_dir, "dxf")
    storage_geojson = os.path.join(storage_dir, "geojson")
    storage_images = os.path.join(storage_dir, "images")
    entry = {
        "filename": display_filename,
        "dxf_path": None,
        "image_path": None,
        "geojson_path": None,
        "transform_path": None,
    }
    dxf_path = os.path.join(storage_dxf, f"{file_id}.dxf")
    geojson_path = os.path.join(storage_geojson, f"{file_id}.geojson")
    image_path = os.path.join(storage_images, f"{file_id}.png")
    transform_path = os.path.join(storage_images, f"{file_id}_transform.json")
    if os.path.exists(dxf_path):
        entry["dxf_path"] = dxf_path
    if os.path.exists(geojson_path):
        entry["geojson_path"] = geojson_path
    if os.path.exists(image_path):
        entry["image_path"] = image_path
    if os.path.exists(transform_path):
        entry["transform_path"] = transform_path
    return {display_filename: entry}


def _extract_active_file_references(history: list[ChatMessage]) -> dict[str, dict]:
    active_filename = _find_active_filename(history)
    if not active_filename:
        return {}
    files = get_available_dxf_files()
    if not files:
        return {}
    normalized = active_filename.strip().lower()
    for file_entry in files:
        candidates = [
            file_entry.get("filename"),
            file_entry.get("source_filename"),
            file_entry.get("id"),
            f"{file_entry.get('id')}.dxf" if file_entry.get("id") else None,
        ]
        if any(candidate and candidate.strip().lower() == normalized for candidate in candidates):
            return _build_file_reference(file_entry)
    return {}


async def _stream_sync_generator_with_result(
    gen,
    result_holder: dict[str, Any],
) -> AsyncGenerator[dict[str, Any], None]:
    queue: asyncio.Queue[tuple[Optional[BaseException], Optional[dict[str, Any]]]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def worker():
        try:
            while True:
                try:
                    item = next(gen)
                except StopIteration as stop:
                    result_holder["value"] = stop.value
                    loop.call_soon_threadsafe(queue.put_nowait, (None, None))
                    break
                loop.call_soon_threadsafe(queue.put_nowait, (None, item))
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, (exc, None))

    threading.Thread(target=worker, daemon=True).start()

    while True:
        error, item = await queue.get()
        if error:
            raise error
        if item is None:
            break
        yield item


async def _run_direct_mode(
    messages: list[dict[str, Any]],
    tools_spec: list[dict[str, Any]],
) -> AsyncGenerator[dict[str, Any], None]:
    task_id = 0
    while True:
        result_holder: dict[str, Any] = {}
        stream = _stream_chat_completion_events(messages, tools_spec)
        async for event in _stream_sync_generator_with_result(stream, result_holder):
            yield event
        msg_dict, tool_calls = result_holder.get("value", ({}, []))
        messages.append(msg_dict)

        if not tool_calls:
            break

        for tool_call in tool_calls:
            name = tool_call["function"]["name"]
            raw_args = tool_call["function"].get("arguments", "")
            try:
                args = json.loads(raw_args or "{}")
            except Exception as exc:
                error_event = _make_event(
                    "error",
                    f"Error parsing tool arguments for '{name}': {exc}. Raw arguments: {raw_args!r}",
                    {},
                )
                yield error_event
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": _tool_result_to_content(error_event["content"]),
                    }
                )
                continue

            task_id += 1
            yield _make_event("task_start", "", {"task_id": task_id, "tool": name, "args": args})
            try:
                result = execute_tool(name, args)

                # If this is the defect analysis tool, try to parse and yield defect events
                if name == "run_design_defect_analysis":
                    try:
                        data = json.loads(result)
                        if "defects" in data and isinstance(data["defects"], list) and data["defects"]:
                            yield _make_event("defect_found", "", {"defects": data["defects"]})
                    except Exception:
                        pass

                summary = _summarize_result_text(result)
                yield _make_event(
                    "task_complete",
                    "",
                    {"task_id": task_id, "result": result, "summary": summary},
                )
            except Exception as exc:
                error_text = f"Error running tool '{name}': {exc}"
                yield _make_event("error", error_text, {})
                result = error_text

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": _tool_result_to_content(result),
                }
            )


async def _chat_event_stream(body: ChatRequest) -> AsyncGenerator[str, None]:
    message_text = body.message
    if message_text.strip() == "/files":
        yield _format_sse(_make_event("token", _format_files_list(), {}))
        return
    if message_text.strip() == "/folders":
        yield _format_sse(_make_event("token", _format_folders_list(), {}))
        return
    history_file_references = _extract_history_file_references(body.history)
    current_file_references = extract_file_references(message_text)
    active_file_references = _extract_active_file_references(body.history)
    file_references = _merge_file_references(history_file_references, current_file_references)
    file_references = _merge_file_references(file_references, active_file_references)
    resolved_message = resolve_file_references(message_text)

    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in body.history:
        if msg.role == "system":
            continue
        messages.append({"role": msg.role, "content": msg.content})

    planning_mode = body.planning_mode if body.planning_mode is not None else _planning_mode_default()

    context_msgs = body.history[-6:] if len(body.history) > 6 else body.history
    conversation_context = "\n".join(
        f"{m.role}: {(m.content or '')[:500]}"
        for m in context_msgs
        if (m.content or "").strip()
    )

    if planning_mode:
        planning_stream = run_planning_agent_stream(
            user_query=resolved_message,
            tools=tools,
            tool_dispatcher=execute_tool,
            file_references=file_references,
            conversation_context=conversation_context,
        )
        planning_result: dict[str, Any] = {}
        async for event in _stream_sync_generator_with_result(planning_stream, planning_result):
            yield _format_sse(event)
        result_value = planning_result.get("value") or {}
        if result_value.get("handled"):
            return
        reason = result_value.get("reason")
        if reason:
            yield _format_sse(_make_event("status", reason, {}))

    messages.append({"role": "user", "content": resolved_message})
    yield _format_sse(_make_event("status", "Direct mode", {}))
    async for event in _run_direct_mode(messages, tools):
        yield _format_sse(event)


@router.post("/chat/stream")
async def chat_stream(body: ChatRequest):
    try:
        return StreamingResponse(_chat_event_stream(body), media_type="text/event-stream")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
