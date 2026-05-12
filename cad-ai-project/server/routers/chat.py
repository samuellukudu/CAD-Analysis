import asyncio
import json
import threading
import os
import uuid
import time
from typing import Any, AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Import from src.chat
# Ensure that the src directory is in the python path if needed, 
# but usually running from root works.
from src.chat import get_available_agents, get_available_contexts, get_cli_system_prompt, run_agent

router = APIRouter()
STREAM_BRIDGE_HEARTBEAT_SECONDS = float(os.getenv("STREAM_BRIDGE_HEARTBEAT_SECONDS", "10"))

class ChatMessage(BaseModel):
    role: str = Field(..., min_length=1)
    content: str = Field(default="")

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)
    # planning_mode is removed as we don't have planning agent yet

def _make_event(event_type: str, content: str = "", payload: Optional[dict] = None) -> dict[str, Any]:
    return {"type": event_type, "content": content, "payload": payload or {}}

def _format_sse(event: dict[str, Any]) -> str:
    # Use ensure_ascii=True to prevent surrogate characters from breaking UTF-8 encoding
    payload = json.dumps(event, ensure_ascii=True)
    return f"data: {payload}\n\n"

def _summarize_result_text(result: str, max_chars: int = 800) -> str:
    text = str(result or "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[truncated]"

async def _stream_sync_generator(gen) -> AsyncGenerator[dict[str, Any], None]:
    """
    Consumes a synchronous generator in a separate thread and yields items asynchronously.
    """
    queue: asyncio.Queue[tuple[Optional[BaseException], Optional[dict[str, Any]]]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def worker():
        try:
            while True:
                try:
                    item = next(gen)
                except StopIteration:
                    loop.call_soon_threadsafe(queue.put_nowait, (None, None))
                    break
                loop.call_soon_threadsafe(queue.put_nowait, (None, item))
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, (exc, None))

    threading.Thread(target=worker, daemon=True).start()

    while True:
        try:
            error, item = await asyncio.wait_for(queue.get(), timeout=STREAM_BRIDGE_HEARTBEAT_SECONDS)
        except asyncio.TimeoutError:
            yield _make_event("status", "Still processing...")
            continue

        if error:
            raise error
        if item is None:
            break
        yield item

async def _chat_event_stream(body: ChatRequest) -> AsyncGenerator[str, None]:
    available_agents = get_available_agents()
    available_contexts = get_available_contexts()
    messages = [{
        "role": "system",
        "content": get_cli_system_prompt(
            available_agents=available_agents,
            available_contexts=available_contexts,
        ),
    }]
    
    # Add history
    for msg in body.history:
        if msg.role == "system":
            continue
        messages.append({"role": msg.role, "content": msg.content})
    
    # Add current message
    messages.append({"role": "user", "content": body.message})

    # Yield status
    yield _format_sse(_make_event("status", "Starting chat...", {}))

    # Run agent
    stream_trace = str(uuid.uuid4())
    # #region agent log
    try:
        with open(
            "/Users/samuellukudu/QilaiCo/End2End/cad-ai-project/.cursor/debug-815a61.log",
            "a",
            encoding="utf-8",
        ) as _df:
            _df.write(
                json.dumps(
                    {
                        "sessionId": "815a61",
                        "requestId": stream_trace,
                        "timestamp": int(time.time() * 1000),
                        "location": "server/routers/chat.py:_chat_event_stream",
                        "message": "sse_stream_opened",
                        "data": {"user_message_len": len(body.message or "")},
                        "hypothesisId": "H-trace",
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
    except Exception:
        pass
    # #endregion
    gen = run_agent(messages, _stream_trace_id=stream_trace)

    task_counter = 0
    active_tasks = {} # tool_call_id -> task_id

    try:
        async for event in _stream_sync_generator(gen):
            event_type = event.get("type")
            
            if event_type == "content":
                chunk = event.get("content", "")
                
                # Transform <thought> tags into Markdown blocks so the UI renderer displays them
                if "<thought>" in chunk:
                    chunk = chunk.replace("<thought>", "\n```text\n🤔 Thought:\n")
                if "</thought>" in chunk:
                    chunk = chunk.replace("</thought>", "\n```\n")
                    
                # Map 'content' to 'token' for frontend compatibility
                yield _format_sse(_make_event("token", chunk))
                
            elif event_type == "tool_start":
                task_counter += 1
                tool_call_id = event.get("tool_call_id")
                
                if not tool_call_id:
                    tool_call_id = f"unknown_{task_counter}"
                
                active_tasks[tool_call_id] = task_counter
                
                tool_name = event.get("tool")
                args = event.get("args")
                yield _format_sse(_make_event("task_start", "", {"task_id": task_counter, "tool": tool_name, "args": args}))
                
            elif event_type == "tool_result":
                result = event.get("result", "")
                summary = _summarize_result_text(result)
                
                tool_call_id = event.get("tool_call_id")
                
                if tool_call_id and tool_call_id in active_tasks:
                    task_id = active_tasks[tool_call_id]
                    yield _format_sse(_make_event("task_complete", "", {"task_id": task_id, "result": result, "summary": summary}))
                    del active_tasks[tool_call_id]
                else:
                    yield _format_sse(_make_event("task_complete", "", {"task_id": task_counter, "result": result, "summary": summary}))
                
            elif event_type == "error":
                yield _format_sse(_make_event("error", event.get("content", "")))
                
            elif event_type == "warning":
                yield _format_sse(_make_event("status", f"Warning: {event.get('content', '')}"))
            elif event_type == "status":
                yield _format_sse(_make_event("status", event.get("content", "")))
            else:
                # #region agent log
                try:
                    with open(
                        "/Users/samuellukudu/QilaiCo/End2End/cad-ai-project/.cursor/debug-815a61.log",
                        "a",
                        encoding="utf-8",
                    ) as _df:
                        _df.write(
                            json.dumps(
                                {
                                    "sessionId": "815a61",
                                    "requestId": stream_trace,
                                    "timestamp": int(time.time() * 1000),
                                    "location": "server/routers/chat.py:_chat_event_stream",
                                    "message": "unhandled run_agent event (not forwarded to SSE)",
                                    "data": {"event_type": event_type},
                                    "hypothesisId": "H-E",
                                },
                                ensure_ascii=True,
                            )
                            + "\n"
                        )
                except Exception:
                    pass
                # #endregion
    except Exception as exc:
        yield _format_sse(_make_event("error", f"Streaming bridge failed: {exc}"))

@router.post("/chat/stream")
async def chat_stream(body: ChatRequest):
    try:
        return StreamingResponse(_chat_event_stream(body), media_type="text/event-stream")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
