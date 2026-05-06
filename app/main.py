import os
import sys
import multiprocessing as mp
import uvicorn
from contextlib import asynccontextmanager
from concurrent.futures import ProcessPoolExecutor
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import dxf
from app.routers import images
from app.routers import masks
from app.routers import embeddings
from app.routers import pdfs
from app.routers import agent
from app.routers import chat

def _server_worker_count() -> int:
    for key in ("UVICORN_WORKERS", "WEB_CONCURRENCY"):
        value = os.getenv(key)
        if value:
            try:
                parsed = int(value)
                if parsed >= 1:
                    return parsed
            except ValueError:
                continue
    return 1

def _executor_max_workers() -> int:
    configured = os.getenv("PROCESS_POOL_MAX_WORKERS")
    if configured:
        try:
            value = int(configured)
            if value >= 1:
                return value
        except ValueError:
            pass
    cpu_count = os.cpu_count() or 1
    base = max(1, cpu_count - 1)
    workers = _server_worker_count()
    if workers > 1:
        return max(1, base // workers)
    return base

def _executor_context():
    if sys.platform.startswith("linux"):
        return mp.get_context("spawn")
    return mp.get_context()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.executor = ProcessPoolExecutor(
        max_workers=_executor_max_workers(),
        mp_context=_executor_context(),
    )
    yield
    app.state.executor.shutdown()

app = FastAPI(title="DXF to GeoJSON Converter", lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
    expose_headers=["Content-Disposition"],
)

app.include_router(dxf.router, prefix="/api", tags=["dxf"])
app.include_router(images.router, prefix="/api", tags=["images"])
app.include_router(masks.router, prefix="/api", tags=["masks"])
app.include_router(pdfs.router, prefix="/api", tags=["pdfs"])
app.include_router(agent.router, prefix="/api", tags=["agent"])
app.include_router(chat.router, prefix="/api", tags=["chat"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the DXF to GeoJSON Converter API. Use POST /api/dxf to upload files."}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port, 
        reload=True,
        timeout_keep_alive=300,  # 5 minutes
    )
