from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.routers import chat, search, files

app = FastAPI(title="CAD AI Agent Server")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router)
app.include_router(search.router)
app.include_router(files.router)

@app.get("/")
async def root():
    return {"status": "ok", "message": "CAD AI Agent Server is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)
