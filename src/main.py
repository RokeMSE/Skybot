import os
import shutil
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional

from src.rag import IngestionPipeline, RAGEngine
from src.config import IMAGE_STORE_DIR, DOCUMENT_STORE_DIR

# Ensure directories exist
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGE_STORE_DIR, exist_ok=True)
os.makedirs(DOCUMENT_STORE_DIR, exist_ok=True)

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="Skybot Backend", version="2.0.0")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

# Initialize Engines
try:
    ingestion_pipeline = IngestionPipeline()
    rag_engine = RAGEngine()
except Exception as e:
    print(f"Error initializing engines: {e}")
    ingestion_pipeline = None
    rag_engine = None

class ChatRequest(BaseModel):
    query: str
    channel: Optional[str] = None

class IngestResponse(BaseModel):
    status: str
    file: str
    chunks: int
    ingest_id: str

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...), channel: str = Form("general")):
    if not ingestion_pipeline:
        raise HTTPException(status_code=500, detail="Ingestion pipeline not initialized.")
        
    try:
        # Save uploaded file to disk
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Run ingestion with channel
        result = ingestion_pipeline.ingest_file(file_path, channel=channel)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/chat")
async def chat(request: ChatRequest):
    if not rag_engine:
        raise HTTPException(status_code=500, detail="RAG engine not initialized.")
        
    try:
        response = rag_engine.query(request.query, channel=request.channel)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.get("/channels")
async def list_channels():
    """Returns a list of available channels from ingested documents."""
    if not rag_engine:
        raise HTTPException(status_code=500, detail="RAG engine not initialized.")
    
    try:
        channels = rag_engine.get_channels()
        return {"channels": channels}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch channels: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
