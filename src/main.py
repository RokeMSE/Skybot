import logging
import os
import shutil
import uvicorn

logging.basicConfig(
    level=logging.INFO,  # change to DEBUG for full prompt logging
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from src.rag import IngestionPipeline, RAGEngine
from src.agents import agent_graph
from src.config import IMAGE_STORE_DIR, DOCUMENT_STORE_DIR

# Ensure directories exist
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGE_STORE_DIR, exist_ok=True)
os.makedirs(DOCUMENT_STORE_DIR, exist_ok=True)

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="Skybot Backend", version="2.0.0")

# Allow Blazor WASM dev server to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


class AgenticChatRequest(BaseModel):
    query: str
    channel: Optional[str] = None
    max_iterations: int = 3
    traceback_uploads_dir: Optional[str] = None   # stains detective: explicit image folder
    traceback_output_dir: Optional[str] = None    # stains detective: where to write panels

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

@app.post("/agentic-chat")
async def agentic_chat(request: AgenticChatRequest):
    """
    Multi-step agentic RAG endpoint backed by LangGraph.

    The orchestrator decides whether to call the Issue Investigation Agent,
    the SOP/Document Agent, or go straight to the Reporting Agent — looping
    until max_iterations is reached or the orchestrator is satisfied.

    Returns the same shape as /chat for frontend compatibility:
      { answer, citations, images }
    """
    try:
        initial_state = {
            "messages": [],
            "user_query": request.query,
            "channel": request.channel,
            "scratchpad": "",
            "sub_query": "",
            "retrieved_docs": [],
            "image_urls": [],
            "citations": [],
            "next_action": "",
            "final_answer": "",
            "iteration": 0,
            "max_iterations": request.max_iterations,
            "traceback_uploads_dir": request.traceback_uploads_dir,
            "traceback_output_dir": request.traceback_output_dir,
        }
        result = agent_graph.invoke(initial_state)
        return {
            "answer": result.get("final_answer", "No answer generated."),
            "citations": result.get("citations", [])[:3],
            "images": result.get("image_urls", [])[:3],
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Agentic chat failed: {str(e)}")


class AriesDataRequest(BaseModel):
    days_back: float = 0.5
    tester_filter: str = "%HXV%"


@app.post("/aries-data")
async def aries_data(request: AriesDataRequest):
    """Query live unit-level test data from the Aries Oracle DB."""
    try:
        from src.services.aries_db import AriesDBService

        svc = AriesDBService()
        df = svc.query_unit_level_data(
            days_back=request.days_back,
            tester_filter=request.tester_filter,
        )
        summary = svc.summarise(df)
        return {
            "rows": len(df),
            "summary": summary,
            "data": df.head(500).to_dict(orient="records"),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Aries query failed: {str(e)}")


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