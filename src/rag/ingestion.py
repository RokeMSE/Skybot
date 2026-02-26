import os
import shutil
import uuid
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image

from ..extractors.base import ContentItem
from ..extractors.pdf import PDFExtractor
from ..extractors.docx import DOCXExtractor
from ..extractors.pptx import PPTXExtractor
from ..extractors.xlsx import XLSXExtractor
from ..extractors.csv_ext import CSVExtractor
from ..extractors.text import TextExtractor
from ..extractors.html_ext import HTMLExtractor
from ..storage.vectordb import get_vector_db
from ..llm.service import get_llm_service
from ..config import VLM_MODEL, LLM_PROVIDER, GEMINI_API_KEY, GEMINI_ENDPOINT, OPENAI_API_KEY, OPENAI_ENDPOINT, ENABLE_VLM_INGESTION, DOCUMENT_STORE_DIR

class IngestionPipeline:
    def __init__(self):
        self.collection = get_vector_db()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        
        # Initialize VLM Service based on configured provider
        if LLM_PROVIDER == "gemini":
            self.vlm_service = get_llm_service(
                provider="gemini", 
                api_key=GEMINI_API_KEY, 
                model_name=VLM_MODEL,
                base_url=GEMINI_ENDPOINT
            )
        elif LLM_PROVIDER == "openai":
            self.vlm_service = get_llm_service(
                provider="openai",
                api_key=OPENAI_API_KEY,
                model_name=VLM_MODEL,
                base_url=OPENAI_ENDPOINT
            )
        else:  # ollama
            self.vlm_service = get_llm_service(
                provider="ollama", 
                model_name=VLM_MODEL
            )
        
        self.extractors = {
            ".pdf": PDFExtractor(),
            ".docx": DOCXExtractor(),
            ".pptx": PPTXExtractor(),
            ".xlsx": XLSXExtractor(),
            ".csv": CSVExtractor(),
            ".txt": TextExtractor(),
            ".md": TextExtractor(),
            ".log": TextExtractor(),
            ".html": HTMLExtractor(),
            ".htm": HTMLExtractor(),
        }

    def ingest_file(self, file_path: str, channel: str = "general") -> Dict[str, Any]:
        """
        Ingests a single file: extracts, chunks, embeds, and stores.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.extractors:
            raise ValueError(f"Unsupported file type: {ext}")
        
        # Copy original file to document store for in-browser viewing
        filename = os.path.basename(file_path)
        doc_dest = os.path.join(DOCUMENT_STORE_DIR, filename)
        if not os.path.exists(doc_dest):
            shutil.copy2(file_path, doc_dest)
            print(f"Copied original file to {doc_dest} for serving.")
            
        extractor = self.extractors[ext]
        print(f"Extracting content from {filename}...")
        items = extractor.extract(file_path)
        
        ingest_id = str(uuid.uuid4())
        documents = []
        metadatas = []
        ids = []
        chunk_counter = 0
        
        for item in items:
            # --- Text Processing ---
            if item.type == "text":
                chunks = self.text_splitter.split_text(item.content)
                for i, chunk in enumerate(chunks):
                    documents.append(chunk)
                    meta = item.metadata.copy()
                    meta.update({
                        "source": item.source,
                        "page": item.page_num,
                        "type": "text",
                        "channel": channel,
                        "ingest_id": ingest_id
                    })
                    metadatas.append(meta)
                    ids.append(f"{ingest_id}_{chunk_counter}")
                    chunk_counter += 1
            
            # --- Image Processing ---
            elif item.type == "image":
                if item.image_path and os.path.exists(item.image_path):
                    try:
                        if ENABLE_VLM_INGESTION:
                            # Full VLM analysis (optional, controlled by config)
                            print(f"Analyzing image from Page {item.page_num} with VLM...")
                            pil_image = Image.open(item.image_path)
                            prompt = (
                                "You are a semiconductor process engineer. Analyze this technical image. "
                                "1. Identify the diagram type (Schematic, Cross-section, Flowchart, UI, Micrograph). "
                                "2. Extract visible text, labels, pin numbers, and component IDs. "
                                "3. Describe connections, material layers, or process steps shown. "
                                "Output concise text for search indexing."
                            )
                            description = self.vlm_service.analyze_image(pil_image, prompt)
                            rich_content = f"[[IMAGE on Page {item.page_num}]]\nDescription: {description}"
                        else:
                            # Metadata-only mode: store placeholder without VLM call
                            print(f"Storing image metadata from Page {item.page_num} (VLM disabled).")
                            rich_content = f"[[IMAGE on Page {item.page_num}]]"
                        
                        documents.append(rich_content)
                        meta = item.metadata.copy()
                        meta.update({
                            "source": item.source,
                            "page": item.page_num,
                            "type": "image_cad",
                            "image_path": item.image_path,
                            "channel": channel,
                            "ingest_id": ingest_id
                        })
                        metadatas.append(meta)
                        ids.append(f"{ingest_id}_{chunk_counter}")
                        chunk_counter += 1
                    except Exception as e:
                        print(f"Failed to process image {item.image_path}: {e}")
        
        if documents:
            print(f"Upserting {len(documents)} chunks to VectorDB...")
            self.collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
        return {"status": "success", "file": filename, "chunks": len(documents), "ingest_id": ingest_id}
