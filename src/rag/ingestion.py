import os
import uuid
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image

from ..extractors.base import ContentItem
from ..extractors.pdf import PDFExtractor
from ..extractors.docx import DOCXExtractor
from ..extractors.pptx import PPTXExtractor
from ..storage.vectordb import get_vector_db
from ..llm.service import get_llm_service
from ..config import VLM_MODEL, LLM_PROVIDER, GEMINI_API_KEY

class IngestionPipeline:
    def __init__(self):
        self.collection = get_vector_db()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        
        # Initialize VLM Service based on configured provider
        if LLM_PROVIDER == "gemini":
            self.vlm_service = get_llm_service(
                provider="gemini", 
                api_key=GEMINI_API_KEY, 
                model_name=VLM_MODEL
            )
        else:  # ollama
            self.vlm_service = get_llm_service(
                provider="ollama", 
                model_name=VLM_MODEL
            )
        
        self.extractors = {
            ".pdf": PDFExtractor(),
            ".docx": DOCXExtractor(),
            ".pptx": PPTXExtractor()
        }

    def ingest_file(self, file_path: str) -> Dict[str, Any]:
        """
        Ingests a single file: extracts, chunks, embeds, and stores.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.extractors:
            raise ValueError(f"Unsupported file type: {ext}")
            
        extractor = self.extractors[ext]
        print(f"Extracting content from {os.path.basename(file_path)}...")
        items = extractor.extract(file_path)
        
        ingest_id = str(uuid.uuid4())
        documents = []
        metadatas = []
        ids = []
        
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
                        "ingest_id": ingest_id
                    })
                    metadatas.append(meta)
                    ids.append(f"{ingest_id}_{item.page_num}_txt_{i}")
            
            # --- Image Processing ---
            elif item.type == "image":
                if item.image_path and os.path.exists(item.image_path):
                    print(f"Analyzing image from Page {item.page_num}...")
                    try:
                        pil_image = Image.open(item.image_path)
                        # Analyze with VLM
                        prompt = (
                            "You are a semiconductor process engineer. Analyze this technical image. "
                            "1. Identify the diagram type (Schematic, Cross-section, Flowchart, UI, Micrograph). "
                            "2. Extract visible text, labels, pin numbers, and component IDs. "
                            "3. Describe connections, material layers, or process steps shown. "
                            "Output concise text for search indexing."
                        )
                        description = self.vlm_service.analyze_image(pil_image, prompt)
                        
                        rich_content = f"[[IMAGE on Page {item.page_num}]]\nDescription: {description}"
                        
                        documents.append(rich_content)
                        meta = item.metadata.copy()
                        meta.update({
                            "source": item.source,
                            "page": item.page_num,
                            "type": "image_cad",
                            "image_path": item.image_path,
                            "ingest_id": ingest_id
                        })
                        metadatas.append(meta)
                        ids.append(f"{ingest_id}_{item.page_num}_img_{uuid.uuid4().hex[:6]}")
                    except Exception as e:
                        print(f"Failed to analyze image {item.image_path}: {e}")
        
        if documents:
            print(f"Upserting {len(documents)} chunks to VectorDB...")
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
        return {"status": "success", "file": os.path.basename(file_path), "chunks": len(documents), "ingest_id": ingest_id}
