import os
import sys

# Add current directory to path so we can import src
sys.path.append(os.getcwd())

from src.rag import IngestionPipeline, RAGEngine
from src.config import VLM_MODEL, CHAT_MODEL

def verify():
    print("--- STARTING VERIFICATION ---")
    print(f"VLM Model: {VLM_MODEL}")
    print(f"Chat Model: {CHAT_MODEL}")
    
    # 1. Test Ingestion
    print("\n[1] Testing Ingestion Pipeline...")
    try:
        pipeline = IngestionPipeline()
        test_file = "test.pdf" 
        if not os.path.exists(test_file):
            print(f"ERROR: {test_file} not found.")
            return
            
        result = pipeline.ingest_file(test_file)
        print("Ingestion Result:", result)
    except Exception as e:
        print(f"Ingestion FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Test Retrieval
    print("\n[2] Testing RAG Engine...")
    try:
        engine = RAGEngine()
        query = "What is the content of the diagram?"
        print(f"Querying: '{query}'")
        response = engine.query(query)
        print("RAG Response:", response['answer'])
        print("Citations:", response['citations'])
    except Exception as e:
        print(f"RAG FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
