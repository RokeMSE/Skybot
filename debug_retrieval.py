
import os
import sys
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Add src to path
sys.path.append(os.getcwd())

try:
    from src.storage.vectordb import get_vector_db
except ImportError:
    # Fallback if src is not in path correctly or if running from root
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    from src.storage.vectordb import get_vector_db

def inspect_db():
    print("Initializing DB...")
    collection = get_vector_db()
    
    # Get total count
    count = collection.count()
    print(f"Total documents in DB: {count}")
    
    if count == 0:
        print("DB is empty.")
        return

    # Peek at some items
    print("\n--- Peeking at first 5 items ---")
    results = collection.peek(limit=5)
    for i, meta in enumerate(results['metadatas']):
        print(f"Item {i}: Type={meta.get('type')}, Source={meta.get('source')}, Page={meta.get('page')}")
        if meta.get('type') == 'image_cad':
            print(f"   Image Path: {meta.get('image_path')}")
            print(f"   Content snippet: {results['documents'][i][:100]}...")

    from src.rag.retrieval import RAGEngine
    rag = RAGEngine()
    
    queries = ["process flow", "diagram", "structure"]
    for q in queries:
        print(f"\n--- Query: '{q}' ---")
        response = rag.query(q, n_results=3)
        print(f"Answer snippet: {response['answer'][:50]}...")
        print(f"Retrieved Images Count: {len(response['images'])}")
        for img in response['images']:
            print(f" - Image: {img}")
        print(f"Retrieved Citations Count: {len(response['citations'])}")

if __name__ == "__main__":
    inspect_db()
