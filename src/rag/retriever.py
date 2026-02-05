from typing import List, Dict, Any
from ..storage import get_vector_db

class Retriever:
    def __init__(self):
        self.collection = get_vector_db()
        
    def retrieve(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        retrieved_items = []
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]
                retrieved_items.append({
                    "content": doc,
                    "metadata": meta,
                    "id": results['ids'][0][i]
                })
        return retrieved_items
