from typing import List, Dict, Any, Optional
from ..storage.vectordb import get_vector_db
from ..llm.service import get_llm_service
from ..config import CHAT_MODEL

class RAGEngine:
    def __init__(self):
        self.collection = get_vector_db()
        # Determine provider based on config model name
        provider = "ollama" if "qwen" in CHAT_MODEL.lower() or "llama" in CHAT_MODEL.lower() else "gemini"
        self.chat_service = get_llm_service(provider=provider, model_name=CHAT_MODEL)

    def query(self, user_query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Retrieves context and generates an answer.
        """
        # 1. Retrieve
        results = self.collection.query(
            query_texts=[user_query],
            n_results=n_results
        )
        
        # 2. Construct Context
        context_str = ""
        retrieved_sources = []
        image_urls = []
        seen_images = set()
        
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]
                source_tag = f"[Source: {meta.get('source', 'Unknown')}, Page {meta.get('page', '?')}]"
                context_str += f"\n--- {source_tag} ---\n{doc}\n"
                retrieved_sources.append(meta)
                
                # Check for images
                if meta.get('type') == 'image_cad' and meta.get('image_path'):
                    import os
                    # Convert absolute path to relative static URL
                    # Assuming we mounted 'static' at /static
                    # And images are in static/images
                    img_path = meta['image_path']
                    img_filename = os.path.basename(img_path)
                    # We need to make sure this path is actually served. 
                    # config.py says IMAGE_STORE_DIR = static/images
                    # So URL should be /static/images/{filename}
                    img_url = f"/static/images/{img_filename}"
                    
                    if img_url not in seen_images:
                        image_urls.append(img_url)
                        seen_images.add(img_url)
        
        if not context_str:
            return {"answer": "I couldn't find any relevant information in the uploaded documents to answer your question.", "citations": [], "images": []}

        # 3. Generate Answer
        system_instruction = (
            "You are an expert Semiconductor Manufacturing Assistant. "
            "Answer the user's question STRICTLY based on the provided context below. "
            "If the context contains a description of an image/diagram, explain it clearly. "
            "Cite the page numbers provided in the context."
        )
        
        final_prompt = f"\nCONTEXT DATA:\n{context_str}\n\nUSER QUESTION: {user_query}"
        
        answer = self.chat_service.generate_response(
            prompt=final_prompt,
            system_instruction=system_instruction
        )
        
        return {
            "answer": answer,
            "citations": retrieved_sources[:3], # Return top 3 unique sources
            "images": list(image_urls)[:3]      # Return top 3 images
        }
