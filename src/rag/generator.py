import ollama
from typing import List, Dict, Any, Tuple
from ..config import CHAT_MODEL

class Generator:
    def generate(self, query: str, context: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]], List[str]]:
        """
        Generates a response based on context.
        Returns: (answer, citations, image_urls)
        """
        context_str = ""
        citations = []
        image_urls = []
        
        seen_images = set()
        
        for item in context:
            meta = item['metadata']
            source = f"{meta.get('source', 'Unknown')} (Page {meta.get('page_num', '?')})"
            context_str += f"\n--- Source: {source} ---\n{item['content']}\n"
            
            citations.append({
                "source": meta.get('source'),
                "page": meta.get('page_num'),
                "type": meta.get('type')
            })
            
            # Collect images
            if meta.get('type') == 'image' and meta.get('image_path'):
                # We need to convert local path to a serveable URL format if needed
                # For now, we'll return the local path or filename so UI can serve it
                # Assuming UI serves from /images/ endpoint mapping to static/images/
                import os
                img_filename = os.path.basename(meta['image_path'])
                img_url = f"/images/{img_filename}"
                
                if img_url not in seen_images:
                    image_urls.append(img_url)
                    seen_images.add(img_url)
        
        system_prompt = (
            "You are an expert Semiconductor Manufacturing Assistant. "
            "Answer the user's question STRICTLY based on the provided context below. "
            "If the context contains descriptions of images or diagrams, explain them clearly. "
            "Refuse to answer if the information is not in the context."
        )
        
        user_prompt = f"CONTEXT:\n{context_str}\n\nQUESTION: {query}"
        
        try:
            response = ollama.chat(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            answer = response['message']['content']
        except Exception as e:
            answer = f"Error generating response: {str(e)}"
            
        return answer, citations, image_urls
