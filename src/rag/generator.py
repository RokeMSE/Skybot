import os
from typing import List, Dict, Any, Tuple
from ..config import CHAT_MODEL, LLM_PROVIDER, GEMINI_API_KEY
from ..llm.service import get_llm_service

class Generator:
    def __init__(self):
        """Initialize the generator with the configured LLM service."""
        # Initialize chat service based on configured provider
        if LLM_PROVIDER == "gemini":
            self.llm_service = get_llm_service(
                provider="gemini",
                api_key=GEMINI_API_KEY,
                model_name=CHAT_MODEL
            )
        else:  # ollama
            self.llm_service = get_llm_service(
                provider="ollama",
                model_name=CHAT_MODEL
            )
    
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
                # Need to convert local path to a serveable URL format if needed
                # TMP FIX: return the local path or filename so UI can serve it
                # Assuming UI serves from /images/ endpoint mapping to static/images/
                img_filename = os.path.basename(meta['image_path'])
                img_url = f"/images/{img_filename}"
                
                if img_url not in seen_images:
                    image_urls.append(img_url)
                    seen_images.add(img_url)
        
        system_prompt = (
            "As an Engineering Assistant, your role is to provide a comprehensive and detailed response to a user's query based on the specific Intel context provided."
            "Answer the user's question STRICTLY based on the provided context below. "
            "Your answer should be thorough and articulate all the necessary steps, policies, and pertinent details without summarizing them, ensuring that the information is presented as it is."
            "When addressing a problem, include a clear description of the issue, detail all the attempted solutions as per the Intel context, and identify any aspects that remain unresolved. Your response should be based on the Intel context alone, without drawing on prior knowledge or external sources."
            "If the context contains descriptions of images or diagrams, explain them clearly. "
            "Refuse to answer if the information is not in the context."
            "Otherwise, you can have conversation with the user casually."
        )
        
        user_prompt = f"CONTEXT:\n{context_str}\n\nQUESTION: {query}"
        
        try:
            answer = self.llm_service.generate_response(user_prompt, system_instruction=system_prompt)
        except Exception as e:
            answer = f"Error generating response: {str(e)}"
            
        return answer, citations, image_urls
