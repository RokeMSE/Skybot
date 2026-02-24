from typing import List, Dict, Any, Optional
from ..storage.vectordb import get_vector_db
from ..llm.service import get_llm_service
from ..config import CHAT_MODEL, LLM_PROVIDER, GEMINI_API_KEY, GEMINI_ENDPOINT

class RAGEngine:
    def __init__(self):
        self.collection = get_vector_db()
        
        # Initialize chat service based on configured provider
        if LLM_PROVIDER == "gemini":
            self.chat_service = get_llm_service(
                provider="gemini",
                api_key=GEMINI_API_KEY,
                model_name=CHAT_MODEL,
                base_url=GEMINI_ENDPOINT
            )
        else:  # ollama
            self.chat_service = get_llm_service(
                provider="ollama",
                model_name=CHAT_MODEL
            )

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
        schema_pages = set()
        
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]
                
                # Collect relevant pages for hybrid image retrieval
                if meta.get('type') == 'text':
                    source = meta.get('source')
                    page = meta.get('page')
                    if source and page:
                        schema_pages.add((source, page))

                source_tag = f"[Source: {meta.get('source', 'Unknown')}, Page {meta.get('page', '?')}]"
                context_str += f"\n--- {source_tag} ---\n{doc}\n"
                retrieved_sources.append(meta)
                
                # Check for images (directly retrieved)
                if meta.get('type') == 'image_cad' and meta.get('image_path'):
                    import os
                    img_path = meta['image_path']
                    img_filename = os.path.basename(img_path)
                    img_url = f"/static/images/{img_filename}"
                    
                    if img_url not in seen_images:
                        image_urls.append(img_url)
                        seen_images.add(img_url)

        # --- Hybrid Retrieval: Fetch images from relevant pages ---
        if schema_pages:
            print(f"Hybrid Retrieval: Checking for images on {len(schema_pages)} pages...")
            try:
                for source, page in schema_pages:
                    # Query for images on this specific page
                    # Note: ChromaDB 'where' filtering is efficient here
                    img_results = self.collection.get(
                        where={
                            "$and": [
                                {"type": "image_cad"},
                                {"source": source},
                                {"page": page}
                            ]
                        }
                    )
                    
                    if img_results['metadatas']:
                        for meta in img_results['metadatas']:
                            if meta.get('image_path'):
                                import os
                                img_path = meta['image_path']
                                img_filename = os.path.basename(img_path)
                                img_url = f"/static/images/{img_filename}"
                                
                                if img_url not in seen_images:
                                    print(f"Hybrid Retrieval: Found related image {img_filename}")
                                    # Insert at the BEGINNING of list to prioritize contextually relevant images
                                    image_urls.insert(0, img_url)
                                    seen_images.add(img_url)
                                    # Add to retrieved sources for citation if not already present
                                    if meta not in retrieved_sources:
                                        retrieved_sources.append(meta)
            except Exception as e:
                print(f"Hybrid retrieval error: {e}")

        if not context_str:
            return {"answer": "I couldn't find any relevant information in the uploaded documents to answer your question.", "citations": [], "images": []}

        # 3. Generate Answer
        # Map pages to image URLs for the LLM to use in markdown
        page_to_image_map = {}
        for meta in retrieved_sources:
             if meta.get('type') == 'image_cad' and meta.get('image_path') and meta.get('page'):
                 import os
                 img_filename = os.path.basename(meta['image_path'])
                 page_to_image_map[meta['page']] = f"/static/images/{img_filename}"

        system_instruction = (
            "You are an expert Semiconductor Manufacturing Assistant. "
            "You are provided with text context (which may contain pre-generated image descriptions) and actual images. "
            "CRITICAL: Prioritize your own visual analysis of the provided images over the pre-generated text descriptions if they conflict. "
            "Answer the user's question STRICTLY based on the provided context (text and images). "
            "Cite the page numbers provided in the context.\n\n"
            "--- INLINE IMAGE RULES ---\n"
            "The context text may contain placeholders like '[[IMAGE on Page X]]'.\n"
            f"Here is the mapping of Page Numbers to available Image URLs:\n{page_to_image_map}\n\n"
            "If you encounter a placeholder '[[IMAGE on Page X]]' and you have a URL for Page X in the mapping above:\n"
            "1. YOU MUST replace the placeholder with a Markdown image link: `![Diagram from Page X](<Image URL>)`.\n"
            "2. Do NOT output the raw `[[IMAGE on Page X]]` text.\n"
            "3. Place the image on its own line.\n"
            "--------------------------"
        )
        
        # Construct multi-modal prompt
        final_prompt_parts = []
        final_prompt_parts.append(f"USER QUESTION: {user_query}\n\nCONTEXT DATA:\n")
        
        # Add text context components
        final_prompt_parts.append(context_str)
        
        # Add images directly to the prompt context if available
        # We load the top 3 images to allow the model to "see" them
        from PIL import Image
        import os
        
        loaded_images_count = 0
        for img_url in image_urls[:3]:
            # Convert URL back to path or use original path from metadata
            # We need to find the original path. Let's look up in retrieved_sources
             for meta in retrieved_sources:
                 if meta.get('type') == 'image_cad' and meta.get('image_path'):
                     if os.path.basename(meta['image_path']) == os.path.basename(img_url):
                         try:
                             if os.path.exists(meta['image_path']):
                                 pil_img = Image.open(meta['image_path'])
                                 # Use a clearer system tag that the model understands is for internal reference
                                 final_prompt_parts.append(f"\n<system_image_attachment name='{os.path.basename(img_url)}'/>\n")
                                 final_prompt_parts.append(pil_img)
                                 loaded_images_count += 1
                         except Exception as e:
                             print(f"Failed to load image for prompt: {e}")
                         break
        
        answer = self.chat_service.generate_response(
            prompt=final_prompt_parts,
            system_instruction=system_instruction
        )
        
        return {
            "answer": answer,
            "citations": retrieved_sources[:3], # Return top 3 unique sources
            "images": list(image_urls)[:3]      # Return top 3 images
        }
