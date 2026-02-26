from typing import List, Dict, Any, Optional
from ..storage.vectordb import get_vector_db
from ..llm.service import get_llm_service
from ..config import CHAT_MODEL, LLM_PROVIDER, GEMINI_API_KEY, GEMINI_ENDPOINT, OPENAI_API_KEY, OPENAI_ENDPOINT

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
        elif LLM_PROVIDER == "openai":
            self.chat_service = get_llm_service(
                provider="openai",
                api_key=OPENAI_API_KEY,
                model_name=CHAT_MODEL,
                base_url=OPENAI_ENDPOINT
            )
        else:  # ollama
            self.chat_service = get_llm_service(
                provider="ollama",
                model_name=CHAT_MODEL
            )

    def get_channels(self) -> List[str]:
        """Returns a list of distinct channel values from the collection."""
        try:
            all_meta = self.collection.get(include=["metadatas"])
            channels = set()
            if all_meta and all_meta.get("metadatas"):
                for meta in all_meta["metadatas"]:
                    ch = meta.get("channel")
                    if ch:
                        channels.add(ch)
            return sorted(channels)
        except Exception as e:
            print(f"Error fetching channels: {e}")
            return []

    def query(self, user_query: str, n_results: int = 5, channel: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieves context and generates an answer.
        Optionally filters by channel.
        """
        # 1. Retrieve — with optional channel filter
        query_kwargs = {
            "query_texts": [user_query],
            "n_results": n_results
        }
        if channel:
            query_kwargs["where"] = {"channel": channel}
        
        results = self.collection.query(**query_kwargs)
        
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
                    where_filter = {
                        "$and": [
                            {"type": "image_cad"},
                            {"source": source},
                            {"page": page}
                        ]
                    }
                    img_results = self.collection.get(where=where_filter)
                    
                    if img_results['metadatas']:
                        for meta in img_results['metadatas']:
                            if meta.get('image_path'):
                                import os
                                img_path = meta['image_path']
                                img_filename = os.path.basename(img_path)
                                img_url = f"/static/images/{img_filename}"
                                
                                if img_url not in seen_images:
                                    print(f"Hybrid Retrieval: Found related image {img_filename}")
                                    image_urls.insert(0, img_url)
                                    seen_images.add(img_url)
                                    if meta not in retrieved_sources:
                                        retrieved_sources.append(meta)
            except Exception as e:
                print(f"Hybrid retrieval error: {e}")

        if not context_str:
            return {"answer": "I couldn't find any relevant information in the uploaded documents to answer your question.", "citations": [], "images": []}

        # 3. Build source-to-document-URL mapping for hyperlinks
        source_doc_links = {}
        for meta in retrieved_sources:
            source = meta.get('source')
            page = meta.get('page')
            if source and page:
                doc_url = f"/static/documents/{source}#page={page}"
                if source not in source_doc_links:
                    source_doc_links[source] = {}
                source_doc_links[source][page] = doc_url

        # 4. Build page-to-image-URL mapping for inline embeds
        page_to_image_map = {}
        for meta in retrieved_sources:
            if meta.get('type') == 'image_cad' and meta.get('image_path') and meta.get('page'):
                import os
                img_filename = os.path.basename(meta['image_path'])
                page_to_image_map[f"{meta.get('source')}_p{meta['page']}"] = f"/static/images/{img_filename}"

        # 5. Generate Answer
        system_instruction = (
            "You are an expert Semiconductor Manufacturing Assistant. "
            "You are provided with text context (which may contain pre-generated image descriptions) and actual images. "
            "CRITICAL: Prioritize your own visual analysis of the provided images over the pre-generated text descriptions if they conflict. "
            "Answer the user's question based on the provided context (text and images). "
            "If the context does not contain relevant information, say so clearly. "
            "Cite the page numbers provided in the context.\n\n"
            "--- IMAGE DISPLAY RULES ---\n"
            "When an extracted image (JPEG/PNG) is available for a page, EMBED it inline using markdown:\n"
            "  ![Description of diagram](/static/images/image_filename.png)\n\n"
            f"Available image URLs by source and page:\n{page_to_image_map}\n\n"
            "If the context references a chart, graph, or diagram that does NOT have an extracted image URL above,\n"
            "then hyperlink to the source document page instead:\n"
            "  📄 [FileName — Page X](/static/documents/FileName#page=X)\n\n"
            f"Available document links:\n{source_doc_links}\n\n"
            "You MAY freely describe or discuss the content of images and diagrams.\n"
            "You MUST NOT generate, draw, or recreate charts, graphs, or diagrams using markdown, ASCII art, or code blocks.\n"
            "--------------------------"
        )
        
        # Construct multi-modal prompt
        final_prompt_parts = []
        final_prompt_parts.append(f"USER QUESTION: {user_query}\n\nCONTEXT DATA:\n")
        
        # Add text context components
        final_prompt_parts.append(context_str)
        
        # Add images directly to the prompt context if available
        from PIL import Image
        import os
        
        loaded_images_count = 0
        for img_url in image_urls[:3]:
             for meta in retrieved_sources:
                 if meta.get('type') == 'image_cad' and meta.get('image_path'):
                     if os.path.basename(meta['image_path']) == os.path.basename(img_url):
                         try:
                             if os.path.exists(meta['image_path']):
                                 pil_img = Image.open(meta['image_path'])
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
            "citations": retrieved_sources[:3],
            "images": list(image_urls)[:3]
        }
