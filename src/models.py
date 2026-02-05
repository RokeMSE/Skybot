import ollama
from chromadb import Documents, EmbeddingFunction, Embeddings
from .config import VLM_MODEL, EMBEDDING_MODEL

class OllamaEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function for ChromaDB using Ollama's local models.
    """
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            try:
                response = ollama.embed(model=EMBEDDING_MODEL, input=text)
                # Ollama.embed returns {'embeddings': [[...]]} for a list or single string
                # We need to handle the response structure carefully
                if 'embeddings' in response and response['embeddings']:
                    embeddings.append(response['embeddings'][0])
                else:
                    # Fallback or error handling
                    print(f"Error embedding text: {text[:50]}...")
                    embeddings.append([0.0] * 768) # Placeholder dimensions, adjust as needed
            except Exception as e:
                print(f"Embedding failed: {e}")
                embeddings.append([0.0] * 768)
        return embeddings

class OllamaVLM:
    """
    Wrapper for Qwen3-VL interactions via Ollama.
    """
    def analyze_image(self, image_path: str) -> str:
        """
        Sends an image to Qwen3-VL for technical analysis.
        """
        prompt = (
            "You are a semiconductor process engineer. Analyze this technical image. "
            "1. Identify the diagram type (Schematic, Cross-section, Flowchart, UI, Micrograph). "
            "2. Extract visible text, labels, pin numbers, and component IDs. "
            "3. Describe connections, material layers, or process steps shown. "
            "Output concise text for search indexing."
        )
        
        try:
            response = ollama.chat(
                model=VLM_MODEL,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_path]
                }]
            )
            return response['message']['content']
        except Exception as e:
            return f"Error analyzing image with {VLM_MODEL}: {str(e)}"
