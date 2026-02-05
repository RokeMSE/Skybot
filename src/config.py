import os

# Ollama Configuration
VLM_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-vl:latest")           # Vision-language model
EMBEDDING_MODEL = "nomic-embed-text" # Local embeddings
CHAT_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-vl:latest")          # Chat model

# ChromaDB Configuration
CHROMA_PERSIST_DIR = os.path.join(os.getcwd(), "chroma_db")
COLLECTION_NAME = "semicon_knowledge_base"

# Storage Configuration
IMAGE_STORE_DIR = os.path.join(os.getcwd(), "static", "images")
os.makedirs(IMAGE_STORE_DIR, exist_ok=True)
