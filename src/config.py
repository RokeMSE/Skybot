import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- LLM Provider Selection ---
# Options: "gemini" (cloud-based, works on CPU) or "ollama" (local inference, requires GPU)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

# --- Gemini Configuration (for cloud-based inference) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# --- Ollama Configuration (for local inference) ---
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-vl:4b")

# --- Legacy aliases for backward compatibility ---
VLM_MODEL = GEMINI_MODEL if LLM_PROVIDER == "gemini" else OLLAMA_MODEL
CHAT_MODEL = VLM_MODEL
EMBEDDING_MODEL = "nomic-embed-text"  # Local embeddings via ChromaDB

# --- ChromaDB Configuration ---
CHROMA_PERSIST_DIR = os.path.join(os.getcwd(), "chroma_db")
COLLECTION_NAME = "semicon_knowledge_base"

# --- Storage Configuration ---
IMAGE_STORE_DIR = os.path.join(os.getcwd(), "static", "images")
os.makedirs(IMAGE_STORE_DIR, exist_ok=True)
