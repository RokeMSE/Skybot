import os
from dotenv import load_dotenv
load_dotenv()

# --- LLM Provider Selection ---
# Options: "gemini" (cloud), "ollama" (local), or "openai" (cloud/Azure)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

# --- Gemini Configuration (for cloud-based inference) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_ENDPOINT = os.getenv("GEMINI_ENDPOINT", None)

# --- Ollama Configuration (for local inference) ---
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-vl:4b")

# --- OpenAI Configuration (standard or Azure-compatible) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT", None)

# --- Legacy aliases for backward compatibility ---
_MODEL_MAP = {"gemini": GEMINI_MODEL, "ollama": OLLAMA_MODEL, "openai": OPENAI_MODEL}
VLM_MODEL = _MODEL_MAP.get(LLM_PROVIDER, GEMINI_MODEL)
CHAT_MODEL = VLM_MODEL
EMBEDDING_MODEL = "nomic-embed-text"  # Local embeddings via ChromaDB
# --- VLM Ingestion Toggle ---
ENABLE_VLM_INGESTION = os.getenv("ENABLE_VLM_INGESTION", "true").lower() == "true"
# --- ChromaDB Configuration ---
CHROMA_PERSIST_DIR = os.path.join(os.getcwd(), "chroma_db")
COLLECTION_NAME = "semicon_knowledge_base"

# --- Storage Configuration ---
IMAGE_STORE_DIR = os.path.join(os.getcwd(), "static", "images")
DOCUMENT_STORE_DIR = os.path.join(os.getcwd(), "static", "documents")
os.makedirs(IMAGE_STORE_DIR, exist_ok=True)
os.makedirs(DOCUMENT_STORE_DIR, exist_ok=True)