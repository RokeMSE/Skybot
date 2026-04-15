import os
from dotenv import load_dotenv
load_dotenv()

"""
NOTE: This is also the template for how to set up a .env file,
just copy the names and values that the system uses (or fallback too) into a .env file, add the API keys
and other config values, and the system will pick them up automatically. 
REMEBER to use dotenv(overides=True).
"""

# --- LLM Provider Selection ---
# Options: "ollama" (local, VERY SLOW AND RESOURCE HEAVY), or "openai"/"gemini" (if allowed)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

# --- Gemini Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_ENDPOINT = os.getenv("GEMINI_ENDPOINT", None)

# --- Ollama Configuration ---
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-vl:4b")

# --- OpenAI / Azure OpenAI Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT", None)
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", None)
OPENAI_REGION = os.getenv("OPENAI_REGION", None) # Technically not needed since the endpoint should include the region but just in case :))

# --- Legacy aliases for backward compatibility ---
_MODEL_MAP = {"gemini": GEMINI_MODEL, "ollama": OLLAMA_MODEL, "openai": OPENAI_MODEL}
VLM_MODEL = _MODEL_MAP.get(LLM_PROVIDER, GEMINI_MODEL)
CHAT_MODEL = VLM_MODEL
# --- Embedding Configuration ---
# Each provider has a sensible default; override via .env if needed.
# NOTE: changing these after ingestion requires deleting chroma_db/ and re-ingesting.
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
LOCAL_EMBEDDING_MODEL  = os.getenv("LOCAL_EMBEDDING_MODEL",  "all-MiniLM-L6-v2")
# Legacy alias kept for any code that still references it
EMBEDDING_MODEL = _MODEL_MAP.get(LLM_PROVIDER, GEMINI_MODEL)
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

# --- Lamas Configuration ---
KEYID_VN_LAMAS = os.getenv("KEYID_VN_LAMAS")
KEYVAL_VN_LAMAS = os.getenv("KEYVAL_VN_LAMAS")
KEYID_CD_LAMAS = os.getenv("KEYID_CD_LAMAS")
KEYVAL_CD_LAMAS = os.getenv("KEYVAL_CD_LAMAS")
KEYID_KM_LAMAS = os.getenv("KEYID_KM_LAMAS")
KEYVAL_KM_LAMAS = os.getenv("KEYVAL_KM_LAMAS")
KEYID_PG_LAMAS = os.getenv("KEYID_PG_LAMAS")
KEYVAL_PG_LAMAS = os.getenv("KEYVAL_PG_LAMAS")

# --- Lot & Unit info data dir ---
LOT_UNIT_DIR = r"\\ssfile1\hdmx_db\lot_info\TC"

# --- Aries Oracle DB ---
ARIES_DB_ENABLED = os.getenv("ARIES_DB_ENABLED", "false").lower() == "true"
ARIES_DB_USER = os.getenv("ARIES_DB_USER", "")
ARIES_DB_PASSWORD = os.getenv("ARIES_DB_PASSWORD", "")
ARIES_DB_DSN = os.getenv("ARIES_DB_DSN", "vn.aries")

# --- Stains Detective — Cloud share for pre-existing traceback results ---
# Folder layout: TRACEBACK_CLOUD_ROOT\{VID}\  (contains OG images, process images, CSVs)
# Override via .env: TRACEBACK_CLOUD_ROOT=\\server\share\path
TRACEBACK_CLOUD_ROOT = os.getenv(
    "TRACEBACK_CLOUD_ROOT",
    r"\\VNATSHFS.intel.com\VNATAnalysis$\MAOATM\VN\Applications\TE\Image_Tracer\result",
)

