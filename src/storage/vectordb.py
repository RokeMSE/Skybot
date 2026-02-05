import chromadb
from chromadb.config import Settings
from ..config import CHROMA_PERSIST_DIR, COLLECTION_NAME
from ..models import OllamaEmbeddingFunction

def get_vector_db():
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=OllamaEmbeddingFunction()
    )
    return collection
