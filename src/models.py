"""
Embedding functions for ChromaDB.

Provider is selected from LLM_PROVIDER in config:
  openai  → Azure OpenAI / OpenAI  (text-embedding-3-small)

Changing the active provider requires deleting chroma_db/ and re-ingesting all documents,
because the embedding dimensions and semantics differ between models.
"""
from chromadb import Documents, EmbeddingFunction, Embeddings

from .config import (
    LOCAL_EMBEDDING_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_API_VERSION,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_ENDPOINT,
)


class OpenAIEmbeddingFunction(EmbeddingFunction):
    """
    Embeddings via OpenAI or Azure OpenAI.
    Batches all texts in a single API call for efficiency.
    """

    def __init__(self):
        if OPENAI_API_VERSION:
            from openai import AzureOpenAI
            import ssl
            import httpx
            ssl_context = ssl.create_default_context()
            self._client = AzureOpenAI(
                api_key=OPENAI_API_KEY,
                azure_endpoint=OPENAI_ENDPOINT,
                api_version=OPENAI_API_VERSION,
                http_client=httpx.Client(verify=ssl_context),
            )
        else:
            from openai import OpenAI
            kwargs = {"api_key": OPENAI_API_KEY}
            if OPENAI_ENDPOINT:
                kwargs["base_url"] = OPENAI_ENDPOINT
            self._client = OpenAI(**kwargs)
        self._model = OPENAI_EMBEDDING_MODEL

    def __call__(self, input: Documents) -> Embeddings:
        try:
            response = self._client.embeddings.create(input=list(input), model=self._model)
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"[OpenAIEmbedding] Error: {e}")
            # Return zero vectors so ChromaDB doesn't crash mid-batch;
            # the embedding dimension for text-embedding-3-small is 1536.
            return [[0.0] * 1536] * len(input)


class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """
    Local embeddings via Sentence Transformers.
    No external API required.
    Model is downloaded once and cached in ~/.cache/huggingface.
    """

    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )

    def __call__(self, input: Documents) -> Embeddings:
        return self._model.encode(list(input), show_progress_bar=False).tolist()


def get_embedding_function() -> EmbeddingFunction:
    """
    Returns the embedding function for the currently configured LLM_PROVIDER.
    """
    if LLM_PROVIDER == "openai":
        return OpenAIEmbeddingFunction()
    else:
        return SentenceTransformerEmbeddingFunction()
