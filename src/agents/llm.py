"""
LangChain model factory for the agent layer.

Separate from src/llm/service.py (which is kept for VLM ingestion).
This factory returns BaseChatModel instances that LangGraph nodes use.
"""
from langchain_openai import AzureChatOpenAI, ChatOpenAI
import ssl
import httpx

from ..config import (
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_ENDPOINT, OPENAI_API_VERSION,
)

def get_chat_model(temperature: float = 0.5):
    """
    Returns a LangChain BaseChatModel for the configured provider.

    Note: structured_output (used by the orchestrator) requires a model
    that supports function/tool calling. GPT-4o/Azure do.
    """
    if OPENAI_API_VERSION:
        ssl_context = ssl.create_default_context()
        return AzureChatOpenAI(
            azure_endpoint=OPENAI_ENDPOINT,
            azure_deployment=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            api_version=OPENAI_API_VERSION,
            temperature=temperature,
            http_client=httpx.Client(verify=ssl_context),
        )
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL,
        base_url=OPENAI_ENDPOINT or None,
        temperature=temperature,
    )
