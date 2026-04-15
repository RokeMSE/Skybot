"""
LangChain model factory for the agent layer.

Separate from src/llm/service.py (which is kept for VLM ingestion).
This factory returns BaseChatModel instances that LangGraph nodes use.
"""
from langchain_openai import AzureChatOpenAI, ChatOpenAI
import ssl # This is for ssl verification (or you could just go ask for the certs :P)
import httpx
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from ..config import (
    LLM_PROVIDER,
    GEMINI_API_KEY, GEMINI_MODEL, GEMINI_ENDPOINT,
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_ENDPOINT, OPENAI_API_VERSION,
    OLLAMA_MODEL,
)

def get_chat_model(temperature: float = 0.5):
    """
    Returns a LangChain BaseChatModel for the configured provider.

    Note: structured_output (used by the orchestrator) requires a model
    that supports function/tool calling. Gemini 2.x and GPT-4o/Azure do.
    Ollama support varies by model — qwen3-vl may not work reliably.
    """
    if LLM_PROVIDER == "gemini":
        kwargs = {
            "model": GEMINI_MODEL,
            "google_api_key": GEMINI_API_KEY,
            "temperature": temperature,
        }
        # Custom base URL (e.g. Intel internal Gemini proxy)
        if GEMINI_ENDPOINT:
            kwargs["client_options"] = {"api_endpoint": GEMINI_ENDPOINT}
        return ChatGoogleGenerativeAI(**kwargs)

    elif LLM_PROVIDER == "openai":
        if OPENAI_API_VERSION:
            ssl_context = ssl.create_default_context()
            # Azure OpenAI deployment
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

    else:  # ollama
        return ChatOllama(model=OLLAMA_MODEL, temperature=temperature)
