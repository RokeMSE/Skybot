from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
from PIL import Image
import os
import io

try:
    from openai import OpenAI, AzureOpenAI

except ImportError:
    OpenAI = None
    AzureOpenAI = None

class VLMService(ABC):
    """
    Abstract base class for Vision-Language Model services.
    """
    @abstractmethod
    def analyze_image(self, image: Image.Image, prompt: str) -> str:
        pass

class ChatService(ABC):
    """
    Abstract base class for Chat/Text Generation services.
    """
    @abstractmethod
    def generate_response(self, prompt: Union[str, List[Any]], system_instruction: Optional[str] = None) -> str:
        pass

# --- OPENAI IMPLEMENTATION ---
class OpenAIService(VLMService, ChatService):
    def __init__(self, api_key: str, model_name: str = "gpt-4o", base_url: Optional[str] = None, api_version: Optional[str] = None):
        # Use AzureOpenAI client when api_version is provided (Azure deployment)
        if api_version and AzureOpenAI:
            import ssl
            import httpx
            ssl_context = ssl.create_default_context()
            self.client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=base_url,
                api_version=api_version,
                http_client=httpx.Client(verify=ssl_context),
            )
        else:
            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            self.client = OpenAI(**client_kwargs)
        self.model_name = model_name

    def _image_to_base64_url(self, image: Image.Image) -> str:
        """Convert a PIL Image to a base64 data URL for the OpenAI vision API."""
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        import base64
        b64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{b64}"

    def analyze_image(self, image: Image.Image, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": self._image_to_base64_url(image)}
                            }
                        ]
                    }
                ],
                temperature=0.5
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error analyzing image with OpenAI: {str(e)}"

    def generate_response(self, prompt: Union[str, List[Any]], system_instruction: Optional[str] = None) -> str:
        try:
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})

            # Build user message content parts
            content_parts = []
            if isinstance(prompt, list):
                for item in prompt:
                    if isinstance(item, str):
                        content_parts.append({"type": "text", "text": item})
                    elif isinstance(item, Image.Image):
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": self._image_to_base64_url(item)}
                        })
            else:
                content_parts.append({"type": "text", "text": prompt})

            messages.append({"role": "user", "content": content_parts})

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.5
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response with OpenAI: {str(e)}"

# --- FACTORY ---
def get_llm_service(provider: str = "openai", **kwargs):
    if provider == "openai":
        return OpenAIService(
            api_key=kwargs.get("api_key"),
            model_name=kwargs.get("model_name", "gpt-4o"),
            base_url=kwargs.get("base_url"),
            api_version=kwargs.get("api_version")
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
