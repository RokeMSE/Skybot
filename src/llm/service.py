from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
from PIL import Image
import os
import io

# --- IMPORTS FOR IMPLEMENTATIONS ---
try:
    import ollama
except ImportError:
    ollama = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

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

# --- GEMINI IMPLEMENTATION ---
class GeminiService(VLMService, ChatService):
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash", base_url: Optional[str] = None):
        if not genai:
            raise ImportError("Google GenAI SDK not installed.")
        
        # Initialize client with custom endpoint if provided
        http_options = None
        if base_url:
            http_options = {'baseUrl': base_url}
            
        self.client = genai.Client(api_key=api_key, http_options=http_options)
        self.model_name = model_name

    def analyze_image(self, image: Image.Image, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[image, prompt],
                config=types.GenerateContentConfig(
                    temperature=0.5
                )
            )
            return response.text
        except Exception as e:
            return f"Error analyzing image with Gemini: {str(e)}"

    def generate_response(self, prompt: Union[str, List[Any]], system_instruction: Optional[str] = None) -> str:
        try:
            config = types.GenerateContentConfig(
                temperature=0.5,
                system_instruction=system_instruction
            )
            
            # Gemini handles list of [text, image] natively
            contents = prompt
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )
            return response.text
        except Exception as e:
            return f"Error generating response with Gemini: {str(e)}"

# --- OLLAMA IMPLEMENTATION ---
class OllamaService(VLMService, ChatService):
    def __init__(self, model_name: str = "qwen3-vl:4b"):
        if not ollama:
            raise ImportError("Ollama library not installed.")
        self.model_name = model_name

    def analyze_image(self, image: Image.Image, prompt: str) -> str:
        try:
            # Ollama expects bytes or path for images
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()

            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [img_bytes]
                    }
                ]
            )
            return response['message']['content']
        except Exception as e:
            return f"Error analyzing image with Ollama: {str(e)}"

    def generate_response(self, prompt: Union[str, List[Any]], system_instruction: Optional[str] = None) -> str:
        try:
            messages = []
            if system_instruction:
                messages.append({'role': 'system', 'content': system_instruction})
            
            # Handle mixed content for Ollama
            user_content = ""
            images = []
            
            if isinstance(prompt, list):
                for item in prompt:
                    if isinstance(item, str):
                        user_content += item
                    elif isinstance(item, Image.Image):
                        # Convert PIL Image to bytes for Ollama
                        img_byte_arr = io.BytesIO()
                        item.save(img_byte_arr, format='PNG')
                        images.append(img_byte_arr.getvalue())
            else:
                user_content = prompt

            user_message = {'role': 'user', 'content': user_content}
            if images:
                user_message['images'] = images
                
            messages.append(user_message)

            response = ollama.chat(
                model=self.model_name,
                messages=messages
            )
            return response['message']['content']
        except Exception as e:
            return f"Error generating response with Ollama: {str(e)}"

# --- OPENAI IMPLEMENTATION ---
class OpenAIService(VLMService, ChatService):
    def __init__(self, api_key: str, model_name: str = "gpt-4o", base_url: Optional[str] = None):
        if not OpenAI:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        
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
def get_llm_service(provider: str = "ollama", **kwargs):
    if provider == "gemini":
        return GeminiService(
            api_key=kwargs.get("api_key"), 
            model_name=kwargs.get("model_name", "gemini-2.5-flash"),
            base_url=kwargs.get("base_url")
        )
    elif provider == "ollama":
        return OllamaService(model_name=kwargs.get("model_name", "qwen3-vl:4b"))
    elif provider == "openai":
        return OpenAIService(
            api_key=kwargs.get("api_key"),
            model_name=kwargs.get("model_name", "gpt-4o"),
            base_url=kwargs.get("base_url")
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
