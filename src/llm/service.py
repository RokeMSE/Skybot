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
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        if not genai:
            raise ImportError("Google GenAI SDK not installed.")
        self.client = genai.Client(api_key=api_key)
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
    def __init__(self, model_name: str = "qwen3-vl"):
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

# --- FACTORY ---
def get_llm_service(provider: str = "ollama", **kwargs):
    if provider == "gemini":
        return GeminiService(api_key=kwargs.get("api_key"), model_name=kwargs.get("model_name", "gemini-2.5-flash"))
    elif provider == "ollama":
        return OllamaService(model_name=kwargs.get("model_name", "qwen3-vl"))
    else:
        raise ValueError(f"Unknown provider: {provider}")
