import os
import uuid
from typing import List
from bs4 import BeautifulSoup
from PIL import Image
import io
import base64
from .base import BaseExtractor, ContentItem
from ..config import IMAGE_STORE_DIR


class HTMLExtractor(BaseExtractor):
    """Extracts text and embedded images from HTML files."""

    def extract(self, file_path: str) -> List[ContentItem]:
        items = []
        filename = os.path.basename(file_path)

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # --- 1. Text Extraction ---
        text = soup.get_text(separator="\n", strip=True)
        if text:
            items.append(ContentItem(
                content=text,
                type="text",
                source=filename,
                page_num=1
            ))

        # --- 2. Embedded Base64 Image Extraction ---
        for img_tag in soup.find_all("img"):
            src = img_tag.get("src", "")
            if src.startswith("data:image"):
                try:
                    # Parse base64 image data
                    header, data = src.split(",", 1)
                    image_bytes = base64.b64decode(data)
                    pil_image = Image.open(io.BytesIO(image_bytes))

                    if pil_image.width < 100 or pil_image.height < 100:
                        continue

                    image_id = f"{uuid.uuid4()}.png"
                    save_path = os.path.join(IMAGE_STORE_DIR, image_id)
                    pil_image.save(save_path)

                    items.append(ContentItem(
                        content=f"[[Image extracted from HTML]]",
                        type="image",
                        source=filename,
                        page_num=1,
                        image_path=save_path,
                        metadata={"width": pil_image.width, "height": pil_image.height}
                    ))
                except Exception as e:
                    print(f"Failed to extract embedded image from HTML: {e}")

        return items
