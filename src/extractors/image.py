import os
import uuid
from typing import List
from PIL import Image
from .base import BaseExtractor, ContentItem
from ..config import IMAGE_STORE_DIR
class ImageExtractor(BaseExtractor):
    """Extracts content from standalone image files (PNG, JPEG, etc.)."""
    def extract(self, file_path: str) -> List[ContentItem]:
        filename = os.path.basename(file_path)
        pil_image = Image.open(file_path)

        # Save a copy to the image store for serving
        image_id = f"{uuid.uuid4()}.png"
        save_path = os.path.join(IMAGE_STORE_DIR, image_id)
        pil_image.save(save_path)

        return [ContentItem(
            content=f"[[Standalone image: {filename}]]",
            type="image",
            source=filename,
            page_num=1,
            image_path=save_path,
            metadata={"width": pil_image.width, "height": pil_image.height}
        )]
