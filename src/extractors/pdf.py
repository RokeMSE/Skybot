import fitz  # PyMuPDF
import os
import uuid
from typing import List
from PIL import Image
import io
from .base import BaseExtractor, ContentItem
from ..config import IMAGE_STORE_DIR

class PDFExtractor(BaseExtractor):
    def extract(self, file_path: str) -> List[ContentItem]:
        items = []
        doc = fitz.open(file_path)
        filename = os.path.basename(file_path)
        
        for page_index, page in enumerate(doc):
            page_num = page_index + 1
            
            # --- 1. Text Extraction ---
            text = page.get_text()
            if text.strip():
                # Naive chunking for now - ideally use a text splitter later
                # For this step, we just capture the whole page text as one item
                # or split by paragraphs. Let's keep it simple for now and rely on
                # downstream splitters if needed, or split by reasonable size here.
                items.append(ContentItem(
                    content=text,
                    type="text",
                    source=filename,
                    page_num=page_num
                ))
            
            # --- 2. Image Extraction ---
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                try:
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    
                    # Filter tiny images (icons, lines)
                    if pil_image.width < 100 or pil_image.height < 100:
                        continue
                        
                    # Save image to dist
                    image_id = f"{uuid.uuid4()}.png"
                    save_path = os.path.join(IMAGE_STORE_DIR, image_id)
                    pil_image.save(save_path)
                    
                    # Create ContentItem for the image
                    items.append(ContentItem(
                        content=f"[[Image extracted from Page {page_num}]]", # Placeholder content
                        type="image",
                        source=filename,
                        page_num=page_num,
                        image_path=save_path,
                        metadata={"width": pil_image.width, "height": pil_image.height}
                    ))
                    
                except Exception as e:
                    print(f"Failed to extract image on page {page_num}: {e}")
                    
        return items
