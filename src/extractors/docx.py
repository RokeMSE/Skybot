import docx
import os
import uuid
from typing import List
from PIL import Image
import io
from .base import BaseExtractor, ContentItem
from ..config import IMAGE_STORE_DIR

class DOCXExtractor(BaseExtractor):
    def extract(self, file_path: str) -> List[ContentItem]:
        items = []
        doc = docx.Document(file_path)
        filename = os.path.basename(file_path)
        
        # --- 1. Text Extraction ---
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text.strip())
        
        # Consolidate text (simplification: treat whole doc text as one or few chunks for now)
        # Ideally we'd chunk by headers.
        if full_text:
            items.append(ContentItem(
                content="\n".join(full_text),
                type="text",
                source=filename,
                page_num=1 # DOCX doesn't have fixed pages like PDF
            ))
            
        # --- 2. Image Extraction ---
        # python-docx doesn't make image extraction easy.
        # We iterate over the part relationships to find images.
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                try:
                    image_part = rel.target_part
                    image_blob = image_part.blob
                    pil_image = Image.open(io.BytesIO(image_blob))
                    
                    if pil_image.width < 100 or pil_image.height < 100:
                        continue
                        
                    image_id = f"{uuid.uuid4()}.png"
                    save_path = os.path.join(IMAGE_STORE_DIR, image_id)
                    pil_image.save(save_path)
                    
                    items.append(ContentItem(
                        content="[[Image extracted from DOCX]]",
                        type="image",
                        source=filename,
                        page_num=1,
                        image_path=save_path,
                        metadata={"width": pil_image.width, "height": pil_image.height}
                    ))
                except Exception as e:
                    print(f"Failed to extract image from DOCX: {e}")
                    
        return items
