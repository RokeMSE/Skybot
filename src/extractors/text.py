import os
from typing import List
from .base import BaseExtractor, ContentItem


class TextExtractor(BaseExtractor):
    """Extracts content from plain text files (.txt, .md, .log, etc.)."""

    def extract(self, file_path: str) -> List[ContentItem]:
        items = []
        filename = os.path.basename(file_path)

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        if text.strip():
            items.append(ContentItem(
                content=text,
                type="text",
                source=filename,
                page_num=1
            ))

        return items
