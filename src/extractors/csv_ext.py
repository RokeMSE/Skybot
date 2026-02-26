import csv
import os
from typing import List
from .base import BaseExtractor, ContentItem


class CSVExtractor(BaseExtractor):
    """
    Extracts text content from CSV files.
    
    Each row becomes its own ContentItem with column headers baked in,
    so downstream text splitters and embeddings retain full context.
    Example output for a row: "product: BTLS8161, mu: 2.54, sigma: 1.47, ..."
    """

    def extract(self, file_path: str) -> List[ContentItem]:
        items = []
        filename = os.path.basename(file_path)

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if len(rows) < 2:
            # Only header or empty — store as-is
            if rows:
                items.append(ContentItem(
                    content=" | ".join(rows[0]),
                    type="text",
                    source=filename,
                    page_num=1
                ))
            return items

        headers = rows[0]

        for row_index, row in enumerate(rows[1:], start=1):
            # Build a natural-language representation: "column: value, column: value, ..."
            pairs = []
            for col, val in zip(headers, row):
                val = val.strip()
                if val:
                    pairs.append(f"{col}: {val}")

            if pairs:
                text = ", ".join(pairs)
                items.append(ContentItem(
                    content=text,
                    type="text",
                    source=filename,
                    page_num=row_index,
                    metadata={"row_index": row_index}
                ))

        return items
