from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

@dataclass
class ContentItem:
    """
    Represents a chunk of content extracted from a document.
    """
    content: str
    type: str  # "text" or "image"
    source: str
    page_num: int
    image_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseExtractor(ABC):
    """
    Abstract base class for document extractors.
    """
    @abstractmethod
    def extract(self, file_path: str) -> List[ContentItem]:
        pass
