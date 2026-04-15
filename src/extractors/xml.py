import os
import json
from typing import List, Dict, Any
import xml.etree.ElementTree as ET

from .base import BaseExtractor, ContentItem


class XMLExtractor(BaseExtractor):
    """
    Extracts content from XML files and can convert them into JSON-friendly structure.

    Each XML file can be converted into a single ContentItem containing the whole
    flattened XML text, or you can use the helper method to export the parsed XML
    as JSON.
    """

    def extract(self, file_path: str) -> List[ContentItem]:
        filename = os.path.basename(file_path)
        tree = ET.parse(file_path)
        root = tree.getroot()
        content = self._flatten_xml(root)

        return [
            ContentItem(
                content=content,
                type="text",
                source=filename,
                page_num=1,
                metadata={
                    "root_tag": root.tag,
                    "file_type": "xml"
                }
            )
        ]

    def _flatten_xml(self, elem: ET.Element, level: int = 0) -> str:
        """
        Recursively flatten XML into a readable text block.
        Captures:
        - tag
        - attributes
        - text
        - children
        - tail text
        """
        indent = "  " * level
        lines = []
        # Opening line with attributes
        if elem.attrib:
            attrs = " | ".join([f"{k}: {v}" for k, v in elem.attrib.items()])
            lines.append(f"{indent}{elem.tag} | {attrs}")
        else:
            lines.append(f"{indent}{elem.tag}")
        # Element text
        text = (elem.text or "").strip()
        if text:
            lines.append(f"{indent}  text: {text}")
        # Children
        for child in list(elem):
            lines.append(self._flatten_xml(child, level + 1))
        # Tail text
        tail = (elem.tail or "").strip()
        if tail:
            lines.append(f"{indent}  tail: {tail}")
        return "\n".join(lines)

    def xml_to_dict(self, file_path: str) -> Dict[str, Any]:
        """
        Convert XML file into nested dictionary.
        """
        tree = ET.parse(file_path)
        root = tree.getroot()
        return {root.tag: self._element_to_dict(root)}

    def _element_to_dict(self, elem: ET.Element) -> Dict[str, Any]:
        """
        Recursively convert XML element to dictionary.
        Handles repeated tags as lists.
        """
        node: Dict[str, Any] = {}
        # Attributes
        if elem.attrib:
            node["@attributes"] = dict(elem.attrib)
        # Text
        text = (elem.text or "").strip()
        if text:
            node["#text"] = text
        # Children
        children = list(elem)
        if children:
            child_map: Dict[str, Any] = {}
            for child in children:
                child_data = self._element_to_dict(child)
                child_tag = child.tag
                child_value = child_data[child_tag]

                if child_tag in child_map:
                    if not isinstance(child_map[child_tag], list):
                        child_map[child_tag] = [child_map[child_tag]]
                    child_map[child_tag].append(child_value)
                else:
                    child_map[child_tag] = child_value

            node.update(child_map)
        return {elem.tag: node}

    def save_json(self, file_path: str, json_path: str):
        """
        Parse XML and save as JSON.
        """
        data = self.xml_to_dict(file_path)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)