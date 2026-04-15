import os
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from src.tools.base_tool import BaseTool, ToolConfig
from src.extractors import XMLExtractor


class GetLotInfoInput(BaseModel):
    file_path: str = Field(..., description="Path to lotinfo XML file")


class GetLotInfoTool(BaseTool):
    """
    Tool to extract LotInfo from XML and return a compact, token-efficient
    summary suitable for LLM agent ingestion.
    """

    INPUT_MODEL = GetLotInfoInput

    def _run(self, validated_input: BaseModel) -> Dict[str, Any]:
        file_path = validated_input.file_path

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        extractor = XMLExtractor()
        xml_dict = extractor.xml_to_dict(file_path)

        compact = format_lot_info(xml_dict)
        return {
            "file_name": os.path.basename(file_path),
            "compact_summary": compact,
        }


def format_lot_info(xml_dict: Dict[str, Any]) -> str:
    """Convert a parsed lotinfo XML dict into a compact text summary.

    Target: ~200 tokens vs ~3,500 for raw flatten — only fields an agent
    needs for lot investigation, yield analysis, and tester diagnostics.
    """
    lot = xml_dict.get("LotInfo", {}).get("LotInfo", {})
    if not lot:
        # Try alternate nesting from xml_to_dict
        lot = xml_dict.get("LotInfo", {})

    def _txt(key: str) -> str:
        """Extract #text from a nested dict value, or return raw string."""
        val = lot.get(key, {})
        if isinstance(val, dict):
            return val.get("#text", "")
        return str(val) if val else ""

    lot_id = _txt("LotID")
    operation = _txt("Operation")
    product = _txt("Product")
    device = _txt("Device")
    rev = _txt("Rev")
    step = _txt("Step")
    facility = _txt("FacilityId")
    sys_id = _txt("SysId")
    operator = _txt("OperatorId")
    program = _txt("Program")
    flow = _txt("FlowName")
    units = _txt("TotalWorkStreamUnits")
    finished = _txt("LotFinished")
    intro_time = _txt("LotIntroTime")[:19]  # trim timezone
    save_time = _txt("LastSaveTime")[:19]
    package = _txt("Package")
    eng_id = _txt("EngID")

    lines = [
        f"LOT: {lot_id} | OP: {operation} | PRODUCT: {product} | DEVICE: {device}-{rev}-{step}",
        f"FACILITY: {facility} | TESTER: {sys_id} | OPERATOR: {operator}",
        f"PROGRAM: {program} | FLOW: {flow} | PKG: {package} | ENG: {eng_id}",
        f"UNITS: {units} | FINISHED: {finished}",
        f"INTRO: {intro_time} | SAVED: {save_time}",
        "",
        "STEPS:",
    ]

    # Process steps
    steps = lot.get("ProcessStep", [])
    if isinstance(steps, dict):
        steps = [steps]
    for ps in steps:
        attrs = ps.get("@attributes", {}) if isinstance(ps, dict) else {}
        name = attrs.get("name", "?")
        order = attrs.get("order", "?")

        summary = ps.get("Summary", {}) if isinstance(ps, dict) else {}
        s_attrs = summary.get("@attributes", {}) if isinstance(summary, dict) else {}
        s_name = s_attrs.get("Name", "")
        ended = s_attrs.get("Ended", "")
        in_progress = s_attrs.get("InProgress", "")

        # Unit count from TrayChunks > Chunk
        unit_count = "?"
        tray_chunks = summary.get("TrayChunks", {}) if isinstance(summary, dict) else {}
        chunk = tray_chunks.get("Chunk", {}) if isinstance(tray_chunks, dict) else {}
        if isinstance(chunk, list):
            unit_count = sum(int(c.get("@attributes", {}).get("UnitCount", 0)) for c in chunk)
        elif isinstance(chunk, dict):
            unit_count = chunk.get("@attributes", {}).get("UnitCount", "?")

        status = "Ended" if ended == "true" else ("InProgress" if in_progress == "true" else "Pending")
        lines.append(f"  {order}. {name} [{s_name}] — {status}, {unit_count} units")

    return "\n".join(lines)
