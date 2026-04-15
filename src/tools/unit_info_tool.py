import os
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from src.tools.base_tool import BaseTool, ToolConfig
from src.extractors import XMLExtractor


class GetUnitInfoInput(BaseModel):
    file_path: str = Field(..., description="Path to unitinfo XML file")


class GetUnitInfoTool(BaseTool):
    """
    Tool to extract UnitInfo from XML and return a compact, token-efficient
    per-unit test summary for LLM agent ingestion.
    """

    INPUT_MODEL = GetUnitInfoInput

    def _run(self, validated_input: BaseModel) -> Dict[str, Any]:
        file_path = validated_input.file_path

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        extractor = XMLExtractor()
        xml_dict = extractor.xml_to_dict(file_path)

        compact = format_unit_info(xml_dict)
        return {
            "file_name": os.path.basename(file_path),
            "compact_summary": compact,
        }


def format_unit_info(xml_dict: Dict[str, Any]) -> str:
    """Convert a parsed unitinfo XML dict into a compact per-unit summary.

    Groups results by VisualId so each unit's full test journey is visible
    in a single block — much easier for an LLM to reason about than the
    step-first layout of the raw XML.

    Target: ~150-300 tokens vs ~2,000+ for raw flatten.
    """
    ui = xml_dict.get("UnitInfo", {}).get("UnitInfo", {})
    if not ui:
        ui = xml_dict.get("UnitInfo", {})

    ui_attrs = ui.get("@attributes", {}) if isinstance(ui, dict) else {}
    lot_id = ui_attrs.get("LotID", "?")
    operation = ui_attrs.get("Operation", "?")

    # Collect per-unit data across all steps
    # {visual_id: [(step_name, summary_name, status, hbin, sbin, tests, retests, history), ...]}
    unit_map: Dict[str, List[tuple]] = {}

    steps = ui.get("ProcessStep", [])
    if isinstance(steps, dict):
        steps = [steps]

    for ps in steps:
        ps_attrs = ps.get("@attributes", {}) if isinstance(ps, dict) else {}
        step_name = ps_attrs.get("name", "?")

        summary = ps.get("Summary", {}) if isinstance(ps, dict) else {}
        s_attrs = summary.get("@attributes", {}) if isinstance(summary, dict) else {}
        s_name = s_attrs.get("Name", "")

        units = summary.get("Units", []) if isinstance(summary, dict) else []
        if isinstance(units, dict):
            units = [units]

        for u in units:
            u_attrs = u.get("@attributes", {}) if isinstance(u, dict) else {}
            vid = u_attrs.get("VisualId", "?")
            status = u_attrs.get("CurrentUnitStatus", "?")
            hbin = u_attrs.get("OrigHardBin", "0")
            sbin = u_attrs.get("OrigSoftBin", "0")
            tests = u_attrs.get("TestCount", "0")
            retests = u_attrs.get("CountRetest", "0")
            history = u_attrs.get("TestHistory", "")

            unit_map.setdefault(vid, []).append(
                (step_name, s_name, status, hbin, sbin, tests, retests, history)
            )

    # Build compact output
    lines = [f"LOT: {lot_id} | OP: {operation}", ""]

    for vid, step_results in unit_map.items():
        lines.append(f"UNIT: {vid}")
        for step_name, s_name, status, hbin, sbin, tests, retests, history in step_results:
            if status == "Skip":
                lines.append(f"  {step_name} [{s_name}] — Skip")
            else:
                parts = [f"{step_name} [{s_name}] — {status}"]
                if hbin != "0" or sbin != "0":
                    parts.append(f"HBin={hbin} SBin={sbin}")
                if tests != "0":
                    test_str = f"Tests={tests}"
                    if retests != "0":
                        test_str += f" Retests={retests}"
                    parts.append(test_str)
                if history:
                    parts.append(f"History: {history}")
                lines.append(f"  {' | '.join(parts)}")
        lines.append("")

    return "\n".join(lines)
