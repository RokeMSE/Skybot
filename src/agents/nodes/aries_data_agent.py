"""
Production Data Agent node.

Gathers live production data from available sources and appends an
LLM-analysed summary to the scratchpad.

Data flow:
  1. Lot/Unit XML  — fetch from network share, extract tester ID
  2. Lamas (ES)    — query equipment alarms for the tester extracted from step 1
"""
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage

from ..llm import get_chat_model
from ..state import AgentState
from ..tools import retrieve_lot_unit_info, list_recent_lots

log = logging.getLogger(__name__)

# ---- Query parameter parsers ----

_LOT_RE = re.compile(r"\blot\s+([A-Z0-9]{2}[A-Z0-9]{5,7})\b", re.IGNORECASE)
_OP_RE = re.compile(r"\bop(?:eration)?\s*(\d{3,5})\b", re.IGNORECASE)
_TESTER_RE = re.compile(r"\b(HXV\d{2,4})\b", re.IGNORECASE)


def _parse_lot_op(text: str) -> tuple[Optional[str], Optional[str]]:
    lot_m = _LOT_RE.search(text)
    op_m = _OP_RE.search(text)
    return (
        lot_m.group(1).upper() if lot_m else None,
        op_m.group(1) if op_m else None,
    )


def _parse_tester(text: str) -> Optional[str]:
    m = _TESTER_RE.search(text)
    return m.group(1).upper() if m else None


def _parse_hours_back(query: str) -> float:
    """Extract a lookback window in hours. Defaults to 12."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:day|days)", query, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 24
    m = re.search(r"(\d+)\s*(?:hour|hours|hr|hrs|h)\b", query, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return 12.0


# ---- Lamas / Elasticsearch helper ----

def _query_lamas(
    tester_id: Optional[str] = None,
    hours_back: float = 12.0,
    site_name: int = 0,
    sample_count: int = 20,
) -> str:
    """Query Lamas Elasticsearch for equipment alarms.

    When *tester_id* is given, filters to that specific tester.
    When omitted, returns the most recent alarms across **all** testers.
    """
    try:
        from ...tools.elastic_alarm_tool import ElasticAlarmTool
        from ...tools.base_tool import ToolConfig
    except ImportError as e:
        return f"Lamas unavailable: {e}"

    # Extract numeric part if a tester ID is provided: HXV053 → 053
    tool_num = re.sub(r"^HXV", "", tester_id, flags=re.IGNORECASE) if tester_id else None

    now = datetime.now(timezone(timedelta(hours=7)))  # VN timezone
    gte = (now - timedelta(hours=hours_back)).isoformat()
    lte = now.isoformat()

    tool = ElasticAlarmTool(config=ToolConfig(
        name="elastic_alarm",
        description="Query Lamas alarms",
        timeout_seconds=120,
    ))

    params: dict = {
        "gte": gte,
        "lte": lte,
        "site_name": site_name,
        "sample_count": sample_count,
    }
    if tool_num:
        params["tool_name"] = tool_num

    try:
        result = tool.execute(params)
        data = result.get("data", {})

        hits = data.get("hits", {}).get("hits", []) if isinstance(data, dict) else []
        scope = f"tester {tester_id}" if tester_id else "all testers"
        if not hits:
            return f"No alarms found for {scope} in the last {hours_back:.0f}h."

        lines = [f"=== Lamas Alarms — {scope} (last {hours_back:.0f}h, {len(hits)} hits) ==="]
        for hit in hits:
            src = hit.get("_source", {})
            ts = src.get("@timestamp", "?")
            msg = src.get("message", "")[:200]
            hostname = (src.get("host", {}).get("hostname")
                        if isinstance(src.get("host"), dict)
                        else src.get("host.hostname", "?"))
            lines.append(f"  [{ts}] {hostname}: {msg}")

        return "\n".join(lines)

    except Exception as e:
        return f"Lamas query failed: {e}"


# ---- Main agent node ----

def aries_data_agent_node(state: AgentState) -> dict:
    sub_query = state.get("sub_query") or state["user_query"]
    iteration = state.get("iteration", 1)

    findings: list[str] = []
    tester_id = _parse_tester(sub_query) or _parse_tester(state["user_query"])

    # --- 1. Lot/Unit XML from network share ---
    lot_id, operation = _parse_lot_op(sub_query)
    if not lot_id:
        lot_id, operation = _parse_lot_op(state["user_query"])

    if lot_id and operation:
        log.info("Data Agent — fetching lot/unit XML: lot=%s op=%s", lot_id, operation)
        lu = retrieve_lot_unit_info(lot_id, operation)
        if lu["lot_summary"] or lu["unit_summary"]:
            parts = []
            if lu["lot_summary"]:
                parts.append(f"=== Lot Info ===\n{lu['lot_summary']}")
            if lu["unit_summary"]:
                parts.append(f"=== Unit Info ===\n{lu['unit_summary']}")
            findings.append("\n\n".join(parts))

            # Extract tester from lot XML if not already in the query
            if not tester_id and lu.get("tester_id"):
                tester_id = lu["tester_id"]
                log.info("Data Agent — extracted tester %s from lot XML", tester_id)

        if lu["error"]:
            findings.append(f"[Lot/Unit XML] {lu['error']}")
    elif lot_id:
        findings.append(f"[Lot/Unit XML] Lot {lot_id} detected but no operation number — cannot look up XML.")

    # --- 2. Lamas / Elasticsearch alarms ---
    hours_back = _parse_hours_back(sub_query)
    if tester_id:
        log.info("Data Agent — querying Lamas: tester=%s, hours_back=%.0f", tester_id, hours_back)
        lamas_result = _query_lamas(tester_id, hours_back=max(hours_back, 12.0))
        findings.append(lamas_result)

    # --- 3. Broad query (no specific IDs) — list recent lots + all alarms ---
    if not lot_id and not tester_id:
        log.info("Data Agent — no specific IDs detected, running broad scan")

        # Recent lots from network share
        recent = list_recent_lots(n=10)
        if recent["lots"]:
            lot_lines = [f"=== Recent Lots (top {len(recent['lots'])}) ==="]
            for entry in recent["lots"]:
                header = f"  {entry['lot_id']} op {entry['operation']} (modified {entry['modified']})"
                if entry.get("tester_id"):
                    header += f" tester={entry['tester_id']}"
                lot_lines.append(header)
                if entry["summary"]:
                    # Include just the first line of the summary for compactness
                    first_line = entry["summary"].split("\n")[0]
                    lot_lines.append(f"    {first_line}")
            findings.append("\n".join(lot_lines))
        elif recent["error"]:
            findings.append(f"[Recent Lots] {recent['error']}")

        # Recent alarms across all testers
        log.info("Data Agent — querying Lamas for all testers, hours_back=%.0f", hours_back)
        lamas_result = _query_lamas(
            tester_id=None,
            hours_back=max(hours_back, 12.0),
            sample_count=30,
        )
        findings.append(lamas_result)

    # --- Combine and analyse ---
    combined = "\n\n".join(findings) if findings else "No data sources returned results."
    has_real_data = findings and not all(
        "failed" in f.lower()[:40] or "unavailable" in f.lower()[:40] or "not found" in f.lower()[:40]
        for f in findings
    )

    if has_real_data:
        analysis_prompt = (
            "You are a Production Data Analysis Agent specialising in semiconductor test data.\n\n"
            f"ORIGINAL QUESTION: {state['user_query']}\n"
            f"SEARCH QUERY USED: {sub_query}\n\n"
            f"PREVIOUS INVESTIGATION NOTES:\n{state.get('scratchpad', 'None')}\n\n"
            "PRODUCTION DATA:\n"
            f"{combined}\n\n"
            "Analyse the data and produce a concise summary covering:\n"
            "1. Lot/unit test results — pass/fail per step, bin patterns\n"
            "2. Equipment alarms or anomalies from Lamas (if available)\n"
            "3. Correlations between alarms and test failures\n"
            "4. Recommendations for follow-up investigation\n"
            "Be precise — cite specific numbers, lot IDs, and timestamps from the data."
        )
        model = get_chat_model(temperature=0.3)
        try:
            response = model.invoke([HumanMessage(content=analysis_prompt)])
            finding = response.content
        except Exception as e:
            finding = f"LLM analysis failed: {e}\n\nRaw data:\n{combined}"
    else:
        finding = combined

    # --- Append to scratchpad ---
    current = state.get("scratchpad", "")
    sources = []
    if lot_id:
        sources.append(f"lot={lot_id}")
    if operation:
        sources.append(f"op={operation}")
    if tester_id:
        sources.append(f"tester={tester_id}")
    source_str = ", ".join(sources) if sources else "general query"

    new_scratchpad = (
        current
        + f"\n\n--- [Aries Data Agent — Step {iteration}] ---\n"
        + f"(Sources: {source_str})\n"
        + finding
    )

    return {
        "scratchpad": new_scratchpad,
        "retrieved_docs": state.get("retrieved_docs", []),
        "image_urls": state.get("image_urls", []),
        "citations": state.get("citations", []),
        "messages": [
            AIMessage(
                content=f"[Aries Data Agent] {finding[:300]}{'...' if len(finding) > 300 else ''}"
            )
        ],
    }
