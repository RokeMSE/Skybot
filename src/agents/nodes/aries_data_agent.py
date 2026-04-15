"""
Production Data Agent node.

Gathers live production data from all available sources and appends an
LLM-analysed summary to the scratchpad.

Data sources (in order):
  1. Lot/Unit XML   — per-lot test results from the network share
  2. Lamas (ES)     — equipment alarms and logs from Elasticsearch
  3. Aries Oracle DB — live unit-level test data (only when ARIES_DB_ENABLED=true)
"""
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage

from ..llm import get_chat_model
from ..state import AgentState
from ..tools import retrieve_lot_unit_info
from ...config import ARIES_DB_ENABLED

log = logging.getLogger(__name__)

# ---- Parsers for extracting query parameters from natural language ----

_LOT_RE = re.compile(r"\b([A-Z0-9]{2}[A-Z0-9]{5,7})\b", re.IGNORECASE)
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


def _parse_days_back(query: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:day|days)", query, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+)\s*(?:hour|hours|hr|hrs|h)\b", query, re.IGNORECASE)
    if m:
        return float(m.group(1)) / 24.0
    return 0.5


# ---- Lamas / Elasticsearch helper ----

def _query_lamas(tester_id: str, hours_back: float = 12.0, site_name: int = 0) -> str:
    """Query Lamas Elasticsearch for equipment alarms around the given tester.

    Returns a text summary or an error message.
    """
    try:
        from ...tools.elastic_alarm_tool import ElasticAlarmTool
        from ...tools.base_tool import ToolConfig
    except ImportError as e:
        return f"Lamas unavailable: {e}"

    # Extract numeric part from tester ID (e.g. HXV053 → 053)
    tool_num = re.sub(r"^HXV", "", tester_id, flags=re.IGNORECASE)

    now = datetime.now(timezone(timedelta(hours=7)))  # VN timezone
    gte = (now - timedelta(hours=hours_back)).isoformat()
    lte = now.isoformat()

    tool = ElasticAlarmTool(config=ToolConfig(
        name="elastic_alarm",
        description="Query Lamas alarms",
        timeout_seconds=120,
    ))

    try:
        result = tool.execute({
            "tool_name": tool_num,
            "gte": gte,
            "lte": lte,
            "site_name": site_name,
            "sample_count": 20,
        })
        data = result.get("data", {})

        # Format hits into a concise summary
        hits = data.get("hits", {}).get("hits", []) if isinstance(data, dict) else []
        if not hits:
            return f"No alarms found for tester {tester_id} in the last {hours_back:.0f}h."

        lines = [f"=== Lamas Alarms — {tester_id} (last {hours_back:.0f}h, {len(hits)} hits) ==="]
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
        if lu["error"]:
            findings.append(f"[Lot/Unit XML] {lu['error']}")
    elif lot_id:
        findings.append(f"[Lot/Unit XML] Lot {lot_id} detected but no operation number found — cannot look up XML.")

    # --- 2. Lamas / Elasticsearch alarms ---
    tester_id = _parse_tester(sub_query) or _parse_tester(state["user_query"])
    if tester_id:
        hours_back = _parse_days_back(sub_query) * 24
        log.info("Data Agent — querying Lamas for tester=%s, hours_back=%.0f", tester_id, hours_back)
        lamas_result = _query_lamas(tester_id, hours_back=max(hours_back, 12.0))
        findings.append(lamas_result)

    # --- 3. Aries Oracle DB (only if enabled) ---
    if ARIES_DB_ENABLED:
        days_back = _parse_days_back(sub_query)
        tester_filter = f"%{tester_id}%" if tester_id else "%HXV%"
        try:
            from ...services.aries_db import AriesDBService
            svc = AriesDBService()
            df = svc.query_unit_level_data(days_back=days_back, tester_filter=tester_filter)
            findings.append(svc.summarise(df))
        except Exception as e:
            log.error("Aries DB query failed: %s", e)
            findings.append(f"[Aries DB] Query failed: {e}")

    # --- Combine and analyse ---
    combined = "\n\n".join(findings) if findings else "No data sources returned results."

    if findings and not all("failed" in f.lower()[:40] or "unavailable" in f.lower()[:40] for f in findings):
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
            "3. Yield metrics and trends (if Aries data available)\n"
            "4. Correlations between alarms and test failures\n"
            "5. Recommendations for follow-up investigation\n"
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
