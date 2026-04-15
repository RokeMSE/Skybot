"""
Issue Investigation Agent node.

Searches the knowledge base for evidence related to a manufacturing issue,
analyses what it finds, and appends structured findings to the scratchpad.

When a lot ID + operation are detected in the query, also fetches lot/unit
XML data from the network share for richer context.
"""
import logging
import re

from langchain_core.messages import AIMessage, HumanMessage

from ..llm import get_chat_model
from ..state import AgentState
from ..tools import retrieve_from_knowledge_base, retrieve_lot_unit_info

log = logging.getLogger(__name__)

# Pattern: 1-2 letters followed by 6-8 alphanumerics  (e.g. 4V56656R, AB1234567)
_LOT_RE = re.compile(r"\b([A-Z0-9]{2}[A-Z0-9]{5,7})\b", re.IGNORECASE)
# Pattern: 4-5 digit operation number (e.g. 5274, 62100)
_OP_RE = re.compile(r"\b(\d{4,5})\b")


def _extract_lot_op(text: str) -> tuple:
    """Try to extract a lot ID and operation from free text."""
    lot_match = _LOT_RE.search(text)
    op_match = _OP_RE.search(text)
    return (
        lot_match.group(1).upper() if lot_match else None,
        op_match.group(1) if op_match else None,
    )


def issue_agent_node(state: AgentState) -> dict:
    sub_query = state.get("sub_query") or state["user_query"]
    channel = state.get("channel")
    iteration = state.get("iteration", 1)

    # 1. Retrieve relevant documents from knowledge base
    retrieval = retrieve_from_knowledge_base(sub_query, channel=channel, n_results=5)

    # 2. Try to fetch lot/unit info if a lot+operation pair is detected
    lot_unit_ctx = ""
    lot_id, operation = _extract_lot_op(sub_query)
    if not lot_id:
        lot_id, operation = _extract_lot_op(state["user_query"])

    if lot_id and operation:
        log.info("Issue Agent — detected lot=%s op=%s, fetching XML", lot_id, operation)
        lu = retrieve_lot_unit_info(lot_id, operation)
        if not lu["error"]:
            parts = []
            if lu["lot_summary"]:
                parts.append(f"=== Lot Info ===\n{lu['lot_summary']}")
            if lu["unit_summary"]:
                parts.append(f"=== Unit Info ===\n{lu['unit_summary']}")
            lot_unit_ctx = "\n\n".join(parts)
        else:
            log.info("Lot/unit lookup: %s", lu["error"])

    # 3. Build context and analyse
    kb_context = retrieval["context"].strip()
    has_context = bool(kb_context) or bool(lot_unit_ctx)

    if not has_context:
        finding = "No relevant documents or lot data found for this issue query."
    else:
        context_block = ""
        if kb_context:
            context_block += f"RETRIEVED DOCUMENTS:\n{kb_context}\n\n"
        if lot_unit_ctx:
            context_block += f"LOT/UNIT DATA:\n{lot_unit_ctx}\n\n"

        analysis_prompt = (
            "You are an Issue Investigation Agent specialising in semiconductor manufacturing.\n\n"
            f"ORIGINAL QUESTION: {state['user_query']}\n"
            f"SEARCH QUERY USED: {sub_query}\n\n"
            f"PREVIOUS INVESTIGATION NOTES:\n{state.get('scratchpad', 'None')}\n\n"
            f"{context_block}"
            "Analyse the retrieved documents and produce a concise technical summary covering:\n"
            "1. Likely root causes or contributing factors\n"
            "2. Relevant data points (process parameters, measurements, lot IDs)\n"
            "3. Recommended next investigation steps\n"
            "Be precise and cite page numbers where possible."
        )
        model = get_chat_model(temperature=0.3)
        try:
            response = model.invoke([HumanMessage(content=analysis_prompt)])
            finding = response.content
        except Exception as e:
            finding = f"Analysis failed: {e}"

    # 4. Append to shared scratchpad
    current = state.get("scratchpad", "")
    new_scratchpad = (current + f"\n\n--- [Issue Agent — Step {iteration}] ---\n" + finding)

    return {
        "scratchpad": new_scratchpad,
        "retrieved_docs": state.get("retrieved_docs", []) + retrieval["docs"],
        "image_urls": list(
            dict.fromkeys(state.get("image_urls", []) + retrieval["images"])
        ),
        "citations": state.get("citations", []) + retrieval["citations"],
        "messages": [
            AIMessage(content=f"[Issue Agent] {finding[:300]}{'...' if len(finding) > 300 else ''}")
        ],
    }
