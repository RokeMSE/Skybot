"""
Orchestrator node — the ReAct "Reason" step.

Uses structured output to decide which specialist agent to invoke next,
or whether enough information has been gathered to produce a final report.

Deterministic shortcuts (bypass the LLM call for unambiguous routing):
  - VID pattern detected  → stains_detective
  - Lot ID + operation    → aries_data  (live DB query comes before KB search)
"""
import logging
import re
from pathlib import Path
from typing import Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from ..llm import get_chat_model
from ..state import AgentState
from .stains_detective_agent import extract_vid

_COMPLIANCE_PATH = Path(__file__).resolve().parent.parent / "compliance.md"
_COMPLIANCE_RULES = _COMPLIANCE_PATH.read_text(encoding="utf-8")

logger = logging.getLogger(__name__)

# Lot ID pattern: 1-2 letters/digits + 5-7 alphanumerics (e.g. 4V56656R)
_LOT_RE = re.compile(r"\blot\s+([A-Z0-9]{2}[A-Z0-9]{5,7})\b", re.IGNORECASE)
# Operation pattern: "op" or "operation" followed by 3-5 digits
_OP_RE = re.compile(r"\bop(?:eration)?\s+(\d{3,5})\b", re.IGNORECASE)


def _extract_lot_op(text: str) -> tuple[Optional[str], Optional[str]]:
    """Try to extract a lot ID and operation from text like 'lot 4V56656R op 5274'."""
    lot_m = _LOT_RE.search(text)
    op_m = _OP_RE.search(text)
    return (
        lot_m.group(1).upper() if lot_m else None,
        op_m.group(1) if op_m else None,
    )


class OrchestratorDecision(BaseModel):
    reasoning: str
    next_action: Literal["issue_agent", "sop_agent", "stains_detective", "aries_data", "general", "reporting"]
    sub_query: str  # focused search query for the chosen agent


def orchestrator_node(state: AgentState) -> dict:
    """
    Reads the current investigation state and decides which specialist to call.
    """
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)
    scratchpad = state.get("scratchpad", "").strip() or "No findings yet."
    user_query = state["user_query"]

    # ------------------------------------------------------------------
    # Deterministic shortcut 1: VID → stains_detective
    # ------------------------------------------------------------------
    vid = extract_vid(user_query)
    if vid and "[Stains Detective" not in scratchpad:
        logger.info(
            "Orchestrator [iter %d/%d] → stains_detective (VID %s detected, direct route)",
            iteration + 1, max_iter, vid,
        )
        return {
            "next_action": "stains_detective",
            "sub_query": user_query,
            "iteration": iteration + 1,
            "messages": [
                AIMessage(content=f"[Orchestrator → stains_detective] VID {vid} detected — routing to defect traceback.")
            ],
        }

    # ------------------------------------------------------------------
    # Deterministic shortcut 2: lot ID + operation → aries_data first
    # ------------------------------------------------------------------
    lot_id, operation = _extract_lot_op(user_query)
    if lot_id and "[Aries Data Agent" not in scratchpad:
        op_str = f" op {operation}" if operation else ""
        logger.info(
            "Orchestrator [iter %d/%d] → aries_data (lot %s%s detected, direct route)",
            iteration + 1, max_iter, lot_id, op_str,
        )
        return {
            "next_action": "aries_data",
            "sub_query": user_query,
            "iteration": iteration + 1,
            "messages": [
                AIMessage(content=f"[Orchestrator → aries_data] Lot {lot_id}{op_str} detected — fetching live production data first.")
            ],
        }

    # ------------------------------------------------------------------
    # LLM-based routing
    # ------------------------------------------------------------------
    prompt = (
        "You are the orchestrator of a semiconductor manufacturing AI system.\n\n"
        "=== COMPLIANCE GUARDRAILS ===\n"
        f"{_COMPLIANCE_RULES}\n"
        "=== END GUARDRAILS ===\n\n"
        f"USER QUESTION:\n{user_query}\n\n"
        f"INVESTIGATION NOTES SO FAR (scratchpad):\n{scratchpad}\n\n"
        f"CURRENT ITERATION: {iteration} of {max_iter} allowed\n\n"
        "Choose the next action:\n"
        "  • aries_data        — use FIRST when the question references a specific lot ID, "
        "operation number, tester name, or asks for live production data (yield, bins, "
        "test results, lot status, tester performance). Also use when the user asks "
        "to list current/recent lots, show recent alarms, or get an overview of "
        "production activity WITHOUT specifying IDs — the agent can scan recent lots "
        "and query all tester alarms. This agent queries the Aries manufacturing "
        "database and fetches lot/unit XML data from the network share. "
        "Include lot IDs, operations, time ranges, and tester IDs in sub_query "
        "when available; leave sub_query as the original question for broad scans.\n"
        "  • issue_agent       — use to search the **knowledge base documents** for known "
        "issues, failure patterns, equipment alarms, RCA reports, and historical analysis. "
        "Best used AFTER aries_data has provided lot-specific context so the search is "
        "more targeted. Do NOT use for fetching live data.\n"
        "  • sop_agent         — use when the question asks for a procedure, SOP, BKM, "
        "checklist, work instruction, or manual reference.\n"
        "  • stains_detective  — use when the question asks to trace back defect origins "
        "across process images, align process images, run a defect traceback pipeline, "
        "or analyse stains / particle origins across inspection steps. "
        "Include any file-system path mentioned by the user verbatim in sub_query.\n"
        "  • general           — use for greetings, chitchat, capability questions "
        "(e.g. 'hello', 'what can you do?', 'who are you?'), or any conversational "
        "query that does not require knowledge-base lookup.\n"
        "  • reporting         — use when the scratchpad already contains enough "
        "information to give a complete, accurate answer to the user.\n\n"
        "ROUTING RULES:\n"
        "1. For lot-specific queries: aries_data FIRST, then issue_agent if KB context is needed.\n"
        "2. Do NOT repeat the same agent if its findings are already in the scratchpad.\n"
        "3. If the scratchpad has data but no KB context and the question needs both, use issue_agent.\n"
        "4. If the scratchpad already has sufficient data + KB context, go to reporting.\n\n"
        "Provide a concise sub_query (≤ 20 words) for the chosen agent."
    )

    logger.debug("Orchestrator prompt:\n%s", prompt)

    model = get_chat_model(temperature=0).with_structured_output(OrchestratorDecision)

    try:
        decision: OrchestratorDecision = model.invoke([HumanMessage(content=prompt)])
    except Exception as e:
        logger.error("Structured output failed: %s — falling back to reporting", e)
        return {
            "next_action": "reporting",
            "sub_query": user_query,
            "iteration": iteration + 1,
            "messages": [AIMessage(content="[Orchestrator] Falling back to reporting.")],
        }

    logger.info(
        "Orchestrator [iter %d/%d] → %s | sub_query: %r | reason: %s",
        iteration + 1,
        max_iter,
        decision.next_action,
        decision.sub_query,
        decision.reasoning,
    )

    return {
        "next_action": decision.next_action,
        "sub_query": decision.sub_query,
        "iteration": iteration + 1,
        "messages": [
            AIMessage(
                content=f"[Orchestrator → {decision.next_action}] {decision.reasoning}"
            )
        ],
    }
