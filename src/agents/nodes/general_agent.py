"""
General / chitchat agent node.

Handles greetings, capability questions, and conversational queries
that do not require a knowledge-base lookup.
"""
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from ..llm import get_chat_model
from ..state import AgentState

_COMPLIANCE_PATH = Path(__file__).resolve().parent.parent / "compliance.md"
_COMPLIANCE_RULES = _COMPLIANCE_PATH.read_text(encoding="utf-8")


def general_agent_node(state: AgentState) -> dict:
    user_query = state["user_query"]

    prompt = (
        "You are SkyBot, an AI assistant specialising in semiconductor manufacturing.\n\n"
        "=== COMPLIANCE GUARDRAILS ===\n"
        f"{_COMPLIANCE_RULES}\n"
        "=== END GUARDRAILS ===\n\n"
        "The user sent a conversational message (greeting, capability question, etc.).\n"
        "Respond in a friendly, professional tone. If relevant, briefly describe your "
        "capabilities:\n"
        "  - Investigating manufacturing defects and root-cause analysis\n"
        "  - Looking up SOPs, BKMs, checklists, and work instructions\n"
        "  - Tracing defect origins across process inspection images\n"
        "  - Answering questions based on your semiconductor knowledge base\n\n"
        "Keep the response concise. Do NOT make up technical information.\n\n"
        f"USER MESSAGE:\n{user_query}"
    )

    model = get_chat_model(temperature=0.7)
    try:
        response = model.invoke([HumanMessage(content=prompt)])
        answer = response.content
    except Exception as e:
        answer = f"Hello! I'm SkyBot, your semiconductor manufacturing assistant. How can I help you today?"

    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
    }
