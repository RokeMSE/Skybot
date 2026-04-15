"""
SOP / Document Agent node.

Searches the knowledge base for procedures, BKMs, checklists, and manuals,
then summarises the relevant steps and appends them to the scratchpad.
"""
from langchain_core.messages import AIMessage, HumanMessage

from ..llm import get_chat_model
from ..state import AgentState
from ..tools import retrieve_from_knowledge_base


def sop_agent_node(state: AgentState) -> dict:
    sub_query = state.get("sub_query") or state["user_query"]
    channel = state.get("channel")
    iteration = state.get("iteration", 1)

    # 1. Retrieve relevant documents
    retrieval = retrieve_from_knowledge_base(sub_query, channel=channel, n_results=5)

    if not retrieval["context"].strip():
        finding = "No relevant SOPs or documents found for this query."
    else:
        # 2. Summarise with a procedure-focused LLM call
        analysis_prompt = (
            "You are a Document & SOP Agent specialising in semiconductor manufacturing.\n\n"
            f"ORIGINAL QUESTION: {state['user_query']}\n"
            f"SEARCH QUERY USED: {sub_query}\n\n"
            f"PREVIOUS INVESTIGATION NOTES:\n{state.get('scratchpad', 'None')}\n\n"
            "RETRIEVED DOCUMENTS:\n"
            f"{retrieval['context']}\n\n"
            "Summarise the relevant procedures, best-known methods (BKMs), or "
            "checklist steps from the retrieved documents. Include:\n"
            "1. The specific steps or instructions applicable to the question\n"
            "2. Any warnings, prerequisites, or critical parameters mentioned\n"
            "3. Document name and page number for each referenced section\n"
            "Be concise and use numbered lists for steps."
        )
        model = get_chat_model(temperature=0.3)
        try:
            response = model.invoke([HumanMessage(content=analysis_prompt)])
            finding = response.content
        except Exception as e:
            finding = f"Analysis failed: {e}"

    # 3. Append to shared scratchpad
    current = state.get("scratchpad", "")
    new_scratchpad = (
        current
        + f"\n\n--- [SOP Agent — Step {iteration}] ---\n"
        + finding
    )

    return {
        "scratchpad": new_scratchpad,
        "retrieved_docs": state.get("retrieved_docs", []) + retrieval["docs"],
        "image_urls": list(
            dict.fromkeys(state.get("image_urls", []) + retrieval["images"])
        ),
        "citations": state.get("citations", []) + retrieval["citations"],
        "messages": [
            AIMessage(content=f"[SOP Agent] {finding[:300]}{'...' if len(finding) > 300 else ''}")
        ],
    }
