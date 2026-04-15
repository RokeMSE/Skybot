"""
Reporting Agent node — the final step.

Compiles the scratchpad and all retrieved sources into a structured,
user-facing answer. Mirrors the system prompt style used by the
original RAGEngine so the frontend renders it the same way.

Adapts citation style based on what data sources contributed:
  - KB documents → cite document name + page
  - Live data (Aries, lot/unit XML) → cite data source + query params
  - Mixed → include both, clearly separated
"""
import os
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from ..llm import get_chat_model
from ..state import AgentState

_COMPLIANCE_PATH = Path(__file__).resolve().parent.parent / "compliance.md"
_COMPLIANCE_RULES = _COMPLIANCE_PATH.read_text(encoding="utf-8")


def reporting_node(state: AgentState) -> dict:
    scratchpad = state.get("scratchpad", "").strip()
    citations = state.get("citations", [])
    image_urls = state.get("image_urls", [])

    # Detect which data sources contributed to the investigation
    has_kb_docs = bool(citations)
    has_live_data = any(
        tag in scratchpad
        for tag in ("[Aries Data Agent", "[Lot Info", "[Unit Info")
    )
    has_stains = any(
        tag in scratchpad
        for tag in ("[Stains Detective",)
    )

    # Build source → document-page URL mapping (only when KB docs exist)
    source_doc_links: dict = {}
    page_to_image_map: dict = {}

    if has_kb_docs:
        for meta in citations:
            source = meta.get("source")
            page = meta.get("page")
            if source and page:
                source_doc_links.setdefault(source, {})[page] = (
                    f"/static/documents/{source}#page={page}"
                )
            if meta.get("type") == "image_cad" and meta.get("image_path") and page:
                key = f"{source}_p{page}"
                page_to_image_map[key] = (
                    f"/static/images/{os.path.basename(meta['image_path'])}"
                )

    # Separate traceback panel images from regular knowledge-base images
    panel_urls = [u for u in image_urls if os.path.basename(u).startswith("PANEL_")]
    kb_image_urls = [u for u in image_urls if not os.path.basename(u).startswith("PANEL_")]

    # Build inline markdown for every panel so the LLM always includes them
    panel_md = ""
    if panel_urls:
        panel_md = (
            "\n\n--- TRACEBACK PANELS ---\n"
            "You MUST embed ALL of the following traceback panel images inline in your answer, "
            "one per defect, in the order listed. Use this exact markdown syntax for each:\n"
            + "\n".join(f"  ![Traceback Panel]({u})" for u in panel_urls)
            + "\n---\n"
        )

    kb_image_map: dict = {}
    for u in kb_image_urls:
        kb_image_map[os.path.basename(u)] = u

    # Build citation instructions based on what sources were used
    citation_rules = _build_citation_rules(
        has_kb_docs=has_kb_docs,
        has_live_data=has_live_data,
        source_doc_links=source_doc_links,
        kb_image_map=kb_image_map,
        panel_md=panel_md,
    )

    system_instruction = (
        "You are an expert Semiconductor Manufacturing Assistant compiling a final answer.\n"
        "Use ONLY the investigation notes below — do not invent information.\n\n"
        "=== COMPLIANCE GUARDRAILS ===\n"
        f"{_COMPLIANCE_RULES}\n"
        "=== END GUARDRAILS ===\n\n"
        f"{citation_rules}"
        "You MUST NOT generate ASCII art, charts, or diagrams.\n"
        "---------------------------"
    )

    if not scratchpad:
        final = "I could not find relevant information in the knowledge base to answer your question."
    else:
        # Adapt report instructions to match the data sources
        cite_instruction = _build_report_cite_instruction(has_kb_docs, has_live_data)

        report_prompt = (
            f"ORIGINAL QUESTION:\n{state['user_query']}\n\n"
            f"INVESTIGATION NOTES:\n{scratchpad}\n\n"
            "Write a comprehensive, well-structured final answer. Include:\n"
            "1. Direct answer to the question\n"
            "2. Supporting evidence and key findings\n"
            f"{cite_instruction}"
            "4. Recommendations or next steps where applicable\n"
        )
        if kb_image_map or panel_md:
            report_prompt += "Embed any available images inline using the rules in the system prompt."

        model = get_chat_model(temperature=0.5)
        try:
            response = model.invoke(
                [HumanMessage(content=system_instruction + "\n\n" + report_prompt)]
            )
            final = response.content
        except Exception as e:
            final = f"Report generation failed: {e}\n\nRaw notes:\n{scratchpad}"

    return {
        "final_answer": final,
        "messages": [AIMessage(content=final)],
    }


def _build_citation_rules(
    has_kb_docs: bool,
    has_live_data: bool,
    source_doc_links: dict,
    kb_image_map: dict,
    panel_md: str,
) -> str:
    """Build the citation/image section of the system instruction."""
    parts = []

    if has_kb_docs and source_doc_links:
        parts.append(
            "--- DOCUMENT CITATION RULES ---\n"
            "When citing knowledge base documents, hyperlink to the document:\n"
            "  [FileName — Page X](/static/documents/FileName#page=X)\n\n"
            f"Available document links:\n{source_doc_links}\n"
        )
        if kb_image_map:
            parts.append(
                "Knowledge-base images — embed inline when relevant:\n"
                "  ![Description](/static/images/filename.png)\n\n"
                f"Available image URLs:\n{kb_image_map}\n"
            )

    if has_live_data:
        parts.append(
            "--- LIVE DATA CITATION RULES ---\n"
            "When citing live data (Aries DB, lot/unit XML), reference the data source "
            "and parameters instead of document pages. Example:\n"
            "  *Source: Aries DB — Lot 4V56656R, Op 5274, HXV testers, last 12h*\n"
            "Do NOT fabricate document page numbers for data that came from live queries.\n"
        )

    if panel_md:
        parts.append(panel_md)

    if not parts:
        parts.append(
            "--- CITATION RULES ---\n"
            "Cite the source of each claim. If no sources are available, state that clearly.\n"
        )

    return "\n".join(parts) + "\n"


def _build_report_cite_instruction(has_kb_docs: bool, has_live_data: bool) -> str:
    """Return the citation line for the report prompt."""
    if has_kb_docs and has_live_data:
        return "3. Source citations: document name + page for KB results; data source + parameters for live data\n"
    elif has_live_data:
        return "3. Data source citations (data source, lot ID, operation, query parameters)\n"
    else:
        return "3. Source citations (document name + page)\n"
