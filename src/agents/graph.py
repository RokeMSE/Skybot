"""
LangGraph graph assembly.

Graph structure:
  START → orchestrator
  orchestrator → issue_agent | sop_agent | stains_detective | aries_data | general | reporting   (conditional)
  issue_agent        → orchestrator
  sop_agent          → orchestrator
  stains_detective   → orchestrator
  aries_data         → orchestrator
  general            → END
  reporting          → END

The orchestrator loops until it decides "reporting" or max_iterations is hit.
"""
from langgraph.graph import END, START, StateGraph

from .nodes import (
    aries_data_agent_node,
    general_agent_node,
    issue_agent_node,
    orchestrator_node,
    reporting_node,
    sop_agent_node,
    stains_detective_node,
)
from .state import AgentState


def _route(state: AgentState) -> str:
    """Routing function called after every orchestrator step."""
    # Hard stop if we've hit the iteration ceiling
    if state.get("iteration", 0) >= state.get("max_iterations", 3):
        return "reporting"
    return state.get("next_action", "reporting")


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("issue_agent", issue_agent_node)
    graph.add_node("sop_agent", sop_agent_node)
    graph.add_node("stains_detective", stains_detective_node)
    graph.add_node("aries_data", aries_data_agent_node)
    graph.add_node("general", general_agent_node)
    graph.add_node("reporting", reporting_node)

    # Entry point
    graph.add_edge(START, "orchestrator")

    # Conditional routing from orchestrator
    graph.add_conditional_edges(
        "orchestrator",
        _route,
        {
            "issue_agent": "issue_agent",
            "sop_agent": "sop_agent",
            "stains_detective": "stains_detective",
            "aries_data": "aries_data",
            "general": "general",
            "reporting": "reporting",
        },
    )

    # Specialist agents loop back to the orchestrator
    graph.add_edge("issue_agent", "orchestrator")
    graph.add_edge("sop_agent", "orchestrator")
    graph.add_edge("stains_detective", "orchestrator")
    graph.add_edge("aries_data", "orchestrator")

    # General and reporting are terminal — no further investigation needed
    graph.add_edge("general", END)
    graph.add_edge("reporting", END)

    return graph.compile()


# Module-level compiled graph — imported by main.py
agent_graph = build_graph()
