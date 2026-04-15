from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # Conversation history — add_messages merges lists instead of overwriting
    messages: Annotated[list, add_messages]

    # The original question from the user
    user_query: str

    # Optional ChromaDB channel filter
    channel: Optional[str]

    # Running investigation notes written by each agent step
    scratchpad: str

    # Focused query the orchestrator sends to the next agent
    sub_query: str

    # Accumulated across all agent calls
    retrieved_docs: list
    image_urls: list
    citations: list

    # Routing decision set by the orchestrator each iteration
    # Values: "issue_agent" | "sop_agent" | "reporting"
    next_action: str

    # Set by the reporting node — the user-facing final answer
    final_answer: str

    # Loop control
    iteration: int
    max_iterations: int

    # Stains detective — set when the orchestrator routes to stains_detective_agent.
    # uploads_dir: folder with OG images, process images, and DVI_box_data.csv
    # output_dir:  where traceback panels are written (defaults to static/images)
    traceback_uploads_dir: Optional[str]
    traceback_output_dir: Optional[str]
