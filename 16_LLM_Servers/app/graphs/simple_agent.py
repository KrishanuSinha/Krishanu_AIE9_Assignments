"""A minimal tool-using agent graph.

The graph:
- Calls a chat model bound to the tool belt.
- If the last message requested tool calls, routes to a ToolNode.
- Otherwise, terminates.
"""

from __future__ import annotations

import time

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from openai import InternalServerError

from app.models import fix_tool_calls, get_chat_model
from app.state import MessagesState
from app.tools import get_tool_belt


def _build_model_with_tools():
    """Return a chat model instance bound to the current tool belt."""
    model = get_chat_model()
    return model.bind_tools(get_tool_belt())


MODEL = _build_model_with_tools()


def call_model(state: MessagesState, config):
    messages = state["messages"]

    max_retries = 12
    delay = 5

    for attempt in range(max_retries):
        try:
            response = fix_tool_calls(MODEL.invoke(messages))
            return {"messages": [response]}

        except InternalServerError as e:
            error_body = getattr(e, "body", {}) or {}
            error_code = (
                error_body.get("error", {}).get("code")
                if isinstance(error_body, dict)
                else None
            )

            if error_code == "DEPLOYMENT_SCALING_UP":
                if attempt == max_retries - 1:
                    raise
                print(f"Fireworks deployment scaling up. Retrying in {delay}s...")
                time.sleep(delay)
                delay = min(int(delay * 1.5), 60)
                continue

            raise


def build_graph():
    """Build an agent graph that interleaves model and tool execution."""
    graph = StateGraph(MessagesState)
    tool_node = ToolNode(get_tool_belt())

    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "action", END: END},
    )
    graph.add_edge("action", "agent")

    return graph


# Export compiled graph for LangGraph
graph = build_graph().compile()

