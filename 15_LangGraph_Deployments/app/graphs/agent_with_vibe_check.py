"""An agent graph with a post-response vibe check loop.

After the agent responds, a secondary evaluator checks whether the answer feels
warm, clear, concise, and actionable. If the answer does not pass, the graph
adds a rewrite instruction and lets the agent try again. A retry counter keeps
this loop from running indefinitely.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.models import get_chat_model
from app.tools import get_tool_belt


class VibeCheckState(TypedDict, total=False):
    """State for the vibe-check graph.

    - messages: conversation history shared across the graph.
    - vibe_retries: how many rewrite attempts have happened.
    - vibe_status: control signal used by the conditional edge after evaluation.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    vibe_retries: int
    vibe_status: Literal["pending", "rewrite", "approved", "stop"]


class VibeCheckResult(BaseModel):
    passes: bool = Field(
        description="Whether the answer is warm, clear, concise, and actionable."
    )
    feedback: str = Field(
        description="One short sentence describing how to improve the answer if it fails. Leave empty when it passes."
    )


_vibe_prompt = ChatPromptTemplate.from_template(
    "You are evaluating the assistant's final answer. "
    "Approve the answer only if it is all of the following:\n"
    "1. Warm and natural in tone\n"
    "2. Clear and easy to follow\n"
    "3. Actionable, with a concrete next step when helpful\n"
    "4. Not robotic or unnecessarily wordy\n\n"
    "User question:\n{initial_query}\n\n"
    "Assistant answer:\n{final_response}"
)


MAX_VIBE_RETRIES = 2


def _build_model_with_tools():
    """Return a chat model instance bound to the current tool belt."""
    model = get_chat_model()
    return model.bind_tools(get_tool_belt())


def call_model(state: VibeCheckState) -> dict:
    """Invoke the model with the accumulated messages and append its response."""
    model = _build_model_with_tools()
    response = model.invoke(state["messages"])
    return {"messages": [response], "vibe_status": "pending"}


def route_after_agent(state: VibeCheckState):
    """Go to tools when needed; otherwise evaluate the final answer's vibe."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "action"
    return "vibe_check"


def vibe_check_node(state: VibeCheckState) -> dict:
    """Evaluate the most recent answer and request a rewrite if needed."""
    retries = state.get("vibe_retries", 0)
    if retries >= MAX_VIBE_RETRIES:
        return {"vibe_status": "stop"}

    initial_query = next(
        (
            getattr(message, "content", "")
            for message in state["messages"]
            if getattr(message, "type", "") == "human"
        ),
        getattr(state["messages"][0], "content", ""),
    )
    final_response = getattr(state["messages"][-1], "content", "")

    evaluator = get_chat_model(model_name="gpt-4.1-mini").with_structured_output(
        VibeCheckResult
    )
    result = (_vibe_prompt | evaluator).invoke(
        {
            "initial_query": initial_query,
            "final_response": final_response,
        }
    )

    if result.passes:
        return {"vibe_status": "approved"}

    feedback = result.feedback.strip() or "Make the answer warmer, clearer, and more actionable."
    rewrite_request = HumanMessage(
        content=(
            "Please rewrite your previous answer. Keep the facts the same, but improve the vibe. "
            "Make it warmer, clearer, and more actionable. "
            f"Specific feedback: {feedback}"
        )
    )
    return {
        "messages": [rewrite_request],
        "vibe_retries": retries + 1,
        "vibe_status": "rewrite",
    }


def route_after_vibe_check(state: VibeCheckState):
    """Loop back for a rewrite or terminate when approved / retry limit is hit."""
    if state.get("vibe_status") == "rewrite":
        return "agent"
    return END


def build_graph():
    """Build an agent graph with a custom vibe-check evaluation loop."""
    graph = StateGraph(VibeCheckState)
    tool_node = ToolNode(get_tool_belt())

    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_node("vibe_check", vibe_check_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"action": "action", "vibe_check": "vibe_check"},
    )
    graph.add_edge("action", "agent")
    graph.add_conditional_edges(
        "vibe_check",
        route_after_vibe_check,
        {"agent": "agent", END: END},
    )
    return graph


graph = build_graph().compile()
