"""Tool helpers for the workflow adapter.

The current graphs do not rely on an open-ended tool-calling loop, but these tools
keep the project ready for future interactive / MCP-backed variants.
"""
from __future__ import annotations

from typing import List

from langchain_core.tools import tool

from app.rag import describe_indexed_sources, retrieve_workpaper_context


WORKFLOW_TEMPLATES = {
    "workpaper_summary": """Return a concise manager-facing workpaper summary with sections for scope, notable facts, open questions, and next actions.""",
    "audit_planning_memo": """Return an audit planning memo with engagement context, risk areas, requested follow-ups, controls observations, and recommended next steps.""",
    "client_letter": """Return a professional client-ready draft that is factual, cautious, and explicit about any information still needed from the client.""",
    "issues_checklist": """Return a checklist grouped by issue category, each item linked to evidence and flagged as complete, missing, or needs follow-up.""",
    "cited_qa": """Answer the user question directly and only use claims supported by source citations.""",
}


@tool
def get_workflow_template(task_type: str) -> str:
    """Look up the preferred output shape for a given workflow task."""
    return WORKFLOW_TEMPLATES.get(
        task_type,
        "Return a structured, review-ready response that cites evidence and highlights missing information.",
    )


@tool
def list_indexed_sources() -> str:
    """List the source files currently available to the workflow adapter."""
    return describe_indexed_sources()


def get_tool_belt() -> List:
    """Expose helper tools for future interactive graphs."""
    return [retrieve_workpaper_context, get_workflow_template, list_indexed_sources]
