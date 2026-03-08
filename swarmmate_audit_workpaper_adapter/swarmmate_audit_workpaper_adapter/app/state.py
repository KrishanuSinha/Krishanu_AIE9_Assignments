"""Shared state schema for the audit workflow adapter."""
from __future__ import annotations

from typing import Annotated, Literal
from typing_extensions import NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

TaskType = Literal[
    "workpaper_summary",
    "audit_planning_memo",
    "client_letter",
    "issues_checklist",
    "cited_qa",
]


class EvidenceItem(TypedDict):
    citation: str
    source: str
    location: str
    query: str
    excerpt: str


class WorkflowAdapterState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    request: NotRequired[str]
    task_type: NotRequired[TaskType]
    problem_frame: NotRequired[str]
    retrieval_queries: NotRequired[list[str]]
    output_sections: NotRequired[list[str]]
    reviewer_checklist: NotRequired[list[str]]
    artifact_title: NotRequired[str]
    artifact_markdown: NotRequired[str]
    artifact_path: NotRequired[str]
    evidence: NotRequired[list[EvidenceItem]]
    evidence_packet: NotRequired[str]
    qa_passed: NotRequired[bool]
    qa_feedback: NotRequired[str]
    qa_missing_requirements: NotRequired[list[str]]
    qa_unsupported_claims: NotRequired[list[str]]
    revision_count: NotRequired[int]
    final_response: NotRequired[str]