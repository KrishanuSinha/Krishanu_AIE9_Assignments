from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.models import get_chat_model
from app.rag import collect_workpaper_evidence, describe_indexed_sources
from app.state import TaskType, WorkflowAdapterState
from app.tools import WORKFLOW_TEMPLATES

MAX_REVISIONS = int(os.environ.get("QUALITY_MAX_REVISIONS", "2"))

def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    return value.strip("-") or "report"


def _save_report(artifact_title: str, content: str) -> str:
    reports_dir = Path(os.environ.get("REPORTS_DIR", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{_slugify(artifact_title)}_{timestamp}.md"
    report_path = reports_dir / filename
    report_path.write_text(content, encoding="utf-8")

    return str(report_path)



class WorkflowPlan(BaseModel):
    task_type: TaskType
    problem_frame: str = Field(description="One or two lines describing what the user is trying to accomplish")
    retrieval_queries: list[str] = Field(description="Two to four grounded retrieval queries for the source folder")
    output_sections: list[str] = Field(description="Ordered sections that should appear in the final artifact")
    reviewer_checklist: list[str] = Field(description="Checks a human reviewer would care about")
    artifact_title: str = Field(description="Short title for the output")


class DraftArtifact(BaseModel):
    artifact_title: str
    artifact_markdown: str = Field(description="Final markdown artifact with source citations like [S1]")


class QualityReview(BaseModel):
    passes: bool
    feedback: str = Field(description="Concrete feedback if the draft needs revision")
    missing_requirements: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)


PLANNER_SYSTEM = """You are the planning layer of SwarmMate, a workflow adapter for audit and tax teams.

Your job is to convert the user's request into a narrow, grounded workflow plan that can be executed against uploaded workpapers.

Rules:
- Classify the request into exactly one supported task type.
- Focus on work that happens before content enters a legacy accounting system.
- Retrieval queries must target the attached source files only.
- Prefer review-ready artifacts over generic chat responses.
- Assume the team cares about groundedness, reviewability, and missing-information flags.
"""


DRAFT_SYSTEM = """You are SwarmMate, a workflow adapter for audit and tax teams.

Write a review-ready artifact using ONLY the evidence packet.

Requirements:
- Use markdown headings and bullet points where useful.
- Cite material claims inline with source tags like [S1], [S2].
- If the evidence is incomplete, explicitly say what is missing.
- Do not invent policies, balances, controls, or client facts.
- Keep the tone professional and useful to a reviewer.
"""


QUALITY_SYSTEM = """You are the grounding and quality gate for SwarmMate.

Decide whether the draft is safe to hand back to a human reviewer.

Reject the draft if any of the following are true:
- material claims are missing source citations
- the draft makes claims not supported by the evidence packet
- the output ignores the requested artifact shape
- the draft fails to flag missing information when the sources are incomplete

Return specific revision feedback when failing the draft.
"""


REVISION_SYSTEM = """You are revising a draft created for an audit / tax workflow.

Revise the artifact using the reviewer feedback while staying fully grounded in the evidence packet.
Keep the useful parts, fix unsupported claims, and improve structure where needed.
"""


def _latest_user_request(state: WorkflowAdapterState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return str(state["messages"][-1].content)


def _format_evidence_packet(evidence: list[dict[str, str]]) -> str:
    if not evidence:
        return "No evidence was retrieved from the current source folder."

    blocks: list[str] = []
    for item in evidence:
        blocks.append(
            f"[{item['citation']}] Source: {item['source']} ({item['location']})\n"
            f"Linked query: {item['query']}\n"
            f"Excerpt: {item['excerpt']}"
        )
    return "\n\n".join(blocks)


def _source_summary(evidence: list[dict[str, str]]) -> str:
    if not evidence:
        return "No source citations were used."

    seen: list[str] = []
    for item in evidence:
        descriptor = f"[{item['citation']}] {item['source']} ({item['location']})"
        if descriptor not in seen:
            seen.append(descriptor)
    return "\n".join(f"- {entry}" for entry in seen)


def plan_request(state: WorkflowAdapterState) -> dict:
    request = _latest_user_request(state)
    planner_model = get_chat_model(
        model_name=os.environ.get("PLANNER_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")),
        temperature=0,
    ).with_structured_output(WorkflowPlan)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PLANNER_SYSTEM),
            (
                "human",
                "Available source files:\n{sources}\n\nUser request:\n{request}",
            ),
        ]
    )

    plan = (prompt | planner_model).invoke(
        {
            "sources": describe_indexed_sources(),
            "request": request,
        }
    )

    plan_summary = (
        f"Plan created for task type `{plan.task_type}`. "
        f"Artifact: {plan.artifact_title}. "
        f"Retrieval queries: {', '.join(plan.retrieval_queries)}"
    )

    return {
        "request": request,
        "task_type": plan.task_type,
        "problem_frame": plan.problem_frame,
        "retrieval_queries": plan.retrieval_queries,
        "output_sections": plan.output_sections,
        "reviewer_checklist": plan.reviewer_checklist,
        "artifact_title": plan.artifact_title,
        "revision_count": state.get("revision_count", 0),
        "messages": [AIMessage(content=plan_summary)],
    }


def retrieve_evidence(state: WorkflowAdapterState) -> dict:
    queries = state.get("retrieval_queries") or [state["request"]]
    evidence = collect_workpaper_evidence(
        queries,
        k=int(os.environ.get("RAG_TOP_K", "4")),
    )
    evidence_packet = _format_evidence_packet(evidence)
    sources_used = len({item["source"] for item in evidence})

    status = (
        f"Retrieved {len(evidence)} evidence snippets across {sources_used} source file(s)."
        if evidence
        else "No evidence snippets were retrieved from the current source folder."
    )

    return {
        "evidence": evidence,
        "evidence_packet": evidence_packet,
        "messages": [AIMessage(content=status)],
    }


def draft_artifact(state: WorkflowAdapterState) -> dict:
    draft_model = get_chat_model(temperature=0).with_structured_output(DraftArtifact)
    task_type = state.get("task_type", "workpaper_summary")
    artifact_hint = WORKFLOW_TEMPLATES.get(task_type, "Return a structured, review-ready artifact.")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", DRAFT_SYSTEM),
            (
                "human",
                "User request:\n{request}\n\n"
                "Problem frame:\n{problem_frame}\n\n"
                "Task type: {task_type}\n"
                "Preferred artifact title: {artifact_title}\n"
                "Artifact shape guidance:\n{artifact_hint}\n\n"
                "Expected sections:\n{output_sections}\n\n"
                "Reviewer checklist:\n{reviewer_checklist}\n\n"
                "Evidence packet:\n{evidence_packet}",
            ),
        ]
    )

    draft = (prompt | draft_model).invoke(
        {
            "request": state["request"],
            "problem_frame": state.get("problem_frame", ""),
            "task_type": task_type,
            "artifact_title": state.get("artifact_title", "Generated Workpaper Artifact"),
            "artifact_hint": artifact_hint,
            "output_sections": "\n".join(f"- {item}" for item in state.get("output_sections", [])),
            "reviewer_checklist": "\n".join(f"- {item}" for item in state.get("reviewer_checklist", [])),
            "evidence_packet": state.get("evidence_packet", "No evidence packet available."),
        }
    )

    return {
        "artifact_title": draft.artifact_title,
        "artifact_markdown": draft.artifact_markdown,
        "messages": [AIMessage(content=f"Draft prepared: {draft.artifact_title}")],
    }


def quality_gate(state: WorkflowAdapterState) -> dict:
    quality_model = get_chat_model(
        model_name=os.environ.get("QUALITY_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")),
        temperature=0,
    ).with_structured_output(QualityReview)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QUALITY_SYSTEM),
            (
                "human",
                "User request:\n{request}\n\n"
                "Task type: {task_type}\n"
                "Expected sections:\n{output_sections}\n\n"
                "Draft:\n{artifact_markdown}\n\n"
                "Evidence packet:\n{evidence_packet}",
            ),
        ]
    )

    review = (prompt | quality_model).invoke(
        {
            "request": state["request"],
            "task_type": state.get("task_type", "workpaper_summary"),
            "output_sections": "\n".join(f"- {item}" for item in state.get("output_sections", [])),
            "artifact_markdown": state.get("artifact_markdown", ""),
            "evidence_packet": state.get("evidence_packet", "No evidence packet available."),
        }
    )

    label = "PASS" if review.passes else "REVISE"
    feedback = review.feedback or "No additional feedback."

    return {
        "qa_passed": review.passes,
        "qa_feedback": feedback,
        "qa_missing_requirements": review.missing_requirements,
        "qa_unsupported_claims": review.unsupported_claims,
        "messages": [AIMessage(content=f"Quality gate result: {label}. {feedback}")],
    }


def quality_decision(state: WorkflowAdapterState) -> Literal["pass", "revise", "stop"]:
    if state.get("qa_passed", False):
        return "pass"
    if state.get("revision_count", 0) >= MAX_REVISIONS:
        return "stop"
    return "revise"


def revise_artifact(state: WorkflowAdapterState) -> dict:
    revision_model = get_chat_model(temperature=0).with_structured_output(DraftArtifact)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REVISION_SYSTEM),
            (
                "human",
                "Original request:\n{request}\n\n"
                "Current draft:\n{artifact_markdown}\n\n"
                "Reviewer feedback:\n{qa_feedback}\n\n"
                "Missing requirements:\n{qa_missing_requirements}\n\n"
                "Unsupported claims:\n{qa_unsupported_claims}\n\n"
                "Evidence packet:\n{evidence_packet}",
            ),
        ]
    )

    revised = (prompt | revision_model).invoke(
        {
            "request": state["request"],
            "artifact_markdown": state.get("artifact_markdown", ""),
            "qa_feedback": state.get("qa_feedback", ""),
            "qa_missing_requirements": "\n".join(
                f"- {item}" for item in state.get("qa_missing_requirements", [])
            ),
            "qa_unsupported_claims": "\n".join(
                f"- {item}" for item in state.get("qa_unsupported_claims", [])
            ),
            "evidence_packet": state.get("evidence_packet", "No evidence packet available."),
        }
    )

    new_revision_count = state.get("revision_count", 0) + 1
    return {
        "artifact_title": revised.artifact_title,
        "artifact_markdown": revised.artifact_markdown,
        "revision_count": new_revision_count,
        "messages": [AIMessage(content=f"Revision {new_revision_count} prepared.")],
    }


def finalize_response(state: WorkflowAdapterState) -> dict:
    artifact_title = state.get("artifact_title", "Generated Workpaper Artifact")
    response = state.get("artifact_markdown", "No artifact was generated.")
    source_block = _source_summary(state.get("evidence", []))

    if not state.get("qa_passed", True):
        response += (
            "\n\n---\n"
            "> Review note: the quality gate still flagged issues. Please review carefully before reuse.\n"
            f"> Feedback: {state.get('qa_feedback', 'No feedback provided.')}"
        )

    response += f"\n\n---\n### Sources used\n{source_block}"

    report_body = (
        f"# {artifact_title}\n\n"
        f"**Request**: {state.get('request', 'N/A')}\n\n"
        f"**Task Type**: {state.get('task_type', 'N/A')}\n\n"
        f"**Quality Gate Passed**: {state.get('qa_passed', True)}\n\n"
        f"---\n\n"
        f"{response}\n"
    )

    artifact_path = _save_report(artifact_title, report_body)

    return {
        "artifact_path": artifact_path,
        "final_response": report_body,
        "messages": [AIMessage(content=f"Report saved to {artifact_path}")],
    }


def build_workpaper_graph(*, enable_quality_gate: bool) -> StateGraph:
    graph = StateGraph(WorkflowAdapterState)
    graph.add_node("plan_request", plan_request)
    graph.add_node("retrieve_evidence", retrieve_evidence)
    graph.add_node("draft_artifact", draft_artifact)
    graph.add_node("finalize_response", finalize_response)

    graph.add_edge(START, "plan_request")
    graph.add_edge("plan_request", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "draft_artifact")

    if enable_quality_gate:
        graph.add_node("quality_gate", quality_gate)
        graph.add_node("revise_artifact", revise_artifact)
        graph.add_edge("draft_artifact", "quality_gate")
        graph.add_conditional_edges(
            "quality_gate",
            quality_decision,
            {
                "pass": "finalize_response",
                "revise": "revise_artifact",
                "stop": "finalize_response",
            },
        )
        graph.add_edge("revise_artifact", "quality_gate")
    else:
        graph.add_edge("draft_artifact", "finalize_response")

    graph.add_edge("finalize_response", END)
    return graph
