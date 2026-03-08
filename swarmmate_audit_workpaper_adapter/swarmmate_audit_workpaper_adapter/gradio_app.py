from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

import gradio as gr
from dotenv import load_dotenv

from app.graphs.workpaper_adapter_with_guardrails import graph
from app import rag as rag_module


load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)


TASK_HINTS = {
    "audit_planning_memo": (
        "Draft an audit planning memo for this client. "
        "Highlight likely risk areas, open requests, and include citations to the source files."
    ),
    "workpaper_summary": (
        "Summarize the uploaded workpapers for reviewer handoff. "
        "Include key facts, missing items, and citations."
    ),
    "issues_checklist": (
        "Create an issues checklist from the uploaded client documents. "
        "Flag follow-ups and cite the sources."
    ),
    "client_letter": (
        "Draft a client-ready letter summarizing the requested information "
        "and open items. Keep it professional and grounded."
    ),
    "cited_qa": (
        "Answer this question using only the uploaded source files and include citations."
    ),
}


def _copy_uploaded_files(files: Iterable[object] | None) -> str | None:
    if not files:
        return None

    temp_dir = Path(tempfile.mkdtemp(prefix="swarmmate_uploads_"))
    copied = 0

    for item in files:
        src_path = Path(getattr(item, "name", str(item)))
        if src_path.exists():
            target = temp_dir / src_path.name
            shutil.copy2(src_path, target)
            copied += 1

    if copied == 0:
        return None

    return str(temp_dir)


def _prepare_data_dir(uploaded_files: list[object] | None) -> tuple[str, str]:
    upload_dir = _copy_uploaded_files(uploaded_files)
    if upload_dir:
        data_dir = upload_dir
        file_count = len([p for p in Path(upload_dir).glob("*") if p.is_file()])
        status = f"Using {file_count} uploaded file(s) from temporary folder: {upload_dir}"
    else:
        data_dir = os.environ.get("RAG_DATA_DIR", "data")
        status = f"Using existing source folder: {data_dir}"

    os.environ["RAG_DATA_DIR"] = data_dir

    # Clear cached vector store so retrieval uses the newly selected folder
    rag_module._get_vector_store.cache_clear()

    return data_dir, status


def _build_request(task_type: str, user_request: str) -> str:
    request = (user_request or "").strip()
    if request:
        return request
    return TASK_HINTS.get(task_type, "Summarize the uploaded workpapers with citations.")


def run_adapter(task_type: str, user_request: str, uploaded_files: list[object] | None):
    if not os.environ.get("OPENAI_API_KEY"):
        return (
            "OPENAI_API_KEY is missing. Add it to your .env file before running the app.",
            "No sources available.",
            "Configuration error",
            None,
        )

    data_dir, source_status = _prepare_data_dir(uploaded_files)
    request = _build_request(task_type, user_request)

    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "human",
                    "content": request,
                }
            ]
        }
    )

    response = result.get("final_response") or "No response generated."
    sources = rag_module.describe_indexed_sources()
    debug = "\n".join(
        [
            f"Task type: {task_type}",
            f"Active data dir: {data_dir}",
            f"Request used: {request}",
            source_status,
            f"Saved report: {result.get('artifact_path', 'Not saved')}",
        ]
    )

    report_file = result.get("artifact_path")
    return response, sources, debug, report_file


with gr.Blocks(title="SwarmMate Audit Workpaper Adapter") as demo:
    gr.Markdown(
        """
# SwarmMate Audit Workpaper Adapter

A workflow-first assistant for pre-CCH audit/tax work.

Upload workpapers, choose a workflow, ask a question, and generate a grounded artifact with citations.
"""
    )

    with gr.Row():
        task_type = gr.Dropdown(
            choices=[
                "audit_planning_memo",
                "workpaper_summary",
                "issues_checklist",
                "client_letter",
                "cited_qa",
            ],
            value="audit_planning_memo",
            label="Workflow",
        )

        upload = gr.Files(
            label="Upload workpapers (optional)",
            file_count="multiple",
            file_types=[".pdf", ".md", ".txt", ".csv", ".xlsx", ".xlsm"],
        )

    request = gr.Textbox(
        label="Task request",
        lines=6,
        placeholder=(
            "Example: Draft an audit planning memo for this client, "
            "highlight revenue recognition risk, and include citations."
        ),
    )

    run_btn = gr.Button("Run workflow", variant="primary")

    output = gr.Markdown(label="Generated artifact")

    with gr.Row():
        sources = gr.Textbox(label="Indexed sources", lines=10)
        debug = gr.Textbox(label="Run details", lines=10)

    report_download = gr.File(label="Generated report file")

    gr.Examples(
        examples=[
            ["audit_planning_memo", TASK_HINTS["audit_planning_memo"]],
            ["workpaper_summary", TASK_HINTS["workpaper_summary"]],
            ["issues_checklist", TASK_HINTS["issues_checklist"]],
            ["client_letter", TASK_HINTS["client_letter"]],
            ["cited_qa", "What support is still missing based on the current PBC list and control notes?"],
        ],
        inputs=[task_type, request],
    )

    run_btn.click(
        fn=run_adapter,
        inputs=[task_type, request, upload],
        outputs=[output, sources, debug, report_download],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=8000, share=False)
