# SwarmMate Audit Workpaper Adapter

A LangGraph deployment that adapts messy **pre-CCH audit / tax work** into a structured AI workflow.

Instead of acting like a generic chatbot, this project behaves like a **workflow adapter** for Steven's use case:

- ingest client workpapers (PDFs, markdown, CSVs, spreadsheets)
- plan the user request
- retrieve grounded evidence from the attached source folder
- draft a review-ready artifact
- optionally run a quality gate to reduce AI slop

## Why this project exists

Steven's firm already proved that people will use AI, but they also surfaced the real risk: **AI slop costs time** when outputs are vague, unsupported, or poorly structured. This MVP addresses that by focusing on a narrow, high-value workflow:

> turn scattered workpapers into a source-grounded deliverable *before* the content enters legacy accounting systems like CCH.

## What this API can produce

The planner classifies each request into one of these workflow types:

- `workpaper_summary`
- `audit_planning_memo`
- `client_letter`
- `issues_checklist`
- `cited_qa`

## Project layout

```text
.
├── .env.example
├── README.md
├── langgraph.json
├── pyproject.toml
├── test_served_graph.py
├── app/
│   ├── __init__.py
│   ├── README.md
│   ├── models.py
│   ├── rag.py
│   ├── state.py
│   ├── tools.py
│   ├── workflow.py
│   └── graphs/
│       ├── __init__.py
│       ├── workpaper_assistant.py
│       └── workpaper_adapter_with_guardrails.py
└── data/
    ├── client_overview.md
    ├── control_notes.md
    ├── pbc_request_list.md
    └── trial_balance.csv
```

## Run locally

1. Copy `.env.example` to `.env` and fill in your OpenAI key.
2. Install dependencies:
   ```bash
   uv sync
   ```
3. Start the LangGraph dev server:
   ```bash
   uv run langgraph dev
   ```
4. In a second terminal, stream a sample run:
   ```bash
   uv run test_served_graph.py
   ```

## Example requests

Use the default `assistant_guarded` assistant for the most realistic demo.

- `Draft an audit planning memo for this client. Focus on key risks, requested follow-ups, and cite the supporting workpapers.`
- `Create a review checklist from the uploaded workpapers and highlight missing items.`
- `Summarize the client documents for a manager who has not seen this file yet.`
- `Answer this question with citations only: what control weaknesses appear in cash reconciliation?`

## How this differs from a generic AI coworker

This API does **not** try to do everything. It is shaped around a specific workflow:

1. **Attach a source folder** (today: local files; later: MCP / connector-backed sources)
2. **Choose a task**
3. **Retrieve evidence from the workpapers**
4. **Produce a structured artifact**
5. **Run a grounded quality gate** before handing the draft back

That makes it a much better fit for Steven's team than a broad assistant surface.
