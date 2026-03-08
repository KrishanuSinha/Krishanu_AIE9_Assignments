## App package structure

This package keeps the LangGraph deployment close to the original course template, but adapts it to a **workflow adapter** use case for audit / tax teams.

### Modules

- `models.py`: central place for LLM construction
- `state.py`: shared state schema used across the workflow graph
- `rag.py`: local-file source adapter that loads PDFs, markdown, CSVs, and spreadsheets into an in-memory Qdrant index
- `tools.py`: helper tools and workflow templates for future interactive graphs / MCP extensions
- `workflow.py`: the real business logic — request planning, evidence gathering, drafting, quality gate, and final response assembly
- `graphs/`: thin graph-export modules registered in `langgraph.json`

### Design choice

This is intentionally **not** a generic chat agent. The graph is deterministic enough to reduce sloppy outputs:

1. classify the task
2. gather evidence from uploaded workpapers
3. draft a structured artifact
4. optionally verify groundedness before returning the answer

That directly addresses Steven's concern that AI should save time rather than create cleanup work for coworkers.
