"""Workflow-first assistant without the explicit quality gate loop."""
from __future__ import annotations

from app.workflow import build_workpaper_graph


graph = build_workpaper_graph(enable_quality_gate=False).compile()
