"""Workflow adapter with an explicit grounded quality gate."""
from __future__ import annotations

from app.workflow import build_workpaper_graph


graph = build_workpaper_graph(enable_quality_gate=True).compile()
