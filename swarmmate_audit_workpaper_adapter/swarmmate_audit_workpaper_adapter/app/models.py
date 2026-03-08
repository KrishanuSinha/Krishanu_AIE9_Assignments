"""Model utilities for constructing chat LLM clients."""
from __future__ import annotations

import os
from typing import Any

from langchain_openai import ChatOpenAI


def get_chat_model(model_name: str | None = None, *, temperature: float = 0) -> Any:
    """Return a configured ChatOpenAI model.

    The project stays provider-agnostic at the workflow layer. For the MVP we use
    OpenAI because it matches the original deployment template and keeps setup simple.
    """
    name = model_name or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
    return ChatOpenAI(model=name, temperature=temperature)
