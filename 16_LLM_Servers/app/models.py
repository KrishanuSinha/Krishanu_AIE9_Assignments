"""Model utilities for constructing provider-specific chat and embedding clients."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Literal

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
Provider = Literal["fireworks", "openai"]


def get_chat_model(
    model_name: str | None = None,
    *,
    provider: Provider = "fireworks",
    temperature: float = 0,
) -> Any:
    """Return a configured chat model for the requested provider."""
    if provider == "fireworks":
        name = model_name or os.environ.get(
            "FIREWORKS_CHAT_MODEL",
            "accounts/fireworks/models/gpt-oss-20b",
        )
        return ChatOpenAI(
            model=name,
            temperature=temperature,
            openai_api_key=os.environ["FIREWORKS_API_KEY"],
            openai_api_base=FIREWORKS_BASE_URL,
        )

    if provider == "openai":
        name = model_name or os.environ.get("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
        return ChatOpenAI(
            model=name,
            temperature=temperature,
            openai_api_key=os.environ["OPENAI_API_KEY"],
        )

    raise ValueError(f"Unsupported provider: {provider}")


def get_embedding_model(
    model_name: str | None = None,
    *,
    provider: Provider = "fireworks",
) -> Any:
    """Return a configured embedding model for the requested provider."""
    if provider == "fireworks":
        name = model_name or os.environ.get(
            "FIREWORKS_EMBEDDING_MODEL",
            "accounts/fireworks/models/gpt-oss-20b",
        )
        kwargs: dict[str, Any] = {
            "model": name,
            "openai_api_key": os.environ["FIREWORKS_API_KEY"],
            "openai_api_base": FIREWORKS_BASE_URL,
            "check_embedding_ctx_length": False,
        }
        dimensions = os.environ.get("FIREWORKS_EMBED_DIMENSIONS")
        if dimensions:
            kwargs["dimensions"] = int(dimensions)
        return OpenAIEmbeddings(**kwargs)

    if provider == "openai":
        name = model_name or os.environ.get(
            "OPENAI_EMBEDDING_MODEL",
            "text-embedding-3-large",
        )
        kwargs = {
            "model": name,
            "openai_api_key": os.environ["OPENAI_API_KEY"],
        }
        dimensions = os.environ.get("OPENAI_EMBED_DIMENSIONS")
        if dimensions:
            kwargs["dimensions"] = int(dimensions)
        return OpenAIEmbeddings(**kwargs)

    raise ValueError(f"Unsupported provider: {provider}")


def fix_tool_calls(response: AIMessage) -> AIMessage:
    """Fix invalid tool calls caused by models appending extra tokens like <|call|>."""
    if not response.invalid_tool_calls:
        return response

    fixed = list(response.tool_calls)
    remaining_invalid = []

    for tc in response.invalid_tool_calls:
        cleaned = re.sub(r"\s*<\|call\|>\s*$", "", tc["args"])
        try:
            parsed = json.loads(cleaned)
            fixed.append(
                {
                    "name": tc["name"],
                    "args": parsed,
                    "id": tc["id"],
                    "type": "tool_call",
                }
            )
        except (json.JSONDecodeError, TypeError):
            remaining_invalid.append(tc)

    response.tool_calls = fixed
    response.invalid_tool_calls = remaining_invalid
    return response