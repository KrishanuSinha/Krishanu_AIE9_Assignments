"""Run Activity 1: compare Fireworks vs OpenAI RAG with RAGAS-style metrics and LangSmith tracing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import truststore
truststore.inject_into_ssl()

import langsmith as ls
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory

try:
    from ragas.metrics.collections import ContextPrecision, Faithfulness, FactualCorrectness
except ImportError as exc: # pragma: no cover
    raise ImportError(
        "This script uses the current RAGAS collections-based API. "
        "Please upgrade ragas to a recent version."
    ) from exc

from app.eval_dataset import EVAL_CASES
from app.rag import answer_question, warm_rag_indexes

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _metric_value(result: Any) -> float:
    value = getattr(result, "value", result)
    return float(value)


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _truncate_text(text: str, max_chars: int = 1200) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _trim_contexts(contexts: list[str], max_contexts: int = 2, max_chars_each: int = 1200) -> list[str]:
    trimmed = []
    for ctx in contexts[:max_contexts]:
        trimmed.append(_truncate_text(ctx, max_chars_each))
    return trimmed


def collect_provider_outputs(provider: str, project_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    with ls.tracing_context(project_name=project_name, enabled=True):
        for case in EVAL_CASES:
            print(f"[EVAL] Running provider={provider} question={case['user_input']}")
            result = answer_question(case["user_input"], provider=provider)
            usage = result.get("usage_metadata", {}) or {}

            rows.append(
                {
                    "provider": provider,
                    "user_input": case["user_input"],
                    "reference": case["reference"],
                    "response": result["answer"],
                    "retrieved_contexts": result["retrieved_contexts"],
                    "source_documents": result["source_documents"],
                    "input_tokens": _safe_int(usage.get("input_tokens")),
                    "output_tokens": _safe_int(usage.get("output_tokens")),
                    "total_tokens": _safe_int(usage.get("total_tokens")),
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(ARTIFACT_DIR / f"{provider}_raw_outputs.csv", index=False)
    return frame


def evaluate_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    evaluator_model = os.environ.get("RAGAS_EVAL_MODEL", "gpt-4.1-mini")

    evaluator_llm = llm_factory(
        evaluator_model,
        client=AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"]),
        max_tokens=6000,
        temperature=0,
    )

    context_precision = ContextPrecision(llm=evaluator_llm)
    factual_correctness = FactualCorrectness(llm=evaluator_llm)

    scored_rows: list[dict[str, Any]] = []

    with ls.tracing_context(enabled=False):
        for row in frame.to_dict(orient="records"):
            trimmed_contexts = _trim_contexts(
                row["retrieved_contexts"],
                max_contexts=2,
                max_chars_each=1200,
            )

            print(f"[RAGAS] Scoring provider={row['provider']} question={row['user_input']}")

            cp = context_precision.score(
                user_input=row["user_input"],
                reference=row["reference"],
                retrieved_contexts=trimmed_contexts,
            )
            fc = factual_correctness.score(
                response=_truncate_text(row["response"], 1500),
                reference=row["reference"],
            )

            scored_rows.append(
                {
                    **row,
                    "retrieved_contexts_trimmed": trimmed_contexts,
                    "context_precision": _metric_value(cp),
                    "faithfulness": None,
                    "factual_correctness": _metric_value(fc),
                }
            )

    detail_df = pd.DataFrame(scored_rows)
    detail_df.to_csv(
        ARTIFACT_DIR / f"{frame.iloc[0]['provider']}_evaluation_detail.csv",
        index=False,
    )

    summary = {
        "context_precision": float(detail_df["context_precision"].mean()),
        "faithfulness": 0.0,
        "factual_correctness": float(detail_df["factual_correctness"].mean()),
        "mean_input_tokens": float(detail_df["input_tokens"].dropna().mean()) if detail_df["input_tokens"].notna().any() else 0.0,
        "mean_output_tokens": float(detail_df["output_tokens"].dropna().mean()) if detail_df["output_tokens"].notna().any() else 0.0,
        "mean_total_tokens": float(detail_df["total_tokens"].dropna().mean()) if detail_df["total_tokens"].notna().any() else 0.0,
    }
    return detail_df, summary


def main() -> None:
    load_dotenv()

    print("FIREWORKS_CHAT_MODEL =", os.environ.get("FIREWORKS_CHAT_MODEL"))
    print("FIREWORKS_EMBEDDING_MODEL =", os.environ.get("FIREWORKS_EMBEDDING_MODEL"))
    print("FIREWORKS_EMBED_DIMENSIONS =", os.environ.get("FIREWORKS_EMBED_DIMENSIONS"))

    required = ["FIREWORKS_API_KEY", "OPENAI_API_KEY", "LANGSMITH_API_KEY"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {missing}")

    base_project = os.environ.get("LANGSMITH_PROJECT", "assignment16-rag")
    fireworks_project = f"{base_project}-fireworks-app"
    openai_project = f"{base_project}-openai-app"

    print("[EVAL] Starting index warm-up...")
    with ls.tracing_context(enabled=False):
        warm_rag_indexes()
    print("[EVAL] Index warm-up complete.")

    print("[EVAL] Collecting Fireworks outputs...")
    fireworks_outputs = collect_provider_outputs("fireworks", fireworks_project)
    print("[EVAL] Fireworks outputs complete.")

    print("[EVAL] Collecting OpenAI outputs...")
    openai_outputs = collect_provider_outputs("openai", openai_project)
    print("[EVAL] OpenAI outputs complete.")

    print("[EVAL] Running RAGAS on Fireworks outputs...")
    fireworks_detail, fireworks_summary = evaluate_frame(fireworks_outputs)
    print("[EVAL] Fireworks evaluation complete.")

    print("[EVAL] Running RAGAS on OpenAI outputs...")
    openai_detail, openai_summary = evaluate_frame(openai_outputs)
    print("[EVAL] OpenAI evaluation complete.")

    summary_df = pd.DataFrame(
        [
            {"provider": "fireworks", **fireworks_summary},
            {"provider": "openai", **openai_summary},
        ]
    )
    summary_df.to_csv(ARTIFACT_DIR / "activity1_summary.csv", index=False)

    print("\n=== ACTIVITY 1 SUMMARY ===")
    print(summary_df.to_string(index=False))
    print("\nDetailed CSVs written to ./artifacts")
    print(f"Fireworks app traces: {fireworks_project}")
    print(f"OpenAI app traces: {openai_project}")
    print(
        "\nIf LangSmith does not show a dollar cost for the Fireworks model, "
        "add the Fireworks model price in LangSmith's model pricing table and rerun."
    )


if __name__ == "__main__":
    main()