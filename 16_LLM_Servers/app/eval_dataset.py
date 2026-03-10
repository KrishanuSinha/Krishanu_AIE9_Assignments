"""Small hand-written evaluation set grounded in the bundled cat-health PDF."""

from __future__ import annotations

EVAL_CASES: list[dict[str, str]] = [
    {
        "user_input": "What are the five feline life stages named in the guideline?",
        "reference": (
            "The five feline life stages are kitten, young adult, mature adult, senior, and end-of-life."
        ),
    },
    {
        "user_input": "Why does the guideline say anesthesia-free dentistry is not appropriate for cats?",
        "reference": (
            "Anesthesia-free dentistry is not appropriate because it causes patient stress, can lead to injury and aspiration risk, and lacks diagnostic capability. "
            "It only cleans the visible tooth surface and can give owners a false sense of benefit."
        ),
    },
]