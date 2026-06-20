"""Promptfoo export helpers."""

from __future__ import annotations

from ragfold.evaluation.runner import Dataset


def to_promptfoo_config(dataset: Dataset, provider: str = "python:ragfold") -> str:
    """Serialize a small promptfoo-compatible YAML config string."""

    lines = [
        "description: ragfold benchmark export",
        "providers:",
        f"  - {provider}",
        "prompts:",
        '  - "{{query}}"',
        "tests:",
    ]
    for query in dataset.queries:
        expected = query.answers[0] if query.answers else ",".join(query.relevant_ids)
        lines.extend(
            [
                "  - vars:",
                f"      query: {query.query!r}",
                "    assert:",
                "      - type: contains",
                f"        value: {expected!r}",
            ]
        )
    return "\n".join(lines) + "\n"
