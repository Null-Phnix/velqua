"""
Memory export for downstream evaluation — produces test contexts from stored facts.

Used by Scalpel's pipeline to feed Anamnesis knowledge into Mirror evaluation.
"""

from __future__ import annotations

from typing import Any, Optional

from ..stores.sqlite_backend import SQLiteBackend


def export_evaluation_context(
    backend: SQLiteBackend,
    topic_filter: Optional[str] = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Export stored facts as evaluation-compatible context dicts.

    Each dict contains:
      - content: The fact text
      - fact_type: The fact category
      - confidence: How confident we are in this fact
      - importance: How important the fact is
      - source: "anamnesis"

    Args:
        backend: SQLiteBackend instance to query
        topic_filter: Optional topic string to filter by (uses FTS search)
        limit: Maximum number of facts to export

    Returns:
        List of dicts suitable for injection into evaluation prompts.
    """
    if topic_filter:
        facts = backend.search_facts(topic_filter, limit=limit)
    else:
        facts = backend.list_facts(limit=limit)

    return [
        {
            "content": f["content"],
            "fact_type": f.get("fact_type", "unknown"),
            "confidence": f.get("confidence", 0.5),
            "importance": f.get("importance", 0.5),
            "source": "anamnesis",
        }
        for f in facts
    ]


def export_as_prompt_context(
    backend: SQLiteBackend,
    topic_filter: Optional[str] = None,
    limit: int = 20,
) -> str:
    """Export facts as a formatted text block for prompt injection.

    Returns a string suitable for including in a system prompt or
    evaluation context window.
    """
    facts = export_evaluation_context(backend, topic_filter=topic_filter, limit=limit)
    if not facts:
        return ""

    lines = ["## Known Facts"]
    for f in facts:
        conf = f["confidence"]
        lines.append(f"- [{f['fact_type']}] {f['content']} (confidence: {conf:.0%})")
    return "\n".join(lines)
