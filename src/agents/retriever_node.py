"""Retriever node: route-aware source-type filtering.

  route=text       : document-only (text + tables + indexed PDF figures).
                     The separately-uploaded user image is excluded so its
                     summary can't leak into a document-focused answer.
  route=image      : user image only.
  route=cross_modal: everything — both the document and the user image.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from src.agents.state import AgentState, new_trace_entry
from src.retrieval.multi_vector import SourceType, get_retriever
from src.utils.logging import get_logger

log = get_logger(__name__)


def _source_types_for_route(route: str | None) -> list[SourceType] | None:
    if route == "image":
        return ["user_image"]
    if route == "text":
        return ["text", "table", "image"]  # document content, excludes user_image
    # cross_modal (or None fallback): everything
    return None


def retriever_node(state: AgentState) -> dict[str, Any]:
    start = time.perf_counter()

    query = state.get("rewritten_query") or state["question"]
    session_id = state["session_id"]
    route = state.get("route")

    source_types = _source_types_for_route(route)
    retriever = get_retriever(session_id)
    items = retriever.retrieve(query, source_types=source_types)

    serialized = [asdict(i) for i in items]

    duration_ms = (time.perf_counter() - start) * 1000
    scope = ",".join(source_types) if source_types else "all"
    return {
        "retrieved": serialized,
        "rewritten_query": None,
        "trace": [
            new_trace_entry(
                "retriever",
                duration_ms=duration_ms,
                input_summary=f"query={query[:80]!r} route={route}",
                output_summary=f"retrieved={len(items)} items · scope={scope}",
            )
        ],
    }
