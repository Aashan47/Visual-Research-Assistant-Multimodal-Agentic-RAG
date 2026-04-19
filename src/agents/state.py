"""LangGraph state schema + trace reducer."""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, Literal, TypedDict

Route = Literal["text", "image", "cross_modal"]


class AgentState(TypedDict, total=False):
    session_id: str
    question: str
    user_image_b64: str | None
    route: Route | None
    rewritten_query: str | None
    retrieved: list[dict[str, Any]]  # serialized RetrievedItem dicts
    answer: str | None
    citations: list[str]
    critique: dict[str, Any] | None
    retry_count: int
    degraded: bool
    trace: Annotated[list[dict[str, Any]], add]


def new_trace_entry(
    node: str,
    *,
    duration_ms: float | None = None,
    input_summary: str | None = None,
    output_summary: str | None = None,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {"node": node}
    if duration_ms is not None:
        entry["duration_ms"] = round(duration_ms, 1)
    if input_summary is not None:
        entry["input"] = input_summary
    if output_summary is not None:
        entry["output"] = output_summary
    if error is not None:
        entry["error"] = error
    if extra:
        entry["extra"] = extra
    return entry
