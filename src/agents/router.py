"""Router node: classifies the question into text | image | cross_modal."""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.prompts import ROUTER_SYS
from src.agents.state import AgentState, new_trace_entry
from src.llm.ollama_client import get_router
from src.utils.logging import get_logger

log = get_logger(__name__)

_VALID = {"text", "image", "cross_modal"}


def router_node(state: AgentState) -> dict[str, Any]:
    start = time.perf_counter()
    question = state["question"]
    has_image = bool(state.get("user_image_b64"))

    user_msg = (
        f"<USER_QUESTION>\n{question}\n</USER_QUESTION>\n"
        f"ATTACHED_IMAGE_PRESENT: {'yes' if has_image else 'no'}\n"
        "Classify:"
    )

    try:
        resp = get_router().invoke(
            [SystemMessage(content=ROUTER_SYS), HumanMessage(content=user_msg)]
        )
        raw = str(resp.content).strip().lower().split()[0] if resp.content else ""
    except Exception as exc:
        log.warning("router_failed_falling_back", error=str(exc))
        raw = ""

    # Robust fallback: if the LLM output is garbage, pick a sensible default.
    if raw not in _VALID:
        raw = "cross_modal" if has_image else "text"
    # Also: if there's no image, cross_modal/image don't make sense.
    if not has_image and raw in {"image", "cross_modal"}:
        raw = "text"

    duration_ms = (time.perf_counter() - start) * 1000
    return {
        "route": raw,
        "trace": [
            new_trace_entry(
                "router",
                duration_ms=duration_ms,
                input_summary=question[:100],
                output_summary=f"route={raw}",
            )
        ],
    }
