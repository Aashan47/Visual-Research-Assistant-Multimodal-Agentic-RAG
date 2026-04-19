"""Critique node: independent evaluator with structured output.

Runs on a different model family from the generator to avoid self-agreement bias.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError as PydanticValidationError

from src.agents.prompts import CRITIQUE_SYS, format_context_block
from src.agents.schemas import CritiqueOutput
from src.agents.state import AgentState, new_trace_entry
from src.llm.ollama_client import get_critique
from src.utils.logging import get_logger

log = get_logger(__name__)


def _fallback_critique(reason: str) -> dict[str, Any]:
    """When the critic itself fails, return a defensively-correct non-retryable verdict.

    We return all_pass-equivalent with `degraded=False` flags but explicit reason,
    so the graph exits the loop rather than spinning. The answer is still surfaced —
    just without confident critique. This is preferable to infinite-looping on critic failure.
    """
    return {
        "grounded": True,
        "relevant": True,
        "hallucination": False,
        "reason": f"critique unavailable ({reason})",
        "rewrite_query": "",
    }


def critique_node(state: AgentState) -> dict[str, Any]:
    start = time.perf_counter()
    question = state["question"]
    answer = state.get("answer") or ""
    retrieved = state.get("retrieved") or []

    prompt = (
        f"<QUESTION>\n{question}\n</QUESTION>\n\n"
        f"{format_context_block(retrieved)}\n\n"
        f"<ANSWER>\n{answer}\n</ANSWER>\n\n"
        "Produce the structured evaluation."
    )

    try:
        llm = get_critique().with_structured_output(CritiqueOutput)
        verdict: CritiqueOutput = llm.invoke(
            [SystemMessage(content=CRITIQUE_SYS), HumanMessage(content=prompt)]
        )
        critique_dict = verdict.model_dump()
        error_note: str | None = None
    except PydanticValidationError as exc:
        log.warning("critique_schema_violation", error=str(exc))
        critique_dict = _fallback_critique("schema violation")
        error_note = "schema violation"
    except Exception as exc:
        log.warning("critique_llm_failed", error=str(exc))
        critique_dict = _fallback_critique("LLM error")
        error_note = str(exc)

    duration_ms = (time.perf_counter() - start) * 1000
    new_retry_count = state.get("retry_count", 0) + 1

    return {
        "critique": critique_dict,
        "retry_count": new_retry_count,
        # On failed critique, stage the rewrite for the next retriever pass.
        "rewritten_query": critique_dict.get("rewrite_query") or None,
        "trace": [
            new_trace_entry(
                "critique",
                duration_ms=duration_ms,
                input_summary=f"answer={len(answer)} chars, context={len(retrieved)} items",
                output_summary=(
                    f"grounded={critique_dict['grounded']} "
                    f"relevant={critique_dict['relevant']} "
                    f"hallucination={critique_dict['hallucination']} "
                    f"retry={new_retry_count}"
                ),
                error=error_note,
            )
        ],
    }
