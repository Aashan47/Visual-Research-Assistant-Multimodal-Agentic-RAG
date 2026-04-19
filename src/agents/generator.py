"""Generator node: answers the question using retrieved text + attached images."""

from __future__ import annotations

import re
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.prompts import GENERATOR_SYS, format_context_block
from src.agents.state import AgentState, new_trace_entry
from src.llm.ollama_client import get_generator, get_text_generator
from src.utils.logging import get_logger

log = get_logger(__name__)

# Match any UUID4 regardless of surrounding decoration — the generator sometimes
# writes [abc-def], sometimes [doc_id="abc-def"], sometimes (abc-def). All valid.
_DOC_ID_CITATION_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)


def _extract_citations(text: str) -> list[str]:
    """Return unique doc_ids cited in the answer, in order of first appearance."""
    seen: set[str] = set()
    ordered: list[str] = []
    for m in _DOC_ID_CITATION_RE.finditer(text):
        d = m.group(0)
        if d not in seen:
            seen.add(d)
            ordered.append(d)
    return ordered


def _build_message_and_pick_model(state: AgentState) -> tuple[HumanMessage, bool]:
    """Build the multimodal user message and decide whether we need a vision model.

    Image-attach policy (driven by the router's route):
      - route=text       : never attach the user image, even if one was uploaded.
                           The question is about the document, the image is a
                           distractor — keep the prompt clean and use the fast
                           text model.
      - route=image      : attach the user image; document retrieval is still
                           passed along but the question is image-focused.
      - route=cross_modal: attach the user image and the document's own figures.

    Returns (message, needs_vision).
    """
    retrieved = state.get("retrieved") or []
    question = state["question"]
    user_img = state.get("user_image_b64")
    route = state.get("route") or "text"

    attach_user_image = bool(user_img) and route in ("image", "cross_modal")
    attach_doc_figures = route in ("image", "cross_modal")

    context_block = format_context_block(retrieved)

    preamble = context_block + "\n\n"
    if attach_user_image:
        preamble += (
            "<USER_IMAGE>\n"
            "The user attached one supplementary image, shown below. It is NOT part of "
            "the document. Reference it because the question is about it.\n"
            "</USER_IMAGE>\n\n"
        )
    preamble += f"<USER_QUESTION>\n{question}\n</USER_QUESTION>\n\nAnswer now."

    image_parts: list[dict[str, Any]] = []
    if attach_doc_figures:
        for item in retrieved:
            if item["source_type"] in ("image", "user_image"):
                b64 = item["original"].get("b64_png")
                if b64:
                    image_parts.append(
                        {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"}
                    )
    if attach_user_image:
        image_parts.append(
            {"type": "image_url", "image_url": f"data:image/png;base64,{user_img}"}
        )

    needs_vision = bool(image_parts)
    if needs_vision:
        content: list[dict[str, Any]] | str = [
            {"type": "text", "text": preamble},
            *image_parts,
        ]
        return HumanMessage(content=content), True
    return HumanMessage(content=preamble), False


def generator_node(state: AgentState) -> dict[str, Any]:
    start = time.perf_counter()

    message, needs_vision = _build_message_and_pick_model(state)
    model = get_generator() if needs_vision else get_text_generator()
    model_tag = "vision" if needs_vision else "text-only"

    try:
        resp = model.invoke([SystemMessage(content=GENERATOR_SYS), message])
        answer = str(resp.content).strip()
        citations = _extract_citations(answer)
        duration_ms = (time.perf_counter() - start) * 1000
        return {
            "answer": answer,
            "citations": citations,
            "trace": [
                new_trace_entry(
                    "generator",
                    duration_ms=duration_ms,
                    input_summary=(
                        f"{model_tag}, "
                        f"question + {len(state.get('retrieved') or [])} context items"
                    ),
                    output_summary=f"{len(answer)} chars, {len(citations)} citations",
                )
            ],
        }
    except Exception as exc:
        log.exception("generator_failed", error=str(exc))
        duration_ms = (time.perf_counter() - start) * 1000
        return {
            "answer": "I could not generate an answer: the generator model failed.",
            "citations": [],
            "trace": [
                new_trace_entry(
                    "generator",
                    duration_ms=duration_ms,
                    error=str(exc),
                )
            ],
        }
