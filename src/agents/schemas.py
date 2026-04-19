"""Pydantic schemas for structured LLM output — no regex parsing."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RouterOutput(BaseModel):
    route: Literal["text", "image", "cross_modal"] = Field(
        description="Question classification: text-only, image-only, or cross-modal"
    )


class CritiqueOutput(BaseModel):
    grounded: bool = Field(
        description="True iff every factual claim in the answer is supported by the retrieved context."
    )
    relevant: bool = Field(
        description="True iff the answer addresses what the user actually asked."
    )
    hallucination: bool = Field(
        description="True if the answer contains any fabricated facts, invented citations, or claims not in context."
    )
    reason: str = Field(
        description="One-sentence rationale explaining the grounded/relevant/hallucination booleans.",
        max_length=500,
    )
    rewrite_query: str = Field(
        default="",
        description=(
            "A substantively different retrieval query targeting the identified gap. "
            "Leave empty if all checks pass."
        ),
        max_length=300,
    )

    @property
    def all_pass(self) -> bool:
        return self.grounded and self.relevant and not self.hallucination
