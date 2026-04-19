"""CritiqueOutput schema unit tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.agents.schemas import CritiqueOutput, RouterOutput


pytestmark = pytest.mark.unit


class TestCritiqueOutput:
    def test_all_pass_shortcut(self) -> None:
        c = CritiqueOutput(
            grounded=True,
            relevant=True,
            hallucination=False,
            reason="looks good",
            rewrite_query="",
        )
        assert c.all_pass is True

    def test_hallucination_fails(self) -> None:
        c = CritiqueOutput(
            grounded=True,
            relevant=True,
            hallucination=True,
            reason="invented a citation",
            rewrite_query="more specific query",
        )
        assert c.all_pass is False

    def test_missing_groundedness_fails(self) -> None:
        c = CritiqueOutput(
            grounded=False,
            relevant=True,
            hallucination=False,
            reason="un-cited claim",
            rewrite_query="narrower query",
        )
        assert c.all_pass is False

    def test_rewrite_query_defaults_empty(self) -> None:
        c = CritiqueOutput(
            grounded=True, relevant=True, hallucination=False, reason="ok"
        )
        assert c.rewrite_query == ""

    def test_rejects_wrong_types(self) -> None:
        # Pydantic v2 coerces strings like "true"/"1" to bool. A dict-of-dict is un-coercible.
        with pytest.raises(ValidationError):
            CritiqueOutput(
                grounded={"nested": "dict"},  # type: ignore[arg-type]
                relevant=True,
                hallucination=False,
                reason="x",
            )


class TestRouterOutput:
    def test_accepts_valid_routes(self) -> None:
        for r in ("text", "image", "cross_modal"):
            assert RouterOutput(route=r).route == r

    def test_rejects_invalid_route(self) -> None:
        with pytest.raises(ValidationError):
            RouterOutput(route="other")  # type: ignore[arg-type]
