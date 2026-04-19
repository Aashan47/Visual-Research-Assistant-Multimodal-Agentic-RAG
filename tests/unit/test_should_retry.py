"""Conditional retry edge logic — tested in isolation without the full graph."""

from __future__ import annotations

import pytest

from config.settings import settings
from src.agents.graph import should_retry


pytestmark = pytest.mark.unit


def _make_state(
    *,
    grounded: bool = True,
    relevant: bool = True,
    hallucination: bool = False,
    retry_count: int = 0,
    critique_present: bool = True,
) -> dict:
    state: dict = {"retry_count": retry_count}
    if critique_present:
        state["critique"] = {
            "grounded": grounded,
            "relevant": relevant,
            "hallucination": hallucination,
        }
    return state


def test_all_pass_returns_done() -> None:
    assert should_retry(_make_state()) == "done"


def test_hallucination_triggers_retry_when_under_cap() -> None:
    assert (
        should_retry(_make_state(hallucination=True, retry_count=0))
        == "retry"
    )


def test_groundedness_failure_alone_does_not_retry() -> None:
    """Small critic models false-flag groundedness too often to be worth a retry."""
    assert (
        should_retry(_make_state(grounded=False, retry_count=0))
        == "done"
    )


def test_irrelevance_alone_does_not_retry() -> None:
    assert (
        should_retry(_make_state(relevant=False, retry_count=0))
        == "done"
    )


def test_retry_cap_exits_with_done() -> None:
    assert (
        should_retry(
            _make_state(hallucination=True, retry_count=settings.max_retries)
        )
        == "done"
    )


def test_missing_critique_is_defensively_done() -> None:
    assert should_retry(_make_state(critique_present=False)) == "done"


def test_retry_cap_is_inclusive() -> None:
    """At retry_count == max_retries we must exit, not loop one more time."""
    assert (
        should_retry(
            _make_state(hallucination=True, retry_count=settings.max_retries + 5)
        )
        == "done"
    )
