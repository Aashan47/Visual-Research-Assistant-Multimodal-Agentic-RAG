"""AgentState + trace reducer unit tests."""

from __future__ import annotations

import pytest

from src.agents.state import new_trace_entry


pytestmark = pytest.mark.unit


def test_new_trace_entry_minimal() -> None:
    entry = new_trace_entry("router")
    assert entry == {"node": "router"}


def test_new_trace_entry_full() -> None:
    entry = new_trace_entry(
        "generator",
        duration_ms=42.357,
        input_summary="query=foo",
        output_summary="234 chars",
        error=None,
        extra={"tokens": 128},
    )
    assert entry["node"] == "generator"
    assert entry["duration_ms"] == 42.4
    assert entry["input"] == "query=foo"
    assert entry["output"] == "234 chars"
    assert entry["extra"] == {"tokens": 128}
    assert "error" not in entry


def test_trace_entry_records_error() -> None:
    entry = new_trace_entry("critique", error="LLM timeout")
    assert entry["error"] == "LLM timeout"
