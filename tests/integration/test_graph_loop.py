"""Inject a forced-failure critique and assert the retry path actually executes."""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.integration


def _always_fail_critique(state: Any) -> dict[str, Any]:
    """Drop-in critique_node replacement that always flags the answer as hallucinating."""
    new_retry = state.get("retry_count", 0) + 1
    return {
        "critique": {
            "grounded": False,
            "relevant": False,
            "hallucination": True,
            "reason": "injected failure for test",
            "rewrite_query": "alternate query",
        },
        "retry_count": new_retry,
        "rewritten_query": "alternate query",
        "trace": [{"node": "critique", "output": f"forced retry={new_retry}"}],
    }


def test_retry_path_runs_until_cap(monkeypatch: pytest.MonkeyPatch, session_id: str) -> None:
    """Verify the retriever is invoked `max_retries + 1` times when critique always fails."""
    from config.settings import settings
    from src.agents import critique as critique_module
    from src.agents import graph as graph_module

    monkeypatch.setattr(critique_module, "critique_node", _always_fail_critique)

    # Reset the cached compiled app so it picks up the patched node.
    graph_module.get_app.cache_clear()

    retriever_calls: list[str] = []
    original_retriever = graph_module.retriever_node

    def _counting_retriever(state: Any) -> dict[str, Any]:
        retriever_calls.append(state.get("rewritten_query") or state["question"])
        return original_retriever(state)

    monkeypatch.setattr(graph_module, "retriever_node", _counting_retriever)
    graph_module.get_app.cache_clear()

    # Also patch generator so we don't need a real model just for this test.
    def _stub_generator(state: Any) -> dict[str, Any]:
        return {
            "answer": "Stubbed answer with no citations.",
            "citations": [],
            "trace": [{"node": "generator", "output": "stub"}],
        }

    monkeypatch.setattr(graph_module, "generator_node", _stub_generator)
    graph_module.get_app.cache_clear()

    # Likewise router — don't require LLM.
    def _stub_router(state: Any) -> dict[str, Any]:
        return {"route": "text", "trace": [{"node": "router", "output": "text"}]}

    monkeypatch.setattr(graph_module, "router_node", _stub_router)
    graph_module.get_app.cache_clear()

    app = graph_module.get_app()
    config = {"configurable": {"thread_id": session_id}}
    state = graph_module.initial_state(session_id, "what does the paper say about X?")

    final = None
    for event in app.stream(state, config=config, stream_mode="values"):
        final = event

    assert final is not None
    assert final["degraded"] is True
    assert len(retriever_calls) == settings.max_retries + 1
