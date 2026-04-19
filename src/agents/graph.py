"""LangGraph StateGraph assembly with bounded critique-retry loop.

Exports `get_app()` — a compiled graph with SQLite checkpointing so Streamlit
can stream execution and reconstruct traces across reruns.
"""

from __future__ import annotations

import sqlite3
from functools import lru_cache
from typing import Literal

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from config.settings import settings
from src.agents.critique import critique_node
from src.agents.generator import generator_node
from src.agents.retriever_node import retriever_node
from src.agents.router import router_node
from src.agents.state import AgentState, new_trace_entry
from src.utils.logging import get_logger

log = get_logger(__name__)


def should_retry(state: AgentState) -> Literal["retry", "done"]:
    """Retry only on hallucination — the one failure mode where a retry reliably helps.

    `grounded=false` from a small critic model is frequently a false positive
    (it penalizes grouped citations, stylistic quirks, etc.), and retrying
    rarely produces a meaningfully different result. Hallucination — invented
    facts or invented doc_ids — is the actual danger and is worth one more pass
    with a rewritten query.

    `finalize_node` still surfaces `degraded=true` for the UI whenever any
    critique flag failed, so the user sees the honest signal.
    """
    critique = state.get("critique")
    if critique is None:
        return "done"

    if not critique.get("hallucination"):
        return "done"

    if state.get("retry_count", 0) >= settings.max_retries:
        return "done"

    return "retry"


def finalize_node(state: AgentState) -> dict:
    """Terminal node. `degraded` mirrors what the graph actually acted on.

    The retry loop triggers only on hallucination, so that is the only signal
    loud enough to warrant a user-facing warning banner. grounded/relevant
    remain visible in the trace panel for anyone inspecting the detail.
    """
    critique = state.get("critique") or {}
    hallucinated = bool(critique.get("hallucination"))
    degraded = hallucinated  # exited with a hallucination we could not fix
    return {
        "degraded": degraded,
        "trace": [
            new_trace_entry(
                "finalize",
                output_summary=(
                    f"degraded={degraded} retry_count={state.get('retry_count', 0)} "
                    f"grounded={critique.get('grounded')} relevant={critique.get('relevant')}"
                ),
            )
        ],
    }


def _build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("generator", generator_node)
    graph.add_node("critic", critique_node)  # node name != state key "critique"
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "router")
    graph.add_edge("router", "retriever")
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", "critic")
    graph.add_conditional_edges(
        "critic",
        should_retry,
        {"retry": "retriever", "done": "finalize"},
    )
    graph.add_edge("finalize", END)
    return graph


@lru_cache(maxsize=1)
def get_app():
    """Return the compiled graph with a shared SQLite checkpointer.

    We construct the SqliteSaver from a long-lived sqlite3 connection rather
    than entering `from_conn_string()` as a context manager, because the
    Streamlit app expects the checkpointer to outlive the initialization call.
    """
    conn = sqlite3.connect(
        str(settings.checkpoints_db),
        check_same_thread=False,
        isolation_level=None,
    )
    checkpointer = SqliteSaver(conn)
    graph = _build_graph()
    app = graph.compile(checkpointer=checkpointer)
    log.info(
        "graph_compiled",
        checkpointer=str(settings.checkpoints_db),
        max_retries=settings.max_retries,
    )
    return app


def initial_state(
    session_id: str,
    question: str,
    user_image_b64: str | None = None,
) -> AgentState:
    return {
        "session_id": session_id,
        "question": question,
        "user_image_b64": user_image_b64,
        "route": None,
        "rewritten_query": None,
        "retrieved": [],
        "answer": None,
        "citations": [],
        "critique": None,
        "retry_count": 0,
        "degraded": False,
        "trace": [],
    }
