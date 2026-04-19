"""End-to-end scenarios against a live Ollama. Slow; run manually or nightly."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e


ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PDF = ROOT / "examples" / "sample_paper.pdf"
SAMPLE_IMAGE = ROOT / "examples" / "sample_chart.png"


@pytest.fixture(scope="module")
def ingested_session() -> str:
    import uuid

    from src.ingestion.pipeline import ingest_pdf, ingest_user_image

    if not SAMPLE_PDF.exists():
        pytest.skip("examples/sample_paper.pdf missing")

    sid = uuid.uuid4().hex
    ingest_pdf(sid, SAMPLE_PDF.read_bytes(), SAMPLE_PDF.name)
    if SAMPLE_IMAGE.exists():
        ingest_user_image(sid, SAMPLE_IMAGE.read_bytes(), SAMPLE_IMAGE.name)
    return sid


def _run(session_id: str, question: str, user_image_b64: str | None = None) -> dict:
    from src.agents.graph import get_app, initial_state

    app = get_app()
    config = {"configurable": {"thread_id": f"{session_id}-{hash(question)}"}}
    state = initial_state(session_id, question, user_image_b64)
    final = None
    for event in app.stream(state, config=config, stream_mode="values"):
        final = event
    assert final is not None
    return final


def test_text_only_question(ingested_session: str) -> None:
    result = _run(
        ingested_session,
        "What dataset does the paper use for evaluation, and what are the reported accuracy numbers?",
    )
    assert result["route"] == "text"
    assert result["answer"]
    assert not result["degraded"], "text question should pass critique first try"


def test_adversarial_question_degrades(ingested_session: str) -> None:
    from config.settings import settings

    result = _run(
        ingested_session,
        "According to the paper, who is the CEO of OpenAI?",
    )
    # The paper cannot answer this; retries should exhaust.
    assert result["retry_count"] == settings.max_retries + 1
    assert result["degraded"] is True
