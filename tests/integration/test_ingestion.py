"""End-to-end ingestion smoke test. Requires Ollama + models pulled."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


SAMPLE_PDF = Path(__file__).resolve().parents[2] / "examples" / "sample_paper.pdf"


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample_paper.pdf missing")
def test_ingest_sample_pdf_indexes_multiple_types(session_id: str) -> None:
    from src.ingestion.pipeline import ingest_pdf
    from src.retrieval.vector_store import get_vector_store

    result = ingest_pdf(session_id, SAMPLE_PDF.read_bytes(), SAMPLE_PDF.name)

    assert not result.skipped
    assert result.n_text_chunks > 0, "expected at least one text chunk"
    # The test asset is chosen to have both a table and a figure.
    assert result.n_image_summaries + result.n_table_summaries > 0

    store = get_vector_store()
    count = store._collection.count()
    assert count >= result.n_text_chunks
