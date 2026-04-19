"""MultiVectorRetriever returns originals (not summaries) — the headline invariant."""

from __future__ import annotations

import uuid

import pytest

pytestmark = pytest.mark.integration


def test_retriever_returns_text_original(session_id: str) -> None:
    from src.retrieval.multi_vector import get_retriever

    retriever = get_retriever(session_id)
    doc_id = str(uuid.uuid4())

    chunk = "Accuracy on the ImageNet validation set reached 78.4% top-1."
    retriever.add_text_chunks(
        chunks=[chunk],
        doc_ids=[doc_id],
        metadatas=[
            {
                "doc_id": doc_id,
                "source_type": "text",
                "page_number": 3,
                "session_id": session_id,
                "pdf_sha": "abcd",
                "chunk_index": 0,
                "ingest_ts": "2026-01-01T00:00:00Z",
            }
        ],
    )

    items = retriever.retrieve("what accuracy did the model hit on ImageNet", k=1)
    assert items, "expected at least one retrieved item"
    item = items[0]
    assert item.doc_id == doc_id
    assert item.source_type == "text"
    assert item.original["kind"] == "text"
    assert item.original["text"] == chunk


def test_retriever_returns_table_original_html(session_id: str) -> None:
    """This is the strongest proof of multi-vector: we embed a summary, get HTML back."""
    from src.retrieval.multi_vector import get_retriever

    retriever = get_retriever(session_id)
    doc_id = str(uuid.uuid4())

    summary = "This table reports accuracy per dataset with our method scoring 78.4%."
    html = "<table><tr><th>dataset</th><th>acc</th></tr><tr><td>imagenet</td><td>78.4</td></tr></table>"

    retriever.add_table_summaries(
        summaries=[summary],
        doc_ids=[doc_id],
        metadatas=[
            {
                "doc_id": doc_id,
                "source_type": "table",
                "page_number": 5,
                "session_id": session_id,
                "pdf_sha": "abcd",
                "ingest_ts": "2026-01-01T00:00:00Z",
            }
        ],
        originals=[{"kind": "table", "html": html, "summary": summary}],
    )

    items = retriever.retrieve("accuracy per dataset", k=1)
    assert items
    assert items[0].original["html"] == html, "retriever must return the raw HTML, not the summary"
