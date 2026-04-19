"""Text chunker unit tests."""

from __future__ import annotations

import pytest

from config.settings import settings
from src.ingestion.chunker import chunk_text


pytestmark = pytest.mark.unit


def test_empty_returns_empty() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\t ") == []


def test_short_text_single_chunk() -> None:
    chunks = chunk_text("A short paragraph that fits in one chunk.")
    assert len(chunks) == 1
    assert "short paragraph" in chunks[0]


def test_long_text_chunks_with_overlap() -> None:
    # Build text significantly longer than chunk_size
    paragraph = "The quick brown fox jumps over the lazy dog. " * 200
    chunks = chunk_text(paragraph)

    assert len(chunks) > 1
    # Every chunk should respect the size budget (with some slack for separator logic).
    for c in chunks:
        assert len(c) <= settings.chunk_size + 100


def test_chunk_overlap_is_nontrivial() -> None:
    # Enough distinct sentences to force multiple chunks.
    text = " ".join(f"Sentence number {i} contains unique token TOKEN_{i}." for i in range(200))
    chunks = chunk_text(text)
    assert len(chunks) >= 2
    # Tail of chunk N and head of chunk N+1 should share some content (overlap).
    for i in range(len(chunks) - 1):
        tail = chunks[i][-settings.chunk_overlap :]
        head = chunks[i + 1][: settings.chunk_overlap]
        # Not exact match due to splitter rebalancing — just require some shared substring.
        shared = any(tok for tok in tail.split() if tok in head)
        assert shared, f"No overlap between chunks {i} and {i + 1}"
