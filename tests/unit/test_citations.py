"""Generator citation extraction unit tests."""

from __future__ import annotations

import pytest

from src.agents.generator import _extract_citations


pytestmark = pytest.mark.unit


def test_extracts_uuid_citations_in_order() -> None:
    text = (
        "Alpha [12345678-1234-1234-1234-123456789abc]. "
        "Beta [abcdef01-2345-6789-abcd-ef0123456789]. "
        "Alpha again [12345678-1234-1234-1234-123456789abc]."
    )
    cits = _extract_citations(text)
    assert cits == [
        "12345678-1234-1234-1234-123456789abc",
        "abcdef01-2345-6789-abcd-ef0123456789",
    ]


def test_extracts_decorated_citations() -> None:
    """Models often add [doc_id="..."] or (...) around the UUID — accept them."""
    text = (
        'First claim [doc_id="12345678-1234-1234-1234-123456789abc"]. '
        "Second claim (abcdef01-2345-6789-abcd-ef0123456789, page 3)."
    )
    cits = _extract_citations(text)
    assert cits == [
        "12345678-1234-1234-1234-123456789abc",
        "abcdef01-2345-6789-abcd-ef0123456789",
    ]


def test_no_citations_returns_empty() -> None:
    assert _extract_citations("No doc ids here.") == []


def test_ignores_non_uuid_text() -> None:
    assert _extract_citations("Some claim [not-a-uuid] and [12345] and [text].") == []
