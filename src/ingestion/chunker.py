"""Text chunking."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings


def build_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
    )


def chunk_text(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    return build_splitter().split_text(text)
