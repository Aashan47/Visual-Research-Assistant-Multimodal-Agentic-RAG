"""Single source of truth for the Chroma vector store client."""

from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from config.settings import settings
from src.llm.ollama_client import get_embeddings


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """Return the shared persistent Chroma collection."""
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=0.1, max=2.0),
    reraise=True,
)
def add_texts_retrying(
    store: Chroma,
    texts: list[str],
    metadatas: list[dict],
    ids: list[str],
) -> None:
    """Write with retry — Chroma's SQLite occasionally throws 'database is locked' on Windows."""
    store.add_texts(texts=texts, metadatas=metadatas, ids=ids)


def purge_session(session_id: str) -> int:
    """Delete all vectors for a session. Returns count deleted."""
    store = get_vector_store()
    collection = store._collection  # underlying chromadb Collection
    got = collection.get(where={"session_id": session_id})
    ids = got.get("ids") or []
    if ids:
        collection.delete(ids=ids)
    return len(ids)
