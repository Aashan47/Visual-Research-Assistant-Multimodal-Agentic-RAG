"""Multi-vector retrieval: embed summaries, return originals.

This is the project's headline feature. The vector store holds summaries
(which are easier for an embedding model to match against queries) but the
retriever returns the *originals* (raw text chunks, raw table HTML, raw
base64 images) so the generator has the full context to reason with.

Session isolation: every vector carries `session_id` metadata; every search
applies a metadata filter so cross-session leakage is architecturally
impossible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.documents import Document

from config.settings import settings
from src.retrieval.docstore import get_docstore
from src.retrieval.vector_store import add_texts_retrying, get_vector_store
from src.utils.logging import get_logger

log = get_logger(__name__)

SourceType = Literal["text", "image", "table", "user_image"]


@dataclass
class RetrievedItem:
    """What the retriever returns to downstream nodes."""

    doc_id: str
    source_type: SourceType
    page_number: int
    summary: str
    original: dict[str, Any]  # {"kind": ..., "text"|"html"|"b64_png": ...}
    score: float | None = None

    def to_langchain_document(self) -> Document:
        """Convert to a LangChain Document — `page_content` is the most useful text."""
        if self.source_type == "text":
            content = self.original.get("text", self.summary)
        elif self.source_type == "table":
            content = f"[TABLE]\n{self.original.get('html', self.summary)}"
        else:  # image or user_image
            content = f"[IMAGE SUMMARY] {self.summary}"
        return Document(
            page_content=content,
            metadata={
                "doc_id": self.doc_id,
                "source_type": self.source_type,
                "page_number": self.page_number,
            },
        )


class MultiVectorRetriever:
    """Session-scoped multi-vector retriever."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.vector_store = get_vector_store()
        self.docstore = get_docstore()

    # ---------------- Writes ----------------

    def add_text_chunks(
        self,
        chunks: list[str],
        doc_ids: list[str],
        metadatas: list[dict],
    ) -> None:
        # For text chunks, the "summary" we embed IS the chunk itself — no need
        # to LLM-summarize, they're already short. Originals stored verbatim.
        add_texts_retrying(self.vector_store, chunks, metadatas, doc_ids)
        self.docstore.mset(
            [(doc_id, {"kind": "text", "text": chunk}) for doc_id, chunk in zip(doc_ids, chunks)]
        )
        log.info("indexed_text_chunks", session_id=self.session_id, n=len(chunks))

    def add_image_summaries(
        self,
        summaries: list[str],
        doc_ids: list[str],
        metadatas: list[dict],
        originals: list[dict],
    ) -> None:
        add_texts_retrying(self.vector_store, summaries, metadatas, doc_ids)
        self.docstore.mset(list(zip(doc_ids, originals)))
        log.info("indexed_image_summaries", session_id=self.session_id, n=len(summaries))

    def add_table_summaries(
        self,
        summaries: list[str],
        doc_ids: list[str],
        metadatas: list[dict],
        originals: list[dict],
    ) -> None:
        add_texts_retrying(self.vector_store, summaries, metadatas, doc_ids)
        self.docstore.mset(list(zip(doc_ids, originals)))
        log.info("indexed_table_summaries", session_id=self.session_id, n=len(summaries))

    # ---------------- Reads ----------------

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        source_types: list[SourceType] | None = None,
    ) -> list[RetrievedItem]:
        """Similarity search with session isolation and optional source_type filter."""
        k = k or settings.retrieval_k

        where: dict[str, Any] = {"session_id": self.session_id}
        if source_types:
            # Chroma metadata filter supports $in
            where = {
                "$and": [
                    {"session_id": self.session_id},
                    {"source_type": {"$in": list(source_types)}},
                ]
            }

        results = self.vector_store.similarity_search_with_score(query, k=k, filter=where)

        if not results:
            log.info(
                "retrieve_empty", session_id=self.session_id, query_preview=query[:80]
            )
            return []

        doc_ids = [doc.metadata["doc_id"] for doc, _ in results]
        originals = self.docstore.mget(doc_ids)

        items: list[RetrievedItem] = []
        for (doc, score), original in zip(results, originals):
            if original is None:
                log.warning("docstore_missing", doc_id=doc.metadata["doc_id"])
                continue
            items.append(
                RetrievedItem(
                    doc_id=doc.metadata["doc_id"],
                    source_type=doc.metadata["source_type"],
                    page_number=int(doc.metadata.get("page_number", -1)),
                    summary=doc.page_content,
                    original=original,
                    score=float(score),
                )
            )
        log.info(
            "retrieve_done",
            session_id=self.session_id,
            query_preview=query[:80],
            returned=len(items),
        )
        return items


_retrievers: dict[str, MultiVectorRetriever] = {}


def get_retriever(session_id: str) -> MultiVectorRetriever:
    """Per-session retriever — cheap to construct but cached for reuse."""
    if session_id not in _retrievers:
        _retrievers[session_id] = MultiVectorRetriever(session_id)
    return _retrievers[session_id]
