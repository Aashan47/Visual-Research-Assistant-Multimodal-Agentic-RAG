"""End-to-end ingestion orchestration.

Entrypoints:
  - `ingest_pdf(session_id, pdf_bytes, declared_name, progress_cb=None)`
  - `ingest_user_image(session_id, image_bytes, declared_name)`

Both return a small summary dict that the UI can display.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from config.settings import settings
from src.ingestion.chunker import chunk_text
from src.ingestion.image_extractor import extract_images
from src.ingestion.pdf_partitioner import partition
from src.ingestion.summarizer import (
    ImageSummary,
    TableSummary,
    summarize_images,
    summarize_tables,
)
from src.ingestion.validators import (
    new_upload_path,
    safe_session_dir,
    validate_image_bytes,
    validate_pdf_bytes,
)
from src.retrieval.multi_vector import get_retriever
from src.utils.errors import IngestionError
from src.utils.hashing import sha256_bytes, sha256_file
from src.utils.image_io import to_base64_png
from src.utils.logging import get_logger

log = get_logger(__name__)

ProgressCallback = Callable[[str, float], None] | None
"""(stage_label, fraction_complete_0_to_1) -> None"""


@dataclass
class IngestSummary:
    session_id: str
    pdf_sha256: str
    n_text_chunks: int
    n_image_summaries: int
    n_table_summaries: int
    pages: int
    skipped: bool = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _progress(cb: ProgressCallback, stage: str, frac: float) -> None:
    if cb is not None:
        try:
            cb(stage, frac)
        except Exception as exc:
            log.warning("progress_callback_failed", error=str(exc))


def _manifest_load() -> dict:
    p = settings.manifest_path
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.warning("manifest_corrupt_resetting", path=str(p))
        return {}


def _manifest_save(data: dict) -> None:
    settings.manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _manifest_seen(session_id: str, sha: str) -> bool:
    data = _manifest_load()
    return data.get(session_id, {}).get(sha) is not None


def _manifest_record(session_id: str, sha: str, entry: dict) -> None:
    data = _manifest_load()
    data.setdefault(session_id, {})[sha] = entry
    _manifest_save(data)


def ingest_pdf(
    session_id: str,
    pdf_bytes: bytes,
    declared_name: str,
    progress_cb: ProgressCallback = None,
) -> IngestSummary:
    """Full pipeline: validate → persist → partition → images → summarize → chunk → index."""
    _progress(progress_cb, "validating", 0.02)
    validate_pdf_bytes(pdf_bytes, declared_name)
    sha = sha256_bytes(pdf_bytes)

    if _manifest_seen(session_id, sha):
        log.info("ingest_skipped_duplicate", session_id=session_id, sha=sha)
        _progress(progress_cb, "already ingested", 1.0)
        return IngestSummary(
            session_id=session_id,
            pdf_sha256=sha,
            n_text_chunks=0,
            n_image_summaries=0,
            n_table_summaries=0,
            pages=0,
            skipped=True,
        )

    # Persist to session dir
    _progress(progress_cb, "persisting upload", 0.05)
    path = new_upload_path(session_id, ".pdf")
    path.write_bytes(pdf_bytes)

    try:
        _progress(progress_cb, "parsing PDF layout", 0.2)
        partition_result = partition(path)

        image_summaries = []
        if settings.index_pdf_images:
            _progress(progress_cb, "extracting images", 0.3)
            images = extract_images(path)

            n_images = len(images)

            def _img_cb(done: int, total: int) -> None:
                # Map per-image progress into the 0.30 .. 0.70 band of the overall bar.
                frac = 0.30 + 0.40 * (done / total if total else 1.0)
                _progress(progress_cb, f"summarizing figures ({done}/{total})", frac)

            if n_images > 0:
                _progress(progress_cb, f"summarizing figures (0/{n_images})", 0.30)
                image_summaries = asyncio.run(summarize_images(images, progress_cb=_img_cb))
            else:
                _progress(progress_cb, "no figures detected", 0.70)
        else:
            _progress(progress_cb, "skipping PDF image indexing (fast mode)", 0.50)

        table_summaries = []
        if settings.index_pdf_tables and partition_result.tables:
            _progress(progress_cb, "summarizing tables", 0.72)
            table_summaries = asyncio.run(summarize_tables(partition_result.tables))

        _progress(progress_cb, "chunking text", 0.82)
        text_blob = "\n\n".join(partition_result.texts)
        text_chunks = chunk_text(text_blob)

        _progress(progress_cb, "embedding + indexing", 0.9)
        _index_all(session_id, text_chunks, image_summaries, table_summaries, sha)
    except IngestionError:
        raise
    except Exception as exc:
        log.exception("ingest_failed", session_id=session_id, error=str(exc))
        raise IngestionError(f"Ingestion failed: {exc}") from exc

    entry = {
        "declared_name": declared_name,
        "stored_path": str(path),
        "ingested_at": _now_iso(),
        "n_text_chunks": len(text_chunks),
        "n_image_summaries": len(image_summaries),
        "n_table_summaries": len(table_summaries),
    }
    _manifest_record(session_id, sha, entry)
    _progress(progress_cb, "done", 1.0)

    return IngestSummary(
        session_id=session_id,
        pdf_sha256=sha,
        n_text_chunks=len(text_chunks),
        n_image_summaries=len(image_summaries),
        n_table_summaries=len(table_summaries),
        pages=max((s.page_number for s in image_summaries), default=0),
    )


def ingest_user_image(
    session_id: str,
    image_bytes: bytes,
    declared_name: str,
) -> str:
    """Validate + summarize + index a standalone user-uploaded image.

    Returns the doc_id under which its summary was indexed.
    """
    validate_image_bytes(image_bytes, declared_name)

    safe_session_dir(session_id)  # ensure it exists
    suffix = Path(declared_name).suffix.lower() or ".png"
    path = new_upload_path(session_id, suffix)
    path.write_bytes(image_bytes)

    from PIL import Image

    img = Image.open(path).convert("RGB")
    b64 = to_base64_png(img)

    # Summarize via generator
    from langchain_core.messages import HumanMessage, SystemMessage
    from src.ingestion.summarizer import IMAGE_SUMMARY_SYS
    from src.llm.ollama_client import get_generator

    resp = get_generator().invoke(
        [
            SystemMessage(content=IMAGE_SUMMARY_SYS),
            HumanMessage(
                content=[
                    {"type": "text", "text": "Describe this user-provided figure for retrieval."},
                    {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"},
                ]
            ),
        ]
    )
    summary = str(resp.content).strip()

    summary_obj = ImageSummary(
        summary=summary,
        b64_png=b64,
        page_number=0,
        source="user_image",
        hash=sha256_file(path),
    )
    doc_id = _index_images(session_id, [summary_obj])[0]
    log.info("user_image_indexed", session_id=session_id, doc_id=doc_id)
    return doc_id


# ---------------------------------------------------------------------------
# Indexing helpers


def _index_all(
    session_id: str,
    text_chunks: list[str],
    image_summaries: list[ImageSummary],
    table_summaries: list[TableSummary],
    pdf_sha: str,
) -> None:
    _index_text(session_id, text_chunks, pdf_sha)
    _index_images(session_id, image_summaries, pdf_sha=pdf_sha)
    _index_tables(session_id, table_summaries, pdf_sha=pdf_sha)


def _index_text(session_id: str, chunks: list[str], pdf_sha: str) -> list[str]:
    if not chunks:
        return []
    retriever = get_retriever(session_id)
    doc_ids: list[str] = []
    metadatas: list[dict] = []
    for i, chunk in enumerate(chunks):
        doc_id = str(uuid.uuid4())
        doc_ids.append(doc_id)
        metadatas.append(
            {
                "doc_id": doc_id,
                "source_type": "text",
                "page_number": -1,
                "session_id": session_id,
                "pdf_sha": pdf_sha,
                "chunk_index": i,
                "ingest_ts": _now_iso(),
            }
        )
    retriever.add_text_chunks(chunks, doc_ids, metadatas)
    return doc_ids


def _index_images(
    session_id: str,
    summaries: list[ImageSummary],
    pdf_sha: str | None = None,
) -> list[str]:
    if not summaries:
        return []
    retriever = get_retriever(session_id)
    doc_ids: list[str] = []
    metadatas: list[dict] = []
    originals: list[dict] = []
    for s in summaries:
        doc_id = str(uuid.uuid4())
        doc_ids.append(doc_id)
        metadatas.append(
            {
                "doc_id": doc_id,
                "source_type": "user_image" if s.source == "user_image" else "image",
                "page_number": s.page_number,
                "session_id": session_id,
                "pdf_sha": pdf_sha or "",
                "image_source": s.source,
                "phash": s.hash,
                "ingest_ts": _now_iso(),
            }
        )
        originals.append({"kind": "image", "b64_png": s.b64_png, "summary": s.summary})
    retriever.add_image_summaries(
        summaries=[s.summary for s in summaries],
        doc_ids=doc_ids,
        metadatas=metadatas,
        originals=originals,
    )
    return doc_ids


def _index_tables(
    session_id: str,
    summaries: list[TableSummary],
    pdf_sha: str | None = None,
) -> list[str]:
    if not summaries:
        return []
    retriever = get_retriever(session_id)
    doc_ids: list[str] = []
    metadatas: list[dict] = []
    originals: list[dict] = []
    for s in summaries:
        doc_id = str(uuid.uuid4())
        doc_ids.append(doc_id)
        metadatas.append(
            {
                "doc_id": doc_id,
                "source_type": "table",
                "page_number": s.page_number if s.page_number is not None else -1,
                "session_id": session_id,
                "pdf_sha": pdf_sha or "",
                "ingest_ts": _now_iso(),
            }
        )
        originals.append({"kind": "table", "html": s.html, "summary": s.summary})
    retriever.add_table_summaries(
        summaries=[s.summary for s in summaries],
        doc_ids=doc_ids,
        metadatas=metadatas,
        originals=originals,
    )
    return doc_ids
