"""Partition a PDF into text blocks and tables using PyMuPDF.

PyMuPDF alone handles our needs here — native text extraction plus `find_tables()`
for layout-detected tables. This keeps us off Unstructured's heavy inference stack
(detectron2, torch) which is overkill for a local demo and fragile on Python 3.13.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.errors import IngestionError
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class PartitionResult:
    texts: list[str] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)  # {html, page_number}
    raw_elements: list[Any] = field(default_factory=list)  # kept for API parity


def _rows_to_html(rows: list[list[str | None]]) -> str:
    def cell(v: str | None) -> str:
        return (v or "").strip().replace("\n", " ")

    parts = ["<table>"]
    if rows:
        parts.append("<tr>" + "".join(f"<th>{cell(c)}</th>" for c in rows[0]) + "</tr>")
        for row in rows[1:]:
            parts.append("<tr>" + "".join(f"<td>{cell(c)}</td>" for c in row) + "</tr>")
    parts.append("</table>")
    return "".join(parts)


def _extract_tables(page: fitz.Page) -> list[dict[str, Any]]:
    """Best-effort table extraction via PyMuPDF's find_tables API."""
    out: list[dict[str, Any]] = []
    try:
        finder = page.find_tables()
    except Exception as exc:
        log.debug("find_tables_unavailable", page=page.number + 1, error=str(exc))
        return out

    # PyMuPDF API variation: .tables attr in some versions, iterable in others.
    tables = getattr(finder, "tables", None) or list(finder)
    for tbl in tables:
        try:
            rows = tbl.extract()
        except Exception as exc:
            log.debug("table_extract_failed", page=page.number + 1, error=str(exc))
            continue
        if not rows:
            continue
        html = _rows_to_html(rows)
        out.append({"html": html, "page_number": page.number + 1})
    return out


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def partition(path: str | Path) -> PartitionResult:
    path = Path(path)
    log.info("partition_pdf_start", path=str(path))
    result = PartitionResult()

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        raise IngestionError(f"could not open {path.name}: {exc}") from exc

    try:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            page_no = page_index + 1

            # 1) Extract tables first so we can later avoid double-counting their text.
            page_tables = _extract_tables(page)
            result.tables.extend(page_tables)

            # 2) Extract narrative text (full page text — acceptable for a retrieval demo;
            #    we still split into chunks downstream).
            text = page.get_text("text") or ""
            text = text.strip()
            if text:
                result.texts.append(text)
    finally:
        doc.close()

    log.info(
        "partition_pdf_done",
        path=str(path),
        n_text_blocks=len(result.texts),
        n_tables=len(result.tables),
    )
    return result
