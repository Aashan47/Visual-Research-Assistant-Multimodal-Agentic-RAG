"""Summarize images (VL model) and tables (text model) for multi-vector retrieval.

These summaries are what we embed and index in Chroma — the originals live in the
docstore and are returned verbatim at retrieval time. This is the MultiVectorRetriever
pattern.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable

from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from src.ingestion.image_extractor import ExtractedImage
from src.llm.ollama_client import get_critique, get_generator
from src.utils.errors import LLMError
from src.utils.image_io import to_base64_png, to_data_uri
from src.utils.logging import get_logger

ProgressCallback = Callable[[int, int], None] | None
"""(done, total) -> None"""

log = get_logger(__name__)


IMAGE_SUMMARY_SYS = (
    "You are an expert technical reader. Describe the provided figure in 100-150 words "
    "so that a downstream retriever can match a user's natural-language question to it. "
    "Cover: what kind of figure it is (chart, diagram, photo, table), axes/legend if any, "
    "variables, visible trends, and any annotations. Be factual — do not speculate beyond "
    "what the image shows. Output plain prose, no markdown."
)

TABLE_SUMMARY_SYS = (
    "You are a retrieval-oriented summarizer. Given a table (HTML or plain text), write a "
    "concise 80-120 word natural-language summary describing its columns, rows, and the "
    "key quantitative claims it supports. A reader of your summary should be able to "
    "decide whether the original table answers a given question. Output plain prose."
)


@dataclass
class ImageSummary:
    summary: str
    b64_png: str
    page_number: int
    source: str
    hash: str


@dataclass
class TableSummary:
    summary: str
    html: str
    page_number: int | None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
async def _summarize_image_async(img: Image.Image) -> tuple[str, str]:
    b64 = to_base64_png(img)
    generator = get_generator()
    msg = HumanMessage(
        content=[
            {"type": "text", "text": "Describe this figure for retrieval."},
            {"type": "image_url", "image_url": to_data_uri(b64)},
        ]
    )
    try:
        resp = await generator.ainvoke([SystemMessage(content=IMAGE_SUMMARY_SYS), msg])
    except Exception as exc:
        raise LLMError(f"image summarization failed: {exc}") from exc
    return str(resp.content).strip(), b64


async def summarize_images(
    images: list[ExtractedImage],
    progress_cb: ProgressCallback = None,
) -> list[ImageSummary]:
    if not images:
        return []

    total = len(images)
    done = 0

    sem = asyncio.Semaphore(settings.summary_concurrency)

    async def _task(item: ExtractedImage) -> ImageSummary | None:
        nonlocal done
        async with sem:
            try:
                summary, b64 = await _summarize_image_async(item.image)
            except LLMError as exc:
                log.warning(
                    "image_summary_skipped",
                    page=item.page_number,
                    source=item.source,
                    error=str(exc),
                )
                result: ImageSummary | None = None
            else:
                result = ImageSummary(
                    summary=summary,
                    b64_png=b64,
                    page_number=item.page_number,
                    source=item.source,
                    hash=item.hash,
                )
            done += 1
            if progress_cb is not None:
                try:
                    progress_cb(done, total)
                except Exception as exc:
                    log.warning("image_progress_cb_failed", error=str(exc))
            return result

    results = await asyncio.gather(*[_task(i) for i in images])
    return [r for r in results if r is not None]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
async def _summarize_table_async(html: str) -> str:
    critique = get_critique()
    try:
        resp = await critique.ainvoke(
            [SystemMessage(content=TABLE_SUMMARY_SYS), HumanMessage(content=html)]
        )
    except Exception as exc:
        raise LLMError(f"table summarization failed: {exc}") from exc
    return str(resp.content).strip()


async def summarize_tables(tables: list[dict]) -> list[TableSummary]:
    if not tables:
        return []

    sem = asyncio.Semaphore(settings.summary_concurrency)

    async def _task(t: dict) -> TableSummary | None:
        async with sem:
            try:
                summary = await _summarize_table_async(t["html"])
            except LLMError as exc:
                log.warning("table_summary_skipped", page=t.get("page_number"), error=str(exc))
                return None
            return TableSummary(summary=summary, html=t["html"], page_number=t.get("page_number"))

    results = await asyncio.gather(*[_task(t) for t in tables])
    return [r for r in results if r is not None]
