"""Extract images from a PDF using PyMuPDF.

Two complementary strategies:
  1. Embedded bitmaps (page.get_images) — catches inline figures.
  2. Full-page rasters — catches charts rendered as vector graphics that have no
     embedded bitmap. We raster every page at 150 DPI and let the VL model decide
     whether there's a figure worth summarizing.

Perceptual-hash dedupe prevents the same chart from being summarized twice.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from src.utils.hashing import phash
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ExtractedImage:
    image: Image.Image
    page_number: int
    source: str  # "embedded" | "page_raster"
    hash: str


def extract_images(
    pdf_path: str | Path,
    render_dpi: int = 100,
    max_images: int = 4,
) -> list[ExtractedImage]:
    """Extract images from a PDF — embedded bitmaps first, page rasters only as fallback.

    For pages that already yielded an embedded bitmap we skip the page raster,
    since VL summarization is the bottleneck and adding redundant coverage on
    the same page isn't worth the latency. `max_images` caps the total count
    so a long paper doesn't blow the ingestion budget.
    """
    path = Path(pdf_path)
    log.info("extract_images_start", path=str(path), dpi=render_dpi, max_images=max_images)
    out: list[ExtractedImage] = []
    seen: set[str] = set()
    pages_with_embedded: set[int] = set()

    doc = fitz.open(str(path))
    try:
        # Pass 1: embedded bitmaps
        for page_index in range(doc.page_count):
            if len(out) >= max_images:
                break
            page = doc.load_page(page_index)
            page_no = page_index + 1

            for img_info in page.get_images(full=True):
                if len(out) >= max_images:
                    break
                xref = img_info[0]
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n - pix.alpha >= 4:  # CMYK → RGB
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                except Exception as exc:
                    log.warning("embedded_image_extract_failed", page=page_no, error=str(exc))
                    continue

                h = phash(img)
                if h in seen:
                    continue
                seen.add(h)
                pages_with_embedded.add(page_no)
                out.append(ExtractedImage(img, page_no, "embedded", h))

        # Pass 2: page rasters only for pages that yielded no embedded bitmap
        # (catches vector-drawn charts).
        for page_index in range(doc.page_count):
            if len(out) >= max_images:
                break
            page_no = page_index + 1
            if page_no in pages_with_embedded:
                continue
            page = doc.load_page(page_index)
            try:
                mat = fitz.Matrix(render_dpi / 72, render_dpi / 72)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                page_img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                h = phash(page_img)
                if h in seen:
                    continue
                seen.add(h)
                out.append(ExtractedImage(page_img, page_no, "page_raster", h))
            except Exception as exc:
                log.warning("page_raster_failed", page=page_no, error=str(exc))
    finally:
        doc.close()

    log.info(
        "extract_images_done",
        path=str(path),
        n_embedded=sum(1 for e in out if e.source == "embedded"),
        n_rasters=sum(1 for e in out if e.source == "page_raster"),
    )
    return out
