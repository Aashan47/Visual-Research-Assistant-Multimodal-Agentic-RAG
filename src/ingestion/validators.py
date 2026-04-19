"""Upload validation: MIME sniffing, size/page caps, filename sanitization, path-traversal guard."""

from __future__ import annotations

import re
import uuid
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, UnidentifiedImageError

from config.settings import settings
from src.utils.errors import ValidationError
from src.utils.logging import get_logger

log = get_logger(__name__)

_PDF_MAGIC = b"%PDF-"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"

_SAFE_ID_RE = re.compile(r"^[0-9a-f]{32}$")  # uuid4 hex
_DOC_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


def _read_head(data: bytes, n: int = 16) -> bytes:
    return data[:n]


def validate_pdf_bytes(data: bytes, declared_name: str) -> None:
    """Raise ValidationError if bytes are not a PDF, too large, or too many pages."""
    size_mb = len(data) / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise ValidationError(
            f"PDF '{declared_name}' is {size_mb:.1f} MB, limit is {settings.max_upload_mb} MB"
        )

    if not _read_head(data, 5).startswith(_PDF_MAGIC):
        raise ValidationError(
            f"File '{declared_name}' does not have a valid PDF header — refusing to persist."
        )

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # fitz raises bare Exception on bad PDFs
        raise ValidationError(f"Could not parse PDF '{declared_name}': {exc}") from exc

    try:
        n_pages = doc.page_count
    finally:
        doc.close()

    if n_pages > settings.max_pdf_pages:
        raise ValidationError(
            f"PDF '{declared_name}' has {n_pages} pages, limit is {settings.max_pdf_pages}"
        )
    if n_pages == 0:
        raise ValidationError(f"PDF '{declared_name}' has no pages")


def validate_image_bytes(data: bytes, declared_name: str) -> None:
    """Raise ValidationError if bytes are not a PNG/JPEG or too large."""
    size_mb = len(data) / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise ValidationError(
            f"Image '{declared_name}' is {size_mb:.1f} MB, limit is {settings.max_upload_mb} MB"
        )

    head = _read_head(data, 8)
    if not (head.startswith(_PNG_MAGIC) or head.startswith(_JPEG_MAGIC)):
        raise ValidationError(
            f"File '{declared_name}' is not a valid PNG or JPEG (magic bytes mismatch)"
        )

    # Round-trip through Pillow to catch truncated/corrupt files.
    try:
        from io import BytesIO

        Image.open(BytesIO(data)).verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValidationError(f"Image '{declared_name}' failed to decode: {exc}") from exc


def safe_session_dir(session_id: str) -> Path:
    """Return the uploads dir for this session, with path-traversal guard."""
    if not _SAFE_ID_RE.match(session_id):
        raise ValidationError("session_id must be a 32-char uuid4 hex string")

    root = settings.uploads_dir.resolve()
    target = (root / session_id).resolve()

    # Python 3.11+: is_relative_to
    if not target.is_relative_to(root):
        raise ValidationError("Session directory escaped the uploads root")

    target.mkdir(parents=True, exist_ok=True)
    return target


def new_upload_path(session_id: str, suffix: str) -> Path:
    """Generate a fresh, collision-free upload path under the session dir."""
    if not suffix.startswith("."):
        suffix = "." + suffix
    suffix = suffix.lower()
    if suffix not in {".pdf", ".png", ".jpg", ".jpeg"}:
        raise ValidationError(f"Unsupported upload suffix: {suffix}")
    return safe_session_dir(session_id) / f"{uuid.uuid4().hex}{suffix}"


def validate_doc_id(doc_id: str) -> None:
    if not _DOC_ID_RE.match(doc_id):
        raise ValidationError("Invalid doc_id — expected a uuid4 string")
