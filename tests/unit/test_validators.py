"""Upload validation unit tests — no network, no LLM."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from src.ingestion.validators import (
    new_upload_path,
    safe_session_dir,
    validate_doc_id,
    validate_image_bytes,
    validate_pdf_bytes,
)
from src.utils.errors import ValidationError


pytestmark = pytest.mark.unit


def _png_bytes(size: tuple[int, int] = (32, 32)) -> bytes:
    img = Image.new("RGB", size, (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes() -> bytes:
    img = Image.new("RGB", (16, 16), (0, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _minimal_pdf_bytes() -> bytes:
    """Smallest valid PDF that fitz will parse."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n"
        b"0000000010 00000 n \n"
        b"0000000053 00000 n \n"
        b"0000000100 00000 n \n"
        b"trailer<</Size 4/Root 1 0 R>>\n"
        b"startxref\n150\n%%EOF\n"
    )


class TestValidatePdfBytes:
    def test_rejects_non_pdf_magic(self) -> None:
        with pytest.raises(ValidationError, match="valid PDF header"):
            validate_pdf_bytes(b"not a pdf", "fake.pdf")

    def test_rejects_oversized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from config.settings import settings

        monkeypatch.setattr(settings, "max_upload_mb", 0.0001)  # effectively zero
        pdf = _minimal_pdf_bytes()
        with pytest.raises(ValidationError, match="limit is"):
            validate_pdf_bytes(pdf, "big.pdf")

    def test_accepts_minimal_pdf(self) -> None:
        validate_pdf_bytes(_minimal_pdf_bytes(), "ok.pdf")

    def test_rejects_too_many_pages(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from config.settings import settings

        monkeypatch.setattr(settings, "max_pdf_pages", 0)
        with pytest.raises(ValidationError, match="pages"):
            validate_pdf_bytes(_minimal_pdf_bytes(), "ok.pdf")


class TestValidateImageBytes:
    def test_accepts_png(self) -> None:
        validate_image_bytes(_png_bytes(), "x.png")

    def test_accepts_jpeg(self) -> None:
        validate_image_bytes(_jpeg_bytes(), "x.jpg")

    def test_rejects_pdf_bytes(self) -> None:
        with pytest.raises(ValidationError, match="valid PNG or JPEG"):
            validate_image_bytes(_minimal_pdf_bytes(), "x.png")

    def test_rejects_garbage(self) -> None:
        with pytest.raises(ValidationError):
            validate_image_bytes(b"\x00\x01\x02\x03", "x.png")


class TestSafeSessionDir:
    def test_accepts_uuid4_hex(self, tmp_uploads_dir) -> None:  # noqa: ARG002
        import uuid

        sid = uuid.uuid4().hex
        path = safe_session_dir(sid)
        assert path.exists()

    def test_rejects_path_traversal(self) -> None:
        with pytest.raises(ValidationError):
            safe_session_dir("../etc/passwd")

    def test_rejects_non_hex(self) -> None:
        with pytest.raises(ValidationError):
            safe_session_dir("not-a-real-uuid")


class TestNewUploadPath:
    def test_rejects_bad_suffix(self, tmp_uploads_dir) -> None:  # noqa: ARG002
        import uuid

        with pytest.raises(ValidationError):
            new_upload_path(uuid.uuid4().hex, ".exe")


class TestValidateDocId:
    def test_accepts_uuid4(self) -> None:
        import uuid

        validate_doc_id(str(uuid.uuid4()))

    def test_rejects_bad_format(self) -> None:
        with pytest.raises(ValidationError):
            validate_doc_id("../etc/passwd")

        with pytest.raises(ValidationError):
            validate_doc_id("1234")
