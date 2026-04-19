"""Image encoding + resizing helpers for VL models."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image

from config.settings import settings


def load_and_resize(source: str | bytes | Path, max_side: int | None = None) -> Image.Image:
    """Load an image from path or bytes and downscale to fit `max_side` on its longest side."""
    max_side = max_side or settings.image_max_side_px
    if isinstance(source, (str, Path)):
        img = Image.open(source)
    else:
        img = Image.open(io.BytesIO(source))

    img = img.convert("RGB")

    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    return img


def to_base64_png(source: str | bytes | Path | Image.Image, max_side: int | None = None) -> str:
    """Return a base64-encoded PNG (no data URI prefix) sized for VL model consumption."""
    if isinstance(source, Image.Image):
        img = source.convert("RGB")
        if max_side and max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    else:
        img = load_and_resize(source, max_side=max_side)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def to_data_uri(b64_png: str) -> str:
    """Wrap a raw base64 PNG string as a data URI (some clients want this form)."""
    return f"data:image/png;base64,{b64_png}"
