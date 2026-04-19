"""Content + perceptual hashing for dedupe."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def phash(img: Image.Image, size: int = 8) -> str:
    """Cheap perceptual hash via downscaled grayscale + mean threshold.

    Sufficient for "is this the same chart image extracted twice?" dedupe,
    which is all we need here. Not a substitute for imagehash for hard cases.
    """
    small = img.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    pixels = list(small.getdata())
    avg = sum(pixels) / len(pixels)
    bits = "".join("1" if p >= avg else "0" for p in pixels)
    return f"{int(bits, 2):0{size * size // 4}x}"
