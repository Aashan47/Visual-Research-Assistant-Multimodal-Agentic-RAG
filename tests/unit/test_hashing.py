"""Hashing helpers unit tests."""

from __future__ import annotations

import pytest
from PIL import Image

from src.utils.hashing import phash, sha256_bytes


pytestmark = pytest.mark.unit


def test_sha256_bytes_stable() -> None:
    assert sha256_bytes(b"hello") == sha256_bytes(b"hello")
    assert sha256_bytes(b"hello") != sha256_bytes(b"world")


def test_phash_identical_images_match() -> None:
    a = Image.new("RGB", (64, 64), (10, 20, 30))
    b = Image.new("RGB", (64, 64), (10, 20, 30))
    assert phash(a) == phash(b)


def test_phash_different_images_differ() -> None:
    a = Image.new("RGB", (64, 64), (0, 0, 0))
    b = Image.new("RGB", (64, 64), (255, 255, 255))
    # Solid images won't differ (all pixels above/below mean identically);
    # use a gradient for a meaningful difference.
    gradient = Image.new("RGB", (64, 64))
    px = gradient.load()
    for y in range(64):
        for x in range(64):
            px[x, y] = (x * 4, x * 4, x * 4)
    assert phash(a) != phash(gradient)
    assert phash(b) != phash(gradient)
