"""Shared pytest fixtures."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest


@pytest.fixture
def session_id() -> str:
    return uuid.uuid4().hex


@pytest.fixture
def tmp_uploads_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect settings.uploads_dir to a temporary path for the duration of the test."""
    from config.settings import settings

    target = tmp_path / "uploads"
    target.mkdir()
    monkeypatch.setattr(settings, "uploads_dir", target)
    return target
