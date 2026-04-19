"""Local file-backed docstore mapping doc_id → original content (JSON-serialized).

We use a small JSON-per-key layout on disk because:
  - zero dependencies beyond stdlib,
  - easy to inspect/debug a single entry,
  - atomic writes via os.replace.

Originals are dicts: `{kind: "text"|"image"|"table", ...}`.
"""

from __future__ import annotations

import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from config.settings import settings
from src.ingestion.validators import validate_doc_id
from src.utils.errors import RetrievalError


class JsonDocStore:
    """Thin JSON-on-disk mapping. One file per doc_id."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, doc_id: str) -> Path:
        validate_doc_id(doc_id)
        # Shard by first 2 chars to keep dir sizes manageable.
        return self.root / doc_id[:2] / f"{doc_id}.json"

    def mset(self, items: list[tuple[str, dict[str, Any]]]) -> None:
        for key, value in items:
            path = self._path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic write
            fd, tmp_name = tempfile.mkstemp(
                prefix=f"{key}.", suffix=".tmp", dir=str(path.parent)
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(value, fh)
                os.replace(tmp_name, path)
            except Exception:
                if os.path.exists(tmp_name):
                    os.remove(tmp_name)
                raise

    def mget(self, keys: list[str]) -> list[dict[str, Any] | None]:
        out: list[dict[str, Any] | None] = []
        for key in keys:
            path = self._path(key)
            if not path.exists():
                out.append(None)
                continue
            try:
                out.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError as exc:
                raise RetrievalError(f"corrupt docstore entry for {key}: {exc}") from exc
        return out

    def mdelete(self, keys: list[str]) -> None:
        for key in keys:
            path = self._path(key)
            if path.exists():
                path.unlink()

    def yield_keys(self) -> list[str]:
        """Not used in hot paths; convenient for tests/debugging."""
        return [p.stem for p in self.root.glob("*/*.json")]


@lru_cache(maxsize=1)
def get_docstore() -> JsonDocStore:
    return JsonDocStore(settings.docstore_dir)
