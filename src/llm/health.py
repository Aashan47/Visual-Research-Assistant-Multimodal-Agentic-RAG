"""Ollama reachability + required-model presence checks."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from config.settings import settings
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class HealthReport:
    ollama_reachable: bool
    missing_models: list[str]
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.ollama_reachable and not self.missing_models

    def help_text(self) -> str:
        if not self.ollama_reachable:
            return (
                f"Ollama is not reachable at {settings.ollama_host}. "
                "Start it with `ollama serve` (or launch the Windows app) and retry."
            )
        if self.missing_models:
            pulls = "\n".join(f"  ollama pull {m}" for m in self.missing_models)
            return f"Required model(s) not installed:\n{pulls}"
        return "All systems nominal."


def check_health(timeout: float = 3.0) -> HealthReport:
    """Ping Ollama and verify required models are available."""
    required = {
        settings.generator_model,
        settings.critique_model,
        settings.embedding_model,
    }

    try:
        resp = httpx.get(f"{str(settings.ollama_host).rstrip('/')}/api/tags", timeout=timeout)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        log.warning("ollama_unreachable", error=str(exc))
        return HealthReport(ollama_reachable=False, missing_models=sorted(required), error=str(exc))

    installed_raw = {m["name"] for m in resp.json().get("models", [])}
    # Ollama stores untagged pulls as `name:latest`. Build a normalized view
    # so `nomic-embed-text` matches `nomic-embed-text:latest`.
    installed_norm = installed_raw | {
        n.split(":", 1)[0] for n in installed_raw if n.endswith(":latest")
    }
    missing = sorted(m for m in required if m not in installed_norm)
    return HealthReport(ollama_reachable=True, missing_models=missing)
