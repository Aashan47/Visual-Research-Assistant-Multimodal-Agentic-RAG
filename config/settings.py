"""Typed, env-driven settings. Imported as `from config.settings import settings`."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, HttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Models (quality-first; GTX 1660 Ti can run these via Ollama auto-swap)
    generator_model: str = "qwen2.5vl:7b"
    generator_num_ctx: int = 3072
    critique_model: str = "llama3.1:8b"
    critique_num_ctx: int = 1536
    embedding_model: str = "nomic-embed-text"

    # Ollama
    ollama_host: HttpUrl = Field(default=HttpUrl("http://localhost:11434"))
    ollama_timeout_s: int = 300
    ollama_max_retries: int = 3
    ollama_keep_alive: str = "10m"

    # Storage (all paths relative to project root by default)
    chroma_persist_dir: Path = Field(default=PROJECT_ROOT / "data" / "chroma_db")
    docstore_dir: Path = Field(default=PROJECT_ROOT / "data" / "docstore")
    uploads_dir: Path = Field(default=PROJECT_ROOT / "data" / "uploads")
    checkpoints_db: Path = Field(default=PROJECT_ROOT / "data" / "checkpoints.db")
    manifest_path: Path = Field(default=PROJECT_ROOT / "data" / "ingested_manifest.json")
    logs_dir: Path = Field(default=PROJECT_ROOT / "logs")

    # Retrieval — tuned for fast inference on modest GPUs; raise `k` if using 8B+ models.
    chunk_size: int = 800
    chunk_overlap: int = 150
    retrieval_k: int = 4
    chroma_collection: str = "visual_research"

    # Upload limits
    max_upload_mb: int = 50
    max_pdf_pages: int = 200

    # Agent graph
    max_retries: int = 1  # one retry; keeps UX snappy while still demoing self-correction
    pdf_strategy: str = "fast"  # "fast" | "hi_res"
    image_max_side_px: int = 1024
    summary_concurrency: int = 1  # single GPU; concurrent calls just queue in Ollama

    # Performance flags — defaults favor responsiveness on modest GPUs.
    # Flip these on when running on a stronger machine or when the demo focus
    # requires indexing figures from the PDF itself.
    index_pdf_images: bool = False
    index_pdf_tables: bool = True

    # Observability
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "visual-research-assistant"

    @field_validator(
        "chroma_persist_dir",
        "docstore_dir",
        "uploads_dir",
        "logs_dir",
        mode="after",
    )
    @classmethod
    def _ensure_dir(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("checkpoints_db", "manifest_path", mode="after")
    @classmethod
    def _ensure_parent_dir(cls, v: Path) -> Path:
        v.parent.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("pdf_strategy")
    @classmethod
    def _validate_strategy(cls, v: str) -> str:
        if v not in {"fast", "hi_res"}:
            raise ValueError("pdf_strategy must be 'fast' or 'hi_res'")
        return v


settings = Settings()
