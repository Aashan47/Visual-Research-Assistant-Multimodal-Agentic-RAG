"""Typed exception hierarchy. Catch these, not `Exception`."""

from __future__ import annotations


class VRAError(Exception):
    """Root of Visual Research Assistant exceptions."""


class ConfigurationError(VRAError):
    """Required config missing or invalid."""


class ValidationError(VRAError):
    """User input failed validation (upload, doc_id, etc.)."""


class IngestionError(VRAError):
    """Ingestion pipeline failure."""


class RetrievalError(VRAError):
    """Vector store or docstore operation failed."""


class LLMError(VRAError):
    """LLM call failed after retries."""


class OllamaUnavailableError(LLMError):
    """Ollama server unreachable or missing required model."""


class AgentError(VRAError):
    """Agent graph execution failed."""
