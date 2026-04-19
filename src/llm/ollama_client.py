"""Centralized ChatOllama + OllamaEmbeddings factories.

All LLM calls in this project MUST go through these factories so tuning
(timeout, keep_alive, num_ctx) lives in one place.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_ollama import ChatOllama, OllamaEmbeddings

from config.settings import settings


@lru_cache(maxsize=8)
def get_chat_model(
    model: str,
    num_ctx: int,
    temperature: float = 0.1,
) -> ChatOllama:
    """Cached ChatOllama instance. Cache key = (model, num_ctx, temperature)."""
    return ChatOllama(
        model=model,
        base_url=str(settings.ollama_host).rstrip("/"),
        num_ctx=num_ctx,
        temperature=temperature,
        keep_alive=settings.ollama_keep_alive,
        timeout=settings.ollama_timeout_s,
    )


def get_generator() -> ChatOllama:
    """Vision-capable generator LLM — use when images are present."""
    return get_chat_model(settings.generator_model, settings.generator_num_ctx, temperature=0.1)


def get_text_generator() -> ChatOllama:
    """Fast text-only generator — use for pure-text questions.

    Routing at the generator node avoids paying the VL inference tax when no
    images are in context. Same prompt, same citation format, much faster."""
    return get_chat_model(settings.critique_model, settings.generator_num_ctx, temperature=0.1)


def get_critique() -> ChatOllama:
    """Text-only critic LLM (different family from generator — independent judgement)."""
    return get_chat_model(settings.critique_model, settings.critique_num_ctx, temperature=0.0)


def get_router() -> ChatOllama:
    """Cheap classifier — uses the critique model at very low temperature."""
    return get_chat_model(settings.critique_model, num_ctx=1024, temperature=0.0)


@lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=str(settings.ollama_host).rstrip("/"),
    )
