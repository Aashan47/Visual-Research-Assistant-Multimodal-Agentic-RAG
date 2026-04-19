"""Hero header with tech-stack badges and the agent pipeline pill row."""

from __future__ import annotations

from config.settings import settings
from src.ui.styles import html

_BADGES = [
    "LangGraph",
    "ChromaDB",
    "Ollama (local)",
    "Streamlit",
]

_PIPELINE = [
    ("1", "Router"),
    ("2", "Retriever"),
    ("3", "Generator"),
    ("4", "Critic"),
    ("5", "Finalize"),
]


def render_header() -> None:
    badges = "".join(
        f'<span class="vra-badge"><span class="dot"></span>{b}</span>' for b in _BADGES
    )
    model_badges = (
        f'<span class="vra-badge">vision · {settings.generator_model}</span>'
        f'<span class="vra-badge">text · {settings.critique_model}</span>'
        f'<span class="vra-badge">embed · {settings.embedding_model}</span>'
    )

    pipeline = '<span class="vra-arrow">›</span>'.join(
        f'<span class="vra-step"><span class="idx">{idx}</span>{name}</span>'
        for idx, name in _PIPELINE
    )

    html(
        f"""
        <div class="vra-hero">
          <div class="title">
            <span class="icon">VR</span>
            <span>Visual Research Assistant</span>
          </div>
          <div class="tagline">
            A multimodal agentic RAG over your PDF and an optional attached image.
            Heterogeneous critique catches hallucinations; multi-vector retrieval
            returns raw originals, not summaries. Runs fully locally.
          </div>
          <div class="badges">{badges}{model_badges}</div>
          <div class="vra-pipeline">{pipeline}</div>
        </div>
        """
    )
