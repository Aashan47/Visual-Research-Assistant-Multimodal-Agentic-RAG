"""Streamlit entrypoint — `main()` called from run.py."""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from config.logging_config import configure_logging
from config.settings import settings
from src.ui.components.chat import render_chat
from src.ui.components.header import render_header
from src.ui.components.sidebar import render_sidebar
from src.ui.session import ensure_session_id
from src.ui.styles import inject_css


def _init_env() -> None:
    load_dotenv()
    configure_logging()
    if settings.langsmith_tracing and settings.langsmith_api_key:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project


def main() -> None:
    _init_env()

    st.set_page_config(
        page_title="Visual Research Assistant",
        page_icon="◆",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": (
                "Multimodal Agentic RAG — LangGraph + Ollama + ChromaDB. "
                "Runs fully locally."
            ),
        },
    )

    inject_css()

    session_id = ensure_session_id()

    render_sidebar()
    render_header()
    render_chat(session_id)


if __name__ == "__main__":
    main()
