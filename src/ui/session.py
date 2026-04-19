"""Streamlit session-state helpers. Keeps session_id generation + reset in one place."""

from __future__ import annotations

import shutil
import uuid
from typing import Any

import streamlit as st

from config.settings import settings
from src.retrieval.vector_store import purge_session
from src.utils.logging import get_logger

log = get_logger(__name__)


def ensure_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    return st.session_state.session_id


def get_messages() -> list[dict[str, Any]]:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    return st.session_state.messages


def reset_session() -> None:
    """Purge on-disk + in-memory state for this session."""
    sid = st.session_state.get("session_id")
    if sid:
        try:
            purged = purge_session(sid)
            log.info("session_purged_chroma", session_id=sid, deleted=purged)
        except Exception as exc:
            log.warning("session_purge_chroma_failed", session_id=sid, error=str(exc))

        upload_dir = settings.uploads_dir / sid
        if upload_dir.exists():
            shutil.rmtree(upload_dir, ignore_errors=True)

    # Wipe all keys
    for key in list(st.session_state.keys()):
        del st.session_state[key]
