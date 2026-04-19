"""Sidebar: session chip, health, uploaders, reset, advanced settings."""

from __future__ import annotations

from html import escape

import streamlit as st

from config.settings import settings
from src.ingestion.pipeline import ingest_pdf, ingest_user_image
from src.llm.health import check_health
from src.ui.session import ensure_session_id, reset_session
from src.ui.styles import html
from src.utils.errors import ValidationError
from src.utils.image_io import to_base64_png
from src.utils.logging import get_logger

log = get_logger(__name__)


def _health_banner() -> None:
    report = check_health()
    if report.ok:
        cls, label = "ok", "Ollama online · all models ready"
    elif not report.ollama_reachable:
        cls, label = "err", "Ollama offline"
    else:
        cls = "warn"
        missing = ", ".join(report.missing_models)
        label = f"Missing: {escape(missing)}"
    html(f'<div class="vra-health {cls}"><span class="dot"></span><span>{label}</span></div>')
    if not report.ok:
        with st.expander("Fix", expanded=False):
            st.code(report.help_text(), language="bash")


def _session_chip(session_id: str) -> None:
    html(f'<div class="vra-session-chip">◆ session {session_id[:8]}</div>')


def _pdf_uploader(session_id: str) -> None:
    st.caption("DOCUMENT")
    uploaded = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        label_visibility="collapsed",
        help=f"Max {settings.max_upload_mb} MB · up to {settings.max_pdf_pages} pages.",
        key="pdf_uploader",
    )
    if uploaded is None:
        return

    file_key = f"ingested:{uploaded.name}:{uploaded.size}"
    if st.session_state.get(file_key):
        st.caption(f"✓ {uploaded.name}")
        return

    data = uploaded.getvalue()
    progress = st.progress(0.0, text="Starting…")

    def _cb(stage: str, frac: float) -> None:
        progress.progress(min(max(frac, 0.0), 1.0), text=stage)

    try:
        result = ingest_pdf(session_id, data, uploaded.name, progress_cb=_cb)
    except ValidationError as exc:
        progress.empty()
        st.error(f"Upload rejected: {exc}")
        return
    except Exception as exc:
        progress.empty()
        log.exception("pdf_ingest_ui_error", error=str(exc))
        st.error(f"Ingestion failed: {exc}")
        return

    progress.empty()
    if result.skipped:
        st.info(f"Already ingested (same file): {uploaded.name}")
    else:
        st.success(
            f"{result.n_text_chunks} chunks · "
            f"{result.n_image_summaries} figures · "
            f"{result.n_table_summaries} tables"
        )
    st.session_state[file_key] = True


def _image_uploader(session_id: str) -> None:
    st.caption("OPTIONAL IMAGE")
    uploaded = st.file_uploader(
        "Optional: attach an image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
        help="Supplementary image — attached only when the question is about the image or cross-modal.",
        key="img_uploader",
    )
    if uploaded is None:
        st.session_state.pop("user_image_b64", None)
        return

    data = uploaded.getvalue()
    ingest_key = f"ingested_img:{uploaded.name}:{uploaded.size}"

    try:
        st.session_state["user_image_b64"] = to_base64_png(data)
        if not st.session_state.get(ingest_key):
            ingest_user_image(session_id, data, uploaded.name)
            st.session_state[ingest_key] = True
            st.success(f"Attached · {uploaded.name}")
        else:
            st.caption(f"✓ {uploaded.name}")
        st.image(data, width='stretch')
    except ValidationError as exc:
        st.error(f"Image rejected: {exc}")
    except Exception as exc:
        log.exception("image_ingest_ui_error", error=str(exc))
        st.error(f"Image indexing failed: {exc}")


def _advanced_settings() -> None:
    with st.expander("Advanced", expanded=False):
        current = settings.index_pdf_images
        new = st.checkbox(
            "Index PDF figures (slow)",
            value=current,
            help=(
                "When ON, each figure in the PDF is sent to the vision model for a retrieval "
                "summary (~30-60 s per figure). OFF by default for fast ingestion."
            ),
        )
        if new != current:
            settings.index_pdf_images = new
            st.caption("Re-upload the PDF to re-index with this setting.")

        st.divider()
        st.caption("Models (change in `.env` and restart)")
        st.markdown(
            f"- **Vision** · `{settings.generator_model}`\n"
            f"- **Text** · `{settings.critique_model}`\n"
            f"- **Embeddings** · `{settings.embedding_model}`\n"
            f"- **Max retries** · `{settings.max_retries}`"
        )


def render_sidebar() -> None:
    with st.sidebar:
        html(
            '<div style="font-weight:700; font-size:15px; color:var(--text); '
            'display:flex; align-items:center; gap:8px; padding: 2px 0 6px 0;">'
            '<span style="width:22px; height:22px; border-radius:6px; '
            'background:linear-gradient(135deg,var(--accent),var(--violet)); '
            'color:#0b0d12; display:inline-flex; align-items:center; justify-content:center; '
            'font-weight:900; font-size:10px;">VR</span>'
            'Visual Research Assistant'
            "</div>"
        )
        session_id = ensure_session_id()
        _session_chip(session_id)
        st.write("")
        _health_banner()
        st.divider()

        _pdf_uploader(session_id)
        st.write("")
        _image_uploader(session_id)
        st.divider()

        if st.button("↺  Reset session", width='stretch'):
            reset_session()
            st.rerun()

        _advanced_settings()
