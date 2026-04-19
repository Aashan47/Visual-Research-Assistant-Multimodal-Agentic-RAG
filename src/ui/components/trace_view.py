"""Timeline-style agent trace view."""

from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from src.ui.styles import html


_NODE_ICONS = {
    "router":    "🧭",
    "retriever": "🔎",
    "generator": "✦",
    "critic":    "⊘",
    "critique":  "⊘",
    "finalize":  "✓",
}


def _row_class(entry: dict[str, Any]) -> str:
    if entry.get("error"):
        return "err"
    if entry.get("node") == "finalize":
        return "end"
    # mark retry rows based on the critic's retry counter if present in output summary
    out = str(entry.get("output") or "")
    if "retry=" in out:
        # retry_count >=2 means we've looped at least once
        try:
            n = int(out.split("retry=")[1].split()[0])
            if n >= 2:
                return "retry"
        except Exception:
            pass
    return "ok"


def _fmt_duration(ms: Any) -> str:
    if ms is None:
        return "—"
    try:
        ms = float(ms)
    except (TypeError, ValueError):
        return "—"
    if ms < 1000:
        return f"{ms:.0f} ms"
    return f"{ms/1000:.1f} s"


def render_trace(trace: list[dict[str, Any]], mermaid_src: str | None = None) -> None:
    if not trace:
        return
    with st.expander("Agent trace", expanded=False):
        st.caption(f"{len(trace)} nodes executed")

        rows_html: list[str] = ['<div class="vra-trace">']
        for entry in trace:
            node = str(entry.get("node", "?"))
            icon = _NODE_ICONS.get(node.lower(), "●")
            cls = _row_class(entry)
            duration = _fmt_duration(entry.get("duration_ms"))

            detail_lines: list[str] = []
            if entry.get("input"):
                detail_lines.append(f"<div>› {escape(str(entry['input']))}</div>")
            if entry.get("output"):
                detail_lines.append(f"<div>{escape(str(entry['output']))}</div>")
            if entry.get("error"):
                detail_lines.append(
                    f'<div class="vra-trace-err">⚠ {escape(str(entry["error"]))}</div>'
                )
            detail_html = "".join(detail_lines) or "<div class='vra-trace-detail'>—</div>"

            rows_html.append(
                f'<div class="vra-trace-row {cls}">'
                f'  <span class="vra-trace-dot"></span>'
                f'  <div class="vra-trace-node">{icon} {escape(node)}</div>'
                f'  <div class="vra-trace-duration">{duration}</div>'
                f'  <div class="vra-trace-detail">{detail_html}</div>'
                f"</div>"
            )
        rows_html.append("</div>")
        html("".join(rows_html))

        if mermaid_src:
            with st.expander("Graph topology (mermaid)", expanded=False):
                st.code(mermaid_src, language="mermaid")
