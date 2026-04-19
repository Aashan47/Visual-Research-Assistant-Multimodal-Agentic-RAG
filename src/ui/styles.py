"""Global CSS injection + HTML helper for a polished recruiter-facing demo.

We use `st.html()` (not `st.markdown(unsafe_allow_html=True)`) because in
Streamlit 1.40+ the markdown path sanitizes `class` attributes, which would
strip every one of our design-token classes. `st.html()` bypasses that.
"""

from __future__ import annotations

import streamlit as st

_GLOBAL_CSS = """
<style>
:root {
  --bg:           #0b0d12;
  --surface:      #141820;
  --surface-2:    #1b2029;
  --border:       #232a36;
  --text:         #e5e7eb;
  --text-dim:     #9ca3af;
  --text-faint:   #6b7280;
  --accent:       #22d3ee;
  --accent-soft:  rgba(34,211,238,0.12);
  --ok:           #10b981;
  --warn:         #f59e0b;
  --err:          #ef4444;
  --violet:       #a78bfa;
  --pink:         #f472b6;
  --radius:       14px;
}

.main .block-container {
  padding-top: 2rem !important;
  max-width: 1180px;
}
[data-testid="stSidebar"]       { background: var(--surface) !important; }
[data-testid="stSidebar"] > div { padding-top: 1rem; }
[data-testid="stHeader"]        { background: transparent; }

h1, h2, h3 { letter-spacing: -0.01em; }

/* Hero */
.vra-hero {
  border: 1px solid var(--border);
  background: radial-gradient(1200px 200px at 0% 0%, rgba(34,211,238,0.10), transparent 60%),
              radial-gradient(1200px 200px at 100% 0%, rgba(167,139,250,0.08), transparent 60%),
              linear-gradient(180deg, var(--surface), var(--surface-2));
  border-radius: var(--radius);
  padding: 22px 26px;
  margin-bottom: 18px;
}
.vra-hero .title {
  display: flex; align-items: center; gap: 12px;
  font-size: 22px; font-weight: 700; color: var(--text);
}
.vra-hero .title .icon {
  width: 34px; height: 34px; border-radius: 9px;
  display: inline-flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, var(--accent), var(--violet));
  color: #0b0d12; font-weight: 900; font-size: 13px;
}
.vra-hero .tagline { color: var(--text-dim); margin-top: 6px; font-size: 14px; max-width: 760px; }
.vra-hero .badges  { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 14px; }

.vra-badge {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 3px 9px; border-radius: 999px;
  background: var(--surface-2); color: var(--text-dim);
  border: 1px solid var(--border);
  font-size: 11px; font-weight: 500; line-height: 1.3;
  white-space: nowrap;
}
.vra-badge .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); }

.vra-pipeline {
  display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
  margin-top: 14px; font-size: 12px;
}
.vra-step {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 5px 11px; border-radius: 999px;
  background: var(--surface-2); border: 1px solid var(--border);
  color: var(--text-dim); font-weight: 500;
}
.vra-step .idx {
  width: 16px; height: 16px; border-radius: 50%;
  background: var(--accent-soft); color: var(--accent);
  display: inline-flex; align-items: center; justify-content: center;
  font-size: 10px; font-weight: 700;
}
.vra-arrow { color: var(--text-faint); font-size: 12px; }

.vra-card {
  background: var(--surface-2); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 14px;
}
.vra-card .card-title {
  font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em;
  color: var(--text-faint); font-weight: 600; margin-bottom: 8px;
}

.vra-health {
  display: flex; align-items: center; gap: 8px;
  padding: 10px 12px; border-radius: 10px;
  background: var(--surface-2); border: 1px solid var(--border);
  font-size: 13px; color: var(--text);
}
.vra-health .dot {
  width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
  background: var(--ok); box-shadow: 0 0 0 4px rgba(16,185,129,0.15);
}
.vra-health.warn .dot { background: var(--warn); box-shadow: 0 0 0 4px rgba(245,158,11,0.15); }
.vra-health.err  .dot { background: var(--err);  box-shadow: 0 0 0 4px rgba(239,68,68,0.15); }

.vra-session-chip {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 4px 10px; border-radius: 999px;
  background: var(--surface-2); border: 1px solid var(--border);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px; color: var(--text-dim);
}

.vra-welcome {
  background: linear-gradient(180deg, var(--surface), var(--surface-2));
  border: 1px solid var(--border); border-radius: var(--radius);
  padding: 26px 28px; margin: 12px 0 18px 0;
}
.vra-welcome h3 { margin: 0 0 6px 0; color: var(--text); font-size: 17px; }
.vra-welcome p  { margin: 0 0 14px 0; color: var(--text-dim); font-size: 13.5px; line-height: 1.55; }
.vra-welcome .prompts { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.vra-welcome .prompt {
  padding: 12px 14px; border-radius: 10px;
  background: var(--surface-2); border: 1px solid var(--border);
  font-size: 13px; color: var(--text);
}
.vra-welcome .prompt .route-tag {
  display: inline-block; margin-bottom: 6px;
  font-size: 10px; padding: 2px 6px; border-radius: 4px;
  background: var(--accent-soft); color: var(--accent);
  text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700;
}

[data-testid="stChatMessage"] {
  background: var(--surface-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 14px 16px !important;
}

.vra-cite {
  display: inline-block; margin: 0 2px;
  padding: 1px 7px; border-radius: 6px;
  background: var(--accent-soft); color: var(--accent);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px; font-weight: 600;
  border: 1px solid rgba(34,211,238,0.28);
}

.vra-degraded {
  margin-top: 10px; padding: 10px 12px; border-radius: 10px;
  background: rgba(245,158,11,0.08);
  border: 1px solid rgba(245,158,11,0.35);
  color: #fbbf24; font-size: 13px;
}

.vra-trace { display: flex; flex-direction: column; gap: 8px; }
.vra-trace-row {
  display: grid; grid-template-columns: 22px 140px 80px 1fr;
  align-items: start; gap: 10px;
  padding: 8px 12px; border-radius: 10px;
  background: var(--surface-2); border: 1px solid var(--border);
}
.vra-trace-row.ok    { border-left: 3px solid var(--ok); }
.vra-trace-row.retry { border-left: 3px solid var(--warn); }
.vra-trace-row.err   { border-left: 3px solid var(--err); }
.vra-trace-row.end   { border-left: 3px solid var(--accent); }
.vra-trace-dot {
  width: 14px; height: 14px; border-radius: 50%;
  background: var(--surface); border: 2px solid var(--border); margin-top: 3px;
}
.vra-trace-row.ok    .vra-trace-dot { border-color: var(--ok); }
.vra-trace-row.retry .vra-trace-dot { border-color: var(--warn); }
.vra-trace-row.err   .vra-trace-dot { border-color: var(--err); }
.vra-trace-row.end   .vra-trace-dot { border-color: var(--accent); }
.vra-trace-node {
  color: var(--text); font-weight: 600; font-size: 13px;
  text-transform: capitalize;
}
.vra-trace-duration {
  color: var(--text-dim);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px; text-align: right;
}
.vra-trace-detail { color: var(--text-dim); font-size: 12px; line-height: 1.45; }
.vra-trace-err    { color: var(--err); font-size: 12px; margin-top: 3px; }

.stButton > button {
  border-radius: 10px !important;
  border: 1px solid var(--border) !important;
  background: var(--surface-2) !important;
  color: var(--text) !important;
}
.stButton > button:hover {
  border-color: var(--accent) !important; color: var(--accent) !important;
}

[data-testid="stFileUploader"] section {
  background: var(--surface-2) !important;
  border: 1px dashed var(--border) !important;
  border-radius: 12px !important;
  padding: 12px !important;
}
[data-testid="stFileUploader"] section:hover { border-color: var(--accent) !important; }

[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  background: var(--surface) !important;
}

[data-testid="stProgress"] > div > div > div > div { background-color: var(--accent) !important; }

footer { visibility: hidden; }
</style>
"""


def inject_css() -> None:
    """Inject the global stylesheet once per session."""
    if not st.session_state.get("_vra_css_injected"):
        st.html(_GLOBAL_CSS)
        st.session_state["_vra_css_injected"] = True


def html(snippet: str) -> None:
    """Render raw HTML inline without Streamlit's markdown sanitizer."""
    st.html(snippet)
