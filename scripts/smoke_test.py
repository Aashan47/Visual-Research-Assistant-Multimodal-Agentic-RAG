"""CI smoke test: ingest examples/sample_paper.pdf and answer one question.

Exits non-zero on failure. Used by the integration workflow as a fast gate
before running the full e2e suite.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.logging_config import configure_logging  # noqa: E402
from src.agents.graph import get_app, initial_state  # noqa: E402
from src.ingestion.pipeline import ingest_pdf  # noqa: E402
from src.llm.health import check_health  # noqa: E402


def main() -> int:
    configure_logging()

    health = check_health()
    if not health.ok:
        print(f"Health check failed: {health.help_text()}", file=sys.stderr)
        return 1

    pdf_path = ROOT / "examples" / "sample_paper.pdf"
    if not pdf_path.exists():
        print(f"Missing {pdf_path} — add a sample PDF and retry.", file=sys.stderr)
        return 2

    session_id = uuid.uuid4().hex
    result = ingest_pdf(session_id, pdf_path.read_bytes(), pdf_path.name)
    print(f"Ingested: {result}")

    app = get_app()
    state = initial_state(session_id, "What is this document about?")
    final = None
    for event in app.stream(
        state, config={"configurable": {"thread_id": session_id}}, stream_mode="values"
    ):
        final = event

    if final is None or not final.get("answer"):
        print("Graph produced no answer", file=sys.stderr)
        return 3

    print("Answer:", final["answer"][:500])
    print(f"degraded={final.get('degraded')}, retry_count={final.get('retry_count')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
