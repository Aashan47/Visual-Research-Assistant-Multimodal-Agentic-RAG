# Demo Questions

Use these to walk a reviewer through the four core behaviors of the system. Upload `sample_paper.pdf` and attach `sample_chart.png` in the sidebar before starting.

| # | Question | Exercises | What to watch in the trace |
|---|---|---|---|
| 1 | What dataset does the paper use for evaluation, and what are the reported accuracy numbers? | Text-only retrieval, multi-vector (returns table HTML, not summary) | `route=text`, single pass, critique passes, citations reference table's `doc_id` |
| 2 | Summarize what the attached chart shows and flag anything unusual. | Image-only path, user-uploaded image indexed at session level | `route=image`, retriever filters `source_type in {user_image, image}` |
| 3 | Does the attached chart's trend match Figure 2 in the paper? | Cross-modal reasoning | `route=cross_modal`, often triggers one retry as critique demands precise grounding |
| 4 (adversarial) | According to the paper, who is the CEO of OpenAI? | Bounded degradation | Two retries exhaust; `degraded=true`; UI shows amber warning banner |

## Recruiter talking points while demoing

1. **Multi-vector retrieval:** "See the table HTML in the expandable context? That's the *original*. What got embedded was a short summary. This is the multi-vector pattern — dense, readable context instead of raw OCR."
2. **Heterogeneous critique:** "The generator runs Qwen2.5-VL. The critic runs Llama 3.1. Different weights → independent judgement. The critic doesn't rubber-stamp its own biases."
3. **Graceful degradation:** "Question 4 is unanswerable. Watch the retry counter. After 2 retries the graph exits with `degraded=true` — a transparent weak answer beats an infinite loop."
4. **Full observability:** "LangSmith link under every answer, structured JSON logs on disk, in-app trace panel with per-node timing."
