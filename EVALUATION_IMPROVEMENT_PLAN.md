# Production-Grade RAG Evaluation Improvement Plan

This file tracks the implementation status for upgrading the RAG evaluator from demo smoke tests to production-oriented regression checks.

## Phase Checklist

- [x] Phase 1: Objective retrieval metrics
  - Add structured retrieval results with stable `source:chunk_index` IDs.
  - Compute `hit_at_5`, `mrr`, `id_context_precision`, and `id_context_recall`.
- [x] Phase 2: Reviewed eval-set schema
  - Add `reference_chunks`, `case_type`, `reviewed`, and `reviewer` fields.
  - Seed `rag_eval_set.reviewed.json` with reviewed normal, ambiguous, adversarial, and negative cases.
- [x] Phase 3: Regression gates
  - Add CLI thresholds for quality, retrieval, latency, and cost.
  - Exit non-zero when `--fail-on-threshold` is set and any configured threshold fails.
- [x] Phase 4: Dashboard trend upgrades
  - Show generation, retrieval, latency, cost, and threshold status while preserving old report compatibility.
- [ ] Phase 5: Expanded reviewed cases
  - Grow the reviewed set from the initial 12-case seed to 50-100 human-reviewed cases.

## Verification Log

- `rag_eval_20260614_224734.json`: retrieval-only gate passed with `hit_at_5=1.00` and `mrr=0.95`.
- `rag_eval_20260614_224808.json`: two-case smoke eval completed with the upgraded report shape.
- `rag_eval_20260614_225852.json`: full reviewed gate wrote a report and failed strict generation thresholds as expected.
- Deliberately impossible one-case gate exited with status 1 for `hit_at_5 actual=1.0 >= 1.1`.
- Four focused evaluator unit tests pass.
- Streamlit app-test rendered the upgraded dashboard with old and new reports and no exceptions.
- Final hardened two-case retrieval gate passed with `case_errors=0`, `hit_at_5=1.00`, and `mrr=1.00`.

## Current Quality Baseline

- Retrieval is strong on the initial reviewed set: `hit_at_5=1.00`, `mrr=0.95`, and `id_context_recall=1.00`.
- Retrieval precision is low at `0.33` because this small three-chunk knowledge base returns multiple overlapping chunks.
- The full reviewed generation gate currently fails: `faithfulness=0.40`, `answer_relevancy=0.40`, and LLM-judged `context_recall=0.58`.
- The next improvement work should focus on answer generation and critic behavior while expanding the reviewed set to 50-100 cases.

## Current Defaults

- Keep `rag_eval_set.synthetic.json` as a draft/synthetic source.
- Use `rag_eval_set.reviewed.json` for regression gates.
- Defer CI wiring until the project has a reliable way to provide local services and API keys in CI.

## Verification Commands

```powershell
uv run evaluate_rag.py .\rag_eval_set.synthetic.json --limit 2 --case-sleep 5
uv run evaluate_rag.py .\rag_eval_set.reviewed.json --retrieval-only --fail-on-threshold --min-hit-at-5 0.90 --min-mrr 0.70
uv run evaluate_rag.py .\rag_eval_set.reviewed.json --fail-on-threshold --min-faithfulness 0.80 --min-answer-relevancy 0.80 --min-context-recall 0.80
```
