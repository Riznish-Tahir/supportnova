# AI Tool Usage Declaration

## Tool
**Claude** (Anthropic) — used in a chat/agentic coding session to scaffold this project.

## Purpose
Generate the initial project skeleton for SupportNova: both pipelines, the rule matrix,
database schema, document processing, comparison engine, sample dataset, sample knowledge
base, tests, and the earlier UI/UX prototype — from the provided SRS document.

## Type of assistance requested
- Architecture and file layout matching the SRS module list
- Implementation of Pipeline 1 (GenAI) and Pipeline 2 (Python ground-truth validation)
- Rule matrix generation, JSON schema, prompt template
- Synthetic complaint dataset generation (754 records) and knowledge-base documents
- Test suite authoring and debugging

## Files affected
All files under `supportnova/` (see README.md for the full tree).

## Changes made after generation
- Fixed a regex bug in the prompt-injection detector that missed
  "disregard the X policy" phrasing (caught by `test_prompt_injection_detected`).
- Added "spark"/"smoke"/"burning" to the objective-risk keyword list after a test
  showed a calmly-worded safety complaint wasn't being force-escalated.
- Fixed the `complaint_id` schema regex, which rejected the hex-based IDs the API
  generates (was numeric-only).

## Testing performed
- `pytest tests/` — 19 unit/integration tests across preprocessing, security,
  both pipelines, and the comparison engine (all passing).
- End-to-end smoke test via FastAPI `TestClient`: complaint submission, detail
  fetch, knowledge-base upload, analytics endpoints.
- Generated and reviewed the GenAI/Python comparison report over 120 sampled
  complaints (100 auto-verified, 20 correctly routed to manual review).

## Verifying team members
_[Add your name(s) here after reviewing the code — per the SRS, AI-generated
source code must be independently reviewed, modified where needed, and
understood before submission. Do not submit this file unedited.]_

## Important note on GenAI Pipeline mode
`genai_pipeline/pipeline.py` ships in **mock mode** by default
(`SUPPORTNOVA_GENAI_MODE=mock`), using deterministic keyword heuristics instead
of a live model call, so the whole system runs without an API key. Set
`SUPPORTNOVA_GENAI_MODE=live` and `ANTHROPIC_API_KEY` to use the real Anthropic
API — see README.md.
