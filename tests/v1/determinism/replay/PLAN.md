# Divergence Replay — Sharpened Spec

Outcome of a /grill-me session, 2026-07-14. Supersedes the original ask
("write a program in ~/Projects/divergence-sim") — everything now lives in
this vLLM checkout, under `tests/v1/determinism/replay/`.

## Goal

Check whether LLM output divergence observed in production traces is
reproducible **in isolation**: same input, replayed n times, strictly
sequentially (one request in flight, closest to bs=1), against the **same
serving endpoint with the exact original config** (model, temperature,
tools, seed).

## Case extraction

- Source: the ~25 `trace_analysis_*.json` files in
  `~/Projects/trace-analyser` (fields: `project_id`, `run_names`,
  `per_event_divergence`, `first_divergence_overall`, …).
- Qualifying cases: **all events** where the input was identical across
  runs but the output diverged — `input_divergence ≈ 0` (≤ noise) and
  `output_divergence > noise` — i.e. the non-determinism sources, not the
  downstream input-divergence consequences (those have differing inputs and
  cannot be replayed "same input").
- Only real LLM calls qualify: **input token count must be non-zero**.
- The analysis JSONs hold only metrics; full payloads (messages, model,
  params) are **re-fetched from LangSmith** by project_id/run name.
- Retention caveat: some source projects may be expired — extractor must
  **skip-and-report**, producing a coverage summary of which cases were
  extractable, never dying on a missing project.

## Replay semantics

- n tries per case, n = `N_TRIES` from `.env` (in this replay directory).
- Strictly sequential calls; exact original request config.
- Divergence measured with **trace_analyser's own `divergence()` function
  and NOISE threshold** (import/reuse it, apples-to-apples with the
  original analysis), including its ID/UUID normalization.
- Verdict: reproduced ⇔ divergence over the n outputs > noise.

## LangSmith logging

- **One LangSmith project per case** (e.g. `repro-<source_hash>-ev<idx>`),
  the n tries logged as n root runs — so `trace_analyser.py` can be pointed
  at that project directly with no changes.

## Test layer

- Pytest, parametrized per case with readable ids, e.g.
  `test_replay_reproduces_divergence[025f879c_ev12_ChatOpenAI]`.
- Assertion direction: **FAIL = divergence reproduced** in isolation
  (revised 2026-07-14): a fully deterministic endpoint is all green. A red
  test means the same input still produces divergent outputs even with
  batching/load removed.

## vLLM determinism suite integration

- The existing suite in `tests/v1/determinism/` (batch invariance:
  engine, online, matmul/RMSNorm/NVFP4 kernels) is **not duplicated** —
  the replay conftest includes it via flags.
- All three tiers **enabled by default** (flag-guarded, disable
  individually); the target machine (user's remote box, where this will
  run) has vLLM + GPU + the serving endpoint locally.

## .env contents

- `N_TRIES` — replay attempts per case.
- `LANGSMITH_API_KEY` — both for re-fetching source runs and logging repro runs.
- Model endpoint URL + API key (original endpoint is reachable).
- Tier flags for the vLLM suite (default true).
- Case-scope overrides optional; default = all qualifying events.

## Non-goals

- No substitute-endpoint mode, no temperature-0 arm, no concurrent replay
  arm (isolation is sequential-only by decision).
