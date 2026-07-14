# Divergence Replay Suite

Checks whether LLM output divergence observed in production traces
(analysed by [trace-analyser](~/Projects/trace-analyser)) is reproducible
**in isolation**: same input, same config, N sequential requests, no
batching. Design decisions are recorded in `PLAN.md`.

**Green = deterministic.** A test fails only when the divergence
reproduces even in isolation.

## Setup

```bash
cd tests/v1/determinism/replay
cp .env.example .env   # fill in keys/endpoint
```

## Workflow

```bash
# 1. Harvest cases: scans trace_analysis_*.json for LLM events with
#    identical input but divergent output, re-fetches exact payloads
#    from LangSmith, writes replay/cases/*.json + coverage_summary.json.
#    Expired LangSmith projects are skipped and reported, never fatal.
python3 extract_cases.py            # --refresh to re-extract

# 2. Replay: one pytest case per harvested event, e.g.
#    test_replay_is_deterministic_in_isolation[025f879c_c0_p3_generate]
#    Each case replays N_TRIES times sequentially against the original
#    endpoint and logs every try to LangSmith project repro-<case_id>.
pytest test_replay_divergence.py -v

# 3. Deep-dive any red case with the trace analyzer (works unchanged,
#    the replay runs are shaped like normal LangChain llm root runs):
python3 ~/Projects/trace-analyser/trace_analyser.py <repro project id>
```

Per-case verdicts also land in `results/<case_id>.json` (divergence,
distinct outputs, LangSmith run ids).

## vLLM batch-invariance tiers

`test_vllm_determinism_suite.py` runs the existing upstream suite in
`tests/v1/determinism/` (engine / online / kernel tiers) via subprocess,
each tier gated by a `.env` flag defaulting to ON.

Run `pytest tests/v1/determinism/replay` as the entry point — running the
parent directory would execute the upstream suite twice.

## Verdict metric

Divergence is computed by importing `divergence()` / `scrub_ids()` /
`NOISE` from `trace_analyser.py` itself (mean pairwise dissimilarity with
volatile IDs stripped, `TRACE_ANALYSER_NOISE` threshold) — identical to
how the original divergence was measured.
