# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated-replay determinism tests, one per harvested divergence case.

Each test replays a production LLM call that diverged across runs despite
identical input: the exact same request is sent N_TRIES times, strictly
sequentially, to the original endpoint with the original config. All tries
are logged to a per-case LangSmith project (repro-<case_id>) so
trace_analyser.py can analyze them directly.

PASS (green) = the endpoint is deterministic in isolation: divergence over
the n outputs is within trace_analyser's NOISE threshold.
FAIL (red)   = the divergence reproduced even without batching/load.

Run:
  python3 extract_cases.py          # once, to harvest cases
  pytest test_replay_divergence.py  # replays every case
"""
import pytest

import replay_lib as rl


def test_cases_extracted():
    """Guard: extraction must have produced at least one case."""
    if not rl.load_cases():
        pytest.skip("no cases in replay/cases/ — run extract_cases.py "
                    "first (or all source LangSmith projects expired; see "
                    "cases/coverage_summary.json)")


def test_replay_is_deterministic_in_isolation(case, replay_env,
                                              replay_results):
    result = replay_results(case)
    assert not result["reproduced"], (
        "Divergence REPRODUCED in isolation for %s: divergence=%.4f > "
        "noise=%.4f over %d sequential tries (%d distinct outputs; "
        "original divergence was %.4f). Inspect LangSmith project %r "
        "with trace_analyser.py." % (
            case["case_id"], result["divergence"],
            result["noise_threshold"], result["n_tries"],
            result["distinct_outputs"],
            result["original_output_divergence"] or 0,
            result["langsmith_project"]))


def test_replay_matches_trace_output_exactly(case, replay_env,
                                             replay_results):
    """Byte-to-byte comparison against the ORIGINAL trace output: every
    replay try must equal the source run's output exactly (rendered the
    same way: content + tool calls). Stricter than the divergence test —
    the endpoint can be self-consistent yet produce a different answer
    than it did in production."""
    if case.get("original_output") is None:
        pytest.skip("case predates original_output capture — re-run "
                    "extract_cases.py --refresh")
    result = replay_results(case)
    n, matched = result["n_tries"], result["exact_match_tries"]
    assert matched == n, (
        "Replay output differs from the original trace for %s: only %d/%d "
        "tries matched byte-to-byte (LangSmith project %r; outputs in "
        "results/%s.json)." % (case["case_id"], matched, n,
                               result["langsmith_project"],
                               case["case_id"]))
