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


def test_replay_is_deterministic_in_isolation(case, replay_env):
    result = rl.replay_case(case)
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
