# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pytest plumbing for the divergence-replay suite.

Parametrizes replay tests over the case files in replay/cases/ (produced
by extract_cases.py). Collection is offline: no network until a test runs.
"""
import pytest

import replay_lib as rl


def pytest_generate_tests(metafunc):
    if "case" in metafunc.fixturenames:
        cases = rl.load_cases()
        metafunc.parametrize("case", cases,
                             ids=[c["case_id"] for c in cases])


@pytest.fixture(scope="session")
def replay_results():
    """Session-wide cache: each case is replayed (paid LLM calls) exactly
    once, even though two tests judge its outputs."""
    cache = {}

    def get(case):
        if case["case_id"] not in cache:
            cache[case["case_id"]] = rl.replay_case(case)
        return cache[case["case_id"]]

    return get


@pytest.fixture(scope="session")
def replay_env():
    """Fail fast with a clear message when replay credentials are absent."""
    missing = []
    if not rl.LS_API_KEY:
        missing.append("LANGSMITH_API_KEY")
    if not rl.MODEL_BASE_URL:
        missing.append("REPLAY_MODEL_BASE_URL")
    if missing:
        pytest.skip("replay/.env is missing: %s" % ", ".join(missing))
