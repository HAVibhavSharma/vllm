# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flag-guarded wrappers over vLLM's existing batch-invariance suite
(tests/v1/determinism/), so one `pytest replay/` run covers both the
isolated-replay cases and all three vLLM determinism tiers.

Flags (replay/.env, all default ON):
  RUN_VLLM_ENGINE_TESTS   end-to-end engine tests (test_batch_invariance.py)
  RUN_VLLM_ONLINE_TESTS   OpenAI-server variant (test_online_batch_invariance.py)
  RUN_VLLM_KERNEL_TESTS   kernel tiers (matmul / RMSNorm / NVFP4)

The upstream tests are invoked in a pytest subprocess (not imported) so
this directory never duplicates them and they keep their own conftest.
NOTE: run `pytest tests/v1/determinism/replay`, not the parent directory,
or the upstream suite executes twice.
"""
import os
import subprocess
import sys

import pytest

import replay_lib as rl

DETERMINISM_DIR = os.path.dirname(rl.REPLAY_DIR)

TIERS = {
    "engine": ("RUN_VLLM_ENGINE_TESTS", ["test_batch_invariance.py"]),
    "online": ("RUN_VLLM_ONLINE_TESTS", ["test_online_batch_invariance.py"]),
    "kernel": ("RUN_VLLM_KERNEL_TESTS", ["test_matmul_batch_invariant.py",
                                         "test_rms_norm_batch_invariant.py",
                                         "test_nvfp4_batch_invariant.py"]),
}


@pytest.mark.parametrize("tier", sorted(TIERS))
def test_vllm_batch_invariance_suite(tier):
    flag, files = TIERS[tier]
    if not rl.cfg_flag(flag, default=True):
        pytest.skip("%s=0 in replay/.env" % flag)
    paths = [os.path.join(DETERMINISM_DIR, f) for f in files]
    missing = [p for p in paths if not os.path.exists(p)]
    assert not missing, "vLLM determinism tests not found: %s" % missing
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *paths],
        cwd=DETERMINISM_DIR, capture_output=True, text=True)
    tail = "\n".join((proc.stdout or "").splitlines()[-30:])
    assert proc.returncode in (0, 5), (
        "vLLM %s-tier determinism tests failed (exit %d):\n%s\n%s"
        % (tier, proc.returncode, tail, (proc.stderr or "")[-2000:]))
    if proc.returncode == 5:
        pytest.skip("vLLM %s-tier tests were all skipped/deselected "
                    "(missing GPU/backend?)" % tier)
