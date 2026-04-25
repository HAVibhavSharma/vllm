#!/usr/bin/env bash
# Run the CacheBlend / KVLink end-to-end benchmark on current vLLM (v1).
#
# Prereqs (see /Users/vibhavsharma/Projects/vllm/AGENTS.md):
#   uv venv --python 3.12
#   source .venv/bin/activate
#   VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
#   uv pip install xformers   # CacheBlend / KVLink path uses xformers
#   uv pip install -r benchmarks_ours/requirements.txt
#
# The --attention-backend is forced to XFORMERS inside main.py, so the
# CacheBlend hooks in vllm/v1/attention/backends/xformers.py fire.
#
# Run from the repo root (so `benchmarks_ours.*` imports resolve):
#   cd /Users/vibhavsharma/Projects/vllm

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
DATASET="${DATASET:-2wikimqa}"
APPROACH="${APPROACH:-cacheblend-15}"

exec .venv/bin/python -m benchmarks_ours.evals.e2e.main \
    --model   "$MODEL" \
    --dataset "$DATASET" \
    --approach "$APPROACH"
