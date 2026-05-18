# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Agent-scoped KV prefix prefetch.

This package implements the building blocks for the
``/v1/agents/chat/completions`` endpoint that proactively warms vLLM's
GPU paged-KV cache from LMCache before a request's prefill stage.

Components in this commit:

* :class:`PrefixDescriptor` -- a single recordable prefix entry.
* :class:`AgentPrefixRegistry` -- per-agent LRU of recent prefix
  descriptors with bounded total size.
* :func:`compute_prefix_hash` -- stable identity hash for a
  (model, cache_salt, token_ids) tuple.
* :func:`chunk_align` -- floor a token-id sequence to a multiple of
  the LMCache chunk size.

The phantom-request submitter and the public HTTP route land in
follow-up commits; see ``plan/agent_prefetch_plan.md``.
"""

from vllm.v1.agent_prefetch.hashing import (
    DEFAULT_CHUNK_SIZE,
    chunk_align,
    compute_prefix_hash,
)
from vllm.v1.agent_prefetch.registry import (
    AgentPrefixRegistry,
    PrefixDescriptor,
)

__all__ = [
    "AgentPrefixRegistry",
    "DEFAULT_CHUNK_SIZE",
    "PrefixDescriptor",
    "chunk_align",
    "compute_prefix_hash",
]
