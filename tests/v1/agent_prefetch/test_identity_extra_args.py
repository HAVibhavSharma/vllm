# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Where the request's identity actually lives.

`to_sampling_params` builds `extra_args` as `vllm_xargs` updated with
`model_extra` (`chat_completion/protocol.py`), so a client may put
`job_id` / `langgraph_node` in either place and engine core cannot tell the
difference. The auto-register hook read only `vllm_xargs`.

That split produced a failure with no error in it: the LangGraph integration
sends identity as top-level `extra_body` fields, which Pydantic collects into
`model_extra`, leaving `vllm_xargs` unset. Engine core therefore saw the full
identity — the `kv_hbm` line attributed evictions to `1:researcher` correctly
— while the registry was never written, so every `/v1/agents/prefetch` came
back `available=0 submitted=0` and answered 200 OK in under a millisecond.

These tests pin the two halves together. Nothing here needs a server.
"""

from types import SimpleNamespace

from vllm.entrypoints.openai.chat_completion.serving import _identity_extra_args
from vllm.v1.agent_prefetch.auto_register import agent_id_for_extra_args

IDENTITY = {
    "job_id": "1",
    "langgraph_node": "researcher",
    "call_type": "researcher",
    "agent_id": "langgraph:researcher",
}


def request(vllm_xargs=None, model_extra=None):
    return SimpleNamespace(vllm_xargs=vllm_xargs, model_extra=model_extra)


def test_identity_sent_as_top_level_extra_body_is_seen():
    """The regression. `extra_body={"langgraph_node": ...}` on the OpenAI
    client lands in `model_extra`, not `vllm_xargs`."""
    merged = _identity_extra_args(request(model_extra=dict(IDENTITY)))
    assert merged["langgraph_node"] == "researcher"
    assert agent_id_for_extra_args(merged) == "langgraph:researcher"


def test_identity_sent_as_vllm_xargs_is_still_seen():
    merged = _identity_extra_args(request(vllm_xargs=dict(IDENTITY)))
    assert agent_id_for_extra_args(merged) == "langgraph:researcher"


def test_model_extra_wins_over_vllm_xargs():
    """Same precedence as `to_sampling_params`. If these disagreed, the
    registry would be keyed on one node and the eviction index on another —
    the exact mismatch class this whole path keeps producing."""
    merged = _identity_extra_args(
        request(
            vllm_xargs={"langgraph_node": "stale"},
            model_extra={"langgraph_node": "researcher"},
        )
    )
    assert merged["langgraph_node"] == "researcher"


def test_both_sources_are_combined():
    merged = _identity_extra_args(
        request(
            vllm_xargs={"job_id": "1"},
            model_extra={"langgraph_node": "researcher"},
        )
    )
    assert merged == {"job_id": "1", "langgraph_node": "researcher"}
    assert agent_id_for_extra_args(merged) == "langgraph:researcher"


def test_no_identity_yields_no_agent_id():
    assert _identity_extra_args(request()) == {}
    assert agent_id_for_extra_args(_identity_extra_args(request())) is None


def test_the_request_is_not_mutated():
    """`to_sampling_params` merges into `self.vllm_xargs` in place. An
    observability read-out must not do the same — it runs on the serving hot
    path, before the sampling params are built."""
    xargs = {"job_id": "1"}
    req = request(vllm_xargs=xargs, model_extra={"langgraph_node": "researcher"})
    _identity_extra_args(req)
    assert xargs == {"job_id": "1"}
    assert req.vllm_xargs is xargs
