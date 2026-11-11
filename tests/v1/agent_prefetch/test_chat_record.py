# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plain `/v1/chat/completions` calls that ask to be recorded are recorded.

B (pitlane's `registry_record`): a ReAct node's next turn extends this turn's
prompt, so the prefetch before it warms this whole prompt -- which only works
if this call put it in the prefix registry. swe-agent sends its calls to
`/v1/chat/completions` with `record_in_registry` and `agent_id` as extra
fields; with `auto_register.py` gone nothing read them, every ReAct prefetch
saw only the system-prompt seed (`available=1`), and arm AB ran as arm A.
"""

from types import SimpleNamespace

from vllm.entrypoints.openai.agent_chat.api_router import (
    record_chat_completion_prefix,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import _identity_extra_args

AGENT = "langgraph:2:swe_architect:conduct_research"
PROMPT = list(range(100))  # six full 16-token chunks and four left over


def _record(state, prompt=PROMPT, **extra):
    return record_chat_completion_prefix(
        state, None, extra_args=extra, model_name="m", prompt_token_ids=prompt
    )


def _entries(state, agent_id=AGENT):
    registry = getattr(state, "agent_prefetch_registry", None)
    return registry.get_all(agent_id) if registry is not None else []


def test_a_call_that_asks_is_recorded_under_its_agent_id():
    state = SimpleNamespace()
    assert _record(state, record_in_registry=True, agent_id=AGENT, job_id="2")
    (entry,) = _entries(state)
    assert entry.token_ids == tuple(PROMPT[:96]), "chunk-aligned, not the tail"
    assert entry.cache_salt == f"agent::{AGENT}", "the salt /v1/agent_chat uses"


def test_a_call_that_does_not_ask_is_not_recorded():
    for extra in ({"agent_id": AGENT},
                  {"agent_id": AGENT, "record_in_registry": False},
                  {"agent_id": AGENT, "record_in_registry": "false"}):
        state = SimpleNamespace()
        assert not _record(state, **extra)
        assert _entries(state) == []


def test_without_a_call_id_nothing_is_recorded():
    for agent_id in (None, "", "langgraph:*:**:conduct_research"):
        state = SimpleNamespace()
        assert not _record(state, record_in_registry=True, agent_id=agent_id)


def test_a_prompt_shorter_than_a_chunk_is_not_recorded():
    state = SimpleNamespace()
    assert not _record(state, prompt=list(range(15)), record_in_registry=True,
                       agent_id=AGENT)
    assert _entries(state) == []


def test_each_turn_is_its_own_entry_and_a_repeat_is_not():
    state = SimpleNamespace()
    turn1, turn2 = list(range(64)), list(range(96))  # turn 2 extends turn 1
    for prompt in (turn1, turn2, turn2):
        _record(state, prompt=prompt, record_in_registry=True, agent_id=AGENT)
    newest_first = [len(e.token_ids) for e in _entries(state)]
    assert newest_first == [96, 64], "the newest, longest prompt comes first"


def test_the_fields_are_read_from_extra_body_and_vllm_xargs():
    # swe-agent sends them as top-level extra_body fields (model_extra).
    request = ChatCompletionRequest.model_validate({
        "model": "m",
        "messages": [{"role": "user", "content": "x"}],
        "agent_id": AGENT,
        "record_in_registry": True,
    })
    extra = _identity_extra_args(request)
    assert extra["agent_id"] == AGENT and extra["record_in_registry"] is True
    request = ChatCompletionRequest.model_validate({
        "model": "m",
        "messages": [{"role": "user", "content": "x"}],
        "vllm_xargs": {"agent_id": AGENT, "record_in_registry": True},
    })
    assert _identity_extra_args(request)["agent_id"] == AGENT
