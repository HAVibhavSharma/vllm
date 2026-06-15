"""Unit tests for extra-body passthrough into SamplingParams.extra_args."""

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


def test_chat_completion_request_promotes_model_extra_fields():
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=10,
        job_id="job-1",
        agent_id="agent-1",
        langgraph_node="planner",
    )

    sampling_params = request.to_sampling_params(
        max_tokens=10,
        default_sampling_params={},
    )

    assert sampling_params.extra_args is not None
    assert sampling_params.extra_args["job_id"] == "job-1"
    assert sampling_params.extra_args["agent_id"] == "agent-1"
    assert sampling_params.extra_args["langgraph_node"] == "planner"


def test_completion_request_promotes_model_extra_fields():
    request = CompletionRequest(
        model="test-model",
        prompt="Hello",
        max_tokens=10,
        job_id="job-1",
        agent_id="agent-1",
        langgraph_node="planner",
    )

    sampling_params = request.to_sampling_params(
        max_tokens=10,
        default_sampling_params={},
    )

    assert sampling_params.extra_args is not None
    assert sampling_params.extra_args["job_id"] == "job-1"
    assert sampling_params.extra_args["agent_id"] == "agent-1"
    assert sampling_params.extra_args["langgraph_node"] == "planner"


def test_responses_request_promotes_model_extra_fields():
    request = ResponsesRequest(
        model="test-model",
        input="Hello",
        max_output_tokens=10,
        job_id="job-1",
        agent_id="agent-1",
        langgraph_node="planner",
    )

    sampling_params = request.to_sampling_params(default_max_tokens=1000)

    assert sampling_params.extra_args is not None
    assert sampling_params.extra_args["job_id"] == "job-1"
    assert sampling_params.extra_args["agent_id"] == "agent-1"
    assert sampling_params.extra_args["langgraph_node"] == "planner"
