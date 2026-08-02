"""Tests for finished request stats export."""

import csv
import json
from pathlib import Path

import pytest

from vllm.v1.engine import FinishReason
from vllm.v1.metrics.loggers import FileStatLogger
from vllm.v1.metrics.stats import (
    FinishedRequestStats,
    IterationStats,
    RequestStateStats,
)


def test_finished_request_stats_include_request_logger_fields():
    iteration_stats = IterationStats()
    req_stats = RequestStateStats(
        arrival_time=100.0,
        queued_ts=100.1,
        scheduled_ts=100.2,
        first_token_ts=100.5,
        last_token_ts=101.1,
    )
    req_stats.num_generation_tokens = 7

    iteration_stats.iteration_timestamp = 101.6
    iteration_stats.update_from_finished_request(
        finish_reason=FinishReason.STOP,
        request_id="chatcmpl-test",
        job_id="job-1",
        agent_id="planner",
        langgraph_node="node-1",
        call_type="tavily:summary",
        input_text="hello prompt",
        output_text="hello output",
        num_prompt_tokens=20,
        max_tokens_param=128,
        req_stats=req_stats,
        num_cached_tokens=5,
    )

    finished_request = iteration_stats.finished_requests[0]
    assert finished_request.request_id == "chatcmpl-test"
    assert finished_request.job_id == "job-1"
    assert finished_request.agent_id == "planner"
    assert finished_request.langgraph_node == "node-1"
    assert finished_request.call_type == "tavily:summary"
    assert finished_request.input_text == "hello prompt"
    assert finished_request.output_text == "hello output"
    assert finished_request.num_cached_tokens == 5
    assert finished_request.prefix_cache_hit_rate == 0.25
    assert finished_request.queued_time == 0.1
    assert finished_request.prefill_time == 0.3
    assert finished_request.inference_time == 0.9
    assert finished_request.decode_time == 0.6
    # Absolute wall-clock timestamps: the Trace Analyser infers tool
    # execution time from the gap between two requests of a job, and a gap
    # cannot be derived from durations.
    assert finished_request.arrival_ts == 100.0
    assert finished_request.finish_ts == pytest.approx(101.6)


def test_file_stat_logger_writes_csv_and_jsonl(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VLLM_REQUEST_STATS_DIR", str(tmp_path))

    logger = FileStatLogger(object(), engine_index=3)
    iteration_stats = IterationStats()
    iteration_stats.finished_requests.append(
        FinishedRequestStats(
            finish_reason=FinishReason.STOP,
            request_id="chatcmpl-123",
            job_id="1",
            agent_id="executor",
            langgraph_node="node-2",
            call_type="plan",
            input_text="user prompt",
            output_text="model output",
            e2e_latency=3.28,
            num_prompt_tokens=489,
            num_generation_tokens=95,
            num_cached_tokens=0,
            prefix_cache_hit_rate=0.0,
            queued_time=0.02,
            prefill_time=0.60,
            inference_time=2.82,
            decode_time=2.22,
            max_tokens_param=10000,
        )
    )

    logger.record(None, iteration_stats)
    logger._close()

    csv_files = sorted(tmp_path.glob("finished_requests_engine3_*.csv"))
    jsonl_files = sorted(tmp_path.glob("finished_requests_engine3_*.jsonl"))
    assert len(csv_files) == 1
    assert len(jsonl_files) == 1

    with csv_files[0].open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == [
        {
            "request_id": "chatcmpl-123",
            "job_id": "1",
            "agent_id": "executor",
            "langgraph_node": "node-2",
            "call_type": "plan",
            "input_text": "user prompt",
            "output_text": "model output",
            "finish_reason": "stop",
            "e2e_latency": "3.28",
            "num_prompt_tokens": "489",
            "num_generation_tokens": "95",
            "num_cached_tokens": "0",
            "num_computed_tokens": "0",
            "num_local_cached_tokens": "0",
            "num_external_cached_tokens": "0",
            "prefix_cache_hit_rate": "0.0",
            "queued_time": "0.02",
            "prefill_time": "0.6",
            "inference_time": "2.82",
            "decode_time": "2.22",
            "max_tokens_param": "10000",
            "arrival_ts": "0.0",
            "finish_ts": "0.0",
        }
    ]

    with jsonl_files[0].open() as f:
        lines = [json.loads(line) for line in f]
    assert lines == [
        {
            "request_id": "chatcmpl-123",
            "job_id": "1",
            "agent_id": "executor",
            "langgraph_node": "node-2",
            "call_type": "plan",
            "input_text": "user prompt",
            "output_text": "model output",
            "finish_reason": "stop",
            "e2e_latency": 3.28,
            "num_prompt_tokens": 489,
            "num_generation_tokens": 95,
            "num_cached_tokens": 0,
            "num_computed_tokens": 0,
            "num_local_cached_tokens": 0,
            "num_external_cached_tokens": 0,
            "prefix_cache_hit_rate": 0.0,
            "queued_time": 0.02,
            "prefill_time": 0.6,
            "inference_time": 2.82,
            "decode_time": 2.22,
            "max_tokens_param": 10000,
            "arrival_ts": 0.0,
            "finish_ts": 0.0,
        }
    ]


def test_file_stat_logger_escapes_multiline_text(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VLLM_REQUEST_STATS_DIR", str(tmp_path))

    logger = FileStatLogger(object(), engine_index=4)
    iteration_stats = IterationStats()
    iteration_stats.finished_requests.append(
        FinishedRequestStats(
            finish_reason=FinishReason.STOP,
            request_id="chatcmpl-escape",
            job_id="2",
            agent_id="agent",
            langgraph_node="node",
            input_text='user said "hello"\nnext line, with comma',
            output_text='model replied "ok"\nfinal line',
            e2e_latency=1.0,
            num_prompt_tokens=10,
            num_generation_tokens=2,
            num_cached_tokens=1,
            prefix_cache_hit_rate=0.1,
            queued_time=0.1,
            prefill_time=0.2,
            inference_time=0.5,
            decode_time=0.3,
            max_tokens_param=20,
        )
    )

    logger.record(None, iteration_stats)
    logger._close()

    csv_files = sorted(tmp_path.glob("finished_requests_engine4_*.csv"))
    assert len(csv_files) == 1

    with csv_files[0].open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["input_text"] == 'user said "hello"\nnext line, with comma'
    assert rows[0]["output_text"] == 'model replied "ok"\nfinal line'
