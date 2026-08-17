# Pinned-mode replay (ODR trace store)

Source: `open_deep_research/trace_store.py`, `open_deep_research/replay_prefetch.py`.

## The path a single chat completion takes

```mermaid
sequenceDiagram
    autonumber
    participant N as ODR node<br/>(supervisor / researcher / …)
    participant LC as ChatOpenAI<br/>+ extra_body{job_id, langgraph_node, agent_id}
    participant T as TraceTransport<br/>(httpx AsyncBaseTransport)
    participant S as TraceStore<br/>(trace.jsonl in RAM)
    participant P as replay_prefetch<br/>(thread pools, non-blocking)
    participant V as vLLM
    participant R as divergence.jsonl

    N->>LC: invoke()
    LC->>T: POST /v1/chat/completions

    T->>T: key = sha256(model, messages, tools,<br/>temperature, seed, max_tokens, …)<br/>job_id = body.job_id
    T->>S: take(job_id, key)

    alt miss (trajectory diverged)
        S-->>T: None
        T->>R: {reason:"miss"}
        alt ON_MISS=strict
            T-->>LC: raise RuntimeError
        else live
            T->>V: real call, real response
        end
    else hit
        S-->>T: entry{response_body, stream, record_latency_ns, _seq}

        Note over T,P: ORACLE — recording already knows<br/>the tool calls, before the live call is sent
        T-)P: schedule(body, entry.response_body, entry, store)
        par successor warm
            P->>P: _tool_names(recording) → _targets()<br/>via graph.transition_prediction rules
            P->>V: POST /v1/agents/prefetch<br/>(agent_id, top_k=1, wait=False)
        and tool-internal warm
            P->>S: following(entry, node=*_tools,<br/>stop_node=producer, limit=4)
            P->>V: POST /v1/agents/prefetch<br/>text=recorded page prompt (seed, serialized)
        end

        T->>T: pinned_body = body<br/>max_(completion_)tokens = recorded completion_tokens<br/>(no ignore_eos — breaks guided decoding)
        T->>V: live call (timing only)
        V-->>T: live body → live_completion_tokens, live_error
        T->>R: {reason:"pinned", live_latency_ns,<br/>pinned_max_tokens, live_completion_tokens,<br/>baseline_latency_ns}
        T-->>LC: DISCARD live output,<br/>return entry.response_body
    end

    LC-->>N: recorded message
    N->>N: tool phase (tavily_search → TAVILY_CACHE_DIR disk read)
    N->>LC: next call
```

## Why the gap between calls is so small on replay

The last two steps of the diagram are where it comes from. Mostly expected, not a bug:

- **The tool phase collapses.** `utils.py:416` serves Tavily from `TAVILY_CACHE_DIR`, so what
  was an HTTP round trip plus page fetches during record becomes a local JSON read. The
  inter-call gap in a record run is dominated by that; in replay it is ~0.
- **Live calls can finish under baseline.** `max_tokens` is capped at the recorded count but
  EOS is deliberately not suppressed (`trace_store.py:433-444`), so a live call that stops
  early returns before the recorded one did. Check `live_completion_tokens` against
  `pinned_max_tokens` in the divergence report — if it is consistently lower, the timing arm
  is under-counting decode.
- **The gap is not doing real work anyway.** Prefetch and seed POSTs run on `_EXECUTOR` /
  `_SEED_EXECUTOR` threads (`replay_prefetch.py:105,116`), so they never occupy the gap; and
  parallel researcher units interleave, compressing wall-clock gaps further.

If replay should reproduce baseline *pacing* rather than just baseline token counts, the
missing piece is that nothing replays `record_latency_ns` — it is recorded, and only ever
written back into the divergence report, never slept on.

## Modes

| `ODR_TRACE_MODE` | Server contact | Response the agent sees |
|---|---|---|
| `off` | live | live |
| `record` | live | live (request/response appended to trace) |
| `pinned` | live, capped at recorded `completion_tokens` | recorded |
| `offline` | none | recorded |
