# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`POST /v1/kv_metrics/reset` — the end-of-warmup marker.

A benchmark that measures KV cache behaviour has to run a warmup pass first,
or every arm is measured against a cache that happens to be empty. The warmup
is then inside the numbers, and there is no way to subtract it after the fact:

* `ttft_ms`, `ttft_p50_ms` and `ttft_p95_ms` are computed over retained
  samples, and a warmup's cold prefills are the tail. The percentiles keep
  reporting them long after the warmup ended.
* `hit_rate` is cumulative over the life of the server. A cold pass drags it
  down permanently, and by an amount that depends on how long the measured run
  happened to be.
* Anything rate-like derived from the log is implicitly over "time since the
  server started", which includes the warmup.

So the workload says when its warmup is done, and the engine starts a new
measurement epoch: counters to zero, `epoch` incremented on every subsequent
`kv_hbm` line, and a `kv_hbm_reset` marker line written at the boundary. The
last pre-reset `kv_hbm` line is emitted first, so the warmup's numbers are
still in the log rather than discarded.

By default this touches no cached data. With `flush_hbm=true` it additionally
drops every resident block from the GPU prefix cache at the same instant --
but **not** a KV connector's store, so with a CPU tier (LMCache and friends)
configured the warm phase starts cold in HBM and warm one level down, and the
prefixes the cold phase produced are pulled back up instead of recomputed.
Without a connector there is nothing underneath and `flush_hbm` simply makes
the warm phase cold; the response reports which case the server is in via
`kv_connector_configured`.

The route is attached unconditionally — unlike `/reset_prefix_cache`, which
lives behind `VLLM_SERVER_DEV_MODE`. Requiring dev mode to mark a measurement
boundary would mean the benchmark server runs a different configuration from
the one being benchmarked.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.responses import JSONResponse

from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


@router.post("/v1/kv_metrics/reset")
async def reset_kv_metrics(
    raw_request: Request,
    label: str = Query(
        default="cold_done",
        description="Recorded on the `kv_hbm_reset` marker line and echoed "
        "on every `kv_hbm` line of the new epoch. Name the phase that just "
        "ended, e.g. `cold_done`.",
    ),
    flush_hbm: bool = Query(
        default=False,
        description="Also drop every resident block from the GPU prefix "
        "cache, leaving any KV connector (CPU tier) untouched. Use to start "
        "the warm phase with cold HBM in front of a warm LM cache.",
    ),
):
    """Zero the KV measurement counters. Optionally empty HBM with them."""
    engine_client = getattr(raw_request.app.state, "engine_client", None)
    reset = getattr(engine_client, "reset_kv_metrics", None)
    if reset is None:
        # Not an error the caller can fix, and not one worth failing a run
        # over: say so and let the harness record that its numbers still
        # contain the warmup.
        logger.warning(
            "kv_metrics: reset requested but this engine client does not "
            "support it (%s)",
            type(engine_client).__name__,
        )
        return JSONResponse(
            content={
                "ok": False,
                "reason": "engine_client_unsupported",
                "engine_client": type(engine_client).__name__,
            },
            status_code=501,
        )

    logger.info(
        "kv_metrics: resetting measurements (label=%s, flush_hbm=%s)",
        label,
        flush_hbm,
    )
    result: dict[str, Any] = await reset(label, flush_hbm)
    if not result.get("ok"):
        # The engine reached, but had nothing to reset — the instrumentation
        # is off. 200 with `ok: false`: the request was handled correctly, and
        # the harness needs the reason, not an exception.
        logger.warning(
            "kv_metrics: nothing to reset (%s)", result.get("reason", "unknown")
        )
    return JSONResponse(content=result)


@router.get("/v1/kv_metrics")
async def get_kv_metrics(raw_request: Request):
    """Read the KV measurements in place, epoch untouched.

    Includes the cross-question reuse matrix when `VLLM_KV_PROVENANCE=1`:
    `reuse_by_source_job` (tokens each job supplied to other jobs),
    `reuse_by_consumer_job` (tokens each job took from other jobs) and
    `reuse_top_pairs` (the largest source -> consumer flows). The per-request
    detail behind those totals goes to `VLLM_KV_PROVENANCE_PATH` as JSONL;
    this endpoint is the rollup.

    Separate from the reset because reading and discarding are different
    acts: a harness that wants the numbers halfway through a run must not
    have to end the measurement window to see them.
    """
    engine_client = getattr(raw_request.app.state, "engine_client", None)
    snapshot = getattr(engine_client, "get_kv_metrics", None)
    if snapshot is None:
        return JSONResponse(
            content={
                "ok": False,
                "reason": "engine_client_unsupported",
                "engine_client": type(engine_client).__name__,
            },
            status_code=501,
        )
    result: dict[str, Any] = await snapshot()
    return JSONResponse(content=result)


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
