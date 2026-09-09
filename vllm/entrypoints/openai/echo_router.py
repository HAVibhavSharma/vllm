# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`POST /v1/echo` — put a client-side event on the server's timeline.

A prefetch experiment has two clocks. The server's log holds everything the
engine does: arrivals, prefills, `kv_hbm_ttft`, the prefetch POSTs themselves.
The client's own decisions live in a separate JSONL with a separate clock, and
joining the two after the fact means trusting that two machines' wall clocks
agree to within the intervals being measured — which for a warm that needs to
land inside a few hundred milliseconds, they do not.

So the client posts its events here and the server logs them in its own
stream, in order, against its own clock. Nothing is stored and nothing is
interpreted: the body is logged and handed back.

The two events this exists for bracket how much lead a prefetch could have
had:

* `min_lead` — the graph runtime has just parsed the tool call that names the
  next node, mid-decode of the producing response. That is the earliest a
  real predictor can know, so it is the *least* lead any prefetch for that
  node can be issued with.
* `max_lead` — the workflow-level oracle issued the warm. It reads the
  recording, so it can know arbitrarily far ahead; this is the *most* lead
  available.

Both carry the `agent_id`, so the pair brackets the window for one target and
the real request's own line closes it. What sits between `min_lead` and the
chat completion is what a predictor-driven arm has to work with; what sits
between `max_lead` and `min_lead` is what the oracle is buying over it.

Unconditional, like `/v1/kv_metrics/reset` and for the same reason: a marker
endpoint behind `VLLM_SERVER_DEV_MODE` would mean the benchmarked server is
configured differently from the one under test.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()

# A marker is a log line, not a payload sink. Anything larger than this is a
# caller sending data rather than an event, and truncating it keeps one bad
# call from burying the log it was meant to annotate.
_MAX_LOGGED_CHARS = 2048


def _format(body: Any) -> str:
    """Render a marker body as one line, most useful fields first.

    A dict is rendered as `key=value` pairs with `event` and `agent_id`
    leading, because those are what the log is grepped on. Anything else is
    rendered as-is -- the endpoint does not require a shape.
    """
    if not isinstance(body, dict):
        return str(body)[:_MAX_LOGGED_CHARS]
    lead = [key for key in ("event", "agent_id") if key in body]
    rest = sorted(key for key in body if key not in lead)
    rendered = " ".join(f"{key}={body[key]}" for key in (*lead, *rest))
    return rendered[:_MAX_LOGGED_CHARS]


@router.post("/v1/echo")
async def echo(raw_request: Request):
    """Log the posted body on the server's timeline and hand it back."""
    try:
        body = await raw_request.json()
    except Exception:  # noqa: BLE001 - a marker must never fail its caller
        raw = (await raw_request.body()).decode("utf-8", errors="replace")
        body = {"raw": raw}
    logger.info("echo: %s", _format(body))
    # `received_at` is the server clock the marker landed on, which is the
    # whole point of routing it through here. Returned as well as logged so a
    # caller can measure its own round trip if it wants to.
    return JSONResponse(
        content={"ok": True, "received_at": time.time(), "echo": body}
    )


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
