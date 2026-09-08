# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Access log filter for uvicorn to exclude specific endpoints from logging.

This module provides a logging filter that can be used to suppress access logs
for specific endpoints (e.g., /health, /metrics) to reduce log noise in
production environments.
"""

import logging
from contextvars import ContextVar
from urllib.parse import urlparse

# The agent an in-flight request belongs to, for the access log.
#
# Uvicorn writes its access line from the ASGI scope, which carries the path
# and nothing about the body, so the endpoint has to hand the identity over
# out of band.
#
# It holds a *mutable dict*, not the id itself, and that is the whole point.
# A plain ContextVar only works when the handler runs in the task uvicorn
# logs from, and `@with_cancellation` breaks exactly that: it runs the
# handler under `asyncio.create_task`, which copies the context, so a value
# set inside is invisible to the caller. Measured on a live server -- the
# undecorated `/v1/agents/prefetch` line carried the agent, the decorated
# `/v1/agents/chat/completions` line did not.
#
# A dict installed *before* the task is created is shared by reference with
# every child task, so a handler can fill it in from wherever it runs and the
# logger still sees it. `AgentRequestContextMiddleware` is what installs one
# per request; without it the fallback below still covers a same-task handler.
request_agent_ctx: ContextVar[dict | None] = ContextVar(
    "request_agent_ctx", default=None
)


def set_request_agent_id(agent_id: str | None) -> None:
    """Name the agent this request belongs to, for its access log line."""
    holder = request_agent_ctx.get()
    if holder is None:
        # No middleware installed one. Works only if this runs in the task
        # uvicorn logs from, which is the case for an undecorated handler.
        holder = {}
        request_agent_ctx.set(holder)
    holder["agent_id"] = agent_id or None


def get_request_agent_id() -> str | None:
    holder = request_agent_ctx.get()
    return holder.get("agent_id") if holder else None


class AgentRequestContextMiddleware:
    """Give each request a holder the access logger can read afterwards.

    Pure ASGI on purpose: `BaseHTTPMiddleware` would run the rest of the
    stack in its own task and reintroduce the isolation this exists to
    defeat. Installed per request and deliberately never reset -- uvicorn
    writes its line *after* this returns, and each request already has its
    own context, so nothing leaks between them.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            request_agent_ctx.set({})
        await self.app(scope, receive, send)


class UvicornAccessLogFilter(logging.Filter):
    """
    A logging filter that excludes access logs for specified endpoint paths.

    This filter is designed to work with uvicorn's access logger. It checks
    the log record's arguments for the request path and filters out records
    matching the excluded paths.

    Uvicorn access log format:
        '%s - "%s %s HTTP/%s" %d'
        (client_addr, method, path, http_version, status_code)

    Example:
        127.0.0.1:12345 - "GET /health HTTP/1.1" 200

    Args:
        excluded_paths: A list of URL paths to exclude from logging.
                       Paths are matched exactly.
                       Example: ["/health", "/metrics"]
    """

    def __init__(self, excluded_paths: list[str] | None = None):
        super().__init__()
        self.excluded_paths = set(excluded_paths or [])

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Determine if the log record should be logged.

        Args:
            record: The log record to evaluate.

        Returns:
            True if the record should be logged, False otherwise.
        """
        if not self.excluded_paths:
            return True

        # This filter is specific to uvicorn's access logs.
        if record.name != "uvicorn.access":
            return True

        # The path is the 3rd argument in the log record's args tuple.
        # See uvicorn's access logging implementation for details.
        log_args = record.args
        if isinstance(log_args, tuple) and len(log_args) >= 3:
            path_with_query = log_args[2]
            # Get path component without query string.
            if isinstance(path_with_query, str):
                path = urlparse(path_with_query).path
                if path in self.excluded_paths:
                    return False

        return True


class AgentAccessFormatter:
    """Uvicorn's access formatter, plus the agent the request belongs to.

    Subclassed lazily (at construction, not import) so this module does not
    import uvicorn -- it is imported by the CLI path too, where uvicorn may
    not be installed.

    The field is appended rather than interpolated into the format string so
    a line for a non-agent endpoint is byte-identical to what it was before.
    """

    def __new__(cls, *args, **kwargs):
        from uvicorn.logging import AccessFormatter

        class _Formatter(AccessFormatter):
            def formatMessage(self, record: logging.LogRecord) -> str:
                line = super().formatMessage(record)
                agent = get_request_agent_id()
                return f"{line} agent={agent}" if agent else line

        return _Formatter(*args, **kwargs)


def create_uvicorn_log_config(
    excluded_paths: list[str] | None = None,
    log_level: str = "info",
) -> dict:
    """
    Create a uvicorn logging configuration with access log filtering.

    This function generates a logging configuration dictionary that can be
    passed to uvicorn's `log_config` parameter. It sets up the access log
    filter to exclude specified paths.

    Args:
        excluded_paths: List of URL paths to exclude from access logs.
        log_level: The log level for uvicorn loggers.

    Returns:
        A dictionary containing the logging configuration.

    Example:
        >>> config = create_uvicorn_log_config(["/health", "/metrics"])
        >>> uvicorn.run(app, log_config=config)
    """
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "access_log_filter": {
                "()": UvicornAccessLogFilter,
                "excluded_paths": excluded_paths or [],
            },
        },
        "formatters": {
            # Timestamped to match vllm.logger's own format. Without this the
            # access lines carry only ordering, so a request cannot be placed
            # against the engine's timeline -- which is the whole reason to
            # read them next to the scheduler and prefetch logs.
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(asctime)s.%(msecs)03d %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
                "use_colors": None,
            },
            "access": {
                "()": "vllm.logging_utils.access_log_filter.AgentAccessFormatter",
                "fmt": '%(levelprefix)s %(asctime)s.%(msecs)03d %(client_addr)s - "%(request_line)s" %(status_code)s',  # noqa: E501
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "filters": ["access_log_filter"],
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["default"],
                "level": log_level.upper(),
                "propagate": False,
            },
            "uvicorn.error": {
                "level": log_level.upper(),
                "handlers": ["default"],
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["access"],
                "level": log_level.upper(),
                "propagate": False,
            },
        },
    }
    return config
