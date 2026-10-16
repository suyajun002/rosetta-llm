"""A client disconnect must not leak the upstream connection.

An abandoned async generator keeps its httpx connection - and therefore its slot
in the provider's connection pool - until the event loop finalizes it, which can be
arbitrarily late. Once the pool filled up, ``pool=10.0`` made every later request
fail with a blank-message 502 ``upstream_error``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import orjson

from rosetta.pipeline import (
    _passthrough_stream_with_recovery,
    _translate_stream,
    _upstream_error_message,
)

SSE_EVENTS: list[dict[str, Any]] = [
    {"type": "response.created", "response": {"model": "test"}},
    {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message"}},
    {"type": "response.output_text.delta", "output_index": 0, "delta": "OK"},
    {"type": "response.output_text.done", "output_index": 0},
    {"type": "response.completed", "response": {"status": "completed"}},
]


class TrackedUpstream:
    """Upstream whose byte streams count how often they were closed."""

    def __init__(self) -> None:
        self.started = 0
        self.closed = 0

    def stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[bytes]:
        del args, kwargs

        async def iterator() -> AsyncIterator[bytes]:
            self.started += 1
            try:
                for event in SSE_EVENTS:
                    yield b"data: " + orjson.dumps(event) + b"\n\n"
                yield b"data: [DONE]\n\n"
            finally:
                self.closed += 1

        return iterator()


class DisconnectingRequest:
    async def is_disconnected(self) -> bool:
        return True


class ConnectedRequest:
    async def is_disconnected(self) -> bool:
        return False


async def test_translate_stream_closes_upstream_when_client_disconnects() -> None:
    upstream = TrackedUpstream()
    chunks = [
        chunk
        async for chunk in _translate_stream(
            "anthropic",
            "openai_responses",
            {},
            "skybj",
            "/responses",
            DisconnectingRequest(),  # type: ignore[arg-type]
            upstream,  # type: ignore[arg-type]
            {},
            {},
        )
    ]

    assert chunks == []
    assert upstream.started == 1
    assert upstream.closed == 1


async def test_translate_stream_closes_upstream_when_it_completes() -> None:
    upstream = TrackedUpstream()
    chunks = [
        chunk
        async for chunk in _translate_stream(
            "anthropic",
            "openai_responses",
            {},
            "skybj",
            "/responses",
            ConnectedRequest(),  # type: ignore[arg-type]
            upstream,  # type: ignore[arg-type]
            {},
            {},
        )
    ]

    assert chunks
    assert upstream.closed == 1


async def test_passthrough_closes_upstream_when_client_disconnects() -> None:
    upstream = TrackedUpstream()
    gen = upstream.stream("skybj", "/responses", {})
    chunks = [
        chunk
        async for chunk in _passthrough_stream_with_recovery(
            gen, DisconnectingRequest(), "skybj", "anthropic", {}  # type: ignore[arg-type]
        )
    ]

    assert chunks == []
    assert upstream.closed == 1


def test_pool_timeout_reports_a_non_empty_reason() -> None:
    # httpcore raises these without a message, so httpx ends up with str() == "".
    message = _upstream_error_message(httpx.PoolTimeout(""))

    assert message
    assert "pool" in message
    assert "ReadError" in _upstream_error_message(httpx.ReadError(""))


def test_error_message_keeps_the_original_text_when_present() -> None:
    assert _upstream_error_message(httpx.ReadError("connection reset")) == "connection reset"
