"""Tests for OpenAI Responses streaming codec behavior."""

from __future__ import annotations

from collections.abc import AsyncIterator

import orjson
import pytest

from rosetta.ir.events import ErrorEvent, MessageStopEvent, PartStopEvent
from rosetta.stream_codecs.openai_responses import parse


async def _sse_events(events: list[dict[str, object]]) -> AsyncIterator[bytes]:
    for event in events:
        yield b"data: " + orjson.dumps(event) + b"\n\n"
    yield b"data: [DONE]\n\n"


def _tool_call_events(deltas: list[str]) -> list[dict[str, object]]:
    """Build a Responses stream whose function-call arguments arrive as `deltas`."""
    return [
        {"type": "response.created", "response": {"model": "gpt-test"}},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "function_call", "call_id": "call_1", "name": "Bash"},
        },
        *(
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "delta": delta,
            }
            for delta in deltas
        ),
        {
            "type": "response.function_call_arguments.done",
            "output_index": 0,
            "arguments": "{}",
        },
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        },
    ]


async def _parse_deltas(deltas: list[str]) -> list[object]:
    return [event async for event in parse(_sse_events(_tool_call_events(deltas)))]


async def test_indented_tool_arguments_do_not_abort_stream() -> None:
    """Indented multi-line tool arguments must not be mistaken for a runaway stream.

    Deeply indented heredoc/script arguments legitimately stream as whitespace-only
    deltas (newline plus indentation). The guard must tolerate them instead of
    aborting the tool call with "Runaway whitespace in tool call arguments".
    """

    deltas = [
        '{"command": "python3 - <<\'PY\'',
        "\n" + " " * 16,
        "\n" + " " * 20,
        "\n" + " " * 24,
        "import ast",
        '\nPY"}',
    ]

    parsed = await _parse_deltas(deltas)

    assert [event for event in parsed if isinstance(event, ErrorEvent)] == []
    assert any(isinstance(event, PartStopEvent) for event in parsed)


async def test_runaway_whitespace_reports_once_past_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuine whitespace runaway is reported once, not once per later delta."""

    monkeypatch.setenv("ROSETTA_MAX_TOOL_ARG_WHITESPACE_RUN", "20")

    deltas = ['{"command": "x', *[" " * 30 for _ in range(4)], '"']

    parsed = await _parse_deltas(deltas)

    errors = [event for event in parsed if isinstance(event, ErrorEvent)]
    assert len(errors) == 1
    assert errors[0].message == "Runaway whitespace in tool call arguments"


async def test_whitespace_limit_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tolerated whitespace run comes from ROSETTA_MAX_TOOL_ARG_WHITESPACE_RUN."""

    deltas = ['{"command": "x', " " * 200, '"y"']

    monkeypatch.delenv("ROSETTA_MAX_TOOL_ARG_WHITESPACE_RUN", raising=False)
    assert [event for event in await _parse_deltas(deltas) if isinstance(event, ErrorEvent)] == []

    monkeypatch.setenv("ROSETTA_MAX_TOOL_ARG_WHITESPACE_RUN", "120")
    errors = [event for event in await _parse_deltas(deltas) if isinstance(event, ErrorEvent)]
    assert len(errors) == 1


async def test_responses_completed_emits_single_message_stop() -> None:
    """Responses completion should close the Anthropic stream exactly once.

    Some providers send both output_text.done and output_item.done, then
    response.completed, then [DONE]. Rosetta should normalize that into one
    content block stop and one message stop for Anthropic clients such as
    Claude Code.
    """

    events = [
        {"type": "response.created", "response": {"model": "gpt-test"}},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "message"},
        },
        {"type": "response.output_text.delta", "output_index": 0, "delta": "OK"},
        {"type": "response.output_text.done", "output_index": 0},
        {"type": "response.output_item.done", "output_index": 0},
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        },
    ]

    parsed = [event async for event in parse(_sse_events(events))]

    assert sum(isinstance(event, PartStopEvent) for event in parsed) == 1
    assert sum(isinstance(event, MessageStopEvent) for event in parsed) == 1
