"""Bound the item count of an OpenAI Responses ``input`` array.

Some gateways reject a request whose ``input`` holds more than a fixed number
of items (a Volcengine-backed route answers ``Maximum of 1000 items allowed in
input``). An agent session emits one item per message, tool call, tool result
and reasoning block, so a long session reaches that cap long before the model's
token budget is exhausted.

Reduction only runs when a provider configures ``max_input_items``:

1. Adjacent message items with the same role are merged; this loses nothing.
2. If the array is still too long, the oldest items are dropped as a prefix.
   The first message item is kept so the original task survives, and the
   remainder never starts with a ``function_call_output`` whose call is gone.
"""

from __future__ import annotations

from typing import Any

_MESSAGE = "message"
_FUNCTION_CALL_OUTPUT = "function_call_output"


def merge_adjacent_messages(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge consecutive message items that share a role, preserving order."""
    merged: list[dict[str, Any]] = []
    for item in items:
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and previous.get("type") == _MESSAGE
            and item.get("type") == _MESSAGE
            and previous.get("role") == item.get("role")
        ):
            merged[-1] = {
                **previous,
                "content": list(previous.get("content") or []) + list(item.get("content") or []),
            }
            continue
        merged.append(item)
    return merged


def fit_input_items(
    items: list[dict[str, Any]], limit: int | None
) -> tuple[list[dict[str, Any]], int]:
    """Return (items, dropped) where ``items`` holds at most ``limit`` entries."""
    if not limit or limit <= 0 or len(items) <= limit:
        return items, 0

    items = merge_adjacent_messages(items)
    if len(items) <= limit:
        return items, 0

    head = 1 if items and items[0].get("type") == _MESSAGE else 0
    cut = max(len(items) - limit + head, head)
    while cut < len(items) and items[cut].get("type") == _FUNCTION_CALL_OUTPUT:
        cut += 1

    kept = items[:head] + items[cut:]
    return kept, len(items) - len(kept)
