"""Input item budgeting for OpenAI Responses upstreams."""

from __future__ import annotations

from rosetta.input_budget import fit_input_items, merge_adjacent_messages


def _message(role: str, text: str) -> dict[str, object]:
    return {
        "type": "message",
        "role": role,
        "content": [{"type": "input_text" if role == "user" else "output_text", "text": text}],
    }


def _call(call_id: str) -> dict[str, object]:
    return {"type": "function_call", "call_id": call_id, "name": "t", "arguments": "{}"}


def _output(call_id: str) -> dict[str, object]:
    return {"type": "function_call_output", "call_id": call_id, "output": "ok"}


def test_leaves_input_untouched_below_limit() -> None:
    items = [_message("user", "hi"), _call("c1"), _output("c1")]
    kept, dropped = fit_input_items(items, 10)
    assert kept == items
    assert dropped == 0


def test_no_limit_is_a_noop() -> None:
    items = [_message("user", "hi")] * 3
    kept, dropped = fit_input_items(items, None)
    assert kept == items
    assert dropped == 0


def test_merges_adjacent_messages_with_same_role_only() -> None:
    items = [
        _message("user", "a"),
        _message("user", "b"),
        _message("assistant", "c"),
        _message("user", "d"),
    ]
    merged = merge_adjacent_messages(items)
    assert len(merged) == 3
    assert [c["text"] for c in merged[0]["content"]] == ["a", "b"]
    assert merged[1]["role"] == "assistant"


def test_merge_does_not_mutate_input_items() -> None:
    first = _message("user", "a")
    items = [first, _message("user", "b")]
    merge_adjacent_messages(items)
    assert len(first["content"]) == 1


def test_trims_oldest_turn_groups_and_keeps_head() -> None:
    items: list[dict[str, object]] = [_message("user", "task")]
    for i in range(20):
        items += [_message("assistant", f"a{i}"), _call(f"c{i}"), _output(f"c{i}")]

    kept, dropped = fit_input_items(items, 10)

    assert len(kept) == 10
    assert dropped == len(items) - len(kept)
    assert kept[0] == items[0]
    assert kept[-1] == items[-1]


def test_never_starts_the_tail_with_an_orphan_tool_output() -> None:
    items: list[dict[str, object]] = []
    for i in range(6):
        items += [_message("assistant", f"a{i}"), _call(f"c{i}"), _output(f"c{i}")]
    items.append(_message("user", "tail"))

    kept, _ = fit_input_items(items, 4)

    assert kept[0]["type"] != "function_call_output"
    assert kept[-1] == items[-1]
