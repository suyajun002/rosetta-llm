"""Cross-protocol extras must not leak into the Responses payload."""

from __future__ import annotations

from rosetta.codecs import anthropic as ac
from rosetta.codecs import openai_responses as orc


def _anthropic_payload(**extra: object) -> dict[str, object]:
    return {
        "model": "deepseek-v4.1-flash-1017",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
        **extra,
    }


def test_anthropic_only_top_level_field_is_not_forwarded() -> None:
    ir = ac.parse_request(_anthropic_payload(safeguards={"mode": "default"}))
    body = orc.render_request(ir)
    assert "safeguards" not in body


def test_known_responses_optional_field_is_still_forwarded() -> None:
    ir = ac.parse_request(_anthropic_payload(truncation="auto"))
    body = orc.render_request(ir)
    assert body["truncation"] == "auto"
