"""Pipeline dispatcher — passthrough fast path vs translate slow path.

Translation invariants:
  - Inbound request bytes are JSON-decoded once.
  - When inbound format == provider format, we forward verbatim (only the
    `model` field is rewritten on a copied dict; original payload untouched).
  - Otherwise we run inbound parser → IR → provider renderer → upstream call →
    provider parser → IR → inbound renderer.
  - Streaming uses per-format state machines that emit IR events; outbound
    state machine consumes IR events and emits target SSE.
  - Ping injection is added only when outbound format == anthropic.
  - Cancellation: a client disconnect aborts the streaming generator,
    which exits the upstream `httpx.stream` context and closes the
    connection.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from rosetta.codecs import anthropic as ac_codec
from rosetta.codecs import openai_chat as oc_codec
from rosetta.codecs import openai_responses as or_codec
from rosetta.config import Config, ProviderConfig, ReasoningEffort
from rosetta.errors import format_error, format_stream_error
from rosetta.input_budget import fit_input_items
from rosetta.ir.request import CanonicalRequest
from rosetta.ir.response import CanonicalResponse
from rosetta.observability import bind_request_context, get_logger
from rosetta.routes.health import record_provider_status
from rosetta.stream_codecs import anthropic as ac_stream
from rosetta.stream_codecs import openai_chat as oc_stream
from rosetta.stream_codecs import openai_responses as or_stream
from rosetta.upstream import UpstreamClient

type ParseRequest = Callable[[dict[str, Any]], CanonicalRequest]
type RenderRequest = Callable[[CanonicalRequest], dict[str, Any]]
type ParseResponse = Callable[[dict[str, Any]], CanonicalResponse]
type RenderResponse = Callable[[CanonicalResponse], dict[str, Any]]
type StreamParse = Callable[[AsyncIterator[bytes]], AsyncIterator[Any]]
type StreamRender = Callable[[AsyncIterator[Any]], AsyncIterator[bytes]]

_PARSE_REQ: dict[str, ParseRequest] = {
    "anthropic": ac_codec.parse_request,
    "openai_chat": oc_codec.parse_request,
    "openai_responses": or_codec.parse_request,
}
_RENDER_REQ: dict[str, RenderRequest] = {
    "anthropic": ac_codec.render_request,
    "openai_chat": oc_codec.render_request,
    "openai_responses": or_codec.render_request,
}
_PARSE_RESP: dict[str, ParseResponse] = {
    "anthropic": ac_codec.parse_response,
    "openai_chat": oc_codec.parse_response,
    "openai_responses": or_codec.parse_response,
}
_RENDER_RESP: dict[str, RenderResponse] = {
    "anthropic": ac_codec.render_response,
    "openai_chat": oc_codec.render_response,
    "openai_responses": or_codec.render_response,
}
_STREAM_PARSE: dict[str, StreamParse] = {
    "anthropic": ac_stream.parse,
    "openai_chat": oc_stream.parse,
    "openai_responses": or_stream.parse,
}
_STREAM_RENDER: dict[str, StreamRender] = {
    "anthropic": ac_stream.render,
    "openai_chat": oc_stream.render,
    "openai_responses": or_stream.render,
}
_FORMAT_PATH: dict[str, str] = {
    "openai_chat": "/chat/completions",
    "openai_responses": "/responses",
    "anthropic": "/messages",
}

# Anthropic headers that MUST be forwarded upstream per the Claude Code LLM Gateway spec.
_ANTHROPIC_FORWARD_HEADERS = frozenset(
    {"anthropic-beta", "anthropic-version", "x-claude-code-session-id"}
)

_STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _forwarded_headers(request: Request) -> dict[str, str]:
    return {k: request.headers[k] for k in _ANTHROPIC_FORWARD_HEADERS if k in request.headers}


_CLAUDE_CODE_GW_PREFIX = "claude-code/"


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    provider: ProviderConfig
    provider_key: str
    upstream_name: str
    model_reasoning_effort: ReasoningEffort | None


def _resolve_model(model_id: str, config: Config) -> ResolvedModel:
    # Claude Code gateway prefix: strip and re-resolve the underlying provider/model.
    if model_id.startswith(_CLAUDE_CODE_GW_PREFIX):
        inner = model_id[len(_CLAUDE_CODE_GW_PREFIX) :]
        if "/" in inner:
            return _resolve_model(inner, config)
        raise ValueError(f"Model id '{model_id}' has gateway prefix but no provider/model")
    if "/" not in model_id:
        raise ValueError(f"Model id '{model_id}' must be in format '<provider>/<model>'")
    provider_key, model_name = model_id.split("/", 1)
    provider = config.providers.get(provider_key)
    if provider is None:
        raise ValueError(f"Unknown provider '{provider_key}'")
    upstream_name = model_name
    model_reasoning_effort = None
    for m in provider.static_models:
        if m.id == model_name:
            upstream_name = m.effective_upstream_name
            model_reasoning_effort = m.reasoning_effort
            break
    return ResolvedModel(
        provider=provider,
        provider_key=provider_key,
        upstream_name=upstream_name,
        model_reasoning_effort=model_reasoning_effort,
    )


def _apply_responses_options(body: dict[str, Any], resolved: ResolvedModel) -> None:
    """Apply provider defaults and explicit model-profile overrides."""
    if resolved.provider.format != "openai_responses":
        return

    input_limit = resolved.provider.max_input_items
    upstream_input = body.get("input")
    if input_limit and isinstance(upstream_input, list):
        fitted, dropped = fit_input_items(upstream_input, input_limit)
        if dropped:
            body["input"] = fitted
            get_logger().warning(
                "input_items_trimmed",
                dropped=dropped,
                kept=len(fitted),
                limit=input_limit,
            )

    prompt_cache_key = resolved.provider.responses_prompt_cache_key
    if prompt_cache_key:
        body.setdefault("prompt_cache_key", prompt_cache_key)

    configured_effort = resolved.model_reasoning_effort
    default_effort = resolved.provider.responses_reasoning_effort
    if configured_effort is None and default_effort is None:
        return

    reasoning = body.get("reasoning")
    if not isinstance(reasoning, dict):
        reasoning = {}
        body["reasoning"] = reasoning
    if configured_effort is not None:
        reasoning["effort"] = configured_effort
    else:
        reasoning.setdefault("effort", default_effort)


async def handle(
    inbound_format: str,
    payload: dict[str, Any],
    request: Request,
) -> Response:
    config: Config = request.app.state.config
    upstream: UpstreamClient = request.app.state.upstream
    status_dict: dict[str, Any] = request.app.state.provider_status
    log = get_logger()

    model_id = payload.get("model", "")
    try:
        resolved = _resolve_model(model_id, config)
    except ValueError as e:
        return JSONResponse(
            content=format_error(inbound_format, 400, "invalid_request_error", str(e)),
            status_code=400,
        )

    provider = resolved.provider
    provider_key = resolved.provider_key
    bind_request_context(provider=provider_key, model=model_id, format=provider.format)

    # Copy and rewrite the model field without mutating the caller's dict.
    upstream_body = {**payload, "model": resolved.upstream_name}
    # Claude Code streaming edit (e.g. clear_thinking_20251015) is an Anthropic-side
    # context-management hint that deepseek/openai_responses upstreams do not support.
    # Drop configured top-level fields (e.g. "edit", "context_management") so they are
    # never forwarded verbatim (covers translate and passthrough).
    for field in provider.strip_top_level_fields:
        upstream_body.pop(field, None)
    is_stream = bool(payload.get("stream", False))
    upstream_path = _FORMAT_PATH[provider.format]

    fwd_headers = _forwarded_headers(request)

    if inbound_format == provider.format:
        _apply_responses_options(upstream_body, resolved)
        return await _passthrough(
            provider_key,
            upstream_path,
            upstream_body,
            is_stream,
            inbound_format,
            request,
            upstream,
            log,
            fwd_headers,
            status_dict,
        )

    return await _translate(
        inbound_format,
        provider.format,
        upstream_body,
        provider_key,
        upstream_path,
        is_stream,
        request,
        upstream,
        log,
        fwd_headers,
        status_dict,
        resolved,
    )


async def _passthrough(
    provider_key: str,
    upstream_path: str,
    body: dict[str, Any],
    is_stream: bool,
    inbound_format: str,
    request: Request,
    upstream: UpstreamClient,
    log: Any,
    fwd_headers: dict[str, str],
    status_dict: dict[str, Any],
) -> Response:
    log.info("passthrough", stream=is_stream)
    if is_stream:
        gen = upstream.stream(provider_key, upstream_path, body, extra_headers=fwd_headers)
        return StreamingResponse(
            _passthrough_stream_with_recovery(
                gen, request, provider_key, inbound_format, status_dict
            ),
            media_type="text/event-stream",
            headers=_STREAM_HEADERS,
        )
    try:
        resp = await upstream.request_json(
            provider_key, upstream_path, body, extra_headers=fwd_headers
        )
        record_provider_status(status_dict, provider_key, ok=resp.is_success)
    except httpx.HTTPError as e:
        log.error(
            "upstream_error",
            error_type=type(e).__name__,
            error=_upstream_error_message(e),
        )
        record_provider_status(status_dict, provider_key, ok=False)
        return JSONResponse(
            content=format_error(
                inbound_format, 502, "upstream_error", _upstream_error_message(e)
            ),
            status_code=502,
        )
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )


def _upstream_error_message(exc: Exception) -> str:
    """Diagnostic text for an httpx transport error.

    httpx builds several transport errors with an empty ``str()`` (``PoolTimeout``
    and friends), so those reached clients as
    ``{"type":"error","error":{"type":"upstream_error","message":""}}`` and could
    not be told apart from an upstream bug. Always return a non-empty reason.
    """
    message = str(exc).strip()
    if message:
        return message
    if isinstance(exc, httpx.PoolTimeout):
        return (
            "upstream connection pool exhausted: every connection of this provider "
            "is still held by an abandoned stream, so the request was never sent"
        )
    return f"{type(exc).__name__}: upstream request failed before any response"


async def _close_open_streams(open_streams: list[AsyncIterator[Any]]) -> None:
    """Close tracked upstream iterators, newest first.

    An abandoned async generator keeps its httpx connection - and with it a slot of
    the provider's connection pool - until the event loop finalizes the generator,
    which can be arbitrarily late. Closing explicitly is what stops a client
    disconnect from permanently eating pool capacity.
    """
    log = get_logger()
    while open_streams:
        stream = open_streams.pop()
        aclose = getattr(stream, "aclose", None)
        if aclose is None:
            continue
        try:
            await aclose()
        except (RuntimeError, StopAsyncIteration) as exc:
            # Already closed, or still driven by wrap_with_ping's producer task.
            log.debug("stream_aclose_skipped", error_type=type(exc).__name__)


async def _passthrough_stream_with_recovery(
    gen: AsyncIterator[bytes],
    request: Request,
    provider_key: str,
    inbound_format: str,
    status_dict: dict[str, Any],
) -> AsyncIterator[bytes]:
    log = get_logger()
    try:
        async for chunk in gen:
            if await request.is_disconnected():
                break
            yield chunk
        record_provider_status(status_dict, provider_key, ok=True)
    except httpx.HTTPError as e:
        log.error(
            "stream_upstream_error",
            error_type=type(e).__name__,
            error=_upstream_error_message(e),
        )
        record_provider_status(status_dict, provider_key, ok=False)
        yield format_stream_error(inbound_format, "upstream_error", _upstream_error_message(e))
    finally:
        await _close_open_streams([gen])


async def _translate(
    inbound_format: str,
    provider_format: str,
    payload: dict[str, Any],
    provider_key: str,
    upstream_path: str,
    is_stream: bool,
    request: Request,
    upstream: UpstreamClient,
    log: Any,
    fwd_headers: dict[str, str],
    status_dict: dict[str, Any],
    resolved: ResolvedModel,
) -> Response:
    log.info(
        "translate", inbound=inbound_format, provider_format=provider_format, stream=is_stream
    )

    try:
        ir = _PARSE_REQ[inbound_format](payload)
        upstream_body = _RENDER_REQ[provider_format](ir)
        # Drop Claude Code streaming-edit hints (e.g. clear_thinking_20251015).
        # render_request can re-insert 'edit'/'context_management' from raw_extras,
        # so strip configured fields here (after render) so they are never forwarded
        # to an unsupporting upstream.
        for field in resolved.provider.strip_top_level_fields:
            upstream_body.pop(field, None)
        _apply_responses_options(upstream_body, resolved)
    except Exception as e:  # noqa: BLE001
        log.error("translate_request_error", error=str(e))
        return JSONResponse(
            content=format_error(inbound_format, 400, "translation_error", str(e)),
            status_code=400,
        )

    if is_stream:
        return StreamingResponse(
            _translate_stream(
                inbound_format,
                provider_format,
                upstream_body,
                provider_key,
                upstream_path,
                request,
                upstream,
                fwd_headers,
                status_dict,
            ),
            media_type="text/event-stream",
            headers=_STREAM_HEADERS,
        )

    try:
        resp = await upstream.request_json(
            provider_key, upstream_path, upstream_body, extra_headers=fwd_headers
        )
    except httpx.HTTPError as e:
        log.error(
            "upstream_error",
            error_type=type(e).__name__,
            error=_upstream_error_message(e),
        )
        record_provider_status(status_dict, provider_key, ok=False)
        return JSONResponse(
            content=format_error(
                inbound_format, 502, "upstream_error", _upstream_error_message(e)
            ),
            status_code=502,
        )

    if not resp.is_success:
        record_provider_status(status_dict, provider_key, ok=False)
        try:
            upstream_err = resp.json()
        except Exception:  # noqa: BLE001
            upstream_err = {"raw": resp.text}
        return JSONResponse(
            content=format_error(
                inbound_format,
                resp.status_code,
                "upstream_error",
                _extract_error_message(upstream_err),
            ),
            status_code=resp.status_code,
        )

    record_provider_status(status_dict, provider_key, ok=True)
    try:
        upstream_data = resp.json()
        ir_resp = _PARSE_RESP[provider_format](upstream_data)
        out = _RENDER_RESP[inbound_format](ir_resp)
    except Exception as e:  # noqa: BLE001
        log.error("translate_response_error", error=str(e))
        return JSONResponse(
            content=format_error(inbound_format, 502, "translation_error", str(e)),
            status_code=502,
        )
    return JSONResponse(content=out)


async def _translate_stream(
    inbound_format: str,
    provider_format: str,
    payload: dict[str, Any],
    provider_key: str,
    upstream_path: str,
    request: Request,
    upstream: UpstreamClient,
    fwd_headers: dict[str, str],
    status_dict: dict[str, Any],
) -> AsyncIterator[bytes]:
    log = get_logger()
    open_streams: list[AsyncIterator[Any]] = []
    try:
        upstream_bytes = upstream.stream(
            provider_key, upstream_path, payload, extra_headers=fwd_headers
        )
        open_streams.append(upstream_bytes)
        ir_events = _STREAM_PARSE[provider_format](upstream_bytes)
        open_streams.append(ir_events)
        if inbound_format == "anthropic":
            ir_events = ac_stream.wrap_with_ping(ir_events)
            open_streams.append(ir_events)
        out_bytes = _STREAM_RENDER[inbound_format](ir_events)
        open_streams.append(out_bytes)

        async for chunk in out_bytes:
            if await request.is_disconnected():
                break
            yield chunk
        record_provider_status(status_dict, provider_key, ok=True)
    except httpx.HTTPError as e:
        log.error(
            "stream_upstream_error",
            error_type=type(e).__name__,
            error=_upstream_error_message(e),
        )
        record_provider_status(status_dict, provider_key, ok=False)
        yield format_stream_error(inbound_format, "upstream_error", _upstream_error_message(e))
    except Exception as e:  # noqa: BLE001
        log.exception("stream_translate_error")
        yield format_stream_error(inbound_format, "translation_error", str(e))
    finally:
        await _close_open_streams(open_streams)


def _extract_error_message(payload: Any) -> str:
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict) and "message" in err:
            return str(err["message"])
        if isinstance(err, str):
            return err
        if "message" in payload:
            return str(payload["message"])
    return str(payload)
