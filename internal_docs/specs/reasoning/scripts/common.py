"""Shared helpers for the reasoning round-trip demo scripts.

The point of these scripts is to prove that a flat dict of OpenInference-style
span attributes contains enough state to reconstruct an LLM API request that
re-invokes the model with the same conversational state — *including* the
opaque per-vendor continuity tokens (OpenAI `encrypted_content`, Anthropic
`signature`, Gemini `thoughtSignature`).

To make that proof load-bearing, each per-provider script:

  1. Makes turn 1 against the live API.
  2. Writes the assistant turn into a Phoenix-tracked LLM span as flat
     `llm.output_messages.{i}.message.contents.{j}.message_content.*` keys.
  3. Force-flushes the OTel pipeline.
  4. Uses `phoenix.client` to fetch the span **back out of Phoenix** by trace_id.
  5. Calls `read_assistant_turn(...)` on the *Phoenix-returned* attributes to
     rebuild the SDK objects for turn 2.
  6. Sends turn 2; asserts the server accepted it.

Attribute keys are string literals — nothing here is wired into the
`openinference-semantic-conventions` Python package. The intent is to validate
shape before promoting to real conventions.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Iterable

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# ---------------------------------------------------------------------------
# Attribute key constants (string-only — not promoted to SpanAttributes yet)
# ---------------------------------------------------------------------------

OPENINFERENCE_SPAN_KIND = "openinference.span.kind"
LLM_PROVIDER = "llm.provider"
LLM_SYSTEM = "llm.system"
LLM_MODEL_NAME = "llm.model_name"
LLM_INVOCATION_PARAMETERS = "llm.invocation_parameters"

# Proposed reasoning request-config conventions.
LLM_REASONING_EFFORT = "llm.reasoning.effort"  # OpenAI: none|low|medium|high|...
LLM_REASONING_BUDGET_TOKENS = "llm.reasoning.budget_tokens"  # Anthropic / Gemini 2.5
LLM_REASONING_LEVEL = "llm.reasoning.level"  # Gemini 3
LLM_REASONING_INCLUDE_SUMMARY = "llm.reasoning.include_summary"  # bool

# Per-content-block discriminators.
CONTENT_TYPE_TEXT = "text"
CONTENT_TYPE_REASONING = "reasoning"
CONTENT_TYPE_REASONING_SUMMARY = "reasoning_summary"  # Gemini thought-summary parts
CONTENT_TYPE_REDACTED_REASONING = "redacted_reasoning"  # Anthropic redacted_thinking
CONTENT_TYPE_TOOL_USE = "tool_use"


def _input_msg_key(i: int) -> str:
    return f"llm.input_messages.{i}.message"


def _output_msg_key(i: int) -> str:
    return f"llm.output_messages.{i}.message"


def _content_key(msg_key: str, j: int) -> str:
    return f"{msg_key}.contents.{j}.message_content"


# ---------------------------------------------------------------------------
# OTel + Phoenix setup
# ---------------------------------------------------------------------------


@dataclass
class TracingCtx:
    tracer: trace.Tracer
    provider: TracerProvider
    project_name: str
    phoenix_base_url: str


def setup_tracing(project_name: str) -> TracingCtx:
    """Wire OTLP/HTTP export to a local Phoenix and return a tracer.

    `PHOENIX_COLLECTOR_ENDPOINT` defaults to `http://localhost:6006`.
    """
    base_url = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006").rstrip("/")
    traces_url = f"{base_url}/v1/traces"

    resource = Resource.create({"openinference.project.name": project_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=traces_url)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    return TracingCtx(
        tracer=provider.get_tracer("reasoning-roundtrip"),
        provider=provider,
        project_name=project_name,
        phoenix_base_url=base_url,
    )


# ---------------------------------------------------------------------------
# Attribute setters (turn 1 → span attributes)
# ---------------------------------------------------------------------------


def set_input_user_message(attrs: dict[str, Any], idx: int, text: str) -> None:
    base = _input_msg_key(idx)
    attrs[f"{base}.role"] = "user"
    attrs[f"{base}.content"] = text


def set_input_tool_result(
    attrs: dict[str, Any],
    idx: int,
    *,
    tool_call_id: str,
    content: str,
) -> None:
    base = _input_msg_key(idx)
    attrs[f"{base}.role"] = "tool"
    attrs[f"{base}.tool_call_id"] = tool_call_id
    attrs[f"{base}.content"] = content


def set_input_assistant_echo(
    attrs: dict[str, Any],
    idx: int,
    contents: list[dict[str, Any]],
    *,
    role: str = "assistant",
) -> None:
    """Write an echoed prior assistant turn into `llm.input_messages.{idx}.*`.

    `contents` is the list of content dicts returned by `read_output_contents`
    — i.e. the round-tripped reasoning / text / tool_use blocks. Each dict's
    keys are written verbatim as `message_content.<key>` siblings, which means
    the input-side and output-side attributes share an identical shape and
    Phoenix renders them in the same way. This is what lets you confirm,
    visually in the UI, that the turn-2 input carries the same reasoning
    block (and the same signature / encrypted_content) that was on the turn-1
    output.
    """
    base = _input_msg_key(idx)
    attrs[f"{base}.role"] = role
    for j, content in enumerate(contents):
        k = f"{base}.contents.{j}.message_content"
        for field, value in content.items():
            attrs[f"{k}.{field}"] = value


def begin_output_message(attrs: dict[str, Any], idx: int, role: str = "assistant") -> str:
    base = _output_msg_key(idx)
    attrs[f"{base}.role"] = role
    return base


def set_output_text(
    attrs: dict[str, Any],
    msg_idx: int,
    j: int,
    text: str,
    *,
    thought_signature: str | None = None,
) -> None:
    k = _content_key(_output_msg_key(msg_idx), j)
    attrs[f"{k}.type"] = CONTENT_TYPE_TEXT
    attrs[f"{k}.text"] = text
    if thought_signature is not None:
        # Gemini-only: a data-bearing text part can carry a signature.
        attrs[f"{k}.thought_signature"] = thought_signature


def set_output_reasoning(
    attrs: dict[str, Any],
    msg_idx: int,
    j: int,
    *,
    text: str | None = None,
    item_id: str | None = None,
    encrypted_content: str | None = None,
    signature: str | None = None,
    thought_signature: str | None = None,
) -> None:
    """Write a reasoning content block.

    Carries any subset of the three vendor continuity-token fields. Only
    populated keys end up on the span — empty strings would round-trip as
    truthy and break echo.
    """
    k = _content_key(_output_msg_key(msg_idx), j)
    attrs[f"{k}.type"] = CONTENT_TYPE_REASONING
    if text is not None:
        attrs[f"{k}.text"] = text
    if item_id is not None:
        attrs[f"{k}.id"] = item_id
    if encrypted_content is not None:
        attrs[f"{k}.encrypted_content"] = encrypted_content
    if signature is not None:
        attrs[f"{k}.signature"] = signature
    if thought_signature is not None:
        attrs[f"{k}.thought_signature"] = thought_signature


def set_output_reasoning_summary(
    attrs: dict[str, Any],
    msg_idx: int,
    j: int,
    *,
    text: str,
) -> None:
    """Gemini-only: a `thought: true` summary part. Never carries a signature."""
    k = _content_key(_output_msg_key(msg_idx), j)
    attrs[f"{k}.type"] = CONTENT_TYPE_REASONING_SUMMARY
    attrs[f"{k}.text"] = text


def set_output_redacted_reasoning(
    attrs: dict[str, Any],
    msg_idx: int,
    j: int,
    *,
    data: str,
) -> None:
    """Anthropic-only: replaces `thinking` text with an opaque `data` blob."""
    k = _content_key(_output_msg_key(msg_idx), j)
    attrs[f"{k}.type"] = CONTENT_TYPE_REDACTED_REASONING
    attrs[f"{k}.redacted_data"] = data


def set_output_tool_use(
    attrs: dict[str, Any],
    msg_idx: int,
    j: int,
    *,
    tool_call_id: str,
    name: str,
    arguments_json: str,
    text: str | None = None,
    thought_signature: str | None = None,
) -> None:
    """Tool-use block on an assistant turn.

    `text` carries any sibling text on the same part (Gemini: rare but legal).
    `thought_signature` is Gemini-only — Anthropic / OpenAI tool calls do not
    carry a signature directly; their signatures ride on the preceding
    reasoning block.
    """
    k = _content_key(_output_msg_key(msg_idx), j)
    attrs[f"{k}.type"] = CONTENT_TYPE_TOOL_USE
    attrs[f"{k}.tool_call.id"] = tool_call_id
    attrs[f"{k}.tool_call.function.name"] = name
    attrs[f"{k}.tool_call.function.arguments"] = arguments_json
    if text is not None:
        attrs[f"{k}.text"] = text
    if thought_signature is not None:
        attrs[f"{k}.thought_signature"] = thought_signature


# ---------------------------------------------------------------------------
# Attribute readers (Phoenix-returned attributes → SDK objects)
# ---------------------------------------------------------------------------


def read_output_contents(attrs: dict[str, Any], msg_idx: int) -> list[dict[str, Any]]:
    """Walk `llm.output_messages.{msg_idx}.message.contents.{j}.message_content.*`
    keys in `attrs` and return an ordered list of content dicts.

    Each dict's keys are the trailing component after `message_content.`, e.g.
    `type`, `text`, `signature`, `tool_call.id`. The per-provider script maps
    these back to its SDK objects.
    """
    prefix = _content_key(_output_msg_key(msg_idx), 0).rsplit(".0.", 1)[0] + "."
    # prefix == "llm.output_messages.{i}.message.contents."
    grouped: dict[int, dict[str, Any]] = {}
    for key, value in attrs.items():
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix):]  # "{j}.message_content.<field...>"
        try:
            j_str, _, field = rest.partition(".")
            j = int(j_str)
        except ValueError:
            continue
        if not field.startswith("message_content."):
            continue
        field = field[len("message_content."):]
        grouped.setdefault(j, {})[field] = value
    return [grouped[j] for j in sorted(grouped)]


# ---------------------------------------------------------------------------
# Phoenix span fetcher
# ---------------------------------------------------------------------------


def fetch_span_attributes(
    ctx: TracingCtx,
    *,
    trace_id_hex: str,
    span_id_hex: str,
    timeout_s: float = 15.0,
) -> dict[str, Any]:
    """Poll Phoenix until our span lands, then return its flat attribute dict.

    `trace_id_hex` and `span_id_hex` are the 32- and 16-char hex strings from
    `span.get_span_context()`. Phoenix's API uses these same hex strings as
    `trace_id` / `id` on the returned `v1.Span` payload.
    """
    # Local import — keeps the helper module importable in environments that
    # do not have arize-phoenix-client installed (each script declares it).
    from phoenix.client import Client

    client = Client(base_url=ctx.phoenix_base_url)
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            spans: Iterable[dict[str, Any]] = client.spans.get_spans(
                project_identifier=ctx.project_name,
                trace_ids=[trace_id_hex],
                limit=20,
            )
            for s in spans:
                if s.get("context", {}).get("span_id") == span_id_hex:
                    return dict(s.get("attributes") or {})
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(0.5)
    raise TimeoutError(
        f"Span {span_id_hex} not visible in Phoenix project "
        f"{ctx.project_name!r} after {timeout_s}s (last err: {last_err!r})"
    )


def flush(ctx: TracingCtx) -> None:
    ctx.provider.force_flush()


# ---------------------------------------------------------------------------
# Pretty-printing for the PASS/FAIL line
# ---------------------------------------------------------------------------


def token_lens(contents: list[dict[str, Any]]) -> dict[str, int]:
    """Continuity-token lengths per block — never the bytes themselves."""
    out: dict[str, int] = {}
    for j, c in enumerate(contents):
        for field in ("signature", "encrypted_content", "thought_signature", "redacted_data"):
            v = c.get(field)
            if isinstance(v, str) and v:
                out[f"block{j}.{field}"] = len(v)
    return out
