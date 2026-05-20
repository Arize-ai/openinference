"""Shared helpers for the reasoning round-trip demo scripts.

Each per-provider script:

  1. Makes turn 1 against the live API.
  2. Writes the assistant turn into an OTel `LLM` span as flat
     `llm.output_messages.{i}.message.contents.{j}.<suffix>` keys, including
     the vendor continuity token.
  3. Force-flushes the OTel pipeline to a local Phoenix.
  4. Uses `phoenix.client` to fetch the span back out by trace_id.
  5. Rebuilds the prior assistant turn from the fetched attribute dict.
  6. Sends turn 2 with the reconstructed history; asserts the server accepted it.

Established attribute keys come from `openinference.semconv.trace`. The
"Proposed" block below holds keys this PR is exploring that are not yet in
`openinference-semantic-conventions`.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

from openinference.instrumentation import get_llm_tool_attributes
from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
    ToolCallAttributes,
)
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import SpanContext

# ---------------------------------------------------------------------------
# Proposed (not yet in openinference-semantic-conventions)
#
# These string literals are what the PR is exploring. If/when they land in
# the semantic-conventions packages, every reference below becomes a normal
# `MessageContentAttributes.<NAME>` import.
# ---------------------------------------------------------------------------

# Additional `message_content.type` discriminators. Every reasoning surface —
# raw thinking, summarized thinking (Gemini thought parts, OpenAI summaries),
# and Anthropic redacted thinking — uses the single value `"reasoning"`. The
# variant is told apart by which sub-fields are present (`text` vs
# `redacted_data`) plus the provider on `llm.provider`, so no extra type is
# needed.
CONTENT_TYPE_TEXT = "text"  # parallels the existing MESSAGE_CONTENT_TEXT field
CONTENT_TYPE_REASONING = "reasoning"
CONTENT_TYPE_TOOL_USE = "tool_use"

# Additional `message_content.*` sub-fields. These carry the per-vendor
# continuity token that the next turn must echo verbatim to keep reasoning
# alive (or, on tool-use turns, to avoid HTTP 400).
MESSAGE_CONTENT_ID = "message_content.id"  # OpenAI reasoning item id
MESSAGE_CONTENT_ENCRYPTED_CONTENT = "message_content.encrypted_content"  # OpenAI
MESSAGE_CONTENT_SIGNATURE = "message_content.signature"  # Anthropic
MESSAGE_CONTENT_THOUGHT_SIGNATURE = "message_content.thought_signature"  # Gemini
MESSAGE_CONTENT_REDACTED_DATA = "message_content.redacted_data"  # Anthropic redacted_thinking

PROPOSED_CONTINUITY_TOKEN_FIELDS: tuple[str, ...] = (
    MESSAGE_CONTENT_SIGNATURE,
    MESSAGE_CONTENT_ENCRYPTED_CONTENT,
    MESSAGE_CONTENT_THOUGHT_SIGNATURE,
    MESSAGE_CONTENT_REDACTED_DATA,
)


# ---------------------------------------------------------------------------
# Key builders
#
# Span attributes are flat dotted strings. These helpers compose them from
# semconv pieces so callers never assemble raw keys by hand.
# ---------------------------------------------------------------------------


# Example output: `output_msg_key(0, MessageAttributes.MESSAGE_ROLE)`
#   → `"llm.output_messages.0.message.role"`.
# Example output: `output_content_key(0, 0, MessageContentAttributes.MESSAGE_CONTENT_TYPE)`
#   → `"llm.output_messages.0.message.contents.0.message_content.type"`.


def output_msg_key(message_index: int, suffix: str) -> str:
    return f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{message_index}.{suffix}"


def input_msg_key(message_index: int, suffix: str) -> str:
    return f"{SpanAttributes.LLM_INPUT_MESSAGES}.{message_index}.{suffix}"


def output_content_key(message_index: int, content_index: int, suffix: str) -> str:
    return (
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{message_index}"
        f".{MessageAttributes.MESSAGE_CONTENTS}.{content_index}.{suffix}"
    )


def input_content_key(message_index: int, content_index: int, suffix: str) -> str:
    return (
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.{message_index}"
        f".{MessageAttributes.MESSAGE_CONTENTS}.{content_index}.{suffix}"
    )


# ---------------------------------------------------------------------------
# OTel + Phoenix setup
# ---------------------------------------------------------------------------


@dataclass
class TracingCtx:
    tracer: trace.Tracer
    provider: TracerProvider
    project_name: str
    phoenix_base_url: str

    def flush(self) -> None:
        self.provider.force_flush()


def setup_tracing(project_name: str) -> TracingCtx:
    """Wire OTLP/HTTP export to a local Phoenix and return a tracer.

    `PHOENIX_COLLECTOR_ENDPOINT` defaults to `http://localhost:6006`.
    """
    base_url = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006").rstrip("/")
    resource = Resource.create({"openinference.project.name": project_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{base_url}/v1/traces"))
    )
    trace.set_tracer_provider(provider)
    return TracingCtx(
        tracer=provider.get_tracer("reasoning-roundtrip"),
        provider=provider,
        project_name=project_name,
        phoenix_base_url=base_url,
    )


# ---------------------------------------------------------------------------
# Attribute writers
# ---------------------------------------------------------------------------


def set_input_user_message(attrs: dict[str, Any], message_index: int, text: str) -> None:
    attrs[input_msg_key(message_index, MessageAttributes.MESSAGE_ROLE)] = "user"
    attrs[input_msg_key(message_index, MessageAttributes.MESSAGE_CONTENT)] = text


def set_input_tool_result(
    attrs: dict[str, Any],
    message_index: int,
    *,
    tool_call_id: str,
    content_text: str,
) -> None:
    attrs[input_msg_key(message_index, MessageAttributes.MESSAGE_ROLE)] = "tool"
    attrs[input_msg_key(message_index, MessageAttributes.MESSAGE_TOOL_CALL_ID)] = tool_call_id
    attrs[input_msg_key(message_index, MessageAttributes.MESSAGE_CONTENT)] = content_text


def set_input_assistant_echo(
    attrs: dict[str, Any],
    message_index: int,
    contents: list[dict[str, Any]],
    *,
    role: str = "assistant",
) -> None:
    """Echo a prior assistant turn into `llm.input_messages.{message_index}.*`.

    `contents` is the list returned by `read_output_contents` — each dict
    maps a full sub-field suffix (already including the `message_content.` or
    `tool_call.` prefix) to its value. They're written back verbatim, which
    is what makes the byte-for-byte continuity-token match visible in
    Phoenix between turn 1's output and turn 2's echoed input.
    """
    attrs[input_msg_key(message_index, MessageAttributes.MESSAGE_ROLE)] = role
    for content_index, content_block in enumerate(contents):
        for suffix, value in content_block.items():
            attrs[input_content_key(message_index, content_index, suffix)] = value


def set_output_role(attrs: dict[str, Any], message_index: int, role: str) -> None:
    attrs[output_msg_key(message_index, MessageAttributes.MESSAGE_ROLE)] = role


def _set_content_block(
    attrs: dict[str, Any],
    message_index: int,
    content_index: int,
    fields_by_suffix: dict[str, Any],
) -> None:
    # Nones are skipped so the presence-or-absence of each field stays a
    # reliable discriminator on the read side.
    for suffix, value in fields_by_suffix.items():
        if value is not None:
            attrs[output_content_key(message_index, content_index, suffix)] = value


def set_output_text(
    attrs: dict[str, Any],
    message_index: int,
    content_index: int,
    text: str,
    *,
    thought_signature: str | None = None,
) -> None:
    _set_content_block(
        attrs,
        message_index,
        content_index,
        {
            MessageContentAttributes.MESSAGE_CONTENT_TYPE: CONTENT_TYPE_TEXT,
            MessageContentAttributes.MESSAGE_CONTENT_TEXT: text,
            # Gemini-only: a text part may carry a thoughtSignature.
            MESSAGE_CONTENT_THOUGHT_SIGNATURE: thought_signature,
        },
    )


def set_output_reasoning(
    attrs: dict[str, Any],
    message_index: int,
    content_index: int,
    *,
    text: str | None = None,
    item_id: str | None = None,
    encrypted_content: str | None = None,
    signature: str | None = None,
    thought_signature: str | None = None,
    redacted_data: str | None = None,
) -> None:
    """A `type: reasoning` block — covers raw thinking, summarized thinking,
    and Anthropic redacted thinking. Every field is optional; the caller
    populates whatever the provider produced:

    - Anthropic thinking:      `text` + `signature`
    - Anthropic redacted:      `redacted_data` (no `text`, no `signature`)
    - OpenAI reasoning item:   `text` + `item_id` + `encrypted_content`
    - Gemini thought summary:  `text` (no continuity token — it rides on the
      sibling data part)
    """
    _set_content_block(
        attrs,
        message_index,
        content_index,
        {
            MessageContentAttributes.MESSAGE_CONTENT_TYPE: CONTENT_TYPE_REASONING,
            MessageContentAttributes.MESSAGE_CONTENT_TEXT: text,
            MESSAGE_CONTENT_ID: item_id,
            MESSAGE_CONTENT_ENCRYPTED_CONTENT: encrypted_content,
            MESSAGE_CONTENT_SIGNATURE: signature,
            MESSAGE_CONTENT_THOUGHT_SIGNATURE: thought_signature,
            MESSAGE_CONTENT_REDACTED_DATA: redacted_data,
        },
    )


def set_output_tool_use(
    attrs: dict[str, Any],
    message_index: int,
    content_index: int,
    *,
    tool_call_id: str,
    name: str,
    arguments_json: str,
    text: str | None = None,
    thought_signature: str | None = None,
) -> None:
    """Tool-use content block.

    `thought_signature` is Gemini-only — OpenAI / Anthropic carry their
    signature on the preceding reasoning block, not on the tool call.
    """
    _set_content_block(
        attrs,
        message_index,
        content_index,
        {
            MessageContentAttributes.MESSAGE_CONTENT_TYPE: CONTENT_TYPE_TOOL_USE,
            ToolCallAttributes.TOOL_CALL_ID: tool_call_id,
            ToolCallAttributes.TOOL_CALL_FUNCTION_NAME: name,
            ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON: arguments_json,
            MessageContentAttributes.MESSAGE_CONTENT_TEXT: text,
            MESSAGE_CONTENT_THOUGHT_SIGNATURE: thought_signature,
        },
    )


# ---------------------------------------------------------------------------
# Tool definitions
#
# Capture each provider's tool schema under the existing
# `llm.tools.{i}.tool.json_schema` convention. The writer reuses the upstream
# `get_llm_tool_attributes` helper directly; only the reader is local because
# the package doesn't ship an inverse.
# ---------------------------------------------------------------------------

_LLM_TOOL_JSON_SCHEMA_SUFFIX = "tool.json_schema"


def set_tools(attrs: dict[str, Any], tool_json_schemas: list[dict[str, Any]]) -> None:
    attrs.update(
        get_llm_tool_attributes(
            tools=[{"json_schema": schema} for schema in tool_json_schemas]
        )
    )


def read_tools(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    prefix = f"{SpanAttributes.LLM_TOOLS}."
    schema_json_by_index: dict[int, str] = {}
    for key, value in attrs.items():
        if not key.startswith(prefix) or not key.endswith(f".{_LLM_TOOL_JSON_SCHEMA_SUFFIX}"):
            continue
        tool_index_str = key[len(prefix):].split(".", 1)[0]
        try:
            schema_json_by_index[int(tool_index_str)] = value
        except ValueError:
            continue
    return [json.loads(schema_json_by_index[tool_index]) for tool_index in sorted(schema_json_by_index)]


# ---------------------------------------------------------------------------
# Attribute reader
# ---------------------------------------------------------------------------


def read_input_message_content(attrs: dict[str, Any], message_index: int) -> str:
    """Return the `message.content` value for an input (user / tool) message."""
    return attrs[input_msg_key(message_index, MessageAttributes.MESSAGE_CONTENT)]


def read_output_contents(attrs: dict[str, Any], message_index: int) -> list[dict[str, Any]]:
    """Group output content-block attributes by index. Returned dicts map the
    trailing key suffix (e.g. `message_content.type`, `tool_call.id`) to its
    value — those suffixes are exactly the semconv constant values."""
    prefix = (
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{message_index}"
        f".{MessageAttributes.MESSAGE_CONTENTS}."
    )
    fields_by_content_index: dict[int, dict[str, Any]] = {}
    for key, value in attrs.items():
        if not key.startswith(prefix):
            continue
        content_index_str, _, suffix = key[len(prefix):].partition(".")
        try:
            content_index = int(content_index_str)
        except ValueError:
            continue
        fields_by_content_index.setdefault(content_index, {})[suffix] = value
    return [fields_by_content_index[content_index] for content_index in sorted(fields_by_content_index)]


def find_tool_use_block(contents: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the first `tool_use` content block, or None if none."""
    return next(
        (
            block
            for block in contents
            if block.get(MessageContentAttributes.MESSAGE_CONTENT_TYPE) == CONTENT_TYPE_TOOL_USE
        ),
        None,
    )


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
    """Poll Phoenix until our span lands, then return its flat attribute dict."""
    from phoenix.client import Client

    client = Client(base_url=ctx.phoenix_base_url)
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            for span in client.spans.get_spans(
                project_identifier=ctx.project_name,
                trace_ids=[trace_id_hex],
                limit=20,
            ):
                if span.get("context", {}).get("span_id") == span_id_hex:
                    return dict(span.get("attributes") or {})
        except Exception as error:  # noqa: BLE001
            last_error = error
        time.sleep(0.5)
    raise TimeoutError(
        f"Span {span_id_hex} not visible in Phoenix project {ctx.project_name!r} "
        f"after {timeout_s}s (last err: {last_error!r})"
    )


def flush_and_fetch(ctx: TracingCtx, span_context: SpanContext) -> dict[str, Any]:
    """Force-flush OTel and pull the span's attributes back out of Phoenix.

    Centralizes the trace/span-id hex formatting so callers don't repeat it.
    """
    ctx.flush()
    return fetch_span_attributes(
        ctx,
        trace_id_hex=f"{span_context.trace_id:032x}",
        span_id_hex=f"{span_context.span_id:016x}",
    )


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------


def continuity_token_lengths(contents: list[dict[str, Any]]) -> dict[str, int]:
    """Length, in characters, of every continuity token present on the
    captured contents. Used for the PASS line — never the bytes themselves,
    since they are large, opaque, and not human-readable."""
    lengths: dict[str, int] = {}
    for content_index, content_block in enumerate(contents):
        for full_suffix in PROPOSED_CONTINUITY_TOKEN_FIELDS:
            value = content_block.get(full_suffix)
            if isinstance(value, str) and value:
                short_name = full_suffix.removeprefix("message_content.")
                lengths[f"block{content_index}.{short_name}"] = len(value)
    return lengths


def print_capture_summary(contents: list[dict[str, Any]]) -> None:
    """One-line summary of captured blocks + continuity-token sizes."""
    print(f"  fetched_blocks={len(contents)} token_lens={continuity_token_lengths(contents)}")
