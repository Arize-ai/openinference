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

# ---------------------------------------------------------------------------
# Proposed (not yet in openinference-semantic-conventions)
#
# These string literals are what the PR is exploring. If/when they land in
# the semantic-conventions packages, every reference below becomes a normal
# `MessageContentAttributes.<NAME>` import.
# ---------------------------------------------------------------------------

# Additional `message_content.type` discriminators.
CONTENT_TYPE_TEXT = "text"  # parallels the existing MESSAGE_CONTENT_TEXT field
CONTENT_TYPE_REASONING = "reasoning"
CONTENT_TYPE_REASONING_SUMMARY = "reasoning_summary"  # Gemini thought-summary parts
CONTENT_TYPE_REDACTED_REASONING = "redacted_reasoning"  # Anthropic redacted_thinking
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


def output_msg_key(message_index: int, suffix: str) -> str:
    """`output_msg_key(0, MessageAttributes.MESSAGE_ROLE)` →
    `llm.output_messages.0.message.role`."""
    return f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{message_index}.{suffix}"


def input_msg_key(message_index: int, suffix: str) -> str:
    return f"{SpanAttributes.LLM_INPUT_MESSAGES}.{message_index}.{suffix}"


def output_content_key(message_index: int, content_index: int, suffix: str) -> str:
    """`output_content_key(0, 0, MessageContentAttributes.MESSAGE_CONTENT_TYPE)`
    → `llm.output_messages.0.message.contents.0.message_content.type`."""
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
    """Write `{suffix: value}` pairs into one content slot, skipping Nones."""
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
) -> None:
    """Reasoning block. Carries any subset of the three vendor continuity
    tokens — presence/absence of each field is the discriminator."""
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
        },
    )


def set_output_reasoning_summary(
    attrs: dict[str, Any],
    message_index: int,
    content_index: int,
    *,
    text: str,
) -> None:
    """Gemini-only: a `thought: true` summary part. Never carries a signature."""
    _set_content_block(
        attrs,
        message_index,
        content_index,
        {
            MessageContentAttributes.MESSAGE_CONTENT_TYPE: CONTENT_TYPE_REASONING_SUMMARY,
            MessageContentAttributes.MESSAGE_CONTENT_TEXT: text,
        },
    )


def set_output_redacted_reasoning(
    attrs: dict[str, Any],
    message_index: int,
    content_index: int,
    *,
    data: str,
) -> None:
    """Anthropic-only: replaces `thinking` text with an opaque `data` blob."""
    _set_content_block(
        attrs,
        message_index,
        content_index,
        {
            MessageContentAttributes.MESSAGE_CONTENT_TYPE: CONTENT_TYPE_REDACTED_REASONING,
            MESSAGE_CONTENT_REDACTED_DATA: data,
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
# The OpenInference spec has `llm.tools.{i}.tool.json_schema` for the raw
# JSON schema of each declared tool. We capture each provider's tool spec
# under that key so turn 2 can rebuild the exact `tools=[...]` argument
# without referring to the original Python literal.
# ---------------------------------------------------------------------------

LLM_TOOL_JSON_SCHEMA_SUFFIX = "tool.json_schema"


def set_tools(attrs: dict[str, Any], tool_json_schemas: list[dict[str, Any]]) -> None:
    for tool_index, schema in enumerate(tool_json_schemas):
        attrs[f"{SpanAttributes.LLM_TOOLS}.{tool_index}.{LLM_TOOL_JSON_SCHEMA_SUFFIX}"] = json.dumps(
            schema
        )


def read_tools(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    prefix = f"{SpanAttributes.LLM_TOOLS}."
    grouped_by_tool_index: dict[int, str] = {}
    for key, value in attrs.items():
        if not key.startswith(prefix) or not key.endswith(f".{LLM_TOOL_JSON_SCHEMA_SUFFIX}"):
            continue
        tool_index_str = key[len(prefix):].split(".", 1)[0]
        try:
            grouped_by_tool_index[int(tool_index_str)] = value
        except ValueError:
            continue
    return [json.loads(grouped_by_tool_index[i]) for i in sorted(grouped_by_tool_index)]


# ---------------------------------------------------------------------------
# Attribute reader
# ---------------------------------------------------------------------------


def read_input_message_content(attrs: dict[str, Any], message_index: int) -> str:
    """Return the `message.content` value for an input (user / tool) message."""
    return attrs[input_msg_key(message_index, MessageAttributes.MESSAGE_CONTENT)]


def read_output_contents(attrs: dict[str, Any], message_index: int) -> list[dict[str, Any]]:
    """Group `llm.output_messages.{message_index}.message.contents.{j}.*`
    keys by `j`. Returned dicts map the trailing suffix (e.g.
    `message_content.type`, `tool_call.id`) to its value — those suffixes
    are exactly the semconv constant values for each field."""
    prefix = (
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{message_index}"
        f".{MessageAttributes.MESSAGE_CONTENTS}."
    )
    grouped_by_content_index: dict[int, dict[str, Any]] = {}
    for key, value in attrs.items():
        if not key.startswith(prefix):
            continue
        content_index_str, _, suffix = key[len(prefix):].partition(".")
        try:
            content_index = int(content_index_str)
        except ValueError:
            continue
        grouped_by_content_index.setdefault(content_index, {})[suffix] = value
    return [grouped_by_content_index[i] for i in sorted(grouped_by_content_index)]


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
