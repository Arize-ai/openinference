"""Shared helpers for the reasoning round-trip demo scripts.

Each per-provider script:

  1. Makes turn 1 against the live API.
  2. Writes the assistant turn into an OTel `LLM` span as flat
     `llm.output_messages.{i}.message.contents.{j}.<suffix>` keys, including
     the vendor continuity token.
  3. Force-flushes the OTel pipeline to a local Phoenix.
  4. Rebuilds the prior assistant turn from the augmented future span attributes.
  6. Sends turn 2 with the reconstructed history; asserts the server accepted it.

Established attribute keys come from `openinference.semconv.trace`. The local
string-literal block below holds keys this script needs while it pins released
semantic-convention packages that may not expose the newest constants yet.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from random import getrandbits, random
from typing import Any, Mapping

from openinference.instrumentation import get_llm_tool_attributes, safe_json_dumps
from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceMimeTypeValues,
    SpanAttributes,
    ToolCallAttributes,
)
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanContext

# ---------------------------------------------------------------------------
# Local semconv suffixes not available in every pinned released package.
#
# These become normal `MessageContentAttributes.<NAME>` /
# `ToolCallAttributes.<NAME>` imports once the script dependencies are bumped.
# ---------------------------------------------------------------------------

# Additional `message_content.type` discriminators. Every reasoning surface —
# raw thinking, summarized thinking (Gemini thought parts, OpenAI summaries),
# and Anthropic redacted thinking — uses the single value `"reasoning"`.
CONTENT_TYPE_TEXT = "text"  # parallels the existing MESSAGE_CONTENT_TEXT field
CONTENT_TYPE_REASONING = "reasoning"
CONTENT_TYPE_TOOL_USE = "tool_use"

# Additional sub-fields. These carry the continuity token that the next turn
# must echo verbatim to keep reasoning alive.
MESSAGE_CONTENT_ID = "message_content.id"  # OpenAI reasoning item id
MESSAGE_CONTENT_SIGNATURE = "message_content.signature"
MESSAGE_CONTENT_DATA = "message_content.data"
MESSAGE_CONTENT_ENCRYPTED_CONTENT = "message_content.encrypted_content"
TOOL_CALL_REASONING_SIGNATURE = "tool_call.reasoning_signature"

PROPOSED_CONTINUITY_TOKEN_FIELDS: tuple[str, ...] = (
    MESSAGE_CONTENT_SIGNATURE,
    MESSAGE_CONTENT_DATA,
    MESSAGE_CONTENT_ENCRYPTED_CONTENT,
    TOOL_CALL_REASONING_SIGNATURE,
)

# Scenario A (text) prompts — shared across provider round-trip scripts.
# Turn 1 is heavy enough to trigger extended/adaptive thinking; turn 2 forces
# reuse of the prior factorization (not a one-liner from the final answer).
FACTORIZE_USER_PROMPT = (
    f"Factorize {int(random() * 1e5)} completely into primes. "
    "Show every trial-division step (or equivalent method) until all factors are prime."
)
FACTORIZE_FOLLOW_UP_PROMPT = (
    "Using only the prime factorization from your previous answer—do not "
    "factor it again—consider every prime factor (including repeats). "
    "Add up every decimal digit that appears when you write each factor. "
    "Is that digit-sum even or odd? Show how you list the factors, extract "
    "each digit, and add them step by step."
)

# Scenario B (tool) prompts — a one-line weather ask rarely triggers thinking;
# these force plan → tool → multi-step interpretation on turn 2.
TOOL_USER_PROMPT = (
    "You must reason step-by-step before calling any tool. "
    "A heat advisory is issued in Paris when the temperature in Celsius, "
    "converted to Fahrenheit via (C × 9/5) + 32, is strictly above 95°F. "
    "Use get_weather for Paris only. "
    "In your reasoning: state the conversion formula, what you need from the tool, "
    "and how you will apply it to decide. Then call the tool."
)
TOOL_FOLLOW_UP_PROMPT = (
    "Using only the weather reading from the tool result, finish your "
    "step-by-step Fahrenheit conversion on the actual temperature and state "
    "whether the Paris heat advisory applies. Show every arithmetic step."
)
TOOL_RESULT_PAYLOAD = {"temperature_c": 14.5}

_DEBUG_LOG_PATH: Path | None = None


def _default_debug_log_path(project_name: str) -> Path:
    script_path = Path(sys.argv[0]).resolve()
    if script_path.exists():
        return script_path.with_suffix(".txt")
    safe_name = "".join(
        char if char.isalnum() or char in "-_" else "-" for char in project_name
    )
    return Path(__file__).with_name(f"{safe_name}.txt")


def reset_debug_log(project_name: str) -> Path:
    global _DEBUG_LOG_PATH
    configured_path = os.environ.get("REASONING_ROUNDTRIP_LOG_PATH")
    _DEBUG_LOG_PATH = (
        Path(configured_path).expanduser()
        if configured_path
        else _default_debug_log_path(project_name)
    )
    _DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _DEBUG_LOG_PATH.write_text(
        f"# Reasoning round-trip debug log: {project_name}\n\n",
        encoding="utf-8",
    )
    print(f"[debug] verbose span output: {_DEBUG_LOG_PATH}")
    return _DEBUG_LOG_PATH


def debug_print(*values: Any, sep: str = " ", end: str = "\n") -> None:
    path = _DEBUG_LOG_PATH or reset_debug_log("reasoning-roundtrip")
    with path.open("a", encoding="utf-8") as stream:
        stream.write(sep.join(str(value) for value in values))
        stream.write(end)


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
class InstrumentedTracingCtx:
    """Capture real instrumentor spans, mutate them, then export to Phoenix."""

    tracer: trace.Tracer
    provider: TracerProvider
    memory_exporter: InMemorySpanExporter
    phoenix_exporter: OTLPSpanExporter
    project_name: str
    phoenix_base_url: str

    def span_count(self) -> int:
        return len(self.memory_exporter.get_finished_spans())

    def latest_span_since(self, previous_count: int) -> ReadableSpan:
        spans = self.memory_exporter.get_finished_spans()
        if len(spans) <= previous_count:
            raise RuntimeError("No finished instrumentor span was captured")
        return spans[-1]

    def span_by_id(self, span_id: int) -> ReadableSpan:
        for span in self.memory_exporter.get_finished_spans():
            if span.context.span_id == span_id:
                return span
        raise RuntimeError(f"Finished span {span_id:016x} was not captured")

    def export(self, span: ReadableSpan) -> None:
        self.phoenix_exporter.export((span,))

    def shutdown(self) -> None:
        self.provider.force_flush()
        self.phoenix_exporter.shutdown()


def setup_instrumented_tracing(project_name: str) -> InstrumentedTracingCtx:
    """Tracer provider for real instrumentors plus manual Phoenix re-export."""
    reset_debug_log(project_name)
    base_url = os.environ.get(
        "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"
    ).rstrip("/")
    resource = Resource.create({"openinference.project.name": project_name})
    provider = TracerProvider(resource=resource)
    memory_exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory_exporter))
    phoenix_exporter = OTLPSpanExporter(endpoint=f"{base_url}/v1/traces")
    return InstrumentedTracingCtx(
        tracer=provider.get_tracer("reasoning-roundtrip"),
        provider=provider,
        memory_exporter=memory_exporter,
        phoenix_exporter=phoenix_exporter,
        project_name=project_name,
        phoenix_base_url=base_url,
    )


def mutate_span_attributes(span: ReadableSpan, additions: Mapping[str, Any]) -> None:
    """Patch a captured finished span in-place for this demo harness.

    ``ReadableSpan`` is normally treated as immutable after export, but the
    in-memory exporter keeps the object around. Mutating ``_attributes`` lets
    us preserve the real instrumentor span shape and add only the proposed
    reasoning fields before re-exporting that same span to Phoenix.
    """
    attributes = dict(span.attributes or {})
    attributes.update(additions)
    _drop_scalar_message_content_when_contents_present(attributes)
    setattr(span, "_attributes", attributes)


def _drop_scalar_message_content_when_contents_present(attrs: dict[str, Any]) -> None:
    """Remove scalar output `message.content` where indexed `message.contents` exists."""
    scalar_suffix = MessageAttributes.MESSAGE_CONTENT
    contents_suffix = MessageAttributes.MESSAGE_CONTENTS
    keys_to_remove: list[str] = []
    for key in attrs:
        if not key.startswith(f"{SpanAttributes.LLM_OUTPUT_MESSAGES}."):
            continue
        if not key.endswith(f".{scalar_suffix}"):
            continue
        parent_prefix = key[: -len(scalar_suffix)]
        contents_prefix = f"{parent_prefix}{contents_suffix}."
        if any(candidate.startswith(contents_prefix) for candidate in attrs):
            keys_to_remove.append(key)
    for key in keys_to_remove:
        attrs.pop(key, None)


def export_original_span(ctx: InstrumentedTracingCtx, span: ReadableSpan) -> None:
    """Export the current instrumentor child span before augmentation."""
    setattr(span, "_name", "current")
    ctx.export(span)


def assign_new_span_id(span: ReadableSpan) -> SpanContext:
    """Patch a finished span's context so its augmented export is distinct."""
    old_context = span.context
    new_span_id = 0
    while new_span_id == 0 or new_span_id == old_context.span_id:
        new_span_id = getrandbits(64)
    new_context = SpanContext(
        trace_id=old_context.trace_id,
        span_id=new_span_id,
        is_remote=old_context.is_remote,
        trace_flags=old_context.trace_flags,
        trace_state=old_context.trace_state,
    )
    setattr(span, "_context", new_context)
    return new_context


def _attribute_value_preview(value: Any, *, limit: int = 25) -> str:
    preview = str(value).replace("\n", "\\n")
    if len(preview) > limit:
        return f"{preview[:limit]}..."
    return preview


def print_attribute_keys(label: str, attrs: Mapping[str, Any]) -> None:
    """Print attribute keys with a short value preview."""
    printable_keys = [
        key
        for key in attrs
        if key.startswith(f"{SpanAttributes.LLM_INPUT_MESSAGES}.")
        or key.startswith(f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.")
    ]
    debug_print(f"\n[{label}] attribute_keys ({len(printable_keys)})")
    for key in sorted(printable_keys):
        value = attrs[key]
        debug_print(f"  {key} = {_attribute_value_preview(value)}")


def export_augmented_span(
    ctx: InstrumentedTracingCtx, span: ReadableSpan
) -> dict[str, Any]:
    """Export a mutated captured span under a new span id."""
    setattr(span, "_name", "future")
    assign_new_span_id(span)
    ctx.export(span)
    return dict(span.attributes or {})


# ---------------------------------------------------------------------------
# Attribute writers
# ---------------------------------------------------------------------------


def _serialize_response_json(response: Any) -> str:
    """JSON-encode an API response the way provider instrumentors do."""
    if hasattr(response, "model_dump_json") and callable(response.model_dump_json):
        try:
            return response.model_dump_json(exclude_none=True)
        except TypeError:
            return response.model_dump_json()
    return safe_json_dumps(response)


def set_span_io_values(
    attrs: dict[str, Any],
    *,
    request: Mapping[str, Any],
    response: Any | None = None,
) -> None:
    """Set ``input.value`` / ``output.value`` like provider instrumentors.

    Request kwargs are ``safe_json_dumps``; responses use ``model_dump_json()``
    when available (Anthropic, OpenAI, Gemini SDK types).
    """
    attrs[SpanAttributes.INPUT_VALUE] = safe_json_dumps(request)
    attrs[SpanAttributes.INPUT_MIME_TYPE] = OpenInferenceMimeTypeValues.JSON.value
    if response is not None:
        attrs[SpanAttributes.OUTPUT_VALUE] = _serialize_response_json(response)
        attrs[SpanAttributes.OUTPUT_MIME_TYPE] = OpenInferenceMimeTypeValues.JSON.value


def set_input_user_message(
    attrs: dict[str, Any], message_index: int, text: str
) -> None:
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
    attrs[input_msg_key(message_index, MessageAttributes.MESSAGE_TOOL_CALL_ID)] = (
        tool_call_id
    )
    attrs[input_msg_key(message_index, MessageAttributes.MESSAGE_CONTENT)] = (
        content_text
    )


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
            MESSAGE_CONTENT_SIGNATURE: thought_signature,
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
    - Anthropic redacted:      `redacted_data` (no `text`)
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
            MESSAGE_CONTENT_SIGNATURE: signature or thought_signature,
            MESSAGE_CONTENT_DATA: redacted_data,
            MESSAGE_CONTENT_ENCRYPTED_CONTENT: encrypted_content,
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
            TOOL_CALL_REASONING_SIGNATURE: thought_signature,
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
        if not key.startswith(prefix) or not key.endswith(
            f".{_LLM_TOOL_JSON_SCHEMA_SUFFIX}"
        ):
            continue
        tool_index_str = key[len(prefix) :].split(".", 1)[0]
        try:
            schema_json_by_index[int(tool_index_str)] = value
        except ValueError:
            continue
    return [
        json.loads(schema_json_by_index[tool_index])
        for tool_index in sorted(schema_json_by_index)
    ]


# ---------------------------------------------------------------------------
# Attribute reader
# ---------------------------------------------------------------------------


def read_input_message_content(attrs: dict[str, Any], message_index: int) -> str:
    """Return the `message.content` value for an input (user / tool) message."""
    return attrs[input_msg_key(message_index, MessageAttributes.MESSAGE_CONTENT)]


def read_input_message_text(attrs: dict[str, Any], message_index: int) -> str:
    """Return input text from either scalar `content` or indexed `contents`.

    Multi-part prompts are preferable for this exercise because they exercise
    the instrumentors' `message.contents.{i}` shape. Keep scalar fallback so we
    can still read spans produced before this script change.
    """
    scalar_key = input_msg_key(message_index, MessageAttributes.MESSAGE_CONTENT)
    if scalar_key in attrs:
        return str(attrs[scalar_key])

    prefix = (
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.{message_index}"
        f".{MessageAttributes.MESSAGE_CONTENTS}."
    )
    text_by_content_index: dict[int, str] = {}
    for key, value in attrs.items():
        if not key.startswith(prefix) or not key.endswith(
            f".{MessageContentAttributes.MESSAGE_CONTENT_TEXT}"
        ):
            continue
        content_index_str = key[len(prefix) :].split(".", 1)[0]
        try:
            text_by_content_index[int(content_index_str)] = str(value)
        except ValueError:
            continue
    if not text_by_content_index:
        raise KeyError(scalar_key)
    return "\n".join(
        text_by_content_index[content_index]
        for content_index in sorted(text_by_content_index)
    )


def read_output_contents(
    attrs: dict[str, Any], message_index: int
) -> list[dict[str, Any]]:
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
        content_index_str, _, suffix = key[len(prefix) :].partition(".")
        try:
            content_index = int(content_index_str)
        except ValueError:
            continue
        fields_by_content_index.setdefault(content_index, {})[suffix] = value
    return [
        fields_by_content_index[content_index]
        for content_index in sorted(fields_by_content_index)
    ]


def find_tool_use_block(contents: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the first `tool_use` content block, or None if none."""
    return next(
        (
            block
            for block in contents
            if block.get(MessageContentAttributes.MESSAGE_CONTENT_TYPE)
            == CONTENT_TYPE_TOOL_USE
        ),
        None,
    )


def find_reasoning_block(contents: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the first `reasoning` content block, or None if none."""
    return next(
        (
            block
            for block in contents
            if block.get(MessageContentAttributes.MESSAGE_CONTENT_TYPE)
            == CONTENT_TYPE_REASONING
        ),
        None,
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
    print(
        f"  fetched_blocks={len(contents)} token_lens={continuity_token_lengths(contents)}"
    )
