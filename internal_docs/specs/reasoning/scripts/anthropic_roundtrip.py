# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "anthropic==0.104.0",
#     "opentelemetry-api==1.42.1",
#     "opentelemetry-sdk==1.42.1",
#     "opentelemetry-exporter-otlp-proto-http==1.42.1",
#     "openinference-semantic-conventions==0.1.29",
#     "openinference-instrumentation-anthropic==1.0.5",
# ]
# ///
"""Anthropic Messages-API Opus 4.7 adaptive-thinking round-trip.

Uses the real Anthropic instrumentor as the baseline span writer. Each live SDK
call is captured with an in-memory exporter, the captured span's private flat
attribute map is augmented with proposed reasoning-continuity attributes, and
that same span is re-exported to Phoenix for round-trip reconstruction.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Literal

import anthropic
from anthropic import BadRequestError
from openinference.instrumentation.anthropic import AnthropicInstrumentor
from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
    ToolCallAttributes,
)

from common import (
    CONTENT_TYPE_REASONING,
    FACTORIZE_FOLLOW_UP_PROMPT,
    FACTORIZE_USER_PROMPT,
    InstrumentedTracingCtx,
    MESSAGE_CONTENT_SIGNATURE,
    TOOL_FOLLOW_UP_PROMPT,
    TOOL_RESULT_PAYLOAD,
    TOOL_USER_PROMPT,
    debug_print,
    export_augmented_span,
    export_original_span,
    find_reasoning_block,
    find_tool_use_block,
    input_content_key,
    mutate_span_attributes,
    output_content_key,
    print_attribute_keys,
    print_capture_summary,
    read_input_message_text,
    read_output_contents,
    read_tools,
    setup_instrumented_tracing,
)

DisplayMode = Literal["omitted", "summarized"]

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-7")
MAX_TOKENS = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "8000"))

WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Current temperature in Celsius.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


def user_text_message(text: str) -> dict[str, Any]:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def adaptive_thinking_config(display: DisplayMode) -> dict[str, Any]:
    return {"type": "adaptive", "display": display}


def base_request(
    *,
    messages: list[dict[str, Any]],
    display: DisplayMode,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "thinking": adaptive_thinking_config(display),
        "messages": messages,
    }
    if tools:
        request["tools"] = tools
    return request


def augment_anthropic_span_from_response(span: Any, message: Any) -> None:
    """Patch proposed reasoning attrs onto the real instrumentor span."""
    baseline = dict(span.attributes or {})
    simulated_future = simulate_anthropic_content_walk(message)
    additions = proposed_additions_from_simulation(simulated_future)

    print_anthropic_content_slots(message)
    print_attribute_keys(f"{span.name} baseline", baseline)
    mutate_span_attributes(span, additions)
    print_attribute_keys(f"{span.name} augmented", dict(span.attributes or {}))


def print_anthropic_content_slots(message: Any) -> None:
    """Print response block indices/types used to place proposed attributes."""
    debug_print("\n[anthropic response.content slots]")
    for index, block in enumerate(message.content):
        debug_print(f"  content[{index}].type={block.type}")


def simulate_anthropic_content_walk(message: Any) -> dict[str, Any]:
    """Simulate a future instrumentor walking Anthropic response.content[].

    The existing Anthropic instrumentor already emits text and tool calls in
    its established shape. This simulation shows the content-block-indexed
    keys a future implementation would see if it walked every Anthropic block,
    including currently skipped thinking/redacted_thinking blocks.
    """
    attrs: dict[str, Any] = {}
    tool_call_index = 0
    for content_index, block in enumerate(message.content):
        if block.type == "thinking":
            attrs[
                output_content_key(
                    0, content_index, MessageContentAttributes.MESSAGE_CONTENT_TYPE
                )
            ] = CONTENT_TYPE_REASONING
            attrs[
                output_content_key(
                    0, content_index, MessageContentAttributes.MESSAGE_CONTENT_TEXT
                )
            ] = block.thinking
            attrs[output_content_key(0, content_index, MESSAGE_CONTENT_SIGNATURE)] = (
                block.signature
            )
        elif block.type == "redacted_thinking":
            attrs[
                output_content_key(
                    0, content_index, MessageContentAttributes.MESSAGE_CONTENT_TYPE
                )
            ] = CONTENT_TYPE_REASONING
            attrs[output_content_key(0, content_index, MESSAGE_CONTENT_SIGNATURE)] = (
                block.data
            )
        elif block.type == "text":
            attrs[
                output_content_key(
                    0, content_index, MessageContentAttributes.MESSAGE_CONTENT_TYPE
                )
            ] = "text"
            attrs[
                output_content_key(
                    0, content_index, MessageContentAttributes.MESSAGE_CONTENT_TEXT
                )
            ] = block.text
        elif block.type == "tool_use":
            attrs[
                output_content_key(
                    0, content_index, MessageContentAttributes.MESSAGE_CONTENT_TYPE
                )
            ] = "tool_use"
            attrs[
                output_content_key(0, content_index, ToolCallAttributes.TOOL_CALL_ID)
            ] = block.id
            attrs[
                output_content_key(
                    0, content_index, ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
                )
            ] = block.name
            attrs[
                output_content_key(
                    0,
                    content_index,
                    ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON,
                )
            ] = json.dumps(block.input)
            attrs[
                f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0."
                f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{tool_call_index}."
                f"{ToolCallAttributes.TOOL_CALL_ID}"
            ] = block.id
            tool_call_index += 1
    return attrs


def proposed_additions_from_simulation(
    simulated_future: dict[str, Any],
) -> dict[str, Any]:
    """Keep only additions needed for the proposed reasoning experiment.

    Text blocks move from today's scalar `message.content` into indexed
    `message.contents.{i}` so ordering and payload stay together. Tool data
    still comes from today's `message.tool_calls` fields, with a compatibility
    patch for `tool_call.id` required to replay Anthropic tool_result messages.
    """
    additions: dict[str, Any] = {}
    for key, value in simulated_future.items():
        if key.endswith(
            f".{MessageContentAttributes.MESSAGE_CONTENT_TYPE}"
        ) and value in {CONTENT_TYPE_REASONING, "text", "tool_use"}:
            additions[key] = value
        elif key.endswith(f".{MessageContentAttributes.MESSAGE_CONTENT_TEXT}"):
            additions[key] = value
        elif key.endswith(f".{MESSAGE_CONTENT_SIGNATURE}"):
            additions[key] = value
        elif f".{MessageAttributes.MESSAGE_TOOL_CALLS}." in key and key.endswith(
            f".{ToolCallAttributes.TOOL_CALL_ID}"
        ):
            additions[key] = value
    return additions


def call_and_export_augmented(
    ctx: InstrumentedTracingCtx,
    client: anthropic.Anthropic,
    request: dict[str, Any],
    *,
    root_name: str,
) -> tuple[Any, dict[str, Any]]:
    before_count = ctx.span_count()
    with ctx.tracer.start_as_current_span(root_name) as root_span:
        root_context = root_span.get_span_context()
        message = client.messages.create(**request)
        span = ctx.latest_span_since(before_count)
    ctx.export(ctx.span_by_id(root_context.span_id))
    export_original_span(ctx, span)
    augment_anthropic_span_from_response(span, message)
    fetched_attrs = export_augmented_span(ctx, span)
    return message, fetched_attrs


def call_expect_bad_request_and_export(
    ctx: InstrumentedTracingCtx,
    client: anthropic.Anthropic,
    request: dict[str, Any],
    *,
    root_name: str,
) -> BadRequestError | None:
    before_count = ctx.span_count()
    error: BadRequestError | None = None
    span: Any | None = None
    with ctx.tracer.start_as_current_span(root_name) as root_span:
        root_context = root_span.get_span_context()
        try:
            client.messages.create(**request)
        except BadRequestError as exc:
            error = exc
            span = ctx.latest_span_since(before_count)
    if error is not None and span is not None:
        ctx.export(ctx.span_by_id(root_context.span_id))
        export_original_span(ctx, span)
        print_attribute_keys(
            f"{span.name} negative_baseline", dict(span.attributes or {})
        )
        augment_anthropic_input_span_from_request(span, request)
        print_attribute_keys(
            f"{span.name} negative_augmented", dict(span.attributes or {})
        )
        export_augmented_span(ctx, span)
        return error
    return None


def augment_anthropic_input_span_from_request(
    span: Any, request: dict[str, Any]
) -> None:
    additions: dict[str, Any] = {}
    messages = request.get("messages") or []
    for message_index, message in enumerate(messages):
        role = message.get("role") if isinstance(message, dict) else None
        if role != "assistant":
            continue
        for content_index, block in enumerate(message.get("content") or []):
            block_type = block.get("type") if isinstance(block, dict) else None
            if block_type == "thinking":
                additions[
                    input_content_key(
                        message_index,
                        content_index,
                        MessageContentAttributes.MESSAGE_CONTENT_TYPE,
                    )
                ] = CONTENT_TYPE_REASONING
                additions[
                    input_content_key(
                        message_index,
                        content_index,
                        MessageContentAttributes.MESSAGE_CONTENT_TEXT,
                    )
                ] = block.get("thinking", "")
                if signature := block.get("signature"):
                    additions[
                        input_content_key(
                            message_index, content_index, MESSAGE_CONTENT_SIGNATURE
                        )
                    ] = signature
            elif block_type == "redacted_thinking":
                additions[
                    input_content_key(
                        message_index,
                        content_index,
                        MessageContentAttributes.MESSAGE_CONTENT_TYPE,
                    )
                ] = CONTENT_TYPE_REASONING
                additions[
                    input_content_key(
                        message_index, content_index, MESSAGE_CONTENT_SIGNATURE
                    )
                ] = block.get("data", "")
            elif block_type == "text":
                additions[
                    input_content_key(
                        message_index,
                        content_index,
                        MessageContentAttributes.MESSAGE_CONTENT_TYPE,
                    )
                ] = "text"
                additions[
                    input_content_key(
                        message_index,
                        content_index,
                        MessageContentAttributes.MESSAGE_CONTENT_TEXT,
                    )
                ] = block.get("text", "")
            elif block_type == "tool_use":
                additions[
                    input_content_key(
                        message_index,
                        content_index,
                        MessageContentAttributes.MESSAGE_CONTENT_TYPE,
                    )
                ] = "tool_use"
                additions[
                    input_content_key(
                        message_index, content_index, ToolCallAttributes.TOOL_CALL_ID
                    )
                ] = block.get("id", "")
                additions[
                    input_content_key(
                        message_index,
                        content_index,
                        ToolCallAttributes.TOOL_CALL_FUNCTION_NAME,
                    )
                ] = block.get("name", "")
                additions[
                    input_content_key(
                        message_index,
                        content_index,
                        ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON,
                    )
                ] = json.dumps(block.get("input") or {})
    mutate_span_attributes(span, additions)


def build_messages_create_request(
    fetched_attrs: dict[str, Any],
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    invocation_parameters = json.loads(
        fetched_attrs[SpanAttributes.LLM_INVOCATION_PARAMETERS]
    )
    request: dict[str, Any] = {
        "model": fetched_attrs[SpanAttributes.LLM_MODEL_NAME],
        "messages": messages,
        **invocation_parameters,
    }
    tools = read_tools(fetched_attrs)
    if tools:
        request["tools"] = tools
    return request


def read_output_tool_calls(fetched_attrs: dict[str, Any]) -> list[dict[str, Any]]:
    prefix = (
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0."
        f"{MessageAttributes.MESSAGE_TOOL_CALLS}."
    )
    by_index: dict[int, dict[str, Any]] = {}
    for key, value in fetched_attrs.items():
        if not key.startswith(prefix):
            continue
        index_str, _, suffix = key[len(prefix) :].partition(".")
        try:
            index = int(index_str)
        except ValueError:
            continue
        by_index.setdefault(index, {})[suffix] = value
    return [by_index[index] for index in sorted(by_index)]


def rebuild_assistant_content_blocks(
    fetched_attrs: dict[str, Any], *, strip_signature: bool = False
) -> list[dict[str, Any]]:
    rebuilt: list[dict[str, Any]] = []
    tool_calls = iter(read_output_tool_calls(fetched_attrs))
    for content_block in read_output_contents(fetched_attrs, 0):
        block_type = content_block.get(MessageContentAttributes.MESSAGE_CONTENT_TYPE)
        if block_type == "text":
            rebuilt.append(
                {
                    "type": "text",
                    "text": content_block[
                        MessageContentAttributes.MESSAGE_CONTENT_TEXT
                    ],
                }
            )
            continue
        if block_type == "tool_use":
            tool_call = next(tool_calls, {})
            tool_call_id = tool_call.get(ToolCallAttributes.TOOL_CALL_ID)
            name = tool_call.get(ToolCallAttributes.TOOL_CALL_FUNCTION_NAME)
            arguments = tool_call.get(
                ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
            )
            if tool_call_id and name and arguments:
                rebuilt.append(
                    {
                        "type": "tool_use",
                        "id": tool_call_id,
                        "name": name,
                        "input": json.loads(arguments),
                    }
                )
            continue

        signature = content_block.get(MESSAGE_CONTENT_SIGNATURE)
        text = content_block.get(MessageContentAttributes.MESSAGE_CONTENT_TEXT)
        if text is None and signature:
            rebuilt.append({"type": "redacted_thinking", "data": signature})
            continue
        thinking_block: dict[str, Any] = {
            "type": "thinking",
            "thinking": text or "",
        }
        if signature and not strip_signature:
            thinking_block["signature"] = signature
        rebuilt.append(thinking_block)
    return rebuilt


def scenario_text(
    client: anthropic.Anthropic, ctx: InstrumentedTracingCtx, *, display: DisplayMode
) -> bool:
    print(f"\n[anthropic:{display}] scenario A: adaptive thinking -> text")
    user_text = FACTORIZE_USER_PROMPT
    turn1_request = base_request(
        messages=[user_text_message(user_text)], display=display
    )
    _, fetched_attrs = call_and_export_augmented(
        ctx, client, turn1_request, root_name=f"anthropic {display} text turn1"
    )
    prior_user_text = read_input_message_text(fetched_attrs, 0)
    captured_contents = read_output_contents(fetched_attrs, 0)
    reasoning_block = find_reasoning_block(captured_contents)
    if reasoning_block is None:
        print("  SKIP: response did not include a replayable thinking block")
        return True

    turn2_messages = [
        user_text_message(prior_user_text),
        {
            "role": "assistant",
            "content": rebuild_assistant_content_blocks(fetched_attrs),
        },
        user_text_message(FACTORIZE_FOLLOW_UP_PROMPT),
    ]
    turn2_request = build_messages_create_request(fetched_attrs, turn2_messages)
    turn2_request["thinking"] = adaptive_thinking_config(display)
    turn2_message, _ = call_and_export_augmented(
        ctx, client, turn2_request, root_name=f"anthropic {display} text turn2"
    )

    final_text = next(
        (block.text for block in turn2_message.content if block.type == "text"), ""
    )
    print_capture_summary(captured_contents)
    print(f"  turn2_text={final_text[:120]!r}")
    return bool(final_text)


def scenario_tool(
    client: anthropic.Anthropic, ctx: InstrumentedTracingCtx, *, display: DisplayMode
) -> tuple[bool, bool]:
    print(
        f"\n[anthropic:{display}] "
        "scenario B: adaptive thinking -> tool_use round-trip (+ negative)"
    )
    user_text = TOOL_USER_PROMPT
    turn1_request = base_request(
        messages=[user_text_message(user_text)],
        tools=[WEATHER_TOOL],
        display=display,
    )
    _, fetched_attrs = call_and_export_augmented(
        ctx, client, turn1_request, root_name=f"anthropic {display} tool turn1"
    )
    prior_user_text = read_input_message_text(fetched_attrs, 0)
    captured_contents = read_output_contents(fetched_attrs, 0)
    tool_use_block = find_tool_use_block(captured_contents)
    tool_calls = read_output_tool_calls(fetched_attrs)
    if (
        tool_use_block is None
        or not tool_calls
        or not tool_calls[0].get(ToolCallAttributes.TOOL_CALL_ID)
    ):
        print("  SKIP: model did not call the tool")
        return True, True

    tool_use_id = tool_calls[0][ToolCallAttributes.TOOL_CALL_ID]
    tool_result_user_message = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": json.dumps(TOOL_RESULT_PAYLOAD),
            }
        ],
    }
    follow_up_user_message = user_text_message(TOOL_FOLLOW_UP_PROMPT)

    turn2_messages = [
        user_text_message(prior_user_text),
        {
            "role": "assistant",
            "content": rebuild_assistant_content_blocks(fetched_attrs),
        },
        tool_result_user_message,
        follow_up_user_message,
    ]
    turn2_request = build_messages_create_request(fetched_attrs, turn2_messages)
    turn2_request["thinking"] = adaptive_thinking_config(display)
    turn2_message, _ = call_and_export_augmented(
        ctx, client, turn2_request, root_name=f"anthropic {display} tool turn2"
    )

    final_text = next(
        (block.text for block in turn2_message.content if block.type == "text"), ""
    )
    print_capture_summary(captured_contents)
    print(f"  positive_turn2_text={final_text[:120]!r}")
    positive_ok = bool(final_text)

    negative_rejected = False
    negative_request = base_request(
        messages=[
            user_text_message(prior_user_text),
            {
                "role": "assistant",
                "content": rebuild_assistant_content_blocks(
                    fetched_attrs, strip_signature=True
                ),
            },
            tool_result_user_message,
            follow_up_user_message,
        ],
        tools=[WEATHER_TOOL],
        display=display,
    )
    error = call_expect_bad_request_and_export(
        ctx,
        client,
        negative_request,
        root_name=f"anthropic {display} tool negative",
    )
    if error is not None:
        negative_rejected = True
        print(f"  negative correctly rejected: {str(error)[:120]!r}")
    if not negative_rejected:
        print("  negative FAILED: stripped-signature request was accepted")
    return positive_ok, negative_rejected


def parse_display_modes() -> list[DisplayMode]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--display",
        "--thinking-display",
        dest="thinking_display",
        choices=["omitted", "summarized", "both"],
        default=os.environ.get("ANTHROPIC_THINKING_DISPLAY", "summarized"),
        help="Set Anthropic adaptive thinking display mode.",
    )
    args = parser.parse_args()
    if args.thinking_display == "both":
        return ["omitted", "summarized"]
    return [args.thinking_display]


def main() -> int:
    ctx = setup_instrumented_tracing("anthropic-reasoning-roundtrip")
    instrumentor = AnthropicInstrumentor()
    instrumentor.instrument(tracer_provider=ctx.provider)
    try:
        client = anthropic.Anthropic()
        results: dict[str, bool] = {}
        for display in parse_display_modes():
            text_ok = scenario_text(client, ctx, display=display)
            tool_positive_ok, tool_negative_ok = scenario_tool(
                client, ctx, display=display
            )
            display_ok = text_ok and tool_positive_ok and tool_negative_ok
            results[display] = display_ok
            print(
                f"\n[anthropic:{display}] text={text_ok}  "
                f"tool_positive={tool_positive_ok}  "
                f"tool_negative={tool_negative_ok}  PASS={display_ok}"
            )
        overall_ok = all(results.values())
        print(f"\n[anthropic] PASS={overall_ok}  results={results}")
        return 0 if overall_ok else 1
    finally:
        instrumentor.uninstrument()
        ctx.shutdown()


if __name__ == "__main__":
    sys.exit(main())
