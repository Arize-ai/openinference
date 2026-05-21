# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai==2.37.0",
#     "opentelemetry-api==1.42.1",
#     "opentelemetry-sdk==1.42.1",
#     "opentelemetry-exporter-otlp-proto-http==1.42.1",
#     "openinference-semantic-conventions==0.1.29",
#     "openinference-instrumentation-openai==0.1.49",
# ]
# ///
"""OpenAI Responses-API reasoning round-trip.

Uses the real OpenAI instrumentor as the baseline span writer. The captured
finished span is augmented with proposed reasoning item fields, then re-exported
to Phoenix and used to reconstruct a stateless follow-up input list.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from openai import OpenAI
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
)
from openinference.semconv.trace import ToolCallAttributes

from common import (
    CONTENT_TYPE_REASONING,
    FACTORIZE_FOLLOW_UP_PROMPT,
    FACTORIZE_USER_PROMPT,
    InstrumentedTracingCtx,
    MESSAGE_CONTENT_ID,
    MESSAGE_CONTENT_SIGNATURE,
    TOOL_FOLLOW_UP_PROMPT,
    TOOL_RESULT_PAYLOAD,
    TOOL_USER_PROMPT,
    debug_print,
    export_augmented_span,
    export_original_span,
    mutate_span_attributes,
    output_content_key,
    print_attribute_keys,
    print_capture_summary,
    read_input_message_text,
    read_tools,
    setup_instrumented_tracing,
)

MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.5")

WEATHER_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Return the current temperature in Celsius for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


def user_text_item(text: str) -> dict[str, Any]:
    return {"role": "user", "content": [{"type": "input_text", "text": text}]}


def build_responses_create_request(
    *,
    model: str,
    input: Any,
    tools: list[dict[str, Any]] | None = None,
    effort: str = "medium",
    summary: str | None = "detailed",
) -> dict[str, Any]:
    reasoning: dict[str, str] = {"effort": effort}
    if summary is not None:
        reasoning["summary"] = summary
    request: dict[str, Any] = {
        "model": model,
        "input": input,
        "reasoning": reasoning,
        "store": False,
        "include": ["reasoning.encrypted_content"],
    }
    if tools:
        request["tools"] = tools
    return request


def simulate_openai_output_walk(response: Any) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for output_index, item in enumerate(response.output or []):
        item_type = getattr(item, "type", None)
        if item_type != "reasoning":
            continue

        # OpenAI's current instrumentor already emits each reasoning summary
        # part as an indexed text content block. The future shape should
        # reclassify those blocks as reasoning, not concatenate duplicate text
        # into content block 0.
        for summary_index, summary_part in enumerate(item.summary or []):
            attrs[
                output_content_key(
                    output_index,
                    summary_index,
                    MessageContentAttributes.MESSAGE_CONTENT_TYPE,
                )
            ] = CONTENT_TYPE_REASONING
            if text := getattr(summary_part, "text", ""):
                attrs[
                    output_content_key(
                        output_index,
                        summary_index,
                        MessageContentAttributes.MESSAGE_CONTENT_TEXT,
                    )
                ] = text

        token_content_index = 0
        attrs[
            output_content_key(output_index, token_content_index, MESSAGE_CONTENT_ID)
        ] = item.id
        encrypted = getattr(item, "encrypted_content", None)
        if encrypted:
            attrs[
                output_content_key(
                    output_index, token_content_index, MESSAGE_CONTENT_SIGNATURE
                )
            ] = encrypted
    return attrs


def print_openai_output_slots(response: Any) -> None:
    debug_print("\n[openai response.output slots]")
    for index, item in enumerate(response.output or []):
        debug_print(f"  output[{index}].type={getattr(item, 'type', None)}")


def augment_openai_span_from_response(span: Any, response: Any) -> None:
    baseline = dict(span.attributes or {})
    additions = simulate_openai_output_walk(response)
    print_openai_output_slots(response)
    print_attribute_keys(f"{span.name} baseline", baseline)
    mutate_span_attributes(span, additions)
    print_attribute_keys(f"{span.name} augmented", dict(span.attributes or {}))


def call_and_export_augmented(
    ctx: InstrumentedTracingCtx,
    client: OpenAI,
    request: dict[str, Any],
    *,
    root_name: str,
) -> tuple[Any, dict[str, Any]]:
    before_count = ctx.span_count()
    with ctx.tracer.start_as_current_span(root_name) as root_span:
        root_context = root_span.get_span_context()
        response = client.responses.create(**request)
        span = ctx.latest_span_since(before_count)
    ctx.export(ctx.span_by_id(root_context.span_id))
    export_original_span(ctx, span)
    augment_openai_span_from_response(span, response)
    fetched_attrs = export_augmented_span(ctx, span)
    return response, fetched_attrs


def output_message_indices(attrs: dict[str, Any]) -> list[int]:
    prefix = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}."
    indices: set[int] = set()
    for key in attrs:
        if not key.startswith(prefix):
            continue
        index_str = key[len(prefix) :].split(".", 1)[0]
        try:
            indices.add(int(index_str))
        except ValueError:
            continue
    return sorted(indices)


def read_content_block(
    attrs: dict[str, Any], message_index: int, content_index: int
) -> dict[str, Any]:
    prefix = (
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{message_index}."
        f"{MessageAttributes.MESSAGE_CONTENTS}.{content_index}."
    )
    block: dict[str, Any] = {}
    for key, value in attrs.items():
        if key.startswith(prefix):
            block[key[len(prefix) :]] = value
    return block


def read_all_output_content_blocks(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for message_index in output_message_indices(attrs):
        block = read_content_block(attrs, message_index, 0)
        if block:
            blocks.append(block)
    return blocks


def read_content_blocks_for_message(
    attrs: dict[str, Any], message_index: int
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    content_index = 0
    while True:
        block = read_content_block(attrs, message_index, content_index)
        if not block:
            break
        blocks.append(block)
        content_index += 1
    return blocks


def read_tool_call(attrs: dict[str, Any], message_index: int) -> dict[str, Any]:
    prefix = (
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{message_index}."
        f"{MessageAttributes.MESSAGE_TOOL_CALLS}.0."
    )
    result: dict[str, Any] = {}
    for key, value in attrs.items():
        if key.startswith(prefix):
            result[key[len(prefix) :]] = value
    return result


def rebuild_input_items(
    fetched_attrs: dict[str, Any], prior_user_text: str
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = [user_text_item(prior_user_text)]
    for message_index in output_message_indices(fetched_attrs):
        role = fetched_attrs.get(
            f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{message_index}.{MessageAttributes.MESSAGE_ROLE}"
        )
        content_blocks = read_content_blocks_for_message(fetched_attrs, message_index)
        if content_blocks and all(
            block.get(MessageContentAttributes.MESSAGE_CONTENT_TYPE)
            == CONTENT_TYPE_REASONING
            for block in content_blocks
        ):
            summary_parts: list[dict[str, str]] = []
            for block in content_blocks:
                if text := block.get(MessageContentAttributes.MESSAGE_CONTENT_TEXT):
                    summary_parts.append({"type": "summary_text", "text": text})
            token_block = content_blocks[0]
            reasoning_item = {
                "type": "reasoning",
                "id": token_block[MESSAGE_CONTENT_ID],
                "summary": summary_parts,
            }
            if encrypted := token_block.get(MESSAGE_CONTENT_SIGNATURE):
                reasoning_item["encrypted_content"] = encrypted
            items.append(reasoning_item)
            continue
        if role == "assistant" and content_blocks:
            items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": block[
                                MessageContentAttributes.MESSAGE_CONTENT_TEXT
                            ],
                        }
                        for block in content_blocks
                        if block.get(MessageContentAttributes.MESSAGE_CONTENT_TYPE)
                        == "text"
                    ],
                }
            )
            continue
        tool_call = read_tool_call(fetched_attrs, message_index)
        if tool_call:
            items.append(
                {
                    "type": "function_call",
                    "call_id": tool_call[ToolCallAttributes.TOOL_CALL_ID],
                    "name": tool_call[ToolCallAttributes.TOOL_CALL_FUNCTION_NAME],
                    "arguments": tool_call[
                        ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
                    ],
                }
            )
    return items


def build_request_from_fetched(
    fetched_attrs: dict[str, Any],
    turn2_input_items: list[dict[str, Any]],
) -> dict[str, Any]:
    invocation_parameters = json.loads(
        fetched_attrs[SpanAttributes.LLM_INVOCATION_PARAMETERS]
    )
    tools = read_tools(fetched_attrs) or None
    return {
        "model": fetched_attrs[SpanAttributes.LLM_MODEL_NAME],
        "input": turn2_input_items,
        "tools": tools,
        **invocation_parameters,
    }


def read_openai_user_input_text(attrs: dict[str, Any]) -> str:
    role_suffix = f".{MessageAttributes.MESSAGE_ROLE}"
    prefix = f"{SpanAttributes.LLM_INPUT_MESSAGES}."
    user_message_indices: list[int] = []
    for key, value in attrs.items():
        if not key.startswith(prefix) or not key.endswith(role_suffix):
            continue
        if value != "user":
            continue
        index_str = key[len(prefix) :].split(".", 1)[0]
        try:
            user_message_indices.append(int(index_str))
        except ValueError:
            continue
    if user_message_indices:
        return read_input_message_text(attrs, min(user_message_indices))
    return read_input_message_text(attrs, 0)


def scenario_text(
    client: OpenAI, ctx: InstrumentedTracingCtx, *, summary: str | None
) -> bool:
    print("\n[openai] scenario A: text round-trip")
    user_text = FACTORIZE_USER_PROMPT
    turn1_request = build_responses_create_request(
        model=MODEL, input=[user_text_item(user_text)], summary=summary
    )
    turn1_response, fetched_attrs = call_and_export_augmented(
        ctx, client, turn1_request, root_name="openai text turn1"
    )
    prior_user_text = read_openai_user_input_text(fetched_attrs)
    captured_contents = read_all_output_content_blocks(fetched_attrs)
    turn2_input_items = rebuild_input_items(fetched_attrs, prior_user_text)
    turn2_input_items.append(user_text_item(FACTORIZE_FOLLOW_UP_PROMPT))

    turn2_request = build_request_from_fetched(fetched_attrs, turn2_input_items)
    turn2_response, _ = call_and_export_augmented(
        ctx, client, turn2_request, root_name="openai text turn2"
    )
    turn2_returned_message = any(
        getattr(item, "type", None) == "message" for item in turn2_response.output
    )
    print_capture_summary(captured_contents)
    print(
        f"  turn2_ok={turn2_returned_message}  "
        f"turn2_text={(turn2_response.output_text or '')[:120]!r}"
    )
    return turn2_returned_message


def scenario_tool(
    client: OpenAI, ctx: InstrumentedTracingCtx, *, summary: str | None
) -> bool:
    print("\n[openai] scenario B: reasoning -> function_call round-trip")
    user_text = TOOL_USER_PROMPT
    turn1_request = build_responses_create_request(
        model=MODEL,
        input=[user_text_item(user_text)],
        tools=[WEATHER_TOOL],
        effort="high",
        summary=summary,
    )
    _, fetched_attrs = call_and_export_augmented(
        ctx, client, turn1_request, root_name="openai tool turn1"
    )
    prior_user_text = read_openai_user_input_text(fetched_attrs)
    captured_contents = read_all_output_content_blocks(fetched_attrs)
    turn2_input_items = rebuild_input_items(fetched_attrs, prior_user_text)
    function_call = next(
        (item for item in turn2_input_items if item.get("type") == "function_call"),
        None,
    )
    if function_call is None:
        print("  SKIP: model did not call the tool")
        return True
    tool_output_json = json.dumps(TOOL_RESULT_PAYLOAD)
    turn2_input_items.append(
        {
            "type": "function_call_output",
            "call_id": function_call["call_id"],
            "output": tool_output_json,
        }
    )
    turn2_input_items.append(user_text_item(TOOL_FOLLOW_UP_PROMPT))

    turn2_request = build_request_from_fetched(fetched_attrs, turn2_input_items)
    turn2_response, _ = call_and_export_augmented(
        ctx, client, turn2_request, root_name="openai tool turn2"
    )
    turn2_returned_message = any(
        getattr(item, "type", None) == "message" for item in turn2_response.output
    )
    print_capture_summary(captured_contents)
    print(f"  turn2_has_final_message={turn2_returned_message}")
    print(f"  turn2_text={(turn2_response.output_text or '')[:120]!r}")
    return turn2_returned_message


def parse_reasoning_summary() -> str | None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reasoning-summary",
        choices=["auto", "concise", "detailed", "off"],
        default=os.environ.get("OPENAI_REASONING_SUMMARY", "detailed"),
        help=(
            "Set OpenAI reasoning.summary. Use 'off' to omit the summary field "
            "from requests."
        ),
    )
    args = parser.parse_args()
    if args.reasoning_summary == "off":
        return None
    return args.reasoning_summary


def main() -> int:
    summary = parse_reasoning_summary()
    ctx = setup_instrumented_tracing("openai-reasoning-roundtrip")
    instrumentor = OpenAIInstrumentor()
    instrumentor.instrument(tracer_provider=ctx.provider)
    try:
        client = OpenAI()
        text_ok = scenario_text(client, ctx, summary=summary)
        tool_ok = scenario_tool(client, ctx, summary=summary)
        print(f"\n[openai] PASS={text_ok and tool_ok}  text={text_ok}  tool={tool_ok}")
        return 0 if (text_ok and tool_ok) else 1
    finally:
        instrumentor.uninstrument()
        ctx.shutdown()


if __name__ == "__main__":
    sys.exit(main())
