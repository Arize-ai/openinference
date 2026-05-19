# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "anthropic>=0.40",
#     "opentelemetry-api",
#     "opentelemetry-sdk",
#     "opentelemetry-exporter-otlp-proto-http",
#     "openinference-semantic-conventions",
#     "arize-phoenix-client",
# ]
# ///
"""Anthropic Messages-API extended-thinking round-trip.

Captures `thinking` / `redacted_thinking` / `tool_use` blocks (with
`signature` / `data`) as flat span attributes, fetches them back out of
Phoenix, and rebuilds `messages=[user, assistant{thinking,...}, ...]` for
the follow-up turn.

Two scenarios:
  A. text:  thinking -> text
  B. tool:  thinking -> tool_use -> tool_result -> text
  Negative assertion (in B): drop the signature and expect HTTP 400.

Env vars: ANTHROPIC_API_KEY, PHOENIX_COLLECTOR_ENDPOINT,
          ANTHROPIC_MODEL (default `claude-opus-4-5`).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import anthropic
from anthropic import BadRequestError
from openinference.semconv.trace import (
    MessageContentAttributes,
    OpenInferenceLLMProviderValues,
    OpenInferenceLLMSystemValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)
from opentelemetry.trace import SpanKind

from common import (
    CONTENT_TYPE_REASONING,
    CONTENT_TYPE_REDACTED_REASONING,
    CONTENT_TYPE_TEXT,
    CONTENT_TYPE_TOOL_USE,
    MESSAGE_CONTENT_REDACTED_DATA,
    MESSAGE_CONTENT_SIGNATURE,
    TracingCtx,
    continuity_token_lengths,
    fetch_span_attributes,
    read_input_message_content,
    read_output_contents,
    read_tools,
    set_input_assistant_echo,
    set_input_tool_result,
    set_input_user_message,
    set_output_reasoning,
    set_output_redacted_reasoning,
    set_output_role,
    set_output_text,
    set_output_tool_use,
    set_tools,
    setup_tracing,
)

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-5")
BUDGET = int(os.environ.get("ANTHROPIC_BUDGET_TOKENS", "4000"))
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


def set_request_attributes(attrs: dict[str, Any]) -> None:
    attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] = OpenInferenceSpanKindValues.LLM.value
    attrs[SpanAttributes.LLM_PROVIDER] = OpenInferenceLLMProviderValues.ANTHROPIC.value
    attrs[SpanAttributes.LLM_SYSTEM] = OpenInferenceLLMSystemValues.ANTHROPIC.value
    attrs[SpanAttributes.LLM_MODEL_NAME] = MODEL
    attrs[SpanAttributes.LLM_INVOCATION_PARAMETERS] = json.dumps(
        {
            "thinking": {"type": "enabled", "budget_tokens": BUDGET},
            "max_tokens": MAX_TOKENS,
        }
    )


def record_assistant_turn(attrs: dict[str, Any], message: Any) -> None:
    set_output_role(attrs, 0, "assistant")
    for content_index, block in enumerate(message.content):
        block_type = block.type
        if block_type == "thinking":
            set_output_reasoning(
                attrs, 0, content_index, text=block.thinking, signature=block.signature
            )
        elif block_type == "redacted_thinking":
            set_output_redacted_reasoning(attrs, 0, content_index, data=block.data)
        elif block_type == "text":
            set_output_text(attrs, 0, content_index, block.text)
        elif block_type == "tool_use":
            set_output_tool_use(
                attrs,
                0,
                content_index,
                tool_call_id=block.id,
                name=block.name,
                arguments_json=json.dumps(block.input),
            )


def rebuild_assistant_content_blocks(
    fetched_attrs: dict[str, Any], *, strip_signature: bool = False
) -> list[dict[str, Any]]:
    rebuilt: list[dict[str, Any]] = []
    for content_block in read_output_contents(fetched_attrs, 0):
        block_type = content_block.get(MessageContentAttributes.MESSAGE_CONTENT_TYPE)
        if block_type == CONTENT_TYPE_REASONING:
            thinking_block: dict[str, Any] = {
                "type": "thinking",
                "thinking": content_block.get(MessageContentAttributes.MESSAGE_CONTENT_TEXT, ""),
            }
            signature = content_block.get(MESSAGE_CONTENT_SIGNATURE)
            if signature and not strip_signature:
                thinking_block["signature"] = signature
            rebuilt.append(thinking_block)
        elif block_type == CONTENT_TYPE_REDACTED_REASONING:
            rebuilt.append(
                {"type": "redacted_thinking", "data": content_block[MESSAGE_CONTENT_REDACTED_DATA]}
            )
        elif block_type == CONTENT_TYPE_TEXT:
            rebuilt.append(
                {
                    "type": "text",
                    "text": content_block[MessageContentAttributes.MESSAGE_CONTENT_TEXT],
                }
            )
        elif block_type == CONTENT_TYPE_TOOL_USE:
            rebuilt.append(
                {
                    "type": "tool_use",
                    "id": content_block[ToolCallAttributes.TOOL_CALL_ID],
                    "name": content_block[ToolCallAttributes.TOOL_CALL_FUNCTION_NAME],
                    "input": json.loads(
                        content_block[ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON]
                    ),
                }
            )
    return rebuilt


def replay_turn2(
    client: anthropic.Anthropic,
    fetched_attrs: dict[str, Any],
    messages: list[dict[str, Any]],
) -> Any:
    """Issue the follow-up call using only Phoenix-fetched config.

    Everything except the `messages` list (which carries the genuinely-new
    follow-up content) is decoded from the fetched attribute dict.
    """
    invocation_parameters = json.loads(fetched_attrs[SpanAttributes.LLM_INVOCATION_PARAMETERS])
    tools = read_tools(fetched_attrs)
    optional_tools = {"tools": tools} if tools else {}
    return client.messages.create(
        model=fetched_attrs[SpanAttributes.LLM_MODEL_NAME],
        messages=messages,
        **optional_tools,
        **invocation_parameters,
    )


def scenario_text(client: anthropic.Anthropic, ctx: TracingCtx) -> bool:
    print("\n[anthropic] scenario A: thinking → text round-trip")
    user_text = "In one short sentence, what's special about the number 1729?"

    with ctx.tracer.start_as_current_span(
        "anthropic.messages.text.turn1.original", kind=SpanKind.CLIENT
    ) as turn1_span:
        turn1_attrs: dict[str, Any] = {}
        set_request_attributes(turn1_attrs)
        set_input_user_message(turn1_attrs, 0, user_text)

        turn1_message = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            thinking={"type": "enabled", "budget_tokens": BUDGET},
            messages=[{"role": "user", "content": user_text}],
        )
        record_assistant_turn(turn1_attrs, turn1_message)
        turn1_span.set_attributes(turn1_attrs)
        turn1_span_context = turn1_span.get_span_context()

    ctx.flush()
    fetched_attrs = fetch_span_attributes(
        ctx,
        trace_id_hex=f"{turn1_span_context.trace_id:032x}",
        span_id_hex=f"{turn1_span_context.span_id:016x}",
    )
    prior_user_text = read_input_message_content(fetched_attrs, 0)
    captured_contents = read_output_contents(fetched_attrs, 0)
    follow_up_text = "Now in one sentence: who proved that property?"

    with ctx.tracer.start_as_current_span(
        "anthropic.messages.text.turn2.roundtrip", kind=SpanKind.CLIENT
    ) as turn2_span:
        turn2_attrs: dict[str, Any] = {}
        set_request_attributes(turn2_attrs)
        set_input_user_message(turn2_attrs, 0, prior_user_text)
        set_input_assistant_echo(turn2_attrs, 1, captured_contents)
        set_input_user_message(turn2_attrs, 2, follow_up_text)

        turn2_message = replay_turn2(
            client,
            fetched_attrs,
            messages=[
                {"role": "user", "content": prior_user_text},
                {"role": "assistant", "content": rebuild_assistant_content_blocks(fetched_attrs)},
                {"role": "user", "content": follow_up_text},
            ],
        )
        record_assistant_turn(turn2_attrs, turn2_message)
        turn2_span.set_attributes(turn2_attrs)

    final_text = next((block.text for block in turn2_message.content if block.type == "text"), "")
    print(f"  fetched_blocks={len(captured_contents)} "
          f"token_lens={continuity_token_lengths(captured_contents)}")
    print(f"  turn2_text={final_text[:120]!r}")
    return bool(final_text)


def scenario_tool(client: anthropic.Anthropic, ctx: TracingCtx) -> tuple[bool, bool]:
    print("\n[anthropic] scenario B: thinking → tool_use round-trip (+ negative)")
    user_text = "What's the weather in Paris? Use the tool."

    with ctx.tracer.start_as_current_span(
        "anthropic.messages.tool.turn1.original", kind=SpanKind.CLIENT
    ) as turn1_span:
        turn1_attrs: dict[str, Any] = {}
        set_request_attributes(turn1_attrs)
        set_tools(turn1_attrs, [WEATHER_TOOL])
        set_input_user_message(turn1_attrs, 0, user_text)

        turn1_message = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            thinking={"type": "enabled", "budget_tokens": BUDGET},
            tools=[WEATHER_TOOL],
            messages=[{"role": "user", "content": user_text}],
        )
        record_assistant_turn(turn1_attrs, turn1_message)
        turn1_span.set_attributes(turn1_attrs)
        turn1_span_context = turn1_span.get_span_context()

    ctx.flush()
    fetched_attrs = fetch_span_attributes(
        ctx,
        trace_id_hex=f"{turn1_span_context.trace_id:032x}",
        span_id_hex=f"{turn1_span_context.span_id:016x}",
    )
    prior_user_text = read_input_message_content(fetched_attrs, 0)
    captured_contents = read_output_contents(fetched_attrs, 0)
    tool_use_block = next(
        (
            block
            for block in captured_contents
            if block.get(MessageContentAttributes.MESSAGE_CONTENT_TYPE) == CONTENT_TYPE_TOOL_USE
        ),
        None,
    )
    if tool_use_block is None:
        print("  SKIP: model did not call the tool")
        return True, True

    tool_use_id = tool_use_block[ToolCallAttributes.TOOL_CALL_ID]
    tool_result_content_text = json.dumps({"temperature_c": 14.5})
    tool_result_user_message = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": tool_result_content_text,
            }
        ],
    }

    with ctx.tracer.start_as_current_span(
        "anthropic.messages.tool.turn2.roundtrip", kind=SpanKind.CLIENT
    ) as turn2_span:
        turn2_attrs: dict[str, Any] = {}
        set_request_attributes(turn2_attrs)
        set_tools(turn2_attrs, [WEATHER_TOOL])
        set_input_user_message(turn2_attrs, 0, prior_user_text)
        set_input_assistant_echo(turn2_attrs, 1, captured_contents)
        set_input_tool_result(
            turn2_attrs, 2, tool_call_id=tool_use_id, content_text=tool_result_content_text
        )

        turn2_message = replay_turn2(
            client,
            fetched_attrs,
            messages=[
                {"role": "user", "content": prior_user_text},
                {"role": "assistant", "content": rebuild_assistant_content_blocks(fetched_attrs)},
                tool_result_user_message,
            ],
        )
        record_assistant_turn(turn2_attrs, turn2_message)
        turn2_span.set_attributes(turn2_attrs)

    final_text = next((block.text for block in turn2_message.content if block.type == "text"), "")
    print(f"  fetched_blocks={len(captured_contents)} "
          f"token_lens={continuity_token_lengths(captured_contents)}")
    print(f"  positive_turn2_text={final_text[:120]!r}")
    positive_ok = bool(final_text)

    negative_rejected = False
    with ctx.tracer.start_as_current_span(
        "anthropic.messages.tool.turn2.roundtrip.negative", kind=SpanKind.CLIENT
    ) as negative_span:
        try:
            client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                thinking={"type": "enabled", "budget_tokens": BUDGET},
                tools=[WEATHER_TOOL],
                messages=[
                    {"role": "user", "content": user_text},
                    {
                        "role": "assistant",
                        "content": rebuild_assistant_content_blocks(
                            fetched_attrs, strip_signature=True
                        ),
                    },
                    tool_result_user_message,
                ],
            )
        except BadRequestError as error:
            negative_rejected = True
            negative_span.set_attribute("error.expected", True)
            negative_span.set_attribute("error.message", str(error)[:200])
            print(f"  negative correctly rejected: {str(error)[:120]!r}")
    if not negative_rejected:
        print("  negative FAILED: stripped-signature request was accepted")

    return positive_ok, negative_rejected


def main() -> int:
    ctx = setup_tracing("reasoning-roundtrip-anthropic")
    client = anthropic.Anthropic()
    text_ok = scenario_text(client, ctx)
    tool_positive_ok, tool_negative_ok = scenario_tool(client, ctx)
    print(
        f"\n[anthropic] text={text_ok}  "
        f"tool_positive={tool_positive_ok}  tool_negative={tool_negative_ok}"
    )
    return 0 if (text_ok and tool_positive_ok and tool_negative_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
