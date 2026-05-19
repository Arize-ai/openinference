# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai>=2.0",
#     "opentelemetry-api",
#     "opentelemetry-sdk",
#     "opentelemetry-exporter-otlp-proto-http",
#     "openinference-semantic-conventions",
#     "arize-phoenix-client",
# ]
# ///
"""OpenAI Responses-API reasoning round-trip.

Proves that the OpenAI reasoning surface (`encrypted_content` + reasoning
item id + summary parts + any following `function_call`) can be captured as
flat string-valued span attributes, fetched back out of Phoenix, and used to
rebuild an `input=[...]` list that the Responses API accepts on a stateless
follow-up turn.

Two scenarios:
  A. text:  reasoning -> output_text
  B. tool:  reasoning -> function_call -> tool result -> final answer
            (asserts turn 2 returns a `message`, not another `function_call`,
            which would mean the prior chain-of-thought wasn't rehydrated).

Env vars: OPENAI_API_KEY, PHOENIX_COLLECTOR_ENDPOINT (default localhost:6006),
          OPENAI_MODEL (default `gpt-5`).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from openai import OpenAI
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
    CONTENT_TYPE_TEXT,
    CONTENT_TYPE_TOOL_USE,
    MESSAGE_CONTENT_ENCRYPTED_CONTENT,
    MESSAGE_CONTENT_ID,
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
    set_output_role,
    set_output_text,
    set_output_tool_use,
    set_tools,
    setup_tracing,
)

MODEL = os.environ.get("OPENAI_MODEL", "gpt-5")

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


def set_request_attributes(attrs: dict[str, Any], *, effort: str, summary: str) -> None:
    attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] = OpenInferenceSpanKindValues.LLM.value
    attrs[SpanAttributes.LLM_PROVIDER] = OpenInferenceLLMProviderValues.OPENAI.value
    attrs[SpanAttributes.LLM_SYSTEM] = OpenInferenceLLMSystemValues.OPENAI.value
    attrs[SpanAttributes.LLM_MODEL_NAME] = MODEL
    attrs[SpanAttributes.LLM_INVOCATION_PARAMETERS] = json.dumps(
        {
            "reasoning": {"effort": effort, "summary": summary},
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
    )


def record_assistant_turn(attrs: dict[str, Any], response: Any) -> None:
    """Walk `response.output[]` and write each item into `attrs` as flat
    output-message attributes."""
    set_output_role(attrs, 0, "assistant")
    content_index = 0
    for output_item in response.output:
        item_type = getattr(output_item, "type", None)
        if item_type == "reasoning":
            summary_text = "\n".join(
                getattr(summary_part, "text", "") for summary_part in (output_item.summary or [])
            )
            set_output_reasoning(
                attrs,
                0,
                content_index,
                text=summary_text or None,
                item_id=output_item.id,
                encrypted_content=getattr(output_item, "encrypted_content", None),
            )
            content_index += 1
        elif item_type == "message":
            for message_part in output_item.content or []:
                if getattr(message_part, "type", None) == "output_text":
                    set_output_text(attrs, 0, content_index, message_part.text)
                    content_index += 1
        elif item_type == "function_call":
            set_output_tool_use(
                attrs,
                0,
                content_index,
                tool_call_id=output_item.call_id,
                name=output_item.name,
                arguments_json=output_item.arguments,
            )
            content_index += 1


def rebuild_input_items(
    fetched_attrs: dict[str, Any], prior_user_text: str
) -> list[dict[str, Any]]:
    """Turn the fetched assistant turn back into a Responses-API
    `input=[...]` list."""
    items: list[dict[str, Any]] = [{"role": "user", "content": prior_user_text}]
    for content_block in read_output_contents(fetched_attrs, 0):
        block_type = content_block.get(MessageContentAttributes.MESSAGE_CONTENT_TYPE)
        if block_type == CONTENT_TYPE_REASONING:
            summary_parts: list[dict[str, str]] = []
            visible_text = content_block.get(MessageContentAttributes.MESSAGE_CONTENT_TEXT)
            if visible_text:
                summary_parts.append({"type": "summary_text", "text": visible_text})
            reasoning_item: dict[str, Any] = {
                "type": "reasoning",
                "id": content_block[MESSAGE_CONTENT_ID],
                "summary": summary_parts,
            }
            encrypted = content_block.get(MESSAGE_CONTENT_ENCRYPTED_CONTENT)
            if encrypted:
                reasoning_item["encrypted_content"] = encrypted
            items.append(reasoning_item)
        elif block_type == CONTENT_TYPE_TEXT:
            items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": content_block[MessageContentAttributes.MESSAGE_CONTENT_TEXT],
                        }
                    ],
                }
            )
        elif block_type == CONTENT_TYPE_TOOL_USE:
            items.append(
                {
                    "type": "function_call",
                    "call_id": content_block[ToolCallAttributes.TOOL_CALL_ID],
                    "name": content_block[ToolCallAttributes.TOOL_CALL_FUNCTION_NAME],
                    "arguments": content_block[ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON],
                }
            )
    return items


def replay_turn2(
    client: OpenAI,
    fetched_attrs: dict[str, Any],
    turn2_input_items: list[dict[str, Any]],
) -> Any:
    """Issue the follow-up call using ONLY values fetched out of Phoenix.

    The Python-side inputs allowed here are turn 2's new conversational
    state (the `input` list, which the demo built from fetched attributes
    plus the genuinely-new follow-up user / tool result). Everything else —
    model, tools, reasoning config, store, include — is decoded from the
    Phoenix-fetched attribute dict.
    """
    invocation_parameters = json.loads(fetched_attrs[SpanAttributes.LLM_INVOCATION_PARAMETERS])
    return client.responses.create(
        model=fetched_attrs[SpanAttributes.LLM_MODEL_NAME],
        input=turn2_input_items,
        tools=read_tools(fetched_attrs) or None,
        **invocation_parameters,
    )


def scenario_text(client: OpenAI, ctx: TracingCtx) -> bool:
    print("\n[openai] scenario A: text round-trip")
    user_text = "In one short sentence, what's special about the number 1729?"

    with ctx.tracer.start_as_current_span(
        "openai.responses.text.turn1.original", kind=SpanKind.CLIENT
    ) as turn1_span:
        turn1_attrs: dict[str, Any] = {}
        set_request_attributes(turn1_attrs, effort="medium", summary="auto")
        set_input_user_message(turn1_attrs, 0, user_text)

        turn1_response = client.responses.create(
            model=MODEL,
            input=user_text,
            reasoning={"effort": "medium", "summary": "auto"},
            store=False,
            include=["reasoning.encrypted_content"],
        )
        record_assistant_turn(turn1_attrs, turn1_response)
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
    turn2_input_items = rebuild_input_items(fetched_attrs, prior_user_text)
    turn2_input_items.append({"role": "user", "content": follow_up_text})

    with ctx.tracer.start_as_current_span(
        "openai.responses.text.turn2.roundtrip", kind=SpanKind.CLIENT
    ) as turn2_span:
        turn2_attrs: dict[str, Any] = {}
        set_request_attributes(turn2_attrs, effort="medium", summary="auto")
        set_input_user_message(turn2_attrs, 0, prior_user_text)
        set_input_assistant_echo(turn2_attrs, 1, captured_contents)
        set_input_user_message(turn2_attrs, 2, follow_up_text)

        turn2_response = replay_turn2(client, fetched_attrs, turn2_input_items)
        record_assistant_turn(turn2_attrs, turn2_response)
        turn2_span.set_attributes(turn2_attrs)

    turn2_returned_message = any(
        getattr(item, "type", None) == "message" for item in turn2_response.output
    )
    print(f"  fetched_blocks={len(captured_contents)} "
          f"token_lens={continuity_token_lengths(captured_contents)}")
    print(f"  turn2_ok={turn2_returned_message}  "
          f"turn2_text={(turn2_response.output_text or '')[:80]!r}")
    return turn2_returned_message


def scenario_tool(client: OpenAI, ctx: TracingCtx) -> bool:
    print("\n[openai] scenario B: reasoning → function_call round-trip")
    user_text = "What's the weather in Paris right now? Use the tool."

    with ctx.tracer.start_as_current_span(
        "openai.responses.tool.turn1.original", kind=SpanKind.CLIENT
    ) as turn1_span:
        turn1_attrs: dict[str, Any] = {}
        set_request_attributes(turn1_attrs, effort="medium", summary="auto")
        set_tools(turn1_attrs, [WEATHER_TOOL])
        set_input_user_message(turn1_attrs, 0, user_text)

        turn1_response = client.responses.create(
            model=MODEL,
            input=user_text,
            tools=[WEATHER_TOOL],
            reasoning={"effort": "medium", "summary": "auto"},
            store=False,
            include=["reasoning.encrypted_content"],
        )
        record_assistant_turn(turn1_attrs, turn1_response)
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
    tool_use_blocks = [
        block
        for block in captured_contents
        if block.get(MessageContentAttributes.MESSAGE_CONTENT_TYPE) == CONTENT_TYPE_TOOL_USE
    ]
    if not tool_use_blocks:
        print("  SKIP: model did not call the tool")
        return True
    tool_call_id = tool_use_blocks[0][ToolCallAttributes.TOOL_CALL_ID]

    turn2_input_items = rebuild_input_items(fetched_attrs, prior_user_text)
    tool_output_json = json.dumps({"temperature_c": 14.5})
    turn2_input_items.append(
        {
            "type": "function_call_output",
            "call_id": tool_call_id,
            "output": tool_output_json,
        }
    )

    with ctx.tracer.start_as_current_span(
        "openai.responses.tool.turn2.roundtrip", kind=SpanKind.CLIENT
    ) as turn2_span:
        turn2_attrs: dict[str, Any] = {}
        set_request_attributes(turn2_attrs, effort="medium", summary="auto")
        set_tools(turn2_attrs, [WEATHER_TOOL])
        set_input_user_message(turn2_attrs, 0, prior_user_text)
        set_input_assistant_echo(turn2_attrs, 1, captured_contents)
        set_input_tool_result(
            turn2_attrs, 2, tool_call_id=tool_call_id, content_text=tool_output_json
        )

        turn2_response = replay_turn2(client, fetched_attrs, turn2_input_items)
        record_assistant_turn(turn2_attrs, turn2_response)
        turn2_span.set_attributes(turn2_attrs)

    turn2_returned_message = any(
        getattr(item, "type", None) == "message" for item in turn2_response.output
    )
    print(f"  fetched_blocks={len(captured_contents)} "
          f"token_lens={continuity_token_lengths(captured_contents)}")
    print(f"  turn2_has_final_message={turn2_returned_message}")
    print(f"  turn2_text={(turn2_response.output_text or '')[:120]!r}")
    return turn2_returned_message


def main() -> int:
    ctx = setup_tracing("reasoning-roundtrip-openai")
    client = OpenAI()
    text_ok = scenario_text(client, ctx)
    tool_ok = scenario_tool(client, ctx)
    print(f"\n[openai] PASS={text_ok and tool_ok}  text={text_ok}  tool={tool_ok}")
    return 0 if (text_ok and tool_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
