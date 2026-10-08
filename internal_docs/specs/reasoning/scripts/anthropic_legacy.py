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
"""Anthropic legacy extended-thinking round-trip.

This preserves the pre-Opus-4.7 flow using ``thinking.type = "enabled"`` with
an explicit token budget, while sharing the same real-instrumentor augmentation
harness as ``anthropic_roundtrip.py``.
"""

from __future__ import annotations

import json
import os
import sys

import anthropic
from openinference.instrumentation.anthropic import AnthropicInstrumentor
from openinference.semconv.trace import ToolCallAttributes

from anthropic_roundtrip import (
    WEATHER_TOOL,
    build_messages_create_request,
    call_and_export_augmented,
    call_expect_bad_request_and_export,
    rebuild_assistant_content_blocks,
    read_output_tool_calls,
    user_text_message,
)
from common import (
    FACTORIZE_FOLLOW_UP_PROMPT,
    FACTORIZE_USER_PROMPT,
    InstrumentedTracingCtx,
    TOOL_FOLLOW_UP_PROMPT,
    TOOL_RESULT_PAYLOAD,
    TOOL_USER_PROMPT,
    find_reasoning_block,
    find_tool_use_block,
    print_capture_summary,
    read_input_message_text,
    read_output_contents,
    setup_instrumented_tracing,
)

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-6")
BUDGET = int(os.environ.get("ANTHROPIC_BUDGET_TOKENS", "4000"))
MAX_TOKENS = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "8000"))


def legacy_thinking_config() -> dict[str, int | str]:
    return {"type": "enabled", "budget_tokens": BUDGET}


def base_request(
    *,
    messages: list[dict[str, object]],
    tools: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    request: dict[str, object] = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "thinking": legacy_thinking_config(),
        "messages": messages,
    }
    if tools:
        request["tools"] = tools
    return request


def scenario_text(client: anthropic.Anthropic, ctx: InstrumentedTracingCtx) -> bool:
    print("\n[anthropic-legacy] scenario A: extended thinking -> text")
    turn1_request = base_request(messages=[user_text_message(FACTORIZE_USER_PROMPT)])
    _, fetched_attrs = call_and_export_augmented(
        ctx,
        client,
        turn1_request,
        root_name="anthropic legacy text turn1",
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
    turn2_message, _ = call_and_export_augmented(
        ctx,
        client,
        turn2_request,
        root_name="anthropic legacy text turn2",
    )
    final_text = next(
        (block.text for block in turn2_message.content if block.type == "text"), ""
    )
    print_capture_summary(captured_contents)
    print(f"  turn2_text={final_text[:120]!r}")
    return bool(final_text)


def scenario_tool(
    client: anthropic.Anthropic, ctx: InstrumentedTracingCtx
) -> tuple[bool, bool]:
    print(
        "\n[anthropic-legacy] "
        "scenario B: extended thinking -> tool_use round-trip (+ negative)"
    )
    turn1_request = base_request(
        messages=[user_text_message(TOOL_USER_PROMPT)],
        tools=[WEATHER_TOOL],
    )
    _, fetched_attrs = call_and_export_augmented(
        ctx,
        client,
        turn1_request,
        root_name="anthropic legacy tool turn1",
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
    turn2_message, _ = call_and_export_augmented(
        ctx,
        client,
        turn2_request,
        root_name="anthropic legacy tool turn2",
    )
    final_text = next(
        (block.text for block in turn2_message.content if block.type == "text"), ""
    )
    print_capture_summary(captured_contents)
    print(f"  positive_turn2_text={final_text[:120]!r}")
    positive_ok = bool(final_text)

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
    )
    error = call_expect_bad_request_and_export(
        ctx,
        client,
        negative_request,
        root_name="anthropic legacy tool negative",
    )
    negative_rejected = error is not None
    if negative_rejected:
        print(f"  negative correctly rejected: {str(error)[:120]!r}")
    else:
        print("  negative FAILED: stripped-signature request was accepted")
    return positive_ok, negative_rejected


def main() -> int:
    ctx = setup_instrumented_tracing("anthropic-legacy-reasoning-roundtrip")
    instrumentor = AnthropicInstrumentor()
    instrumentor.instrument(tracer_provider=ctx.provider)
    try:
        client = anthropic.Anthropic()
        text_ok = scenario_text(client, ctx)
        tool_positive_ok, tool_negative_ok = scenario_tool(client, ctx)
        overall_ok = text_ok and tool_positive_ok and tool_negative_ok
        print(
            f"\n[anthropic-legacy] text={text_ok}  "
            f"tool_positive={tool_positive_ok}  "
            f"tool_negative={tool_negative_ok}  PASS={overall_ok}"
        )
        return 0 if overall_ok else 1
    finally:
        instrumentor.uninstrument()
        ctx.shutdown()


if __name__ == "__main__":
    sys.exit(main())
