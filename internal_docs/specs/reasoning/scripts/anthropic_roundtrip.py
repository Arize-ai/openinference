# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "anthropic>=0.40",
#     "opentelemetry-api",
#     "opentelemetry-sdk",
#     "opentelemetry-exporter-otlp-proto-http",
#     "arize-phoenix-client",
# ]
# ///
"""Anthropic Messages-API extended-thinking round-trip.

Captures `thinking` / `redacted_thinking` / `tool_use` blocks (with their
`signature` / `data`) as flat span attributes, fetches them back out of
Phoenix, and rebuilds `messages=[user, assistant{thinking,...}, ...]` for the
next turn.

Two scenarios:
  A. text:  thinking -> text                      (signature echoed, request accepted)
  B. tool:  thinking -> tool_use -> tool_result   (signature is load-bearing; if
            dropped the API returns HTTP 400)
  Negative assertion: scenario B with signature stripped → expect API error.

Phoenix is required.

Env vars: ANTHROPIC_API_KEY, PHOENIX_COLLECTOR_ENDPOINT, ANTHROPIC_MODEL
(default `claude-opus-4-5`).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import anthropic
from anthropic import BadRequestError
from opentelemetry.trace import SpanKind

from common import (
    CONTENT_TYPE_REASONING,
    CONTENT_TYPE_REDACTED_REASONING,
    CONTENT_TYPE_TEXT,
    CONTENT_TYPE_TOOL_USE,
    LLM_INVOCATION_PARAMETERS,
    LLM_MODEL_NAME,
    LLM_PROVIDER,
    LLM_SYSTEM,
    OPENINFERENCE_SPAN_KIND,
    begin_output_message,
    fetch_span_attributes,
    flush,
    read_output_contents,
    set_input_assistant_echo,
    set_input_tool_result,
    set_input_user_message,
    set_output_reasoning,
    set_output_redacted_reasoning,
    set_output_text,
    set_output_tool_use,
    setup_tracing,
    token_lens,
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


def _record_assistant_turn(attrs: dict[str, Any], message: Any) -> None:
    begin_output_message(attrs, 0, role="assistant")
    for j, block in enumerate(message.content):
        t = block.type
        if t == "thinking":
            set_output_reasoning(attrs, 0, j, text=block.thinking, signature=block.signature)
        elif t == "redacted_thinking":
            set_output_redacted_reasoning(attrs, 0, j, data=block.data)
        elif t == "text":
            set_output_text(attrs, 0, j, block.text)
        elif t == "tool_use":
            set_output_tool_use(
                attrs,
                0,
                j,
                tool_call_id=block.id,
                name=block.name,
                arguments_json=json.dumps(block.input),
            )


def _rebuild_assistant_content(
    attrs: dict[str, Any], *, strip_signature: bool = False
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for c in read_output_contents(attrs, 0):
        t = c.get("type")
        if t == CONTENT_TYPE_REASONING:
            block: dict[str, Any] = {"type": "thinking", "thinking": c.get("text", "")}
            if not strip_signature and c.get("signature"):
                block["signature"] = c["signature"]
            out.append(block)
        elif t == CONTENT_TYPE_REDACTED_REASONING:
            out.append({"type": "redacted_thinking", "data": c["redacted_data"]})
        elif t == CONTENT_TYPE_TEXT:
            out.append({"type": "text", "text": c["text"]})
        elif t == CONTENT_TYPE_TOOL_USE:
            out.append(
                {
                    "type": "tool_use",
                    "id": c["tool_call.id"],
                    "name": c["tool_call.function.name"],
                    "input": json.loads(c["tool_call.function.arguments"]),
                }
            )
    return out


def _set_request_attrs(attrs: dict[str, Any]) -> None:
    attrs[OPENINFERENCE_SPAN_KIND] = "LLM"
    attrs[LLM_PROVIDER] = "anthropic"
    attrs[LLM_SYSTEM] = "anthropic"
    attrs[LLM_MODEL_NAME] = MODEL
    attrs[LLM_INVOCATION_PARAMETERS] = json.dumps(
        {
            "thinking": {"type": "enabled", "budget_tokens": BUDGET},
            "max_tokens": MAX_TOKENS,
        }
    )


def scenario_text(client: anthropic.Anthropic, ctx) -> bool:
    print("\n[anthropic] scenario A: thinking → text round-trip")
    user_text = "In one short sentence, what's special about the number 1729?"

    with ctx.tracer.start_as_current_span(
        "anthropic.messages.text.turn1.original", kind=SpanKind.CLIENT
    ) as sp:
        attrs: dict[str, Any] = {}
        _set_request_attrs(attrs)
        set_input_user_message(attrs, 0, user_text)

        msg = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            thinking={"type": "enabled", "budget_tokens": BUDGET},
            messages=[{"role": "user", "content": user_text}],
        )
        _record_assistant_turn(attrs, msg)
        for k, v in attrs.items():
            sp.set_attribute(k, v)
        ctx_ids = sp.get_span_context()

    flush(ctx)
    fetched = fetch_span_attributes(
        ctx,
        trace_id_hex=f"{ctx_ids.trace_id:032x}",
        span_id_hex=f"{ctx_ids.span_id:016x}",
    )
    contents = read_output_contents(fetched, 0)
    follow_up = "Now in one sentence: who proved that property?"

    with ctx.tracer.start_as_current_span(
        "anthropic.messages.text.turn2.roundtrip", kind=SpanKind.CLIENT
    ) as sp2:
        attrs2: dict[str, Any] = {}
        _set_request_attrs(attrs2)
        set_input_user_message(attrs2, 0, user_text)
        set_input_assistant_echo(attrs2, 1, contents)
        set_input_user_message(attrs2, 2, follow_up)

        msg2 = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            thinking={"type": "enabled", "budget_tokens": BUDGET},
            messages=[
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": _rebuild_assistant_content(fetched)},
                {"role": "user", "content": follow_up},
            ],
        )
        _record_assistant_turn(attrs2, msg2)
        for k, v in attrs2.items():
            sp2.set_attribute(k, v)

    final_text = next((b.text for b in msg2.content if b.type == "text"), "")
    print(f"  fetched_blocks={len(contents)} token_lens={token_lens(contents)}")
    print(f"  turn2_text={final_text[:120]!r}")
    return bool(final_text)


def scenario_tool(client: anthropic.Anthropic, ctx) -> tuple[bool, bool]:
    print("\n[anthropic] scenario B: thinking → tool_use round-trip (+ negative)")
    user_text = "What's the weather in Paris? Use the tool."

    with ctx.tracer.start_as_current_span(
        "anthropic.messages.tool.turn1.original", kind=SpanKind.CLIENT
    ) as sp:
        attrs: dict[str, Any] = {}
        _set_request_attrs(attrs)
        set_input_user_message(attrs, 0, user_text)

        msg = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            thinking={"type": "enabled", "budget_tokens": BUDGET},
            tools=[WEATHER_TOOL],
            messages=[{"role": "user", "content": user_text}],
        )
        _record_assistant_turn(attrs, msg)
        for k, v in attrs.items():
            sp.set_attribute(k, v)
        ctx_ids = sp.get_span_context()

    flush(ctx)
    fetched = fetch_span_attributes(
        ctx,
        trace_id_hex=f"{ctx_ids.trace_id:032x}",
        span_id_hex=f"{ctx_ids.span_id:016x}",
    )
    contents = read_output_contents(fetched, 0)
    tool_block = next((c for c in contents if c.get("type") == CONTENT_TYPE_TOOL_USE), None)
    if tool_block is None:
        print("  SKIP: model did not call the tool")
        return True, True

    tool_use_id = tool_block["tool_call.id"]
    tool_result_content = json.dumps({"temperature_c": 14.5})
    tool_result_msg = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": tool_result_content,
            }
        ],
    }

    # Positive: full echo.
    with ctx.tracer.start_as_current_span(
        "anthropic.messages.tool.turn2.roundtrip", kind=SpanKind.CLIENT
    ) as sp2:
        attrs2: dict[str, Any] = {}
        _set_request_attrs(attrs2)
        set_input_user_message(attrs2, 0, user_text)
        set_input_assistant_echo(attrs2, 1, contents)
        set_input_tool_result(
            attrs2, 2, tool_call_id=tool_use_id, content=tool_result_content
        )

        msg2 = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            thinking={"type": "enabled", "budget_tokens": BUDGET},
            tools=[WEATHER_TOOL],
            messages=[
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": _rebuild_assistant_content(fetched)},
                tool_result_msg,
            ],
        )
        _record_assistant_turn(attrs2, msg2)
        for k, v in attrs2.items():
            sp2.set_attribute(k, v)

    final_text = next((b.text for b in msg2.content if b.type == "text"), "")
    print(f"  fetched_blocks={len(contents)} token_lens={token_lens(contents)}")
    print(f"  positive_turn2_text={final_text[:120]!r}")
    positive_ok = bool(final_text)

    # Negative: drop the signature and expect HTTP 400.
    negative_rejected = False
    with ctx.tracer.start_as_current_span(
        "anthropic.messages.tool.turn2.roundtrip.negative", kind=SpanKind.CLIENT
    ) as sp_neg:
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
                        "content": _rebuild_assistant_content(fetched, strip_signature=True),
                    },
                    tool_result_msg,
                ],
            )
        except BadRequestError as e:
            negative_rejected = True
            sp_neg.set_attribute("error.expected", True)
            sp_neg.set_attribute("error.message", str(e)[:200])
            print(f"  negative correctly rejected: {str(e)[:120]!r}")
    if not negative_rejected:
        print("  negative FAILED: stripped-signature request was accepted")

    return positive_ok, negative_rejected


def main() -> int:
    ctx = setup_tracing("reasoning-roundtrip-anthropic")
    client = anthropic.Anthropic()
    a = scenario_text(client, ctx)
    b_pos, b_neg = scenario_tool(client, ctx)
    print(f"\n[anthropic] text={a}  tool_positive={b_pos}  tool_negative={b_neg}")
    return 0 if (a and b_pos and b_neg) else 1


if __name__ == "__main__":
    sys.exit(main())
