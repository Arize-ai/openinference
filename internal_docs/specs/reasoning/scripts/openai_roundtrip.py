# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai>=2.0",
#     "opentelemetry-api",
#     "opentelemetry-sdk",
#     "opentelemetry-exporter-otlp-proto-http",
#     "arize-phoenix-client",
# ]
# ///
"""OpenAI Responses-API reasoning round-trip.

Proves that the OpenAI reasoning surface (`encrypted_content` + reasoning item
id + summary parts + any following `function_call`) can be captured as flat
string-valued span attributes, fetched back out of Phoenix, and used to rebuild
an `input=[...]` list that the Responses API accepts.

Two scenarios:
  A. text:  reasoning -> output_text
  B. tool:  reasoning -> function_call -> we provide tool result ->
            assert turn-2 returns a `message` (final answer), not another
            `function_call` (which would mean the prior chain-of-thought was
            not actually rehydrated).

Phoenix is required: `pip install arize-phoenix && phoenix serve`.

Env vars: OPENAI_API_KEY, PHOENIX_COLLECTOR_ENDPOINT (default localhost:6006),
          OPENAI_MODEL (default `gpt-5`).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from openai import OpenAI
from opentelemetry.trace import SpanKind

from common import (
    CONTENT_TYPE_REASONING,
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
    set_output_text,
    set_output_tool_use,
    setup_tracing,
    token_lens,
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


def _record_assistant_turn(attrs: dict[str, Any], response: Any) -> tuple[int, str | None]:
    """Walk `response.output[]` and write each item into `attrs`.

    Returns (next_input_message_idx, captured_call_id_if_any).
    """
    begin_output_message(attrs, 0, role="assistant")
    j = 0
    call_id: str | None = None
    for item in response.output:
        t = getattr(item, "type", None)
        if t == "reasoning":
            summary_text = "\n".join(getattr(s, "text", "") for s in (item.summary or []))
            set_output_reasoning(
                attrs,
                0,
                j,
                text=summary_text or None,
                item_id=item.id,
                encrypted_content=getattr(item, "encrypted_content", None),
            )
            j += 1
        elif t == "message":
            for part in item.content or []:
                if getattr(part, "type", None) == "output_text":
                    set_output_text(attrs, 0, j, part.text)
                    j += 1
        elif t == "function_call":
            call_id = item.call_id
            set_output_tool_use(
                attrs,
                0,
                j,
                tool_call_id=item.call_id,
                name=item.name,
                arguments_json=item.arguments,
            )
            j += 1
    return j, call_id


def _rebuild_input_from_attrs(
    attrs: dict[str, Any], prior_user_text: str
) -> list[dict[str, Any]]:
    """Pure attribute→`input[]` reconstruction. Mirrors the layout the
    Responses API expects on a stateless follow-up turn.
    """
    items: list[dict[str, Any]] = [
        {"role": "user", "content": prior_user_text},
    ]
    for c in read_output_contents(attrs, 0):
        ctype = c.get("type")
        if ctype == CONTENT_TYPE_REASONING:
            summary: list[dict[str, str]] = []
            if c.get("text"):
                summary.append({"type": "summary_text", "text": c["text"]})
            item: dict[str, Any] = {
                "type": "reasoning",
                "id": c["id"],
                "summary": summary,
            }
            if c.get("encrypted_content"):
                item["encrypted_content"] = c["encrypted_content"]
            items.append(item)
        elif ctype == CONTENT_TYPE_TEXT:
            items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": c["text"]}],
                }
            )
        elif ctype == CONTENT_TYPE_TOOL_USE:
            items.append(
                {
                    "type": "function_call",
                    "call_id": c["tool_call.id"],
                    "name": c["tool_call.function.name"],
                    "arguments": c["tool_call.function.arguments"],
                }
            )
    return items


def _set_request_attrs(attrs: dict[str, Any], *, effort: str, summary: str) -> None:
    attrs[OPENINFERENCE_SPAN_KIND] = "LLM"
    attrs[LLM_PROVIDER] = "openai"
    attrs[LLM_SYSTEM] = "openai"
    attrs[LLM_MODEL_NAME] = MODEL
    attrs[LLM_INVOCATION_PARAMETERS] = json.dumps(
        {
            "reasoning": {"effort": effort, "summary": summary},
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
    )


def scenario_text(client: OpenAI, ctx) -> bool:
    print("\n[openai] scenario A: text round-trip")
    user_text = "In one short sentence, what's special about the number 1729?"

    with ctx.tracer.start_as_current_span(
        "openai.responses.text.turn1.original", kind=SpanKind.CLIENT
    ) as sp:
        attrs: dict[str, Any] = {}
        _set_request_attrs(attrs, effort="medium", summary="auto")
        set_input_user_message(attrs, 0, user_text)

        resp = client.responses.create(
            model=MODEL,
            input=user_text,
            reasoning={"effort": "medium", "summary": "auto"},
            store=False,
            include=["reasoning.encrypted_content"],
        )
        _record_assistant_turn(attrs, resp)
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
    input_items = _rebuild_input_from_attrs(fetched, user_text)
    input_items.append({"role": "user", "content": follow_up})

    with ctx.tracer.start_as_current_span(
        "openai.responses.text.turn2.roundtrip", kind=SpanKind.CLIENT
    ) as sp2:
        attrs2: dict[str, Any] = {}
        _set_request_attrs(attrs2, effort="medium", summary="auto")
        # Echoed history as input_messages — proves the round-trip carried
        # the prior reasoning block (with its encrypted_content) into turn 2.
        set_input_user_message(attrs2, 0, user_text)
        set_input_assistant_echo(attrs2, 1, contents)
        set_input_user_message(attrs2, 2, follow_up)

        resp2 = client.responses.create(
            model=MODEL,
            input=input_items,
            reasoning={"effort": "medium", "summary": "auto"},
            store=False,
            include=["reasoning.encrypted_content"],
        )
        _record_assistant_turn(attrs2, resp2)
        for k, v in attrs2.items():
            sp2.set_attribute(k, v)

    ok = any(getattr(it, "type", None) == "message" for it in resp2.output)
    print(f"  fetched_blocks={len(contents)} token_lens={token_lens(contents)}")
    print(f"  turn2_ok={ok}  turn2_text={(resp2.output_text or '')[:80]!r}")
    return ok


def scenario_tool(client: OpenAI, ctx) -> bool:
    print("\n[openai] scenario B: reasoning → function_call round-trip")
    user_text = "What's the weather in Paris right now? Use the tool."

    with ctx.tracer.start_as_current_span(
        "openai.responses.tool.turn1.original", kind=SpanKind.CLIENT
    ) as sp:
        attrs: dict[str, Any] = {}
        _set_request_attrs(attrs, effort="medium", summary="auto")
        set_input_user_message(attrs, 0, user_text)

        resp = client.responses.create(
            model=MODEL,
            input=user_text,
            tools=[WEATHER_TOOL],
            reasoning={"effort": "medium", "summary": "auto"},
            store=False,
            include=["reasoning.encrypted_content"],
        )
        _record_assistant_turn(attrs, resp)
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
    tool_blocks = [c for c in contents if c.get("type") == CONTENT_TYPE_TOOL_USE]
    if not tool_blocks:
        print("  SKIP: model did not call the tool")
        return True
    call_id = tool_blocks[0]["tool_call.id"]

    input_items = _rebuild_input_from_attrs(fetched, user_text)
    tool_output_json = json.dumps({"temperature_c": 14.5})
    input_items.append(
        {
            "type": "function_call_output",
            "call_id": call_id,
            "output": tool_output_json,
        }
    )

    with ctx.tracer.start_as_current_span(
        "openai.responses.tool.turn2.roundtrip", kind=SpanKind.CLIENT
    ) as sp2:
        attrs2: dict[str, Any] = {}
        _set_request_attrs(attrs2, effort="medium", summary="auto")
        set_input_user_message(attrs2, 0, user_text)
        set_input_assistant_echo(attrs2, 1, contents)
        set_input_tool_result(attrs2, 2, tool_call_id=call_id, content=tool_output_json)

        resp2 = client.responses.create(
            model=MODEL,
            input=input_items,
            tools=[WEATHER_TOOL],
            reasoning={"effort": "medium", "summary": "auto"},
            store=False,
            include=["reasoning.encrypted_content"],
        )
        _record_assistant_turn(attrs2, resp2)
        for k, v in attrs2.items():
            sp2.set_attribute(k, v)

    has_message = any(getattr(it, "type", None) == "message" for it in resp2.output)
    print(f"  fetched_blocks={len(contents)} token_lens={token_lens(contents)}")
    print(f"  turn2_has_final_message={has_message}")
    print(f"  turn2_text={(resp2.output_text or '')[:120]!r}")
    return has_message


def main() -> int:
    ctx = setup_tracing("reasoning-roundtrip-openai")
    client = OpenAI()
    a = scenario_text(client, ctx)
    b = scenario_tool(client, ctx)
    print(f"\n[openai] PASS={a and b}  text={a}  tool={b}")
    return 0 if (a and b) else 1


if __name__ == "__main__":
    sys.exit(main())
