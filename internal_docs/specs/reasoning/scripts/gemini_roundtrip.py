# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "google-genai>=1.0",
#     "opentelemetry-api",
#     "opentelemetry-sdk",
#     "opentelemetry-exporter-otlp-proto-http",
#     "arize-phoenix-client",
# ]
# ///
"""Google Gemini thinking-surface round-trip.

Captures `parts[]` — thought-summary parts (`thought: true`) and data parts
(text / functionCall) along with their `thoughtSignature` — as flat span
attributes, fetches them back out of Phoenix, and rebuilds the `contents=[...]`
list for the next turn.

`thought_signature` is `bytes` on the SDK type but base64-encoded on the wire;
we base64 it for storage (OTel attribute values must be primitive types) and
decode back to bytes when echoing.

Two scenarios:
  A. text:  thought summary + thought-signed answer part → follow-up
  B. tool:  thought summary + signed functionCall → functionResponse →
            final answer
  Optional negative (Gemini 3 only): drop the signature on the functionCall
  and expect HTTP 400 ("Function call ... is missing a thought_signature.").

Phoenix is required.

Env vars: GOOGLE_API_KEY, PHOENIX_COLLECTOR_ENDPOINT, GEMINI_MODEL
(default `gemini-2.5-pro`).
"""

from __future__ import annotations

import base64
import json
import os
import sys
from typing import Any

from google import genai
from google.genai import types as gtypes
from opentelemetry.trace import SpanKind

from common import (
    CONTENT_TYPE_REASONING_SUMMARY,
    CONTENT_TYPE_TEXT,
    CONTENT_TYPE_TOOL_USE,
    LLM_INVOCATION_PARAMETERS,
    LLM_MODEL_NAME,
    LLM_PROVIDER,
    LLM_REASONING_BUDGET_TOKENS,
    LLM_REASONING_INCLUDE_SUMMARY,
    LLM_SYSTEM,
    OPENINFERENCE_SPAN_KIND,
    begin_output_message,
    fetch_span_attributes,
    flush,
    read_output_contents,
    set_input_assistant_echo,
    set_input_tool_result,
    set_input_user_message,
    set_output_reasoning_summary,
    set_output_text,
    set_output_tool_use,
    setup_tracing,
    token_lens,
)

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
BUDGET = int(os.environ.get("GEMINI_THINKING_BUDGET", "4000"))

WEATHER_TOOL = gtypes.Tool(
    function_declarations=[
        gtypes.FunctionDeclaration(
            name="get_weather",
            description="Current temperature in Celsius.",
            parameters_json_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )
    ]
)


def _b64(b: bytes | None) -> str | None:
    if b is None:
        return None
    return base64.b64encode(b).decode("ascii")


def _unb64(s: str | None) -> bytes | None:
    if not s:
        return None
    return base64.b64decode(s)


def _record_assistant_turn(attrs: dict[str, Any], response: Any) -> None:
    begin_output_message(attrs, 0, role="model")
    candidate = response.candidates[0]
    j = 0
    for part in candidate.content.parts or []:
        sig_b64 = _b64(part.thought_signature)
        if part.thought:
            # Thought-summary parts never carry a signature — they're a sibling
            # surface to the data part that does.
            set_output_reasoning_summary(attrs, 0, j, text=part.text or "")
        elif part.function_call is not None:
            set_output_tool_use(
                attrs,
                0,
                j,
                tool_call_id=part.function_call.id or f"call_{j}",
                name=part.function_call.name or "",
                arguments_json=json.dumps(dict(part.function_call.args or {})),
                thought_signature=sig_b64,
            )
        elif part.text is not None:
            set_output_text(attrs, 0, j, part.text, thought_signature=sig_b64)
        else:
            continue
        j += 1


def _rebuild_model_content(
    attrs: dict[str, Any], *, strip_signature: bool = False
) -> gtypes.Content:
    parts: list[gtypes.Part] = []
    for c in read_output_contents(attrs, 0):
        t = c.get("type")
        sig = None if strip_signature else _unb64(c.get("thought_signature"))
        if t == CONTENT_TYPE_REASONING_SUMMARY:
            parts.append(gtypes.Part(text=c.get("text", ""), thought=True))
        elif t == CONTENT_TYPE_TEXT:
            parts.append(gtypes.Part(text=c["text"], thought_signature=sig))
        elif t == CONTENT_TYPE_TOOL_USE:
            fc = gtypes.FunctionCall(
                id=c["tool_call.id"],
                name=c["tool_call.function.name"],
                args=json.loads(c["tool_call.function.arguments"]),
            )
            parts.append(gtypes.Part(function_call=fc, thought_signature=sig))
    return gtypes.Content(role="model", parts=parts)


def _set_request_attrs(attrs: dict[str, Any]) -> None:
    attrs[OPENINFERENCE_SPAN_KIND] = "LLM"
    attrs[LLM_PROVIDER] = "google"
    attrs[LLM_SYSTEM] = "gemini"
    attrs[LLM_MODEL_NAME] = MODEL
    attrs[LLM_REASONING_BUDGET_TOKENS] = BUDGET
    attrs[LLM_REASONING_INCLUDE_SUMMARY] = True
    attrs[LLM_INVOCATION_PARAMETERS] = json.dumps(
        {
            "generationConfig": {
                "thinkingConfig": {"thinkingBudget": BUDGET, "includeThoughts": True},
            }
        }
    )


def _config(*, tools: list[gtypes.Tool] | None = None) -> gtypes.GenerateContentConfig:
    return gtypes.GenerateContentConfig(
        thinking_config=gtypes.ThinkingConfig(
            thinking_budget=BUDGET,
            include_thoughts=True,
        ),
        tools=tools,
    )


def scenario_text(client: genai.Client, ctx) -> bool:
    print("\n[gemini] scenario A: text round-trip")
    user_text = "In one short sentence, what's special about the number 1729?"

    with ctx.tracer.start_as_current_span(
        "gemini.generate_content.text.turn1.original", kind=SpanKind.CLIENT
    ) as sp:
        attrs: dict[str, Any] = {}
        _set_request_attrs(attrs)
        set_input_user_message(attrs, 0, user_text)

        resp = client.models.generate_content(
            model=MODEL,
            contents=user_text,
            config=_config(),
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

    model_content = _rebuild_model_content(fetched)
    with ctx.tracer.start_as_current_span(
        "gemini.generate_content.text.turn2.roundtrip", kind=SpanKind.CLIENT
    ) as sp2:
        attrs2: dict[str, Any] = {}
        _set_request_attrs(attrs2)
        set_input_user_message(attrs2, 0, user_text)
        set_input_assistant_echo(attrs2, 1, contents, role="model")
        set_input_user_message(attrs2, 2, follow_up)

        resp2 = client.models.generate_content(
            model=MODEL,
            contents=[
                gtypes.Content(role="user", parts=[gtypes.Part(text=user_text)]),
                model_content,
                gtypes.Content(role="user", parts=[gtypes.Part(text=follow_up)]),
            ],
            config=_config(),
        )
        _record_assistant_turn(attrs2, resp2)
        for k, v in attrs2.items():
            sp2.set_attribute(k, v)

    text2 = (resp2.text or "")[:120]
    print(f"  fetched_blocks={len(contents)} token_lens={token_lens(contents)}")
    print(f"  turn2_text={text2!r}")
    return bool(text2)


def scenario_tool(client: genai.Client, ctx) -> tuple[bool, bool | None]:
    print("\n[gemini] scenario B: thinking → functionCall round-trip (+ optional negative)")
    user_text = "What's the weather in Paris? Use the tool."

    with ctx.tracer.start_as_current_span(
        "gemini.generate_content.tool.turn1.original", kind=SpanKind.CLIENT
    ) as sp:
        attrs: dict[str, Any] = {}
        _set_request_attrs(attrs)
        set_input_user_message(attrs, 0, user_text)

        resp = client.models.generate_content(
            model=MODEL,
            contents=user_text,
            config=_config(tools=[WEATHER_TOOL]),
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
    tool_block = next((c for c in contents if c.get("type") == CONTENT_TYPE_TOOL_USE), None)
    if tool_block is None:
        print("  SKIP: model did not call the tool")
        return True, None

    tool_response_payload = {"temperature_c": 14.5}
    tool_response = gtypes.Content(
        role="user",
        parts=[
            gtypes.Part(
                function_response=gtypes.FunctionResponse(
                    id=tool_block["tool_call.id"],
                    name=tool_block["tool_call.function.name"],
                    response=tool_response_payload,
                )
            )
        ],
    )

    # Positive echo.
    model_content = _rebuild_model_content(fetched)
    with ctx.tracer.start_as_current_span(
        "gemini.generate_content.tool.turn2.roundtrip", kind=SpanKind.CLIENT
    ) as sp2:
        attrs2: dict[str, Any] = {}
        _set_request_attrs(attrs2)
        set_input_user_message(attrs2, 0, user_text)
        set_input_assistant_echo(attrs2, 1, contents, role="model")
        set_input_tool_result(
            attrs2,
            2,
            tool_call_id=tool_block["tool_call.id"],
            content=json.dumps(tool_response_payload),
        )

        resp2 = client.models.generate_content(
            model=MODEL,
            contents=[
                gtypes.Content(role="user", parts=[gtypes.Part(text=user_text)]),
                model_content,
                tool_response,
            ],
            config=_config(tools=[WEATHER_TOOL]),
        )
        _record_assistant_turn(attrs2, resp2)
        for k, v in attrs2.items():
            sp2.set_attribute(k, v)

    final_text = (resp2.text or "")[:120]
    print(f"  fetched_blocks={len(contents)} token_lens={token_lens(contents)}")
    print(f"  positive_turn2_text={final_text!r}")
    positive_ok = bool(final_text)

    # Negative — only meaningful on Gemini 3 (HTTP 400). On 2.5, silent
    # quality loss; skip with a warning.
    if not MODEL.startswith("gemini-3"):
        print("  negative skipped (Gemini 2.5: stripping signature is a silent quality loss, not a 400)")
        return positive_ok, None

    stripped = _rebuild_model_content(fetched, strip_signature=True)
    negative_rejected = False
    with ctx.tracer.start_as_current_span(
        "gemini.generate_content.tool.turn2.roundtrip.negative", kind=SpanKind.CLIENT
    ) as sp_neg:
        try:
            client.models.generate_content(
                model=MODEL,
                contents=[
                    gtypes.Content(role="user", parts=[gtypes.Part(text=user_text)]),
                    stripped,
                    tool_response,
                ],
                config=_config(tools=[WEATHER_TOOL]),
            )
        except Exception as e:  # google.genai raises various error types
            negative_rejected = True
            sp_neg.set_attribute("error.expected", True)
            sp_neg.set_attribute("error.message", str(e)[:200])
            print(f"  negative correctly rejected: {str(e)[:120]!r}")
    if not negative_rejected:
        print("  negative FAILED: stripped-signature request was accepted on Gemini 3")
    return positive_ok, negative_rejected


def main() -> int:
    ctx = setup_tracing("reasoning-roundtrip-gemini")
    client = genai.Client()
    a = scenario_text(client, ctx)
    b_pos, b_neg = scenario_tool(client, ctx)
    overall = a and b_pos and (b_neg is not False)
    print(f"\n[gemini] text={a}  tool_positive={b_pos}  tool_negative={b_neg}  PASS={overall}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
