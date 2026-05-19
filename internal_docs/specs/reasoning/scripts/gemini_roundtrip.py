# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "google-genai>=1.0",
#     "opentelemetry-api",
#     "opentelemetry-sdk",
#     "opentelemetry-exporter-otlp-proto-http",
#     "openinference-semantic-conventions",
#     "arize-phoenix-client",
# ]
# ///
"""Google Gemini thinking-surface round-trip.

Captures `parts[]` — thought-summary parts (`thought: true`) and data parts
(text / functionCall) along with their `thoughtSignature` — as flat span
attributes, fetches them back out of Phoenix, and rebuilds `contents=[...]`
for the next turn.

`thought_signature` is `bytes` on the SDK type but base64-encoded on the
wire. We base64 it for storage (OTel attribute values must be primitive)
and decode back to bytes when echoing.

Two scenarios:
  A. text:  thought summary + (optionally thought-signed) answer
  B. tool:  thought summary + signed functionCall → functionResponse → final
  Optional negative (Gemini 3 only): drop the signature on functionCall and
  expect HTTP 400 ("Function call ... is missing a thought_signature.").

Env vars: GOOGLE_API_KEY, PHOENIX_COLLECTOR_ENDPOINT,
          GEMINI_MODEL (default `gemini-2.5-pro`).
"""

from __future__ import annotations

import base64
import json
import os
import sys
from typing import Any

from google import genai
from google.genai import types as gtypes
from openinference.semconv.trace import (
    MessageContentAttributes,
    OpenInferenceLLMProviderValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)
from opentelemetry.trace import SpanKind

from common import (
    CONTENT_TYPE_REASONING_SUMMARY,
    CONTENT_TYPE_TEXT,
    CONTENT_TYPE_TOOL_USE,
    MESSAGE_CONTENT_THOUGHT_SIGNATURE,
    TracingCtx,
    continuity_token_lengths,
    fetch_span_attributes,
    read_input_message_content,
    read_output_contents,
    read_tools,
    set_input_assistant_echo,
    set_input_tool_result,
    set_input_user_message,
    set_output_reasoning_summary,
    set_output_role,
    set_output_text,
    set_output_tool_use,
    set_tools,
    setup_tracing,
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
WEATHER_TOOL_SCHEMA = WEATHER_TOOL.model_dump(exclude_none=True, by_alias=True)


def base64_encode(raw_bytes: bytes | None) -> str | None:
    if raw_bytes is None:
        return None
    return base64.b64encode(raw_bytes).decode("ascii")


def base64_decode(encoded: str | None) -> bytes | None:
    if not encoded:
        return None
    return base64.b64decode(encoded)


def set_request_attributes(attrs: dict[str, Any]) -> None:
    attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] = OpenInferenceSpanKindValues.LLM.value
    attrs[SpanAttributes.LLM_PROVIDER] = OpenInferenceLLMProviderValues.GOOGLE.value
    # OpenInferenceLLMSystemValues has no gemini member yet; use vertexai
    # (the closest existing value) on llm.system so it parses as a known enum.
    attrs[SpanAttributes.LLM_SYSTEM] = "gemini"
    attrs[SpanAttributes.LLM_MODEL_NAME] = MODEL
    attrs[SpanAttributes.LLM_INVOCATION_PARAMETERS] = json.dumps(
        {
            "generationConfig": {
                "thinkingConfig": {"thinkingBudget": BUDGET, "includeThoughts": True},
            }
        }
    )


def thinking_config() -> gtypes.GenerateContentConfig:
    return gtypes.GenerateContentConfig(
        thinking_config=gtypes.ThinkingConfig(
            thinking_budget=BUDGET,
            include_thoughts=True,
        ),
        tools=[WEATHER_TOOL],
    )


def text_only_config() -> gtypes.GenerateContentConfig:
    return gtypes.GenerateContentConfig(
        thinking_config=gtypes.ThinkingConfig(
            thinking_budget=BUDGET,
            include_thoughts=True,
        ),
    )


def config_from_fetched(fetched_attrs: dict[str, Any]) -> gtypes.GenerateContentConfig:
    """Build the request config purely from Phoenix-fetched attributes."""
    invocation_parameters = json.loads(fetched_attrs[SpanAttributes.LLM_INVOCATION_PARAMETERS])
    thinking = invocation_parameters["generationConfig"]["thinkingConfig"]
    fetched_tools = read_tools(fetched_attrs)
    return gtypes.GenerateContentConfig(
        thinking_config=gtypes.ThinkingConfig(
            thinking_budget=thinking.get("thinkingBudget"),
            include_thoughts=thinking.get("includeThoughts"),
        ),
        tools=fetched_tools or None,
    )


def replay_turn2(
    client: genai.Client,
    fetched_attrs: dict[str, Any],
    contents: list[gtypes.Content],
) -> Any:
    return client.models.generate_content(
        model=fetched_attrs[SpanAttributes.LLM_MODEL_NAME],
        contents=contents,
        config=config_from_fetched(fetched_attrs),
    )


def record_assistant_turn(attrs: dict[str, Any], response: Any) -> None:
    set_output_role(attrs, 0, "model")
    candidate = response.candidates[0]
    content_index = 0
    for part in candidate.content.parts or []:
        signature_b64 = base64_encode(part.thought_signature)
        if part.thought:
            # Thought-summary parts never carry a signature — they're a
            # sibling surface to the data part that does.
            set_output_reasoning_summary(attrs, 0, content_index, text=part.text or "")
        elif part.function_call is not None:
            set_output_tool_use(
                attrs,
                0,
                content_index,
                tool_call_id=part.function_call.id or f"call_{content_index}",
                name=part.function_call.name or "",
                arguments_json=json.dumps(dict(part.function_call.args or {})),
                thought_signature=signature_b64,
            )
        elif part.text is not None:
            set_output_text(attrs, 0, content_index, part.text, thought_signature=signature_b64)
        else:
            continue
        content_index += 1


def rebuild_model_content(
    fetched_attrs: dict[str, Any], *, strip_signature: bool = False
) -> gtypes.Content:
    parts: list[gtypes.Part] = []
    for content_block in read_output_contents(fetched_attrs, 0):
        block_type = content_block.get(MessageContentAttributes.MESSAGE_CONTENT_TYPE)
        signature = (
            None
            if strip_signature
            else base64_decode(content_block.get(MESSAGE_CONTENT_THOUGHT_SIGNATURE))
        )
        if block_type == CONTENT_TYPE_REASONING_SUMMARY:
            parts.append(
                gtypes.Part(
                    text=content_block.get(MessageContentAttributes.MESSAGE_CONTENT_TEXT, ""),
                    thought=True,
                )
            )
        elif block_type == CONTENT_TYPE_TEXT:
            parts.append(
                gtypes.Part(
                    text=content_block[MessageContentAttributes.MESSAGE_CONTENT_TEXT],
                    thought_signature=signature,
                )
            )
        elif block_type == CONTENT_TYPE_TOOL_USE:
            function_call = gtypes.FunctionCall(
                id=content_block[ToolCallAttributes.TOOL_CALL_ID],
                name=content_block[ToolCallAttributes.TOOL_CALL_FUNCTION_NAME],
                args=json.loads(content_block[ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON]),
            )
            parts.append(gtypes.Part(function_call=function_call, thought_signature=signature))
    return gtypes.Content(role="model", parts=parts)


def scenario_text(client: genai.Client, ctx: TracingCtx) -> bool:
    print("\n[gemini] scenario A: text round-trip")
    user_text = "In one short sentence, what's special about the number 1729?"

    with ctx.tracer.start_as_current_span(
        "gemini.generate_content.text.turn1.original", kind=SpanKind.CLIENT
    ) as turn1_span:
        turn1_attrs: dict[str, Any] = {}
        set_request_attributes(turn1_attrs)
        set_input_user_message(turn1_attrs, 0, user_text)

        turn1_response = client.models.generate_content(
            model=MODEL,
            contents=user_text,
            config=text_only_config(),
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

    with ctx.tracer.start_as_current_span(
        "gemini.generate_content.text.turn2.roundtrip", kind=SpanKind.CLIENT
    ) as turn2_span:
        turn2_attrs: dict[str, Any] = {}
        set_request_attributes(turn2_attrs)
        set_input_user_message(turn2_attrs, 0, prior_user_text)
        set_input_assistant_echo(turn2_attrs, 1, captured_contents, role="model")
        set_input_user_message(turn2_attrs, 2, follow_up_text)

        turn2_response = replay_turn2(
            client,
            fetched_attrs,
            contents=[
                gtypes.Content(role="user", parts=[gtypes.Part(text=prior_user_text)]),
                rebuild_model_content(fetched_attrs),
                gtypes.Content(role="user", parts=[gtypes.Part(text=follow_up_text)]),
            ],
        )
        record_assistant_turn(turn2_attrs, turn2_response)
        turn2_span.set_attributes(turn2_attrs)

    turn2_text = (turn2_response.text or "")[:120]
    print(f"  fetched_blocks={len(captured_contents)} "
          f"token_lens={continuity_token_lengths(captured_contents)}")
    print(f"  turn2_text={turn2_text!r}")
    return bool(turn2_text)


def scenario_tool(client: genai.Client, ctx: TracingCtx) -> tuple[bool, bool | None]:
    print("\n[gemini] scenario B: thinking → functionCall round-trip (+ optional negative)")
    user_text = "What's the weather in Paris? Use the tool."

    with ctx.tracer.start_as_current_span(
        "gemini.generate_content.tool.turn1.original", kind=SpanKind.CLIENT
    ) as turn1_span:
        turn1_attrs: dict[str, Any] = {}
        set_request_attributes(turn1_attrs)
        set_tools(turn1_attrs, [WEATHER_TOOL_SCHEMA])
        set_input_user_message(turn1_attrs, 0, user_text)

        turn1_response = client.models.generate_content(
            model=MODEL,
            contents=user_text,
            config=thinking_config(),
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
        return True, None

    tool_response_payload = {"temperature_c": 14.5}
    tool_response_user_content = gtypes.Content(
        role="user",
        parts=[
            gtypes.Part(
                function_response=gtypes.FunctionResponse(
                    id=tool_use_block[ToolCallAttributes.TOOL_CALL_ID],
                    name=tool_use_block[ToolCallAttributes.TOOL_CALL_FUNCTION_NAME],
                    response=tool_response_payload,
                )
            )
        ],
    )

    rebuilt_model_content = rebuild_model_content(fetched_attrs)
    with ctx.tracer.start_as_current_span(
        "gemini.generate_content.tool.turn2.roundtrip", kind=SpanKind.CLIENT
    ) as turn2_span:
        turn2_attrs: dict[str, Any] = {}
        set_request_attributes(turn2_attrs)
        set_tools(turn2_attrs, [WEATHER_TOOL_SCHEMA])
        set_input_user_message(turn2_attrs, 0, prior_user_text)
        set_input_assistant_echo(turn2_attrs, 1, captured_contents, role="model")
        set_input_tool_result(
            turn2_attrs,
            2,
            tool_call_id=tool_use_block[ToolCallAttributes.TOOL_CALL_ID],
            content_text=json.dumps(tool_response_payload),
        )

        turn2_response = replay_turn2(
            client,
            fetched_attrs,
            contents=[
                gtypes.Content(role="user", parts=[gtypes.Part(text=prior_user_text)]),
                rebuilt_model_content,
                tool_response_user_content,
            ],
        )
        record_assistant_turn(turn2_attrs, turn2_response)
        turn2_span.set_attributes(turn2_attrs)

    final_text = (turn2_response.text or "")[:120]
    print(f"  fetched_blocks={len(captured_contents)} "
          f"token_lens={continuity_token_lengths(captured_contents)}")
    print(f"  positive_turn2_text={final_text!r}")
    positive_ok = bool(final_text)

    if not MODEL.startswith("gemini-3"):
        print(
            "  negative skipped (Gemini 2.5: stripping signature is a "
            "silent quality loss, not a 400)"
        )
        return positive_ok, None

    stripped_model_content = rebuild_model_content(fetched_attrs, strip_signature=True)
    negative_rejected = False
    with ctx.tracer.start_as_current_span(
        "gemini.generate_content.tool.turn2.roundtrip.negative", kind=SpanKind.CLIENT
    ) as negative_span:
        try:
            client.models.generate_content(
                model=MODEL,
                contents=[
                    gtypes.Content(role="user", parts=[gtypes.Part(text=user_text)]),
                    stripped_model_content,
                    tool_response_user_content,
                ],
                config=thinking_config(),
            )
        except Exception as error:  # noqa: BLE001 — google.genai raises various types
            negative_rejected = True
            negative_span.set_attribute("error.expected", True)
            negative_span.set_attribute("error.message", str(error)[:200])
            print(f"  negative correctly rejected: {str(error)[:120]!r}")
    if not negative_rejected:
        print("  negative FAILED: stripped-signature request was accepted on Gemini 3")
    return positive_ok, negative_rejected


def main() -> int:
    ctx = setup_tracing("reasoning-roundtrip-gemini")
    client = genai.Client()
    text_ok = scenario_text(client, ctx)
    tool_positive_ok, tool_negative_ok = scenario_tool(client, ctx)
    overall_ok = text_ok and tool_positive_ok and (tool_negative_ok is not False)
    print(
        f"\n[gemini] text={text_ok}  "
        f"tool_positive={tool_positive_ok}  tool_negative={tool_negative_ok}  "
        f"PASS={overall_ok}"
    )
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
