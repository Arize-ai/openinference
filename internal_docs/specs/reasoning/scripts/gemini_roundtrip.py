# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "google-genai==2.5.0",
#     "opentelemetry-api==1.42.1",
#     "opentelemetry-sdk==1.42.1",
#     "opentelemetry-exporter-otlp-proto-http==1.42.1",
#     "openinference-semantic-conventions==0.1.29",
#     "openinference-instrumentation-google-genai==1.0.2",
# ]
# ///
"""Google Gemini thinking-surface round-trip.

Uses the real Google GenAI instrumentor as the baseline span writer. Captured
finished spans are augmented with proposed thought-summary/signature fields and
re-exported to Phoenix before replay.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from typing import Any

from google import genai
from google.genai import types as gtypes
from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
)
from openinference.semconv.trace import ToolCallAttributes

from common import (
    CONTENT_TYPE_REASONING,
    CONTENT_TYPE_TEXT,
    CONTENT_TYPE_TOOL_USE,
    FACTORIZE_FOLLOW_UP_PROMPT,
    FACTORIZE_USER_PROMPT,
    InstrumentedTracingCtx,
    MESSAGE_CONTENT_SIGNATURE,
    TOOL_FOLLOW_UP_PROMPT,
    TOOL_RESULT_PAYLOAD,
    TOOL_USER_PROMPT,
    TOOL_CALL_REASONING_SIGNATURE,
    debug_print,
    export_augmented_span,
    export_original_span,
    input_content_key,
    mutate_span_attributes,
    output_content_key,
    print_attribute_keys,
    print_capture_summary,
    read_input_message_text,
    read_tools,
    setup_instrumented_tracing,
)

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
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


def user_text_content(text: str) -> gtypes.Content:
    return gtypes.Content(role="user", parts=[gtypes.Part(text=text)])


def base64_encode(raw_bytes: bytes | None) -> str | None:
    if raw_bytes is None:
        return None
    return base64.b64encode(raw_bytes).decode("ascii")


def base64_decode(encoded: str | None) -> bytes | None:
    if not encoded:
        return None
    return base64.b64decode(encoded)


def make_config(
    *, with_tools: bool, include_thoughts: bool
) -> gtypes.GenerateContentConfig:
    return gtypes.GenerateContentConfig(
        thinking_config=gtypes.ThinkingConfig(
            thinking_budget=BUDGET,
            include_thoughts=include_thoughts,
        ),
        tools=[WEATHER_TOOL] if with_tools else None,
    )


def serialize_gemini_contents(contents: Any) -> Any:
    if isinstance(contents, str):
        return contents
    return [item.model_dump(exclude_none=True, by_alias=True) for item in contents]


def build_generate_content_request(
    *,
    model: str,
    contents: Any,
    config: gtypes.GenerateContentConfig,
) -> dict[str, Any]:
    return {
        "model": model,
        "contents": serialize_gemini_contents(contents),
        "config": config.model_dump(exclude_none=True, by_alias=True),
    }


def config_from_fetched(fetched_attrs: dict[str, Any]) -> gtypes.GenerateContentConfig:
    invocation_parameters = json.loads(
        fetched_attrs[SpanAttributes.LLM_INVOCATION_PARAMETERS]
    )
    config = invocation_parameters.get("config") or invocation_parameters
    thinking = (
        config.get("thinkingConfig")
        or config.get("thinking_config")
        or config.get("generationConfig", {}).get("thinkingConfig")
        or {}
    )
    fetched_tools = gemini_tools_from_fetched(fetched_attrs)
    return gtypes.GenerateContentConfig(
        thinking_config=gtypes.ThinkingConfig(
            thinking_budget=thinking.get("thinkingBudget")
            or thinking.get("thinking_budget"),
            include_thoughts=thinking.get("includeThoughts")
            or thinking.get("include_thoughts"),
        ),
        tools=fetched_tools or None,
    )


def gemini_tools_from_fetched(fetched_attrs: dict[str, Any]) -> list[gtypes.Tool]:
    tools: list[gtypes.Tool] = []
    for schema in read_tools(fetched_attrs):
        declarations = (
            schema.get("function_declarations")
            or schema.get("functionDeclarations")
            or [schema]
        )
        function_declarations: list[gtypes.FunctionDeclaration] = []
        for declaration in declarations:
            parameters = (
                declaration.get("parameters_json_schema")
                or declaration.get("parametersJsonSchema")
                or declaration.get("parameters")
            )
            function_declarations.append(
                gtypes.FunctionDeclaration(
                    name=declaration.get("name", ""),
                    description=declaration.get("description"),
                    parameters_json_schema=parameters,
                )
            )
        tools.append(gtypes.Tool(function_declarations=function_declarations))
    return tools


def simulate_gemini_content_walk(response: Any) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    candidate = response.candidates[0]
    tool_call_index = 0
    for content_index, part in enumerate(candidate.content.parts or []):
        signature = base64_encode(part.thought_signature)
        if part.thought:
            attrs[
                output_content_key(
                    0, content_index, MessageContentAttributes.MESSAGE_CONTENT_TYPE
                )
            ] = CONTENT_TYPE_REASONING
            attrs[
                output_content_key(
                    0, content_index, MessageContentAttributes.MESSAGE_CONTENT_TEXT
                )
            ] = part.text or ""
        elif part.function_call is not None:
            attrs[
                output_content_key(
                    0, content_index, MessageContentAttributes.MESSAGE_CONTENT_TYPE
                )
            ] = CONTENT_TYPE_TOOL_USE
            if signature:
                attrs[
                    output_content_key(0, content_index, TOOL_CALL_REASONING_SIGNATURE)
                ] = signature
            call = part.function_call
            call_id = call.id or f"call_{content_index}"
            attrs[
                output_content_key(0, content_index, ToolCallAttributes.TOOL_CALL_ID)
            ] = call_id
            attrs[
                output_content_key(
                    0, content_index, ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
                )
            ] = call.name or ""
            attrs[
                output_content_key(
                    0,
                    content_index,
                    ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON,
                )
            ] = json.dumps(dict(call.args or {}))
            attrs[
                f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0."
                f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{tool_call_index}."
                f"{ToolCallAttributes.TOOL_CALL_ID}"
            ] = call_id
            tool_call_index += 1
        elif part.text is not None:
            attrs[
                output_content_key(
                    0, content_index, MessageContentAttributes.MESSAGE_CONTENT_TYPE
                )
            ] = CONTENT_TYPE_TEXT
            attrs[
                output_content_key(
                    0, content_index, MessageContentAttributes.MESSAGE_CONTENT_TEXT
                )
            ] = part.text
            if signature:
                attrs[
                    output_content_key(0, content_index, MESSAGE_CONTENT_SIGNATURE)
                ] = signature
    return attrs


def _part_signature(part: dict[str, Any]) -> str | None:
    signature = part.get("thoughtSignature") or part.get("thought_signature")
    if isinstance(signature, bytes):
        return base64_encode(signature)
    if isinstance(signature, str):
        return signature
    return None


def _part_function_call(part: dict[str, Any]) -> dict[str, Any] | None:
    return part.get("functionCall") or part.get("function_call")


def simulate_gemini_input_walk(request: dict[str, Any]) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    contents = request.get("contents") or []
    if isinstance(contents, str):
        contents = [{"role": "user", "parts": [{"text": contents}]}]

    for message_index, content in enumerate(contents):
        parts = content.get("parts") or []
        for content_index, part in enumerate(parts):
            signature = _part_signature(part)
            if part.get("thought"):
                attrs[
                    input_content_key(
                        message_index,
                        content_index,
                        MessageContentAttributes.MESSAGE_CONTENT_TYPE,
                    )
                ] = CONTENT_TYPE_REASONING
                attrs[
                    input_content_key(
                        message_index,
                        content_index,
                        MessageContentAttributes.MESSAGE_CONTENT_TEXT,
                    )
                ] = part.get("text", "")
            elif "text" in part:
                attrs[
                    input_content_key(
                        message_index,
                        content_index,
                        MessageContentAttributes.MESSAGE_CONTENT_TYPE,
                    )
                ] = CONTENT_TYPE_TEXT
                attrs[
                    input_content_key(
                        message_index,
                        content_index,
                        MessageContentAttributes.MESSAGE_CONTENT_TEXT,
                    )
                ] = part["text"]
                if signature:
                    attrs[
                        input_content_key(
                            message_index,
                            content_index,
                            MESSAGE_CONTENT_SIGNATURE,
                        )
                    ] = signature
            elif function_call := _part_function_call(part):
                attrs[
                    input_content_key(
                        message_index,
                        content_index,
                        MessageContentAttributes.MESSAGE_CONTENT_TYPE,
                    )
                ] = CONTENT_TYPE_TOOL_USE
                if signature:
                    attrs[
                        input_content_key(
                            message_index,
                            content_index,
                            TOOL_CALL_REASONING_SIGNATURE,
                        )
                    ] = signature
                attrs[
                    input_content_key(
                        message_index, content_index, ToolCallAttributes.TOOL_CALL_ID
                    )
                ] = function_call.get("id", f"call_{content_index}")
                attrs[
                    input_content_key(
                        message_index,
                        content_index,
                        ToolCallAttributes.TOOL_CALL_FUNCTION_NAME,
                    )
                ] = function_call.get("name", "")
                attrs[
                    input_content_key(
                        message_index,
                        content_index,
                        ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON,
                    )
                ] = json.dumps(function_call.get("args") or {})
    return attrs


def omit_redundant_scalar_input_additions(
    baseline: dict[str, Any], additions: dict[str, Any]
) -> dict[str, Any]:
    """Do not replace existing scalar input text with equivalent text contents."""
    filtered = dict(additions)
    input_prefix = f"{SpanAttributes.LLM_INPUT_MESSAGES}."
    scalar_suffix = f".{MessageAttributes.MESSAGE_CONTENT}"
    by_message_index: dict[int, list[str]] = {}
    for key in additions:
        if not key.startswith(input_prefix):
            continue
        index_str, _, remainder = key[len(input_prefix) :].partition(".")
        if not remainder.startswith(f"{MessageAttributes.MESSAGE_CONTENTS}."):
            continue
        try:
            message_index = int(index_str)
        except ValueError:
            continue
        by_message_index.setdefault(message_index, []).append(key)

    for message_index, keys in by_message_index.items():
        scalar_key = (
            f"{SpanAttributes.LLM_INPUT_MESSAGES}.{message_index}{scalar_suffix}"
        )
        if scalar_key not in baseline:
            continue
        suffixes = {
            key.split(f"{MessageAttributes.MESSAGE_CONTENTS}.", 1)[1].split(".", 1)[1]
            for key in keys
        }
        content_types = {
            additions[key]
            for key in keys
            if key.endswith(f".{MessageContentAttributes.MESSAGE_CONTENT_TYPE}")
        }
        if suffixes <= {
            MessageContentAttributes.MESSAGE_CONTENT_TYPE,
            MessageContentAttributes.MESSAGE_CONTENT_TEXT,
        } and content_types <= {CONTENT_TYPE_TEXT}:
            for key in keys:
                filtered.pop(key, None)
    return filtered


def print_gemini_content_slots(response: Any) -> None:
    debug_print("\n[gemini response.content.parts slots]")
    for index, part in enumerate(response.candidates[0].content.parts or []):
        if part.thought:
            kind = "thought"
        elif part.function_call is not None:
            kind = "function_call"
        elif part.text is not None:
            kind = "text"
        else:
            kind = "unknown"
        debug_print(f"  parts[{index}].type={kind}")


def augment_gemini_span_from_response(
    span: Any, response: Any, request: dict[str, Any]
) -> None:
    baseline = dict(span.attributes or {})
    output_additions = simulate_gemini_content_walk(response)
    input_additions = omit_redundant_scalar_input_additions(
        baseline, simulate_gemini_input_walk(request)
    )
    additions = {**input_additions, **output_additions}
    print_gemini_content_slots(response)
    print_attribute_keys(f"{span.name} baseline", baseline)
    mutate_span_attributes(span, additions)
    print_attribute_keys(f"{span.name} augmented", dict(span.attributes or {}))


def call_and_export_augmented(
    ctx: InstrumentedTracingCtx,
    client: genai.Client,
    request: dict[str, Any],
    *,
    root_name: str,
) -> tuple[Any, dict[str, Any]]:
    before_count = ctx.span_count()
    with ctx.tracer.start_as_current_span(root_name) as root_span:
        root_context = root_span.get_span_context()
        response = client.models.generate_content(**request)
        span = ctx.latest_span_since(before_count)
    ctx.export(ctx.span_by_id(root_context.span_id))
    export_original_span(ctx, span)
    augment_gemini_span_from_response(span, response, request)
    fetched_attrs = export_augmented_span(ctx, span)
    return response, fetched_attrs


def read_output_contents(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    prefix = (
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENTS}."
    )
    by_index: dict[int, dict[str, Any]] = {}
    for key, value in attrs.items():
        if not key.startswith(prefix):
            continue
        index_str, _, suffix = key[len(prefix) :].partition(".")
        try:
            index = int(index_str)
        except ValueError:
            continue
        by_index.setdefault(index, {})[suffix] = value
    return [by_index[index] for index in sorted(by_index)]


def read_output_tool_calls(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    prefix = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}."
    by_index: dict[int, dict[str, Any]] = {}
    for key, value in attrs.items():
        if not key.startswith(prefix):
            continue
        index_str, _, suffix = key[len(prefix) :].partition(".")
        try:
            index = int(index_str)
        except ValueError:
            continue
        by_index.setdefault(index, {})[suffix] = value
    return [by_index[index] for index in sorted(by_index)]


def rebuild_model_content(
    fetched_attrs: dict[str, Any], *, strip_signature: bool = False
) -> gtypes.Content:
    parts: list[gtypes.Part] = []
    tool_calls = iter(read_output_tool_calls(fetched_attrs))
    for content_block in read_output_contents(fetched_attrs):
        block_type = content_block.get(MessageContentAttributes.MESSAGE_CONTENT_TYPE)
        if block_type == CONTENT_TYPE_REASONING:
            parts.append(
                gtypes.Part(
                    text=content_block.get(
                        MessageContentAttributes.MESSAGE_CONTENT_TEXT, ""
                    ),
                    thought=True,
                )
            )
        elif block_type == CONTENT_TYPE_TEXT:
            signature = (
                None
                if strip_signature
                else base64_decode(content_block.get(MESSAGE_CONTENT_SIGNATURE))
            )
            parts.append(
                gtypes.Part(
                    text=content_block[MessageContentAttributes.MESSAGE_CONTENT_TEXT],
                    thought_signature=signature,
                )
            )
        elif block_type == CONTENT_TYPE_TOOL_USE:
            signature = (
                None
                if strip_signature
                else base64_decode(content_block.get(TOOL_CALL_REASONING_SIGNATURE))
            )
            tool_call = next(tool_calls, {})
            function_call = gtypes.FunctionCall(
                id=tool_call.get(ToolCallAttributes.TOOL_CALL_ID),
                name=tool_call.get(ToolCallAttributes.TOOL_CALL_FUNCTION_NAME),
                args=json.loads(
                    tool_call.get(
                        ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON, "{}"
                    )
                ),
            )
            parts.append(
                gtypes.Part(function_call=function_call, thought_signature=signature)
            )
    return gtypes.Content(role="model", parts=parts)


def build_request_from_fetched(
    fetched_attrs: dict[str, Any], contents: list[gtypes.Content]
) -> dict[str, Any]:
    return build_generate_content_request(
        model=fetched_attrs[SpanAttributes.LLM_MODEL_NAME],
        contents=contents,
        config=config_from_fetched(fetched_attrs),
    )


def scenario_text(
    client: genai.Client, ctx: InstrumentedTracingCtx, *, include_thoughts: bool
) -> bool:
    print("\n[gemini] scenario A: text round-trip")
    user_text = FACTORIZE_USER_PROMPT
    turn1_request = build_generate_content_request(
        model=MODEL,
        contents=[user_text_content(user_text)],
        config=make_config(with_tools=False, include_thoughts=include_thoughts),
    )
    _, fetched_attrs = call_and_export_augmented(
        ctx, client, turn1_request, root_name="gemini text turn1"
    )
    prior_user_text = read_input_message_text(fetched_attrs, 0)
    turn2_contents = [
        user_text_content(prior_user_text),
        rebuild_model_content(fetched_attrs),
        user_text_content(FACTORIZE_FOLLOW_UP_PROMPT),
    ]
    turn2_response, _ = call_and_export_augmented(
        ctx,
        client,
        build_request_from_fetched(fetched_attrs, turn2_contents),
        root_name="gemini text turn2",
    )
    turn2_text = (turn2_response.text or "")[:120]
    print_capture_summary(read_output_contents(fetched_attrs))
    print(f"  turn2_text={turn2_text!r}")
    return bool(turn2_text)


def scenario_tool(
    client: genai.Client, ctx: InstrumentedTracingCtx, *, include_thoughts: bool
) -> tuple[bool, bool | None]:
    print(
        "\n[gemini] scenario B: thinking -> functionCall round-trip (+ optional negative)"
    )
    user_text = TOOL_USER_PROMPT
    turn1_request = build_generate_content_request(
        model=MODEL,
        contents=[user_text_content(user_text)],
        config=make_config(with_tools=True, include_thoughts=include_thoughts),
    )
    _, fetched_attrs = call_and_export_augmented(
        ctx, client, turn1_request, root_name="gemini tool turn1"
    )
    prior_user_text = read_input_message_text(fetched_attrs, 0)
    tool_calls = read_output_tool_calls(fetched_attrs)
    if not tool_calls:
        print("  SKIP: model did not call the tool")
        return True, None
    tool_response_payload = dict(TOOL_RESULT_PAYLOAD)
    tool_response_user_content = gtypes.Content(
        role="user",
        parts=[
            gtypes.Part(
                function_response=gtypes.FunctionResponse(
                    id=tool_calls[0].get(ToolCallAttributes.TOOL_CALL_ID),
                    name=tool_calls[0][ToolCallAttributes.TOOL_CALL_FUNCTION_NAME],
                    response=tool_response_payload,
                )
            )
        ],
    )
    turn2_contents = [
        user_text_content(prior_user_text),
        rebuild_model_content(fetched_attrs),
        tool_response_user_content,
        user_text_content(TOOL_FOLLOW_UP_PROMPT),
    ]
    turn2_response, _ = call_and_export_augmented(
        ctx,
        client,
        build_request_from_fetched(fetched_attrs, turn2_contents),
        root_name="gemini tool turn2",
    )
    final_text = (turn2_response.text or "")[:120]
    print_capture_summary(read_output_contents(fetched_attrs))
    print(f"  positive_turn2_text={final_text!r}")
    positive_ok = bool(final_text)

    if not MODEL.startswith("gemini-3"):
        print("  negative skipped (Gemini 2.5: stripping signature is not a 400)")
        return positive_ok, None
    try:
        negative_contents = [
            user_text_content(user_text),
            rebuild_model_content(fetched_attrs, strip_signature=True),
            tool_response_user_content,
            user_text_content(TOOL_FOLLOW_UP_PROMPT),
        ]
        call_and_export_augmented(
            ctx,
            client,
            build_request_from_fetched(fetched_attrs, negative_contents),
            root_name="gemini tool negative",
        )
    except Exception as error:  # noqa: BLE001
        print(f"  negative correctly rejected: {str(error)[:120]!r}")
        return positive_ok, True
    print("  negative FAILED: stripped-signature request was accepted on Gemini 3")
    return positive_ok, False


def parse_include_thoughts() -> bool:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-thoughts",
        choices=["on", "off", "true", "false"],
        default=os.environ.get("GEMINI_INCLUDE_THOUGHTS", "on").lower(),
        help="Set Gemini thinkingConfig.includeThoughts.",
    )
    args = parser.parse_args()
    return args.include_thoughts in {"on", "true"}


def main() -> int:
    include_thoughts = parse_include_thoughts()
    ctx = setup_instrumented_tracing("gemini-reasoning-roundtrip")
    instrumentor = GoogleGenAIInstrumentor()
    instrumentor.instrument(tracer_provider=ctx.provider)
    try:
        client = genai.Client()
        text_ok = scenario_text(client, ctx, include_thoughts=include_thoughts)
        tool_positive_ok, tool_negative_ok = scenario_tool(
            client, ctx, include_thoughts=include_thoughts
        )
        overall_ok = text_ok and tool_positive_ok and (tool_negative_ok is not False)
        print(
            f"\n[gemini] text={text_ok}  "
            f"tool_positive={tool_positive_ok}  tool_negative={tool_negative_ok}  "
            f"PASS={overall_ok}"
        )
        return 0 if overall_ok else 1
    finally:
        instrumentor.uninstrument()
        ctx.shutdown()


if __name__ == "__main__":
    sys.exit(main())
