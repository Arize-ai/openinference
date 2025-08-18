# openlit_to_openinference.py

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, Tuple

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.util.types import AttributeValue

import openinference.instrumentation as oi
import openinference.semconv.trace as sc
from openinference.instrumentation import (
    get_input_attributes,
    get_llm_attributes,
    get_output_attributes,
)

__all__ = ["OpenInferenceSpanProcessor"]

_PROVIDERS = {
    "openai",
    "ollama",
    "anthropic",
    "gpt4all",
    "cohere",
    "mistral",
    "github_models",
    "vllm",
    "azure_openai",
    "azure_ai_inference",
    "huggingface",
    "amazon_bedrock",
    "vertex_ai",
    "google_ai_studio",
    "groq",
    "nvidia_nim",
    "xai",
    "elevenlabs",
    "ai21",
    "together",
    "assembly_ai",
    "featherless",
    "reka_ai",
    "ola_krutrim",
    "titan_ml",
    "prem_ai",
    "vector_dbs",
}


def _is_wrapper_span(attrs: Dict[str, Any]) -> bool:
    """
    Heuristic: True = a framework / chain / tool span (not the real LLM HTTP call).
    """
    sys_name = str(attrs.get("gen_ai.system", "")).lower()
    addr = attrs.get("server.address")
    port = attrs.get("server.port")

    if sys_name in _PROVIDERS:
        return False

    if addr in (None, "", "NOT_FOUND") or port in (None, "", "NOT_FOUND"):
        return True

    return False


def _safe_int(v: Any) -> int | None:
    try:
        return int(v)
    except Exception:
        return None


def _tool_call_to_dict(tc: Any) -> Dict[str, Any]:
    """
    Convert an oi.ToolCall OR a raw dict into the JSON shape Phoenix expects.
    """
    if isinstance(tc, dict):
        return tc

    try:
        fn = tc["function"]
        args = fn["arguments"] if isinstance(fn, dict) else ""
        name = fn["name"] if isinstance(fn, dict) else ""
        return {
            "id": tc.get("id", ""),
            "type": "function",
            "function": {"name": name, "arguments": args},
        }
    except Exception:
        return {}


def _parse_prompt_from_events(events: Any) -> str | None:
    for ev in events or []:
        if ev.name == "gen_ai.content.prompt":
            result = ev.attributes.get("gen_ai.prompt")
            return str(result) if result is not None else None
    return None


def _parse_completion_from_events(events: Any) -> str | None:
    for ev in events or []:
        if ev.name == "gen_ai.content.completion":
            result = ev.attributes.get("gen_ai.completion")
            return str(result) if result is not None else None
    return None


_ROLE_LINE_RE = re.compile(r"^(user|assistant|system|tool):\s*(.*)$", re.IGNORECASE)


def _unflatten_prompt(flat: str) -> list[Dict[str, str]]:
    """
    Turn the newline-joined OpenLIT prompt back into a list of
    {'role': str, 'content': str} dictionaries.
    """
    msgs: list[Dict[str, str]] = []
    for line in flat.splitlines():
        m = _ROLE_LINE_RE.match(line.strip())
        if not m:
            if msgs:
                msgs[-1]["content"] += "\n" + line
            continue
        role, content = m.groups()
        msgs.append({"role": role.lower(), "content": content})
    return msgs


def _load_tool_calls(raw: Any) -> list[Dict[str, Any]]:
    """
    Parse the value that OpenLIT stores in
    `gen_ai.response.tool_calls`.

    * already a list      → return as-is
    * JSON string         → json.loads(...)
    * Python-literal str  → ast.literal_eval(...)
    * anything else       → []
    """
    if raw is None:
        return []

    # 1) direct list
    if isinstance(raw, list):
        return [tc if isinstance(tc, dict) else _tool_call_to_dict(tc) for tc in raw]

    # 2) JSON string
    if isinstance(raw, str):
        try:
            result = json.loads(raw)
            if isinstance(result, list):
                return [tc if isinstance(tc, dict) else _tool_call_to_dict(tc) for tc in result]
            return []
        except Exception:
            # 3) python literal – OpenLIT sometimes dumps repr(list)
            try:
                result = ast.literal_eval(raw)
                if isinstance(result, list):
                    return [tc if isinstance(tc, dict) else _tool_call_to_dict(tc) for tc in result]
                return []
            except Exception:
                return []

    # 4) fallback
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return [tc if isinstance(tc, dict) else _tool_call_to_dict(tc) for tc in result]
        return []
    except Exception:
        return []


def _build_messages(
    prompt: str | None,
    completion: str | None,
    tool_calls_json: Any,
) -> Tuple[list[oi.Message], list[oi.Message]]:
    """
    Convert OpenLIT's plain-text fields into OpenInference message objects.
    Returns (input_messages, output_messages).
    """

    input_msgs: list[oi.Message] = []
    if prompt:
        for m in _unflatten_prompt(prompt):
            content = m.get("content", "")
            if content != "None" and content is not None:
                input_msgs.append(oi.Message(role=m["role"], content=str(content)))

    assistant = oi.Message(role="assistant")
    if completion == "None":
        assistant["content"] = ""
    else:
        assistant["content"] = str(completion) if completion is not None else ""
    tc_list = _load_tool_calls(tool_calls_json)
    if tc_list:
        assistant["tool_calls"] = [
            oi.ToolCall(
                id=tc.get("id", ""),
                function=oi.ToolCallFunction(
                    name=tc.get("function", {}).get("name", ""),
                    arguments=tc.get("function", {}).get("arguments", ""),
                ),
            )
            for tc in tc_list
        ]

    output_msgs = [assistant]
    return input_msgs, output_msgs


def parse_messages(events: Any, attrs: Dict[str, Any]) -> Tuple[list[oi.Message], list[oi.Message]]:
    prompt_text = _parse_prompt_from_events(events)
    completion_text = _parse_completion_from_events(events)
    tool_calls_json = attrs.get("gen_ai.response.tool_calls")

    return _build_messages(prompt_text, completion_text, tool_calls_json)


def convert_to_oi_tool_attributes(span: ReadableSpan) -> dict[str, Any]:
    openlit_attrs = dict(getattr(span, "_attributes", {}))
    oi_attrs = {
        sc.SpanAttributes.OPENINFERENCE_SPAN_KIND: sc.OpenInferenceSpanKindValues.TOOL.value,
    }

    tool_name = openlit_attrs.get("gen_ai.tool.name")
    tool_description = openlit_attrs.get("gen_ai.tool.description")

    if tool_name:
        oi_attrs[sc.SpanAttributes.TOOL_NAME] = tool_name
    if tool_description:
        oi_attrs[sc.SpanAttributes.TOOL_DESCRIPTION] = tool_description

    prompt_txt = _parse_prompt_from_events(span.events)
    completion_txt = _parse_completion_from_events(span.events)
    if prompt_txt:
        oi_attrs.update(
            {
                sc.SpanAttributes.INPUT_MIME_TYPE: sc.OpenInferenceMimeTypeValues.TEXT.value,
                sc.SpanAttributes.INPUT_VALUE: prompt_txt,
            }
        )
    if completion_txt:
        oi_attrs.update(
            {
                sc.SpanAttributes.OUTPUT_MIME_TYPE: sc.OpenInferenceMimeTypeValues.TEXT.value,
                sc.SpanAttributes.OUTPUT_VALUE: completion_txt,
            }
        )
    print(oi_attrs)

    return oi_attrs


def is_openlit_tool_span(attrs: dict[str, Any]) -> bool:
    operation_name = attrs.get("gen_ai.operation.name")
    tool_name = attrs.get("gen_ai.tool.name")
    return operation_name == "execute_tool" or tool_name is not None


def find_invocation_parameters(attrs: Dict[str, Any]) -> Dict[str, Any]:
    invocation: Dict[str, Any] = {}
    for knob in ("model", "temperature", "max_tokens", "top_p", "top_k"):
        val = attrs.get(f"gen_ai.request.{knob}")
        if val not in (None, "", -1):
            invocation[knob] = val
    return invocation


def build_output_messages(output_msgs: list[oi.Message]) -> list[Dict[str, Any]]:
    out_msgs_json: list[Dict[str, Any]] = []
    for m in output_msgs:
        try:
            mdict = dict(m)
        except (TypeError, ValueError):
            mdict = {}

        # drop falsy content keys
        if not mdict.get("content"):
            mdict.pop("content", None)

        # normalise tool-calls …
        if "tool_calls" in mdict:
            tool_calls = mdict["tool_calls"]
            if hasattr(tool_calls, "__iter__") and not isinstance(tool_calls, (str, bytes)):
                mdict["tool_calls"] = [_tool_call_to_dict(tc) for tc in tool_calls]
            else:
                # If tool_calls is not iterable, remove it
                mdict.pop("tool_calls", None)

        out_msgs_json.append(mdict)
    return out_msgs_json


class OpenInferenceSpanProcessor(SpanProcessor):
    """
    Converts OpenLIT GenAI spans → OpenInference attributes in-place.
    Add to your tracer-provider:

        provider.add_span_processor(OpenInferenceSpanProcessor())
    """

    def on_end(self, span: ReadableSpan) -> None:
        attrs: Dict[str, Any] = dict(getattr(span, "_attributes", {}))

        oi_attrs: Dict[str, AttributeValue] = {
            sc.SpanAttributes.OPENINFERENCE_SPAN_KIND: sc.OpenInferenceSpanKindValues.CHAIN.value
        }

        if is_openlit_tool_span(attrs):
            oi_attrs = convert_to_oi_tool_attributes(span)
            if span._attributes:
                span._attributes = {**span._attributes, **oi_attrs}
            return

        if _is_chain_span(attrs):

            prompt_txt = _parse_prompt_from_events(span.events)
            completion_txt = _parse_completion_from_events(span.events)
            if prompt_txt:
                oi_attrs.update(
                    {
                        sc.SpanAttributes.INPUT_MIME_TYPE: (
                            sc.OpenInferenceMimeTypeValues.TEXT.value
                        ),
                        sc.SpanAttributes.INPUT_VALUE: prompt_txt,
                    }
                )
            if completion_txt:
                oi_attrs.update(
                    {
                        sc.SpanAttributes.OUTPUT_MIME_TYPE: (
                            sc.OpenInferenceMimeTypeValues.TEXT.value
                        ),
                        sc.SpanAttributes.OUTPUT_VALUE: completion_txt,
                    }
                )
            oi_attrs.update(
                {
                    sc.SpanAttributes.OPENINFERENCE_SPAN_KIND: (sc.OpenInferenceSpanKindValues.CHAIN.value),
                }
            )

            if span._attributes:
                if attrs.get("gen_ai.system"):
                    span._name = attrs.get("gen_ai.system")
                span._attributes = {**span._attributes, **oi_attrs}
            return

        if "gen_ai.system" not in attrs:
            return

        input_msgs, output_msgs = parse_messages(span.events, attrs)

        prompt_tokens = _safe_int(attrs.get("gen_ai.usage.input_tokens"))
        completion_tokens = _safe_int(attrs.get("gen_ai.usage.output_tokens"))
        total_tokens = _safe_int(attrs.get("gen_ai.usage.total_tokens"))

        token_count = oi.TokenCount()
        if prompt_tokens:
            token_count["prompt"] = prompt_tokens
        if completion_tokens:
            token_count["completion"] = completion_tokens
        if total_tokens:
            token_count["total"] = total_tokens

        invocation_params = find_invocation_parameters(attrs)

        output_messages = build_output_messages(output_msgs)

        oi_attrs = {
            **oi_attrs,
            **get_llm_attributes(
                provider=attrs.get("gen_ai.system", "").lower(),
                system=attrs.get("gen_ai.system", "").lower(),
                model_name=attrs.get("gen_ai.request.model") or attrs.get("gen_ai.response.model"),
                input_messages=input_msgs,
                output_messages=output_msgs,
                token_count=token_count,
                invocation_parameters=invocation_params or None,
            ),
            **get_input_attributes(
                {
                    "messages": [
                        {"role": m.get("role"), "content": m.get("content", "")} for m in input_msgs
                    ],
                    "model": attrs.get("gen_ai.request.model"),
                    **invocation_params,
                }
            ),
            **get_output_attributes(
                {
                    "id": attrs.get("gen_ai.response.id"),
                    "messages": output_messages,
                }
            ),
            sc.SpanAttributes.OPENINFERENCE_SPAN_KIND: (sc.OpenInferenceSpanKindValues.LLM.value),
        }

        if invocation_params:
            oi_attrs[sc.SpanAttributes.LLM_INVOCATION_PARAMETERS] = json.dumps(
                invocation_params, separators=(",", ":")
            )
        if span._attributes:
            span._attributes = {**span._attributes, **oi_attrs}
