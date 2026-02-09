import ast
import json
import re
from typing import Any, Dict, Optional, Tuple

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor

import openinference.instrumentation as oi
import openinference.semconv.trace as sc
from openinference.instrumentation import (
    get_input_attributes,
    get_llm_attributes,
    get_output_attributes,
    get_span_kind_attributes,
)

__all__ = ["OpenInferenceSpanProcessor"]

_LLM_PROVIDERS = {
    "ai21",
    "amazon_bedrock",
    "anthropic",
    "assembly_ai",
    "azure_ai_inference",
    "azure_openai",
    "cohere",
    "elevenlabs",
    "featherless",
    "github_models",
    "google_ai_studio",
    "gpt4all",
    "groq",
    "huggingface",
    "mistral",
    "nvidia_nim",
    "ola_krutrim",
    "ollama",
    "openai",
    "prem_ai",
    "reka_ai",
    "titan_ml",
    "together",
    "vector_dbs",
    "vertex_ai",
    "vllm",
    "xai",
}


class OpenInferenceSpanProcessor(SpanProcessor):
    """
    Augments OpenLIT GenAI spans with the corresponding OpenInference attributes.
    Add to your tracer-provider:
        provider.add_span_processor(OpenInferenceSpanProcessor())
    """

    def on_end(self, span: ReadableSpan) -> None:
        attrs: Dict[str, Any] = dict(getattr(span, "_attributes", {}))
        oi_attrs = {}

        if _is_tool_span(attrs):
            oi_attrs = _get_oi_tool_attributes(span)
        elif _is_chain_span(attrs):
            oi_attrs = _get_oi_chain_attributes(span)
        elif _is_llm_span(attrs):
            oi_attrs = _get_llm_attributes(span)

        if oi_attrs and span._attributes:
            span._attributes = {**span._attributes, **oi_attrs}


def _is_chain_span(attrs: Dict[str, Any]) -> bool:
    """
    Heuristic:
    True = a framework / chain / tool span.
    False: LLM span.
    """
    sys_name = str(attrs.get("gen_ai.system", "")).lower()
    addr = attrs.get("server.address")
    port = attrs.get("server.port")

    if sys_name in _LLM_PROVIDERS:
        return False

    if addr in (None, "", "NOT_FOUND") or port in (None, "", "NOT_FOUND"):
        return True

    return False


def _is_tool_span(attrs: dict[str, Any]) -> bool:
    operation_name = attrs.get("gen_ai.operation.name")
    tool_name = attrs.get("gen_ai.tool.name")
    return operation_name == "execute_tool" or tool_name is not None


def _is_llm_span(attrs: dict[str, Any]) -> bool:
    return "gen_ai.system" in attrs


def _int_safe(v: Any) -> int | None:
    try:
        return int(v)
    except Exception:
        return None


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


def _load_tool_calls(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []

    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
        return []
    except Exception:
        # python literal – OpenLIT sometimes dumps repr(list)
        try:
            result = ast.literal_eval(raw)
            if isinstance(result, list):
                return result
            return []
        except Exception:
            return []


def _build_messages(
    prompt: str | None,
    completion: str | None,
    tool_calls_json: Any | None,
) -> Tuple[list[oi.Message], list[oi.Message]]:
    """
    Convert OpenLIT's plain-text fields into OpenInference message objects.
    Returns (input_messages, output_messages).
    """

    input_msgs: list[oi.Message] = []
    if prompt:
        for m in _unflatten_prompt(prompt):
            content = m.get("content", "")
            if content != "" and content is not None:
                input_msgs.append(oi.Message(role=m["role"], content=str(content)))

    assistant = oi.Message(role="assistant")
    if completion == "None":
        assistant["content"] = ""
    else:
        assistant["content"] = str(completion) if completion is not None else ""
    tc_list = _load_tool_calls(tool_calls_json)
    if tc_list:
        assistant["tool_calls"] = tc_list  # type: ignore

    output_msgs = [assistant]
    return input_msgs, output_msgs


def parse_messages(events: Any, attrs: Dict[str, Any]) -> Tuple[list[oi.Message], list[oi.Message]]:
    prompt_text = _parse_prompt_from_events(events)
    completion_text = _parse_completion_from_events(events)
    tool_calls_json = attrs.get("gen_ai.response.tool_calls")
    return _build_messages(prompt_text, completion_text, tool_calls_json)


def _get_oi_tool_attributes(span: ReadableSpan) -> dict[str, Any]:
    openlit_attrs = dict(getattr(span, "_attributes", {}))
    oi_attrs: dict[str, Any] = {
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

    oi_attrs.update(get_span_kind_attributes("tool"))
    return oi_attrs


def find_invocation_parameters(attrs: Dict[str, Any]) -> Dict[str, Any]:
    invocation: Dict[str, Any] = {}
    for parameter in ("model", "temperature", "max_tokens", "top_p", "top_k"):
        val = attrs.get(f"gen_ai.request.{parameter}")
        if val not in (None, ""):
            invocation[parameter] = val
    return invocation


def _get_oi_chain_attributes(span: ReadableSpan) -> Dict[str, Any]:
    oi_attrs = {}
    prompt_txt = _parse_prompt_from_events(span.events)
    completion_txt = _parse_completion_from_events(span.events)
    if prompt_txt:
        oi_attrs.update(
            get_input_attributes(prompt_txt, mime_type=sc.OpenInferenceMimeTypeValues.TEXT.value)
        )
    if completion_txt:
        oi_attrs.update(
            get_output_attributes(
                completion_txt, mime_type=sc.OpenInferenceMimeTypeValues.TEXT.value
            )
        )
    oi_attrs.update(get_span_kind_attributes("chain"))

    return oi_attrs


def _get_llm_attributes(span: ReadableSpan) -> dict[str, Any]:
    attrs = dict(getattr(span, "_attributes", {}))
    input_msgs, output_msgs = parse_messages(span.events, attrs)
    prompt_tokens = _int_safe(attrs.get("gen_ai.usage.input_tokens"))
    completion_tokens = _int_safe(attrs.get("gen_ai.usage.output_tokens"))
    total_tokens = _int_safe(attrs.get("gen_ai.usage.total_tokens"))

    token_count = oi.TokenCount()
    if prompt_tokens:
        token_count["prompt"] = prompt_tokens
    if completion_tokens:
        token_count["completion"] = completion_tokens
    if total_tokens:
        token_count["total"] = total_tokens

    invocation_params = find_invocation_parameters(attrs)
    provider_val, system_val = _extract_llm_provider_and_system(attrs)

    oi_attrs = {
        **get_llm_attributes(
            provider=provider_val,
            system=system_val,
            model_name=(
                attrs.get("gen_ai.llm.model")
                or attrs.get("gen_ai.request.model")
                or attrs.get("gen_ai.response.model")
            ),
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
                "messages": output_msgs,
            }
        ),
        **get_span_kind_attributes("llm"),
    }
    return oi_attrs


def _extract_llm_provider_and_system(
    attrs: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    """Extract validated OpenInference LLM provider and system values from span attributes."""
    provider_val: Optional[str] = str(attrs.get("gen_ai.llm.provider", "unknown")).lower()
    if provider_val not in {v.value for v in sc.OpenInferenceLLMProviderValues}:
        provider_val = None

    system_val: Optional[str] = str(attrs.get("gen_ai.system", "unknown")).lower()
    if system_val not in {v.value for v in sc.OpenInferenceLLMSystemValues}:
        system_val = None

    return provider_val, system_val
