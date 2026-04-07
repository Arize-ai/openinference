"""
OpenLLMetry → OpenInference Span Processor

This module provides a SpanProcessor that converts OpenLLMetry (TraceLoop) spans
into OpenInference semantic conventions for Phoenix observability.
"""

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

# Import OpenLLMetry constants from the official package
from opentelemetry.semconv_ai import SpanAttributes, TraceloopSpanKindValues

import openinference.instrumentation as oi

# Import OpenInference constants
import openinference.semconv.trace as sc
from openinference.instrumentation import (
    get_input_attributes,
    get_llm_attributes,
    get_output_attributes,
)

__all__ = ["OpenInferenceSpanProcessor"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SPAN_KIND_MAPPING: Dict[str, str] = {
    TraceloopSpanKindValues.WORKFLOW.value: sc.OpenInferenceSpanKindValues.CHAIN.value,
    TraceloopSpanKindValues.TASK.value: sc.OpenInferenceSpanKindValues.TOOL.value,
    TraceloopSpanKindValues.AGENT.value: sc.OpenInferenceSpanKindValues.AGENT.value,
    TraceloopSpanKindValues.TOOL.value: sc.OpenInferenceSpanKindValues.TOOL.value,
    "llm": sc.OpenInferenceSpanKindValues.LLM.value,
    TraceloopSpanKindValues.UNKNOWN.value: sc.OpenInferenceSpanKindValues.UNKNOWN.value,
}

_INVOCATION_PARAMETER_KEYS: List[str] = [
    GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS,
    GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE,
    GenAIAttributes.GEN_AI_REQUEST_TOP_P,
    SpanAttributes.LLM_TOP_K,
    SpanAttributes.LLM_CHAT_STOP_SEQUENCES,
    SpanAttributes.LLM_REQUEST_REPETITION_PENALTY,
    SpanAttributes.LLM_FREQUENCY_PENALTY,
    SpanAttributes.LLM_PRESENCE_PENALTY,
]

_OPENINF_TOOL_LIST_KEY = "llm.tools"

# GenAI semconv attribute keys (v0.55.0+ default format)
_GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
_GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
_GEN_AI_TOOL_DEFINITIONS = "gen_ai.tool.definitions"

_VALID_LLM_PROVIDERS = frozenset(v.value for v in sc.OpenInferenceLLMProviderValues)
_VALID_LLM_SYSTEMS = frozenset(v.value for v in sc.OpenInferenceLLMSystemValues)

_TOOL_KEY_CANDIDATES = [
    SpanAttributes.LLM_REQUEST_FUNCTIONS,
    "llm.request.tools",
    _GEN_AI_TOOL_DEFINITIONS,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _as_json_str(value: Any) -> str:
    """
    Ensure the given value is serialized as a compact JSON string.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"))


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _map_generic_span(attrs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert TraceLoop 'workflow' / 'task' / 'agent' / 'tool' spans
    to OpenInference semantic conventions.
    """
    raw_kind = str(attrs.get(SpanAttributes.TRACELOOP_SPAN_KIND, "unknown")).lower()
    kind_val = _SPAN_KIND_MAPPING.get(raw_kind, sc.OpenInferenceSpanKindValues.UNKNOWN.value)

    mapped: Dict[str, Any] = {"openinference.span.kind": kind_val}

    input_raw = attrs.get(SpanAttributes.TRACELOOP_ENTITY_INPUT)
    if input_raw is not None:
        mapped.update(
            {
                "input.mime_type": "application/json",
                "input.value": _as_json_str(input_raw),
            }
        )

    output_raw = attrs.get(SpanAttributes.TRACELOOP_ENTITY_OUTPUT)
    if output_raw is not None:
        mapped.update(
            {
                "output.mime_type": "application/json",
                "output.value": _as_json_str(output_raw),
            }
        )

    return mapped


def _parse_messages_from_attributes(
    attrs: Dict[str, Any], prefix: str
) -> tuple[List[oi.Message], List[Optional[str]]]:
    """
    Reconstruct a list of OpenInference Messages from span attributes
    using the given prefix ("gen_ai.prompt." or "gen_ai.completion.").
    """
    buckets: Dict[int, Dict[str, Any]] = defaultdict(dict)

    for key, val in attrs.items():
        if not key.startswith(prefix):
            continue
        parts = key.split(".")
        if len(parts) < 4 or not parts[2].isdigit():
            continue
        idx = int(parts[2])
        field = parts[3]

        # Handle tool_calls.* entries
        if field == "tool_calls" and len(parts) >= 6 and parts[4].isdigit():
            tool_idx = int(parts[4])
            sub_field = ".".join(parts[5:])
            tc_bucket = buckets[idx].setdefault("tool_calls", defaultdict(dict))
            tc_bucket[tool_idx][sub_field] = val
            continue

        buckets[idx][field] = val

    messages: List[oi.Message] = []
    finish_reasons: List[Optional[str]] = []

    for idx in sorted(buckets):
        raw = buckets[idx]
        role = raw.get("role", "user")
        msg = oi.Message(role=role)
        if "content" in raw:
            msg["content"] = raw["content"]
        # Note: finish_reason is not part of the Message TypedDict, so we handle it separately
        finish_reason = raw.get("finish_reason")

        # Build tool_calls if present

        if calls := raw.get("tool_calls"):
            oi_calls: List[oi.ToolCall] = []
            for tidx in sorted(calls):
                entry = calls[tidx]
                name = entry.get("function.name") or entry.get("name")
                args = entry.get("function.arguments") or entry.get("arguments")
                call_id = entry.get("id") or entry.get("tool_call.id")
                if not name:
                    continue
                oi_calls.append(
                    oi.ToolCall(id=call_id, function=oi.ToolCallFunction(name=name, arguments=args))
                )
            if oi_calls:
                msg["tool_calls"] = oi_calls

        messages.append(msg)
        finish_reasons.append(finish_reason)

    return messages, finish_reasons


def _parse_messages_from_json(
    raw_json: str,
) -> tuple[List[oi.Message], List[Optional[str]]]:
    """
    Parse the updated ``gen_ai.input.messages`` / ``gen_ai.output.messages``
    JSON-string attribute (OTel GenAI semconv 0.5.1+).

    Each message is ``{"role": "...", "parts": [{"type": "text", "content": "..."},
    {"type": "tool_call", ...}], "finish_reason": "..."}``
    """
    try:
        items = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
    except Exception:
        return [], []
    if not isinstance(items, list):
        items = [items]

    messages: List[oi.Message] = []
    finish_reasons: List[Optional[str]] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        role = item.get("role", "user")
        msg = oi.Message(role=role)

        parts = item.get("parts") or []
        text_parts: List[str] = []
        tool_calls: List[oi.ToolCall] = []

        for part in parts:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type", "")
            if ptype == "text":
                text_parts.append(part.get("content", ""))
            elif ptype == "tool_call":
                tc = oi.ToolCall(
                    function=oi.ToolCallFunction(
                        name=part.get("name", ""),
                        arguments=part.get("arguments", ""),
                    )
                )
                if part.get("id"):
                    tc["id"] = part["id"]
                tool_calls.append(tc)
            elif ptype == "tool_call_response":
                # tool role messages carry the response as content
                text_parts.append(str(part.get("response", "")))

        # If no parts array, fall back to top-level content
        if not parts and "content" in item:
            text_parts.append(str(item["content"]))

        if text_parts:
            msg["content"] = "\n".join(text_parts) if len(text_parts) > 1 else text_parts[0]
        if tool_calls:
            msg["tool_calls"] = tool_calls

        messages.append(msg)
        finish_reasons.append(item.get("finish_reason"))

    return messages, finish_reasons


def _handle_tool_list(raw: Any, dst: Dict[str, Any]) -> List[oi.Tool]:
    """
    Convert OpenLLMetry functions/tools list into OpenInference tools list
    and set the appropriate span attributes in dst.
    """
    try:
        tools_py = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return []
    if not isinstance(tools_py, list):
        tools_py = [tools_py]

    dst[_OPENINF_TOOL_LIST_KEY] = json.dumps(tools_py, separators=(",", ":"))
    oi_tools: List[oi.Tool] = []

    for idx, tool in enumerate(tools_py):
        if isinstance(tool, dict):
            base = f"{_OPENINF_TOOL_LIST_KEY}.{idx}"
            for k, v in tool.items():
                key = f"{base}.{k}"
                dst[key] = (
                    json.dumps(v, separators=(",", ":")) if isinstance(v, (list, dict)) else v
                )
            oi_tools.append(oi.Tool(json_schema=tool))

    return oi_tools


def _extract_llm_provider_and_system(
    attrs: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    """Extract validated OpenInference LLM provider and system values from span attributes."""
    provider_val: Optional[str] = str(
        attrs.get(GenAIAttributes.GEN_AI_PROVIDER_NAME, "unknown")
    ).lower()
    if provider_val not in _VALID_LLM_PROVIDERS:
        provider_val = None

    # gen_ai.system is deprecated (OTel semconv v1.37.0); v0.55.0+ only emits
    # gen_ai.provider.name, so system_val will be None for newer spans.
    system_val: Optional[str] = str(attrs.get(GenAIAttributes.GEN_AI_SYSTEM, "unknown")).lower()
    if system_val not in _VALID_LLM_SYSTEMS:
        system_val = None

    return provider_val, system_val


class OpenInferenceSpanProcessor(SpanProcessor):
    """
    SpanProcessor that converts OpenLLMetry spans to OpenInference attributes.
    """

    def on_end(self, span: Any) -> None:
        attrs: Dict[str, Any] = getattr(span, "_attributes", {})

        kind = attrs.get(SpanAttributes.TRACELOOP_SPAN_KIND)
        if kind and kind.lower() != "llm":
            generic = _map_generic_span(attrs)
            attrs.clear()
            attrs.update(generic)
            return

        # Detect which message format is present.
        # The JSON-based format (v0.55.0+) is the default; the legacy
        # attribute-per-field format is kept only as a fallback.
        has_json_messages = _GEN_AI_INPUT_MESSAGES in attrs
        has_legacy_attributes = any(k.startswith("gen_ai.prompt.") for k in attrs)

        # Skip if no LLM prompt data in either format
        if not has_json_messages and not has_legacy_attributes:
            return

        # Reconstruct messages, preferring the current format
        if has_json_messages:
            inputs, input_finish_reasons = _parse_messages_from_json(
                attrs.get(_GEN_AI_INPUT_MESSAGES, "[]")
            )
            outputs, output_finish_reasons = _parse_messages_from_json(
                attrs.get(_GEN_AI_OUTPUT_MESSAGES, "[]")
            )
        else:
            # Fallback for older OpenLLMetry versions (< 0.55.0)
            inputs, input_finish_reasons = _parse_messages_from_attributes(attrs, "gen_ai.prompt.")
            outputs, output_finish_reasons = _parse_messages_from_attributes(
                attrs, "gen_ai.completion."
            )

        # Token usage
        prompt_toks = _safe_int(attrs.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)) or 0
        comp_toks = _safe_int(attrs.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS)) or 0
        total_toks = _safe_int(attrs.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)) or (
            prompt_toks + comp_toks
        )
        cache_read = _safe_int(attrs.get(SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS)) or 0
        token_count = oi.TokenCount(
            prompt=prompt_toks,
            completion=comp_toks,
            total=total_toks,
            prompt_details={"cache_read": cache_read},
        )

        # Invocation parameters
        invocation_params: Dict[str, Any] = {}
        for key in _INVOCATION_PARAMETER_KEYS:
            if key in attrs:
                invocation_params[key.rsplit(".", 1)[-1]] = attrs[key]
        if GenAIAttributes.GEN_AI_REQUEST_MODEL in attrs:
            invocation_params.setdefault("model", attrs[GenAIAttributes.GEN_AI_REQUEST_MODEL])
        # Tools
        tool_key = next((k for k in _TOOL_KEY_CANDIDATES if k in attrs), None)
        oi_tools: List[oi.Tool] = []
        if tool_key:
            oi_tools = _handle_tool_list(attrs[tool_key], attrs)

        # Build bodies for OpenInference
        request_body = {
            "messages": [{"role": m.get("role"), "content": m.get("content", "")} for m in inputs],
            "model": attrs.get(GenAIAttributes.GEN_AI_REQUEST_MODEL),
            "max_tokens": invocation_params.get("max_tokens"),
            "temperature": invocation_params.get("temperature"),
            "top_p": invocation_params.get("top_p"),
            "tools": json.loads(attrs[tool_key]) if tool_key else None,
        }
        assistant_text = outputs[0].get("content", "") if outputs else ""
        finish_reason = output_finish_reasons[0] if output_finish_reasons else "stop"
        response_body = {
            "id": attrs.get("gen_ai.response.id"),
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {
                        "role": outputs[0].get("role", "assistant") if outputs else "assistant",
                        "content": assistant_text,
                        "annotations": [],
                    },
                }
            ],
            "model": attrs.get(GenAIAttributes.GEN_AI_RESPONSE_MODEL)
            or attrs.get(GenAIAttributes.GEN_AI_REQUEST_MODEL),
            "usage": {
                "prompt_tokens": prompt_toks,
                "completion_tokens": comp_toks,
                "total_tokens": total_toks,
            },
        }

        provider_val, system_val = _extract_llm_provider_and_system(attrs)

        # Span kind
        span_val = (
            _SPAN_KIND_MAPPING.get(str(kind).lower(), sc.OpenInferenceSpanKindValues.LLM.value)
            if kind
            else sc.OpenInferenceSpanKindValues.LLM.value
        )

        # Assemble OpenInference attributes
        oi_attrs = {
            sc.SpanAttributes.OPENINFERENCE_SPAN_KIND: span_val,
            **get_llm_attributes(
                provider=provider_val,
                system=system_val,
                model_name=request_body["model"],
                input_messages=inputs,
                output_messages=outputs,
                token_count=token_count,
                invocation_parameters=invocation_params,
                tools=oi_tools,
            ),
            **get_input_attributes(request_body),
            **get_output_attributes(response_body),
        }

        attrs.update(oi_attrs)
