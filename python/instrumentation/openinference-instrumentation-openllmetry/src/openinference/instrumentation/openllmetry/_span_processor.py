"""
OpenLLMetry → OpenInference Span Processor

This module provides a SpanProcessor that converts OpenLLMetry (TraceLoop) spans
into OpenInference semantic conventions for Phoenix observability.
"""

import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional
from opentelemetry.sdk.trace import SpanProcessor
import openinference.instrumentation as oi
from openinference.instrumentation import (
    get_llm_attributes,
    get_input_attributes,
    get_output_attributes,
)

__all__ = ["OpenLLToOIProcessor"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SPAN_KIND_MAPPING: Dict[str, str] = {
    "workflow": "CHAIN",
    "task":     "TOOL",
    "agent":    "AGENT",
    "tool":     "TOOL",
    "llm":      "LLM",
    "unknown":  "UNKNOWN",
}

_INVOCATION_PARAMETER_KEYS: List[str] = [
    "gen_ai.request.max_tokens",
    "gen_ai.request.temperature",
    "gen_ai.request.top_p",
    "gen_ai.request.top_k",
    "gen_ai.request.stop_sequences",
    "llm.request.max_tokens",
    "llm.request.temperature",
    "llm.request.top_p",
    "llm.request.top_k",
    "llm.request.repetition_penalty",
    "llm.request.stop_sequences",
    "llm.frequency_penalty",
    "llm.presence_penalty",
    "llm.top_k",
    "llm.chat.stop_sequences",
]

_OPENINF_TOOL_LIST_KEY = "llm.tools"

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
    raw_kind = str(attrs.get("traceloop.span.kind", "unknown")).lower()
    kind_val = _SPAN_KIND_MAPPING.get(raw_kind, "UNKNOWN")

    mapped: Dict[str, Any] = {"openinference.span.kind": kind_val}

    input_raw = attrs.get("traceloop.entity.input")
    if input_raw is not None:
        mapped.update({
            "input.mime_type": "application/json",
            "input.value": _as_json_str(input_raw),
        })

    output_raw = attrs.get("traceloop.entity.output")
    if output_raw is not None:
        mapped.update({
            "output.mime_type": "application/json",
            "output.value": _as_json_str(output_raw),
        })

    return mapped


def _collect_oi_messages(attrs: Dict[str, Any], prefix: str) -> List[oi.Message]:
    """
    Reconstruct a list of OpenInference Messages from span attributes
    using the given prefix ("gen_ai.prompt." or "gen_ai.completion.").
    """
    buckets: Dict[int, Dict[str, Any]] = defaultdict(dict)

    for key, val in attrs.items():
        if not key.startswith(prefix):
            continue
        parts = key.split('.')
        if len(parts) < 4 or not parts[2].isdigit():
            continue
        idx = int(parts[2])
        field = parts[3]

        # Handle tool_calls.* entries
        if field == 'tool_calls' and len(parts) >= 6 and parts[4].isdigit():
            tool_idx = int(parts[4])
            sub_field = '.'.join(parts[5:])
            tc_bucket = buckets[idx].setdefault('tool_calls', defaultdict(dict))
            tc_bucket[tool_idx][sub_field] = val
            continue

        buckets[idx][field] = val

    messages: List[oi.Message] = []
    for idx in sorted(buckets):
        raw = buckets[idx]
        role = raw.get('role', 'user')
        msg = oi.Message(role=role)
        if 'content' in raw:
            msg['content'] = raw['content']
        if 'finish_reason' in raw:
            msg['finish_reason'] = raw['finish_reason']

        # Build tool_calls if present

        if calls:= raw.get('tool_calls'):
            oi_calls: List[oi.ToolCall] = []
            for tidx in sorted(calls):
                entry = calls[tidx]
                name = entry.get('function.name') or entry.get('name')
                args = entry.get('function.arguments') or entry.get('arguments')
                call_id = entry.get('id') or entry.get('tool_call.id')
                if not name:
                    continue
                oi_calls.append(
                    oi.ToolCall(
                        id=call_id,
                        function=oi.ToolCallFunction(name=name, arguments=args)
                    )
                )
            if oi_calls:
                msg['tool_calls'] = oi_calls

        messages.append(msg)

    return messages


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
                dst[key] = json.dumps(v, separators=(",", ":")) if isinstance(v, (list, dict)) else v
            oi_tools.append(oi.Tool(json_schema=tool))

    return oi_tools


class OpenLLToOIProcessor(SpanProcessor):
    """
    SpanProcessor that converts OpenLLMetry spans to OpenInference attributes.
    """

    def on_end(self, span) -> None:
        attrs: Dict[str, Any] = getattr(span, '_attributes', {})

        kind = attrs.get('traceloop.span.kind')
        if kind and kind.lower() != 'llm':
            generic = _map_generic_span(attrs)
            attrs.clear()
            attrs.update(generic)
            return

        # Skip if no LLM prompt data
        if not any(k.startswith('gen_ai.prompt.') for k in attrs):
            return

        # Reconstruct messages
        inputs = _collect_oi_messages(attrs, 'gen_ai.prompt.')
        outputs = _collect_oi_messages(attrs, 'gen_ai.completion.')

        # Token usage
        prompt_toks = _safe_int(attrs.get('gen_ai.usage.prompt_tokens')) or 0
        comp_toks   = _safe_int(attrs.get('gen_ai.usage.completion_tokens')) or 0
        total_toks  = _safe_int(attrs.get('llm.usage.total_tokens')) or (prompt_toks + comp_toks)
        cache_read  = _safe_int(attrs.get('gen_ai.usage.cache_read_input_tokens')) or 0
        token_count = oi.TokenCount(
            prompt=prompt_toks,
            completion=comp_toks,
            total=total_toks,
            prompt_details={'cache_read': cache_read},
        )

        # Invocation parameters
        invocation_params: Dict[str, Any] = {}
        for key in _INVOCATION_PARAMETER_KEYS:
            if key in attrs:
                invocation_params[key.rsplit('.', 1)[-1]] = attrs[key]
        if 'gen_ai.request.model' in attrs:
            invocation_params.setdefault('model', attrs['gen_ai.request.model'])

        # Tools
        tool_key = 'llm.request.functions' if 'llm.request.functions' in attrs else (
                   'llm.request.tools' if 'llm.request.tools' in attrs else None)
        oi_tools: List[oi.Tool] = []
        if tool_key:
            oi_tools = _handle_tool_list(attrs[tool_key], attrs)

        # Build bodies for OpenInference
        request_body = {
            'messages': [{'role': m.get('role'), 'content': m.get('content', '')} for m in inputs],
            'model': attrs.get('gen_ai.request.model'),
            'max_tokens': invocation_params.get('llm.request.max_tokens'),
            'temperature': invocation_params.get('llm.request.temperature'),
            'top_p': invocation_params.get('llm.request.top_p'),
            'tools': json.loads(attrs[tool_key]) if tool_key else None,
        }
        assistant_text = outputs[0].get('content', '') if outputs else ''
        response_body = {
            'id': attrs.get('gen_ai.response.id'),
            'choices': [{
                'index': 0,
                'finish_reason': outputs[0].get('finish_reason', 'stop') if outputs else None,
                'message': {
                    'role': outputs[0].get('role', 'assistant') if outputs else 'assistant',
                    'content': assistant_text,
                    'annotations': [],
                }
            }],
            'model': attrs.get('gen_ai.response.model') or attrs.get('gen_ai.request.model'),
            'usage': {'prompt_tokens': prompt_toks, 'completion_tokens': comp_toks, 'total_tokens': total_toks},
        }

        system_val   = attrs.get('gen_ai.system', 'unknown')
        provider_val = attrs.get('gen_ai.provider', system_val)

        # Span kind
        span_val = _SPAN_KIND_MAPPING.get(str(kind).lower(), 'LLM') if kind else 'LLM'

        # Assemble OpenInference attributes
        oi_attrs = {
            'openinference.span.kind': span_val,
            **get_llm_attributes(
                provider=provider_val,
                system=system_val,
                model_name=request_body['model'],
                input_messages=inputs,
                output_messages=outputs,
                token_count=token_count,
                invocation_parameters=invocation_params,
                tools=oi_tools,
            ),
            **get_input_attributes(request_body),
            **get_output_attributes(response_body),
            'llm.invocation_parameters': json.dumps(invocation_params, separators=(",", ":")),
        }

        attrs.clear()
        attrs.update(oi_attrs)
