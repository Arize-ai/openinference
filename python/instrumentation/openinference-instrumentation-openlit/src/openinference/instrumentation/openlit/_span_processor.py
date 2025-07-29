# openlit_to_openinference.py

from __future__ import annotations

import json, ast, re
from typing import Any, Dict, Tuple

from opentelemetry.sdk.trace import SpanProcessor, ReadableSpan
import openinference.instrumentation as oi
from openinference.instrumentation import (
    get_llm_attributes,
    get_input_attributes,
    get_output_attributes,
)

import openinference.semconv.trace as sc

__all__ = ["OpenInferenceSpanProcessor"]

# ────────────────────────────────────────────────────────────────────────────────
# 1.  CONSTANTS / "DICTIONARY" MAPPINGS
# ────────────────────────────────────────────────────────────────────────────────
_DIRECT_MAPPING = {
    "gen_ai.system": sc.SpanAttributes.LLM_SYSTEM,          # openai, anthropic, …
    "gen_ai.request.model": sc.SpanAttributes.LLM_MODEL_NAME,
    "gen_ai.response.model": sc.SpanAttributes.LLM_MODEL_NAME,          # collapse to one key
    "gen_ai.operation.name": "llm.request.type",        # chat / embeddings / …
    # token counts
    "gen_ai.usage.input_tokens":  sc.SpanAttributes.LLM_TOKEN_COUNT_PROMPT,
    "gen_ai.usage.output_tokens": sc.SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
    "gen_ai.usage.total_tokens":  sc.SpanAttributes.LLM_TOKEN_COUNT_TOTAL,
}

_INVOC_PARAM_PREFIX = "gen_ai.request."
_EXCLUDE_INVOC_KEYS  = {
    "model", "max_tokens", "temperature", "top_p", "top_k",
    "frequency_penalty", "presence_penalty", "stop_sequences",
}

_PROVIDERS = {
    "openai", 
    "ollama"
    "anthropic",
    "deepseek",
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

def _is_wrapper_span(attrs: dict) -> bool:
    """
    Heuristic: True = a framework / chain / tool span (not the real LLM HTTP call).
    """
    sys_name = str(attrs.get("gen_ai.system", "")).lower()
    addr     = attrs.get("server.address")
    port     = attrs.get("server.port")

    # If we know the provider → it's an LLM span
    if sys_name in _PROVIDERS:
        return False

    # No network endpoint recorded → very likely a local wrapper/tool
    if addr in (None, "", "NOT_FOUND") or port in (None, "", "NOT_FOUND"):
        return True

    # Fallback: treat as LLM
    return False


# ────────────────────────────────────────────────────────────────────────────────
# 2.  HELPERS
# ────────────────────────────────────────────────────────────────────────────────
def _safe_int(v: Any) -> int | None:
    try:
        return int(v)
    except Exception:
        return None
    

# ── ADD just above the span-processor class ──────────────────────────────────
def _tool_call_to_dict(tc: Any) -> dict:
    """
    Convert an oi.ToolCall OR a raw dict into the JSON shape Phoenix expects.
    """
    # already the right shape
    if isinstance(tc, dict):
        return tc

    # oi.ToolCall  ->  dict
    try:
        fn   = tc["function"]
        args = fn["arguments"] if isinstance(fn, dict) else ""
        name = fn["name"]       if isinstance(fn, dict) else ""
        return {
            "id": tc.get("id", ""),
            "type": "function",
            "function": {"name": name, "arguments": args},
        }
    except Exception:
        # fallback – keep something rather than fail the span processor
        return {}


def _parse_prompt_from_events(events) -> str | None:
    for ev in events or []:
        if ev.name == "gen_ai.content.prompt":
            return ev.attributes.get("gen_ai.prompt")
    return None


def _parse_completion_from_events(events) -> str | None:
    for ev in events or []:
        if ev.name == "gen_ai.content.completion":
            return ev.attributes.get("gen_ai.completion")
    return None


_ROLE_LINE_RE = re.compile(r"^(user|assistant|system|tool):\s*(.*)$",
                           re.IGNORECASE)

def _unflatten_prompt(flat: str) -> list[dict]:
    """
    Turn the newline-joined OpenLIT prompt back into a list of
    {'role': str, 'content': str} dictionaries.
    """
    msgs: list[dict] = []
    for line in flat.splitlines():
        m = _ROLE_LINE_RE.match(line.strip())
        if not m:
            if msgs:
                msgs[-1]["content"] += "\n" + line
            continue
        role, content = m.groups()
        msgs.append({"role": role.lower(), "content": content})
    return msgs



def _load_tool_calls(raw: Any) -> list[dict]:
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
        return raw

    # 2) JSON string
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            # 3) python literal – OpenLIT sometimes dumps repr(list)
            try:
                return ast.literal_eval(raw)
            except Exception:
                return []

    # 4) fallback
    try:
        return json.loads(raw)
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

    # ── 1.  INPUT (user) ────────────────────────────────────────────────────
    input_msgs: list[oi.Message] = []
    if prompt:
        for m in _unflatten_prompt(prompt):
            if m.get("content") != "None":
                input_msgs.append(oi.Message(role=m["role"], content=m.get("content", "")))

    # ── 2.  OUTPUT (assistant  +  optional tool_calls) ──────────────────────
    assistant = oi.Message(role="assistant")
    if completion == "None":
        assistant["content"] = ""
    else:
        assistant["content"] = completion
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

    # ── 3.  DONE ────────────────────────────────────────────────────────────
    return input_msgs, output_msgs


# ────────────────────────────────────────────────────────────────────────────────
# 3.  SPAN-PROCESSOR
# ────────────────────────────────────────────────────────────────────────────────
class OpenInferenceSpanProcessor(SpanProcessor):
    """
    Converts OpenLIT GenAI spans → OpenInference attributes in-place.
    Add to your tracer-provider:

        provider.add_span_processor(OpenInferenceSpanProcessor())
    """

    # ---- lifecycle -----------------------------------------------------------
    def on_start(self, span: ReadableSpan, parent_context=None):
        pass

    def shutdown(self):
        return True

    def force_flush(self, timeout_millis: int | None = None):
        return True

    # ---- main work -----------------------------------------------------------
    def on_end(self, span: ReadableSpan) -> None:            # noqa: C901
        attrs: Dict[str, Any] = dict(getattr(span, "_attributes", {}))

        # Set default span kind as backup - will be overridden by specific logic below
        default_oi_attrs = {sc.SpanAttributes.OPENINFERENCE_SPAN_KIND: sc.OpenInferenceSpanKindValues.CHAIN.value}

        # ── Handle tool spans first ──────────────────────────────────
        operation_name = attrs.get("gen_ai.operation.name")
        tool_name = attrs.get("gen_ai.tool.name")
        tool_description = attrs.get("gen_ai.tool.description")
        
        if operation_name == "execute_tool" or (tool_name and tool_description):
            oi_attrs = {
                sc.SpanAttributes.OPENINFERENCE_SPAN_KIND: sc.OpenInferenceSpanKindValues.TOOL.value,
                "span.name": tool_name or operation_name or "tool_execution",
            }
            
            # Set tool attributes in OpenInference format
            if tool_name:
                # oi_attrs["llm.tools.0.name"] = tool_name
                oi_attrs[sc.SpanAttributes.TOOL_NAME] = tool_name
            if tool_description:
                # oi_attrs["llm.tools.0.description"] = tool_description
                oi_attrs[sc.SpanAttributes.TOOL_DESCRIPTION] = tool_description
                
            # Add input/output from events if available
            prompt_txt = _parse_prompt_from_events(span.events)
            completion_txt = _parse_completion_from_events(span.events)
            if prompt_txt:
                oi_attrs.update({
                    sc.SpanAttributes.INPUT_MIME_TYPE: sc.OpenInferenceMimeTypeValues.TEXT.value,
                    sc.SpanAttributes.INPUT_VALUE: prompt_txt,
                })
            if completion_txt:
                oi_attrs.update({
                    sc.SpanAttributes.OUTPUT_MIME_TYPE: sc.OpenInferenceMimeTypeValues.TEXT.value,
                    sc.SpanAttributes.OUTPUT_VALUE: completion_txt,
                })

            span._attributes.clear()                 # type: ignore[attr-defined]
            span._attributes.update(oi_attrs)        # type: ignore[attr-defined]
            return                                    # ✅ done – tool span processed

        # ── Handle wrapper / chain spans quickly ───────────────
        if _is_wrapper_span(attrs):
            oi_attrs = {
                **default_oi_attrs,
                "openinference.span.name": attrs.get("gen_ai.system"),
            }

            prompt_txt     = _parse_prompt_from_events(span.events)
            completion_txt = _parse_completion_from_events(span.events)
            if prompt_txt:
                oi_attrs.update({
                    sc.SpanAttributes.INPUT_MIME_TYPE:  sc.OpenInferenceMimeTypeValues.TEXT.value,
                    sc.SpanAttributes.INPUT_VALUE:      prompt_txt,
                })
            if completion_txt:
                oi_attrs.update({
                    sc.SpanAttributes.OUTPUT_MIME_TYPE: sc.OpenInferenceMimeTypeValues.TEXT.value,
                    sc.SpanAttributes.OUTPUT_VALUE:     completion_txt,
                })

            span._attributes.clear()                 # type: ignore[attr-defined]
            span._attributes.update(oi_attrs)        # type: ignore[attr-defined]
            return                                    # ✅ done – don't run the LLM logic

        # ── Otherwise: this is a provider call → run LLM conversion
        if "gen_ai.system" not in attrs:
            # Set backup attributes for spans without gen_ai.system
            span._attributes.clear()                 # type: ignore[attr-defined]
            span._attributes.update(default_oi_attrs)        # type: ignore[attr-defined]
            return

        # ── 3.1  PROMPT / COMPLETION  ────────────────────────────────────────
        prompt_text      = _parse_prompt_from_events(span.events)
        completion_text  = _parse_completion_from_events(span.events)
        tool_calls_json  = attrs.get("gen_ai.response.tool_calls")

        input_msgs, output_msgs = _build_messages(
            prompt_text,
            completion_text,
            tool_calls_json,
        )

        # ── 3.2  TOKEN-COUNTS  ───────────────────────────────────────────────
        token_count = oi.TokenCount(
            prompt     = _safe_int(attrs.get("gen_ai.usage.input_tokens")),
            completion = _safe_int(attrs.get("gen_ai.usage.output_tokens")),
            total      = _safe_int(attrs.get("gen_ai.usage.total_tokens")),
        )

        # ── 3.3  INVOCATION PARAMETERS (everything under gen_ai.request.*) ───
        invocation: Dict[str, Any] = {}
        for k, v in attrs.items():
            if k.startswith(_INVOC_PARAM_PREFIX):
                leaf = k.split(".", 2)[-1]
                if leaf not in _EXCLUDE_INVOC_KEYS:
                    invocation[leaf] = v

        # add the "simple" knobs so Phoenix shows them
        for knob in ("max_tokens", "temperature", "top_p"):
            val = attrs.get(f"gen_ai.request.{knob}")
            if val not in (None, "", -1):
                invocation[knob] = val

        out_msgs_json: list[dict] = []
        for m in output_msgs:
            mdict = dict(m)

            # drop falsy content keys
            if not mdict.get("content"):
                mdict.pop("content", None)

            # normalise tool-calls …
            if "tool_calls" in mdict:
                mdict["tool_calls"] = [_tool_call_to_dict(tc) for tc in mdict["tool_calls"]]

            out_msgs_json.append(mdict)

        

        # ── 3.4  OPENINFERENCE ATTRIBUTE PACKS  ──────────────────────────────
        oi_attrs = {
            # Start with default attributes as backup
            **default_oi_attrs,
            
            # 3.4-1 direct key moves (model, system, token counts …)
            **{
                _DIRECT_MAPPING[k]: v
                for k, v in attrs.items()
                if k in _DIRECT_MAPPING
            },

            # 3.4-2 llm.*
            **get_llm_attributes(
                provider      = attrs.get("gen_ai.system", "").lower(),
                system        = attrs.get("gen_ai.system", "").lower(),
                model_name    = attrs.get("gen_ai.request.model")
                                or attrs.get("gen_ai.response.model"),
                input_messages  = input_msgs,
                output_messages = output_msgs,
                token_count     = token_count,
                invocation_parameters = invocation or None,
            ),

            # 3.4-3 input.* / output.*
            **get_input_attributes(
                {
                    "messages": [
                        {"role": m.get("role"), "content": m.get("content", "")}
                        for m in input_msgs
                    ],
                    "model": attrs.get("gen_ai.request.model"),
                    **invocation,                      # include the knobs
                }
            ),
            **get_output_attributes(
                {              
                    "id": attrs.get("gen_ai.response.id"),
                    "messages": out_msgs_json,
                }
            ),

            # 3.4-4 span kind for Phoenix navigation panel
            sc.SpanAttributes.OPENINFERENCE_SPAN_KIND: sc.OpenInferenceSpanKindValues.LLM.value,
        }

        # explicit JSON dump for invocation_parameters (UI convenience)
        if invocation:
            oi_attrs[sc.SpanAttributes.LLM_INVOCATION_PARAMETERS] = json.dumps(
                invocation, separators=(",", ":")
            )

        # ── 3.5  REPLACE ORIGINAL ATTRIBUTES  ────────────────────────────────
        span._attributes.clear()                        # type: ignore[attr-defined]
        span._attributes.update(oi_attrs)               # type: ignore[attr-defined]
