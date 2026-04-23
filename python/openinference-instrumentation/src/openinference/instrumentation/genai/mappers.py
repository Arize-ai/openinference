# ruff: noqa: E501
"""
Pure-functional mappers from OpenInference flattened span attributes to
OpenTelemetry GenAI semantic convention attributes.

Each mapper:

* takes a ``Mapping[str, AttributeValue]`` of OpenInference attributes,
* returns a new ``Dict[str, AttributeValue]`` of GenAI attributes,
* does not mutate its input,
* returns an empty dict when the inputs it needs are absent.

Mappers are intentionally narrow so they can be unit-tested in isolation and
composed via :func:`convert_oi_to_genai`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, Optional, Tuple

from opentelemetry.util.types import AttributeValue

from openinference.semconv.trace import (
    DocumentAttributes,
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)

from .attributes import GenAIAttributes as GA
from .values import (
    GenAIMessagePartTypeValues,
    GenAIOperationNameValues,
    GenAIProviderNameValues,
)

OIMap = Mapping[str, AttributeValue]
GAMap = Dict[str, AttributeValue]

# ---------------------------------------------------------------------------
# span kind / operation
# ---------------------------------------------------------------------------

_SPAN_KIND_TO_OPERATION: Dict[str, GenAIOperationNameValues] = {
    "EMBEDDING": GenAIOperationNameValues.EMBEDDINGS,
    "RETRIEVER": GenAIOperationNameValues.RETRIEVAL,
    "TOOL": GenAIOperationNameValues.EXECUTE_TOOL,
    "AGENT": GenAIOperationNameValues.INVOKE_AGENT,
}


def map_span_kind(attrs: OIMap) -> GAMap:
    """openinference.span.kind -> gen_ai.operation.name."""
    kind = attrs.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
    if not isinstance(kind, str):
        return {}
    kind = kind.upper()
    if kind == "LLM":
        # OI uses a single LLM kind; disambiguate to text_completion when we
        # see the flattened prompts/choices attributes (the completions API).
        if _has_any_key_prefix(attrs, SpanAttributes.LLM_PROMPTS):
            return {
                GA.GEN_AI_OPERATION_NAME: GenAIOperationNameValues.TEXT_COMPLETION.value
            }
        return {GA.GEN_AI_OPERATION_NAME: GenAIOperationNameValues.CHAT.value}
    op = _SPAN_KIND_TO_OPERATION.get(kind)
    if op is None:
        return {}
    return {GA.GEN_AI_OPERATION_NAME: op.value}


# ---------------------------------------------------------------------------
# model name
# ---------------------------------------------------------------------------


def map_model_name(attrs: OIMap) -> GAMap:
    """llm.model_name / embedding.model_name -> gen_ai.request.model."""
    model = attrs.get(SpanAttributes.LLM_MODEL_NAME) or attrs.get(
        SpanAttributes.EMBEDDING_MODEL_NAME
    )
    if isinstance(model, str) and model:
        return {GA.GEN_AI_REQUEST_MODEL: model}
    return {}


# ---------------------------------------------------------------------------
# provider
# ---------------------------------------------------------------------------

# (system, provider) -> genai provider name
_COMPOSITE_PROVIDER_MAP: Dict[Tuple[Optional[str], Optional[str]], GenAIProviderNameValues] = {
    ("openai", "openai"): GenAIProviderNameValues.OPENAI,
    ("openai", "azure"): GenAIProviderNameValues.AZURE_AI_OPENAI,
    ("anthropic", "anthropic"): GenAIProviderNameValues.ANTHROPIC,
    ("anthropic", "aws"): GenAIProviderNameValues.AWS_BEDROCK,
    ("anthropic", "google"): GenAIProviderNameValues.GCP_VERTEX_AI,
    ("vertexai", "google"): GenAIProviderNameValues.GCP_VERTEX_AI,
    ("cohere", "cohere"): GenAIProviderNameValues.COHERE,
    ("mistralai", "mistralai"): GenAIProviderNameValues.MISTRAL_AI,
    ("deepseek", "deepseek"): GenAIProviderNameValues.DEEPSEEK,
    ("openai", "groq"): GenAIProviderNameValues.GROQ,
}

_SYSTEM_ONLY_PROVIDER_MAP: Dict[str, GenAIProviderNameValues] = {
    "openai": GenAIProviderNameValues.OPENAI,
    "anthropic": GenAIProviderNameValues.ANTHROPIC,
    "cohere": GenAIProviderNameValues.COHERE,
    "mistralai": GenAIProviderNameValues.MISTRAL_AI,
    "vertexai": GenAIProviderNameValues.GCP_VERTEX_AI,
    "deepseek": GenAIProviderNameValues.DEEPSEEK,
    "xai": GenAIProviderNameValues.X_AI,
}

_PROVIDER_ONLY_PROVIDER_MAP: Dict[str, GenAIProviderNameValues] = {
    "openai": GenAIProviderNameValues.OPENAI,
    "anthropic": GenAIProviderNameValues.ANTHROPIC,
    "cohere": GenAIProviderNameValues.COHERE,
    "mistralai": GenAIProviderNameValues.MISTRAL_AI,
    "deepseek": GenAIProviderNameValues.DEEPSEEK,
    "groq": GenAIProviderNameValues.GROQ,
    "perplexity": GenAIProviderNameValues.PERPLEXITY,
    "xai": GenAIProviderNameValues.X_AI,
    "azure": GenAIProviderNameValues.AZURE_AI_OPENAI,
    "aws": GenAIProviderNameValues.AWS_BEDROCK,
    "google": GenAIProviderNameValues.GCP_VERTEX_AI,
}


def map_provider(attrs: OIMap) -> GAMap:
    """(llm.system, llm.provider) -> gen_ai.provider.name."""
    raw_system = attrs.get(SpanAttributes.LLM_SYSTEM)
    raw_provider = attrs.get(SpanAttributes.LLM_PROVIDER)
    system = raw_system.lower() if isinstance(raw_system, str) else None
    provider = raw_provider.lower() if isinstance(raw_provider, str) else None

    resolved: Optional[GenAIProviderNameValues] = None
    if system and provider:
        resolved = _COMPOSITE_PROVIDER_MAP.get((system, provider))
    if resolved is None and system:
        resolved = _SYSTEM_ONLY_PROVIDER_MAP.get(system)
    if resolved is None and provider:
        resolved = _PROVIDER_ONLY_PROVIDER_MAP.get(provider)
    if resolved is None:
        return {}
    return {GA.GEN_AI_PROVIDER_NAME: resolved.value}


# ---------------------------------------------------------------------------
# token counts
# ---------------------------------------------------------------------------

_TOKEN_COUNT_MAP: Dict[str, str] = {
    SpanAttributes.LLM_TOKEN_COUNT_PROMPT: GA.GEN_AI_USAGE_INPUT_TOKENS,
    SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: GA.GEN_AI_USAGE_OUTPUT_TOKENS,
    SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ: GA.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
    SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE: GA.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
}


def map_token_counts(attrs: OIMap) -> GAMap:
    out: GAMap = {}
    for oi_key, genai_key in _TOKEN_COUNT_MAP.items():
        if (value := attrs.get(oi_key)) is not None:
            out[genai_key] = value
    return out


# ---------------------------------------------------------------------------
# invocation parameters
# ---------------------------------------------------------------------------

_INVOCATION_PARAM_ALIASES: Dict[str, Tuple[str, Callable[[Any], Any]]] = {
    "temperature": (GA.GEN_AI_REQUEST_TEMPERATURE, float),
    "top_p": (GA.GEN_AI_REQUEST_TOP_P, float),
    "top_k": (GA.GEN_AI_REQUEST_TOP_K, float),
    "max_tokens": (GA.GEN_AI_REQUEST_MAX_TOKENS, int),
    "max_completion_tokens": (GA.GEN_AI_REQUEST_MAX_TOKENS, int),
    "max_output_tokens": (GA.GEN_AI_REQUEST_MAX_TOKENS, int),
    "frequency_penalty": (GA.GEN_AI_REQUEST_FREQUENCY_PENALTY, float),
    "presence_penalty": (GA.GEN_AI_REQUEST_PRESENCE_PENALTY, float),
    "seed": (GA.GEN_AI_REQUEST_SEED, int),
    "n": (GA.GEN_AI_REQUEST_CHOICE_COUNT, int),
    "choice_count": (GA.GEN_AI_REQUEST_CHOICE_COUNT, int),
}


def map_invocation_parameters(attrs: OIMap) -> GAMap:
    raw = attrs.get(SpanAttributes.LLM_INVOCATION_PARAMETERS)
    if not isinstance(raw, str):
        return {}
    try:
        params = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    if not isinstance(params, dict):
        return {}

    out: GAMap = {}
    for oi_name, (ga_name, caster) in _INVOCATION_PARAM_ALIASES.items():
        if oi_name in params and ga_name not in out:
            try:
                out[ga_name] = caster(params[oi_name])
            except (TypeError, ValueError):
                continue

    # `stop` can be a string or list; GenAI expects a string[].
    if "stop" in params:
        stop = params["stop"]
        if isinstance(stop, str):
            out[GA.GEN_AI_REQUEST_STOP_SEQUENCES] = (stop,)
        elif isinstance(stop, (list, tuple)) and all(isinstance(s, str) for s in stop):
            out[GA.GEN_AI_REQUEST_STOP_SEQUENCES] = tuple(stop)
    return out


# ---------------------------------------------------------------------------
# messages
# ---------------------------------------------------------------------------

_INPUT_MSG_RE = re.compile(r"^llm\.input_messages\.(\d+)\.(.+)$")
_OUTPUT_MSG_RE = re.compile(r"^llm\.output_messages\.(\d+)\.(.+)$")


def map_messages(attrs: OIMap) -> GAMap:
    input_groups = _group_messages(attrs, _INPUT_MSG_RE)
    output_groups = _group_messages(attrs, _OUTPUT_MSG_RE)

    out: GAMap = {}
    system_parts: List[Dict[str, Any]] = []
    input_messages: List[Dict[str, Any]] = []
    for _, fields in sorted(input_groups.items()):
        msg = _build_message(fields)
        if msg is None:
            continue
        if msg.get("role") == "system":
            system_parts.extend(msg.get("parts", []))
            continue
        input_messages.append(msg)

    if system_parts:
        out[GA.GEN_AI_SYSTEM_INSTRUCTIONS] = json.dumps(system_parts, ensure_ascii=False)
    if input_messages:
        out[GA.GEN_AI_INPUT_MESSAGES] = json.dumps(input_messages, ensure_ascii=False)

    output_messages: List[Dict[str, Any]] = []
    for _, fields in sorted(output_groups.items()):
        msg = _build_message(fields)
        if msg is not None:
            output_messages.append(msg)
    if output_messages:
        out[GA.GEN_AI_OUTPUT_MESSAGES] = json.dumps(output_messages, ensure_ascii=False)
    return out


def _group_messages(
    attrs: OIMap, pattern: "re.Pattern[str]"
) -> Dict[int, Dict[str, AttributeValue]]:
    groups: Dict[int, Dict[str, AttributeValue]] = {}
    for key, value in attrs.items():
        m = pattern.match(key)
        if not m:
            continue
        idx = int(m.group(1))
        groups.setdefault(idx, {})[m.group(2)] = value
    return groups


def _build_message(fields: Mapping[str, AttributeValue]) -> Optional[Dict[str, Any]]:
    role = fields.get(MessageAttributes.MESSAGE_ROLE)
    if not isinstance(role, str):
        return None

    parts: List[Dict[str, Any]] = []

    # tool role response
    tool_call_id = fields.get(MessageAttributes.MESSAGE_TOOL_CALL_ID)
    if role == "tool" and isinstance(tool_call_id, str):
        response = fields.get(MessageAttributes.MESSAGE_CONTENT)
        parts.append(
            {
                "type": GenAIMessagePartTypeValues.TOOL_CALL_RESPONSE.value,
                "id": tool_call_id,
                "response": response,
            }
        )
        return {"role": role, "parts": parts}

    # single string content
    content = fields.get(MessageAttributes.MESSAGE_CONTENT)
    if isinstance(content, str):
        parts.append(
            {
                "type": GenAIMessagePartTypeValues.TEXT.value,
                "content": content,
            }
        )

    # structured message.contents
    content_block_fields: Dict[int, Dict[str, AttributeValue]] = {}
    prefix = MessageAttributes.MESSAGE_CONTENTS + "."
    for k, v in fields.items():
        if not k.startswith(prefix):
            continue
        remainder = k[len(prefix) :]
        # remainder looks like "N.rest"
        idx_str, _, rest = remainder.partition(".")
        if not idx_str.isdigit() or not rest:
            continue
        content_block_fields.setdefault(int(idx_str), {})[rest] = v

    for _, block in sorted(content_block_fields.items()):
        part = _build_content_part(block)
        if part is not None:
            parts.append(part)

    # assistant tool calls
    tool_call_fields: Dict[int, Dict[str, AttributeValue]] = {}
    tc_prefix = MessageAttributes.MESSAGE_TOOL_CALLS + "."
    for k, v in fields.items():
        if not k.startswith(tc_prefix):
            continue
        remainder = k[len(tc_prefix) :]
        idx_str, _, rest = remainder.partition(".")
        if not idx_str.isdigit() or not rest:
            continue
        tool_call_fields.setdefault(int(idx_str), {})[rest] = v

    for _, tc in sorted(tool_call_fields.items()):
        part = _build_tool_call_part(tc)
        if part is not None:
            parts.append(part)

    if not parts:
        return None
    return {"role": role, "parts": parts}


def _build_content_part(block: Mapping[str, AttributeValue]) -> Optional[Dict[str, Any]]:
    content_type = block.get(MessageContentAttributes.MESSAGE_CONTENT_TYPE)
    if content_type == "text":
        text = block.get(MessageContentAttributes.MESSAGE_CONTENT_TEXT)
        if isinstance(text, str):
            return {
                "type": GenAIMessagePartTypeValues.TEXT.value,
                "content": text,
            }
        return None
    if content_type == "image":
        url_key = (
            MessageContentAttributes.MESSAGE_CONTENT_IMAGE
            + "."
            + ImageAttributes.IMAGE_URL
        )
        url = block.get(url_key)
        if isinstance(url, str):
            return {
                "type": GenAIMessagePartTypeValues.URI.value,
                "modality": "image",
                "uri": url,
            }
        return None
    if content_type == "audio":
        url = block.get("audio.url") or block.get("message_content.audio.url")
        if isinstance(url, str):
            return {
                "type": GenAIMessagePartTypeValues.URI.value,
                "modality": "audio",
                "uri": url,
            }
        return None
    return None


def _build_tool_call_part(tc: Mapping[str, AttributeValue]) -> Optional[Dict[str, Any]]:
    call_id = tc.get(ToolCallAttributes.TOOL_CALL_ID)
    name = tc.get(ToolCallAttributes.TOOL_CALL_FUNCTION_NAME)
    args_raw = tc.get(ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON)
    if not isinstance(call_id, str) and not isinstance(name, str):
        return None
    part: Dict[str, Any] = {"type": GenAIMessagePartTypeValues.TOOL_CALL.value}
    if isinstance(call_id, str):
        part["id"] = call_id
    if isinstance(name, str):
        part["name"] = name
    if isinstance(args_raw, str):
        try:
            part["arguments"] = json.loads(args_raw)
        except (ValueError, TypeError):
            part["arguments"] = args_raw
    return part


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------

_TOOL_RE = re.compile(r"^llm\.tools\.(\d+)\.tool\.json_schema$")


def map_tools(attrs: OIMap) -> GAMap:
    schemas: List[Tuple[int, Any]] = []
    for key, value in attrs.items():
        m = _TOOL_RE.match(key)
        if not m:
            continue
        if not isinstance(value, str):
            continue
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            parsed = value
        schemas.append((int(m.group(1)), parsed))
    if not schemas:
        return {}
    schemas.sort(key=lambda x: x[0])
    return {
        GA.GEN_AI_TOOL_DEFINITIONS: json.dumps(
            [schema for _, schema in schemas], ensure_ascii=False
        )
    }


# ---------------------------------------------------------------------------
# tool call (execute_tool spans)
# ---------------------------------------------------------------------------


def map_tool_call(attrs: OIMap) -> GAMap:
    if attrs.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) != "TOOL":
        return {}
    out: GAMap = {}
    if isinstance(name := attrs.get(SpanAttributes.TOOL_NAME), str):
        out[GA.GEN_AI_TOOL_NAME] = name
    if isinstance(desc := attrs.get(SpanAttributes.TOOL_DESCRIPTION), str):
        out[GA.GEN_AI_TOOL_DESCRIPTION] = desc
    if isinstance(tool_id := attrs.get(SpanAttributes.TOOL_ID), str):
        out[GA.GEN_AI_TOOL_CALL_ID] = tool_id
    if (args := attrs.get(SpanAttributes.INPUT_VALUE)) is not None:
        out[GA.GEN_AI_TOOL_CALL_ARGUMENTS] = args
    if (result := attrs.get(SpanAttributes.OUTPUT_VALUE)) is not None:
        out[GA.GEN_AI_TOOL_CALL_RESULT] = result
    return out


# ---------------------------------------------------------------------------
# conversation / agent
# ---------------------------------------------------------------------------


def map_conversation(attrs: OIMap) -> GAMap:
    out: GAMap = {}
    if isinstance(sid := attrs.get(SpanAttributes.SESSION_ID), str):
        out[GA.GEN_AI_CONVERSATION_ID] = sid
    return out


def map_agent(attrs: OIMap) -> GAMap:
    out: GAMap = {}
    if isinstance(name := attrs.get(SpanAttributes.AGENT_NAME), str):
        out[GA.GEN_AI_AGENT_NAME] = name
    return out


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------

_RETRIEVAL_DOC_RE = re.compile(r"^retrieval\.documents\.(\d+)\.document\.(.+)$")
_DOC_FIELD_MAP = {
    DocumentAttributes.DOCUMENT_ID.split(".")[-1]: "id",
    DocumentAttributes.DOCUMENT_CONTENT.split(".")[-1]: "content",
    DocumentAttributes.DOCUMENT_METADATA.split(".")[-1]: "metadata",
    DocumentAttributes.DOCUMENT_SCORE.split(".")[-1]: "score",
}


def map_retrieval(attrs: OIMap) -> GAMap:
    docs: Dict[int, Dict[str, Any]] = {}
    for key, value in attrs.items():
        m = _RETRIEVAL_DOC_RE.match(key)
        if not m:
            continue
        idx = int(m.group(1))
        field = m.group(2)
        target = _DOC_FIELD_MAP.get(field)
        if target is None:
            continue
        docs.setdefault(idx, {})[target] = value
    if not docs:
        return {}
    ordered = [docs[i] for i in sorted(docs)]
    return {GA.GEN_AI_RETRIEVAL_DOCUMENTS: json.dumps(ordered, ensure_ascii=False)}


# ---------------------------------------------------------------------------
# top-level composition
# ---------------------------------------------------------------------------

MAPPERS: Tuple[Callable[[OIMap], GAMap], ...] = (
    map_span_kind,
    map_model_name,
    map_provider,
    map_token_counts,
    map_invocation_parameters,
    map_messages,
    map_tools,
    map_tool_call,
    map_conversation,
    map_agent,
    map_retrieval,
)


def convert_oi_to_genai(attrs: OIMap) -> GAMap:
    """
    Run every registered mapper over ``attrs`` and merge their outputs.

    Pure: does not mutate ``attrs`` and produces a fresh dict each call.
    """
    out: GAMap = {}
    for mapper in MAPPERS:
        out.update(mapper(attrs))
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _has_any_key_prefix(attrs: OIMap, prefix: str) -> bool:
    prefix_dot = prefix + "."
    for k in attrs.keys():
        if k == prefix or k.startswith(prefix_dot):
            return True
    return False
