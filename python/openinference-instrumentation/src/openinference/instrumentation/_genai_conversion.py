import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple

from opentelemetry.util.types import AttributeValue

from openinference.semconv.trace import (
    ChoiceAttributes,
    DocumentAttributes,
    EmbeddingAttributes,
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceLLMProviderValues,
    OpenInferenceLLMSystemValues,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    PromptAttributes,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)

from ._genai_attributes import (
    GenAIAttributes,
    GenAIMessagePartTypeValues,
    GenAIModalityValues,
    GenAIOperationNameValues,
    GenAIOutputTypeValues,
    GenAIProviderNameValues,
    GenAIRoleValues,
    GenAIToolTypeValues,
)
from .helpers import safe_json_dumps

_DATA_URL_PATTERN = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<content>.+)$")


def get_genai_attributes(attributes: Mapping[str, AttributeValue]) -> Dict[str, AttributeValue]:
    output_messages = _get_output_messages(attributes)
    return {
        **get_genai_base_attributes(attributes),
        **get_genai_request_attributes(attributes),
        **get_genai_usage_attributes(attributes),
        **get_genai_message_attributes(attributes, _output_messages=output_messages),
        **get_genai_response_attributes(attributes, _output_messages=output_messages),
        **get_genai_tool_attributes(attributes),
        **get_genai_retrieval_attributes(attributes),
        **get_genai_embedding_attributes(attributes),
    }


def get_genai_base_attributes(
    attributes: Mapping[str, AttributeValue],
) -> Dict[str, AttributeValue]:
    genai_attributes: Dict[str, AttributeValue] = {}
    if operation_name := _get_genai_operation_name(attributes):
        genai_attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] = operation_name
    if provider_name := _get_genai_provider_name(attributes):
        genai_attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] = provider_name
    if conversation_id := _as_optional_str(attributes.get(SpanAttributes.SESSION_ID)):
        genai_attributes[GenAIAttributes.GEN_AI_CONVERSATION_ID] = conversation_id
    return genai_attributes


def get_genai_request_attributes(
    attributes: Mapping[str, AttributeValue],
) -> Dict[str, AttributeValue]:
    genai_attributes: Dict[str, AttributeValue] = {}

    llm_parameters = _parse_json_mapping(attributes.get(SpanAttributes.LLM_INVOCATION_PARAMETERS))
    embedding_parameters = _parse_json_mapping(
        attributes.get(SpanAttributes.EMBEDDING_INVOCATION_PARAMETERS)
    )
    invocation_parameters = {**embedding_parameters, **llm_parameters}

    if request_model := _get_request_model(attributes, llm_parameters, embedding_parameters):
        genai_attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] = request_model

    for genai_key, raw_value in _iter_request_parameter_mappings(invocation_parameters):
        genai_attributes[genai_key] = raw_value

    if output_type := _get_output_type(invocation_parameters):
        genai_attributes[GenAIAttributes.GEN_AI_OUTPUT_TYPE] = output_type

    return genai_attributes


def get_genai_usage_attributes(
    attributes: Mapping[str, AttributeValue],
) -> Dict[str, AttributeValue]:
    genai_attributes: Dict[str, AttributeValue] = {}

    for oi_key, genai_key in (
        (SpanAttributes.LLM_TOKEN_COUNT_PROMPT, GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS),
        (SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS),
        (
            SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ,
            GenAIAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
        ),
        (
            SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE,
            GenAIAttributes.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
        ),
    ):
        value = attributes.get(oi_key)
        if value is not None:
            genai_attributes[genai_key] = value

    return genai_attributes


def get_genai_message_attributes(
    attributes: Mapping[str, AttributeValue],
    *,
    _output_messages: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, AttributeValue]:
    genai_attributes: Dict[str, AttributeValue] = {}

    input_messages = _get_input_messages(attributes)
    output_messages = _output_messages if _output_messages is not None else _get_output_messages(
        attributes
    )

    if input_messages:
        genai_attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES] = safe_json_dumps(input_messages)
    if output_messages:
        genai_attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES] = safe_json_dumps(output_messages)

    return genai_attributes


def get_genai_response_attributes(
    attributes: Mapping[str, AttributeValue],
    *,
    _output_messages: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, AttributeValue]:
    finish_reasons = _get_response_finish_reasons(attributes, _output_messages=_output_messages)
    if not finish_reasons:
        return {}
    return {
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS: tuple(finish_reasons),
    }


def get_genai_tool_attributes(
    attributes: Mapping[str, AttributeValue],
) -> Dict[str, AttributeValue]:
    genai_attributes: Dict[str, AttributeValue] = {}

    tool_definitions = _get_tool_definitions(attributes)
    if tool_definitions:
        genai_attributes[GenAIAttributes.GEN_AI_TOOL_DEFINITIONS] = safe_json_dumps(
            tool_definitions
        )

    if _get_oi_span_kind(attributes) != OpenInferenceSpanKindValues.TOOL.value:
        return genai_attributes

    if tool_name := _as_optional_str(attributes.get(SpanAttributes.TOOL_NAME)):
        genai_attributes[GenAIAttributes.GEN_AI_TOOL_NAME] = tool_name
        genai_attributes[GenAIAttributes.GEN_AI_TOOL_TYPE] = GenAIToolTypeValues.FUNCTION.value
    if tool_description := _as_optional_str(attributes.get(SpanAttributes.TOOL_DESCRIPTION)):
        genai_attributes[GenAIAttributes.GEN_AI_TOOL_DESCRIPTION] = tool_description

    tool_call_id = _as_optional_str(attributes.get(SpanAttributes.TOOL_ID)) or _as_optional_str(
        attributes.get(ToolCallAttributes.TOOL_CALL_ID)
    )
    if tool_call_id:
        genai_attributes[GenAIAttributes.GEN_AI_TOOL_CALL_ID] = tool_call_id

    if tool_arguments := _get_tool_call_arguments(attributes):
        genai_attributes[GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS] = tool_arguments
    if tool_result := _get_tool_call_result(attributes):
        genai_attributes[GenAIAttributes.GEN_AI_TOOL_CALL_RESULT] = tool_result

    return genai_attributes


def get_genai_retrieval_attributes(
    attributes: Mapping[str, AttributeValue],
) -> Dict[str, AttributeValue]:
    if _get_oi_span_kind(attributes) != OpenInferenceSpanKindValues.RETRIEVER.value:
        return {}

    genai_attributes: Dict[str, AttributeValue] = {}
    if documents := _get_documents(attributes, SpanAttributes.RETRIEVAL_DOCUMENTS):
        genai_attributes[GenAIAttributes.GEN_AI_RETRIEVAL_DOCUMENTS] = safe_json_dumps(documents)

    query_text = _get_retrieval_query_text(attributes)
    if query_text is not None:
        genai_attributes[GenAIAttributes.GEN_AI_RETRIEVAL_QUERY_TEXT] = query_text

    return genai_attributes


def get_genai_embedding_attributes(
    attributes: Mapping[str, AttributeValue],
) -> Dict[str, AttributeValue]:
    if _get_oi_span_kind(attributes) != OpenInferenceSpanKindValues.EMBEDDING.value:
        return {}

    if dimension_count := _get_embedding_dimension_count(attributes):
        return {
            GenAIAttributes.GEN_AI_EMBEDDINGS_DIMENSION_COUNT: dimension_count,
        }
    return {}


def _get_oi_span_kind(attributes: Mapping[str, AttributeValue]) -> Optional[str]:
    span_kind = attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
    if isinstance(span_kind, str):
        return span_kind.upper()
    return None


def _get_genai_operation_name(attributes: Mapping[str, AttributeValue]) -> Optional[str]:
    span_kind = _get_oi_span_kind(attributes)
    if span_kind == OpenInferenceSpanKindValues.TOOL.value:
        return GenAIOperationNameValues.EXECUTE_TOOL.value
    if span_kind == OpenInferenceSpanKindValues.EMBEDDING.value:
        return GenAIOperationNameValues.EMBEDDINGS.value
    if span_kind == OpenInferenceSpanKindValues.RETRIEVER.value:
        return GenAIOperationNameValues.RETRIEVAL.value
    if span_kind == OpenInferenceSpanKindValues.AGENT.value:
        return GenAIOperationNameValues.INVOKE_AGENT.value
    if span_kind == OpenInferenceSpanKindValues.LLM.value:
        if _has_prompt_completion_attributes(attributes):
            return GenAIOperationNameValues.TEXT_COMPLETION.value
        return GenAIOperationNameValues.CHAT.value
    return None


def _has_prompt_completion_attributes(attributes: Mapping[str, AttributeValue]) -> bool:
    if SpanAttributes.LLM_PROMPTS in attributes or SpanAttributes.LLM_CHOICES in attributes:
        return True
    return any(
        key.startswith(f"{SpanAttributes.LLM_PROMPTS}.")
        or key.startswith(f"{SpanAttributes.LLM_CHOICES}.")
        for key in attributes
    )


def _get_genai_provider_name(attributes: Mapping[str, AttributeValue]) -> Optional[str]:
    provider = _normalize_provider_or_system(attributes.get(SpanAttributes.LLM_PROVIDER))
    system = _normalize_provider_or_system(attributes.get(SpanAttributes.LLM_SYSTEM))

    if provider == OpenInferenceLLMProviderValues.AZURE.value:
        if system == OpenInferenceLLMSystemValues.OPENAI.value:
            return GenAIProviderNameValues.AZURE_AI_OPENAI.value
        return GenAIProviderNameValues.AZURE_AI_INFERENCE.value

    if provider == OpenInferenceLLMProviderValues.AWS.value:
        return GenAIProviderNameValues.AWS_BEDROCK.value

    if system == OpenInferenceLLMSystemValues.VERTEXAI.value or (
        provider == OpenInferenceLLMProviderValues.GOOGLE.value
        and system == OpenInferenceLLMSystemValues.VERTEXAI.value
    ):
        return GenAIProviderNameValues.GCP_VERTEX_AI.value

    if provider == OpenInferenceLLMProviderValues.GOOGLE.value:
        return GenAIProviderNameValues.GCP_GEN_AI.value

    if provider == OpenInferenceLLMProviderValues.OPENAI.value or (
        provider is None and system == OpenInferenceLLMSystemValues.OPENAI.value
    ):
        return GenAIProviderNameValues.OPENAI.value

    if provider == OpenInferenceLLMProviderValues.ANTHROPIC.value or (
        provider is None and system == OpenInferenceLLMSystemValues.ANTHROPIC.value
    ):
        return GenAIProviderNameValues.ANTHROPIC.value

    if provider == OpenInferenceLLMProviderValues.COHERE.value or (
        provider is None and system == OpenInferenceLLMSystemValues.COHERE.value
    ):
        return GenAIProviderNameValues.COHERE.value

    if provider == OpenInferenceLLMProviderValues.DEEPSEEK.value:
        return GenAIProviderNameValues.DEEPSEEK.value

    if provider == OpenInferenceLLMProviderValues.GROQ.value:
        return GenAIProviderNameValues.GROQ.value

    if provider == OpenInferenceLLMProviderValues.PERPLEXITY.value:
        return GenAIProviderNameValues.PERPLEXITY.value

    if provider == OpenInferenceLLMProviderValues.XAI.value:
        return GenAIProviderNameValues.X_AI.value

    if provider == OpenInferenceLLMProviderValues.MISTRALAI.value or (
        provider is None and system == OpenInferenceLLMSystemValues.MISTRALAI.value
    ):
        return GenAIProviderNameValues.MISTRAL_AI.value

    return system or provider


def _normalize_provider_or_system(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value.lower()
    return None


def _get_request_model(
    attributes: Mapping[str, AttributeValue],
    llm_parameters: Mapping[str, Any],
    embedding_parameters: Mapping[str, Any],
) -> Optional[str]:
    if model_name := _as_optional_str(attributes.get(SpanAttributes.LLM_MODEL_NAME)):
        return model_name
    if model_name := _as_optional_str(attributes.get(SpanAttributes.EMBEDDING_MODEL_NAME)):
        return model_name
    return _as_optional_str(llm_parameters.get("model")) or _as_optional_str(
        embedding_parameters.get("model")
    )


def _iter_request_parameter_mappings(
    invocation_parameters: Mapping[str, Any],
) -> List[Tuple[str, AttributeValue]]:
    mapped: List[Tuple[str, AttributeValue]] = []
    for genai_key, source_keys, coercer in _PARAMETER_MAPPINGS:
        for source_key in source_keys:
            if source_key not in invocation_parameters:
                continue
            coerced_value = coercer(invocation_parameters[source_key])
            if coerced_value is None:
                continue
            if genai_key == GenAIAttributes.GEN_AI_REQUEST_CHOICE_COUNT and coerced_value == 1:
                continue
            mapped.append((genai_key, coerced_value))
            break
    return mapped


def _get_output_type(invocation_parameters: Mapping[str, Any]) -> Optional[str]:
    response_format = invocation_parameters.get("response_format")
    if isinstance(response_format, str):
        normalized = response_format.lower()
        if normalized in {"json", "json_object", "json_schema"}:
            return GenAIOutputTypeValues.JSON.value
        if normalized == "text":
            return GenAIOutputTypeValues.TEXT.value
    if isinstance(response_format, Mapping):
        normalized_type = _as_optional_str(response_format.get("type"))
        if normalized_type in {"json", "json_object", "json_schema"}:
            return GenAIOutputTypeValues.JSON.value
        if normalized_type == "text":
            return GenAIOutputTypeValues.TEXT.value
    return None


def _get_input_messages(attributes: Mapping[str, AttributeValue]) -> List[Dict[str, Any]]:
    messages = _extract_oi_messages(attributes, SpanAttributes.LLM_INPUT_MESSAGES, is_output=False)
    if messages:
        return messages
    prompts = _get_prompt_texts(attributes)
    return [_build_text_message(GenAIRoleValues.USER.value, prompt) for prompt in prompts]


def _get_output_messages(attributes: Mapping[str, AttributeValue]) -> List[Dict[str, Any]]:
    messages = _extract_oi_messages(attributes, SpanAttributes.LLM_OUTPUT_MESSAGES, is_output=True)
    if messages:
        return messages
    choices = _get_choice_texts(attributes)
    return [
        _build_text_message(
            GenAIRoleValues.ASSISTANT.value,
            choice,
            finish_reason=_get_default_output_finish_reason(attributes),
        )
        for choice in choices
    ]


def _extract_oi_messages(
    attributes: Mapping[str, AttributeValue],
    prefix: str,
    *,
    is_output: bool,
) -> List[Dict[str, Any]]:
    root_messages = _parse_marshaled_oi_messages(attributes.get(prefix), is_output=is_output)
    buckets: Dict[int, Dict[str, Any]] = defaultdict(dict)

    for key, value in attributes.items():
        if not key.startswith(f"{prefix}."):
            continue
        suffix = key[len(prefix) + 1 :]
        index, _, remainder = suffix.partition(".")
        if not index.isdigit() or not remainder:
            continue
        buckets[int(index)][remainder] = value

    if not buckets:
        return root_messages

    fallback_finish_reason = (
        _get_default_output_finish_reason(attributes) if is_output else None
    )
    messages: List[Dict[str, Any]] = []
    for index in sorted(buckets):
        if message := _build_genai_message_from_bucket(
            buckets[index],
            is_output=is_output,
            fallback_finish_reason=fallback_finish_reason,
        ):
            messages.append(message)
    return messages


def _parse_marshaled_oi_messages(
    value: Any,
    *,
    is_output: bool,
) -> List[Dict[str, Any]]:
    parsed_value = _maybe_parse_json(value)
    if not isinstance(parsed_value, list):
        return []

    messages: List[Dict[str, Any]] = []
    for item in parsed_value:
        if not isinstance(item, Mapping):
            continue
        if message := _build_genai_message_from_marshaled(item, is_output=is_output):
            messages.append(message)
    return messages


def _build_genai_message_from_marshaled(
    message: Mapping[str, Any],
    *,
    is_output: bool,
) -> Optional[Dict[str, Any]]:
    if any(key.startswith("message.") for key in message):
        return _build_genai_message_from_bucket(
            dict(message),
            is_output=is_output,
            fallback_finish_reason=_infer_finish_reason_from_marshaled_message(message),
        )

    role = _normalize_message_role(_as_optional_str(message.get("role")))
    if role is None:
        return None

    parts: List[Dict[str, Any]] = []
    raw_parts = message.get("parts")
    if isinstance(raw_parts, list) and all(isinstance(part, Mapping) for part in raw_parts):
        parts = [dict(part) for part in raw_parts]
    elif content := _as_optional_str(message.get("content")):
        parts = [{"type": GenAIMessagePartTypeValues.TEXT.value, "content": content}]
    elif isinstance(contents := message.get("contents"), Sequence) and not isinstance(
        contents, str
    ):
        parts = _convert_marshaled_contents(contents)

    if tool_calls := message.get("tool_calls"):
        parts.extend(_convert_marshaled_tool_calls(tool_calls))

    tool_call_id = _as_optional_str(message.get("tool_call_id"))
    if role == GenAIRoleValues.TOOL.value and tool_call_id:
        response = _get_marshaled_tool_response(message)
        parts = [
            {
                "type": GenAIMessagePartTypeValues.TOOL_CALL_RESPONSE.value,
                "id": tool_call_id,
                "response": response,
            }
        ]

    if not parts:
        return None

    genai_message: Dict[str, Any] = {
        "role": role,
        "parts": parts,
    }
    if name := _as_optional_str(message.get("name")):
        genai_message["name"] = name
    elif name := _as_optional_str(message.get("message.name")):
        genai_message["name"] = name

    finish_reason = _as_optional_str(message.get("finish_reason"))
    if is_output:
        genai_message["finish_reason"] = _normalize_finish_reason(
            finish_reason
        ) or _default_finish_reason(genai_message)

    return genai_message


def _build_genai_message_from_bucket(
    bucket: Mapping[str, Any],
    *,
    is_output: bool,
    fallback_finish_reason: Optional[str],
) -> Optional[Dict[str, Any]]:
    role = _normalize_message_role(_as_optional_str(bucket.get(MessageAttributes.MESSAGE_ROLE)))
    if role is None:
        return None

    contents = _get_message_contents(bucket)
    tool_calls = _get_message_tool_calls(bucket)
    content = _as_optional_str(bucket.get(MessageAttributes.MESSAGE_CONTENT))
    tool_call_id = _as_optional_str(bucket.get(MessageAttributes.MESSAGE_TOOL_CALL_ID))

    parts: List[Dict[str, Any]] = []
    if role == GenAIRoleValues.TOOL.value and tool_call_id:
        parts.append(
            {
                "type": GenAIMessagePartTypeValues.TOOL_CALL_RESPONSE.value,
                "id": tool_call_id,
                "response": _get_bucket_tool_response(content, contents),
            }
        )
    else:
        parts.extend(contents)
        if content is not None and not _content_is_duplicate(content, contents):
            parts.insert(
                0,
                {
                    "type": GenAIMessagePartTypeValues.TEXT.value,
                    "content": content,
                },
            )
        parts.extend(tool_calls)

    if not parts:
        return None

    genai_message: Dict[str, Any] = {
        "role": role,
        "parts": parts,
    }

    if name := _as_optional_str(bucket.get(MessageAttributes.MESSAGE_NAME)):
        genai_message["name"] = name
    elif name := _as_optional_str(bucket.get(MessageAttributes.MESSAGE_FUNCTION_CALL_NAME)):
        genai_message["name"] = name

    if is_output:
        finish_reason = _normalize_finish_reason(
            _as_optional_str(bucket.get("message.finish_reason"))
        )
        genai_message["finish_reason"] = (
            finish_reason or fallback_finish_reason or _default_finish_reason(genai_message)
        )

    return genai_message


def _convert_marshaled_contents(contents: Sequence[Any]) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    for content in contents:
        if not isinstance(content, Mapping):
            continue
        content_type = _as_optional_str(content.get("type"))
        if content_type == "text":
            if text := _as_optional_str(content.get("text")):
                parts.append({"type": GenAIMessagePartTypeValues.TEXT.value, "content": text})
        elif content_type == "image":
            if image_part := _image_part_from_url(
                _as_optional_str(
                    (content.get("image") or {}).get(ImageAttributes.IMAGE_URL)
                    if isinstance(content.get("image"), Mapping)
                    else None
                )
            ):
                parts.append(image_part)
    return parts


def _convert_marshaled_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
    if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, str):
        return []
    parts: List[Dict[str, Any]] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, Mapping):
            continue
        function = tool_call.get("function")
        if not isinstance(function, Mapping):
            continue
        part: Dict[str, Any] = {
            "type": GenAIMessagePartTypeValues.TOOL_CALL.value,
            "name": _as_optional_str(function.get("name")) or "",
            "arguments": _jsonish_python_value(function.get("arguments")),
        }
        if tool_call_id := _as_optional_str(tool_call.get("id")):
            part["id"] = tool_call_id
        parts.append(part)
    return parts


def _infer_finish_reason_from_marshaled_message(message: Mapping[str, Any]) -> Optional[str]:
    if finish_reason := _as_optional_str(message.get("finish_reason")):
        return _normalize_finish_reason(finish_reason)
    if isinstance(parts := message.get("parts"), Sequence) and not isinstance(parts, str):
        if any(
            isinstance(part, Mapping)
            and _as_optional_str(part.get("type")) == GenAIMessagePartTypeValues.TOOL_CALL.value
            for part in parts
        ):
            return "tool_call"
    return None


def _get_message_contents(bucket: Mapping[str, Any]) -> List[Dict[str, Any]]:
    contents: Dict[int, Dict[str, Any]] = defaultdict(dict)
    for key, value in bucket.items():
        if not key.startswith(f"{MessageAttributes.MESSAGE_CONTENTS}."):
            continue
        suffix = key[len(MessageAttributes.MESSAGE_CONTENTS) + 1 :]
        index, _, remainder = suffix.partition(".")
        if not index.isdigit() or not remainder:
            continue
        contents[int(index)][remainder] = value

    parts: List[Dict[str, Any]] = []
    for index in sorted(contents):
        content_type = _as_optional_str(
            contents[index].get(MessageContentAttributes.MESSAGE_CONTENT_TYPE)
        )
        if content_type == "text":
            text = _as_optional_str(
                contents[index].get(MessageContentAttributes.MESSAGE_CONTENT_TEXT)
            )
            if text is not None:
                parts.append(
                    {
                        "type": GenAIMessagePartTypeValues.TEXT.value,
                        "content": text,
                    }
                )
        elif content_type == "image":
            image_key = (
                f"{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}.{ImageAttributes.IMAGE_URL}"
            )
            if image_part := _image_part_from_url(_as_optional_str(contents[index].get(image_key))):
                parts.append(image_part)
    return parts


def _get_message_tool_calls(bucket: Mapping[str, Any]) -> List[Dict[str, Any]]:
    tool_calls: Dict[int, Dict[str, Any]] = defaultdict(dict)
    for key, value in bucket.items():
        if not key.startswith(f"{MessageAttributes.MESSAGE_TOOL_CALLS}."):
            continue
        suffix = key[len(MessageAttributes.MESSAGE_TOOL_CALLS) + 1 :]
        index, _, remainder = suffix.partition(".")
        if not index.isdigit() or not remainder:
            continue
        tool_calls[int(index)][remainder] = value

    parts: List[Dict[str, Any]] = []
    for index in sorted(tool_calls):
        function_name = _as_optional_str(
            tool_calls[index].get(ToolCallAttributes.TOOL_CALL_FUNCTION_NAME)
        )
        if function_name is None:
            continue
        part: Dict[str, Any] = {
            "type": GenAIMessagePartTypeValues.TOOL_CALL.value,
            "name": function_name,
            "arguments": _jsonish_python_value(
                tool_calls[index].get(ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON)
            ),
        }
        if tool_call_id := _as_optional_str(tool_calls[index].get(ToolCallAttributes.TOOL_CALL_ID)):
            part["id"] = tool_call_id
        parts.append(part)
    return parts


def _content_is_duplicate(content: str, parts: Sequence[Mapping[str, Any]]) -> bool:
    text_parts = [
        part.get("content")
        for part in parts
        if _as_optional_str(part.get("type")) == GenAIMessagePartTypeValues.TEXT.value
    ]
    return len(text_parts) == 1 and text_parts[0] == content


def _image_part_from_url(url: Optional[str]) -> Optional[Dict[str, Any]]:
    if url is None:
        return None
    if data_url_match := _DATA_URL_PATTERN.match(url):
        return {
            "type": GenAIMessagePartTypeValues.BLOB.value,
            "modality": GenAIModalityValues.IMAGE.value,
            "mime_type": data_url_match.group("mime"),
            "content": data_url_match.group("content"),
        }
    return {
        "type": GenAIMessagePartTypeValues.URI.value,
        "modality": GenAIModalityValues.IMAGE.value,
        "uri": url,
    }


def _get_bucket_tool_response(
    content: Optional[str],
    contents: Sequence[Mapping[str, Any]],
) -> Any:
    if content is not None:
        return _jsonish_python_value(content)
    if len(contents) == 1 and contents[0].get("type") == GenAIMessagePartTypeValues.TEXT.value:
        return contents[0].get("content", "")
    return contents


def _get_marshaled_tool_response(message: Mapping[str, Any]) -> Any:
    if content := _as_optional_str(message.get("content")):
        return _jsonish_python_value(content)
    if isinstance(parts := message.get("parts"), Sequence) and not isinstance(parts, str):
        return list(parts)
    return ""


def _build_text_message(
    role: str,
    content: str,
    *,
    finish_reason: Optional[str] = None,
) -> Dict[str, Any]:
    message: Dict[str, Any] = {
        "role": role,
        "parts": [{"type": GenAIMessagePartTypeValues.TEXT.value, "content": content}],
    }
    if finish_reason is not None:
        message["finish_reason"] = finish_reason
    return message


def _normalize_message_role(role: Optional[str]) -> Optional[str]:
    if role is None:
        return None
    lowered_role = role.lower()
    if lowered_role == "agent":
        return GenAIRoleValues.ASSISTANT.value
    if lowered_role == "function":
        return GenAIRoleValues.TOOL.value
    return lowered_role


def _get_prompt_texts(attributes: Mapping[str, AttributeValue]) -> List[str]:
    prompts = attributes.get(SpanAttributes.LLM_PROMPTS)
    parsed_prompts = _maybe_parse_json(prompts)
    if isinstance(parsed_prompts, list):
        return [str(prompt) for prompt in parsed_prompts]
    if isinstance(prompts, Sequence) and not isinstance(prompts, str):
        return [str(prompt) for prompt in prompts]

    buckets: Dict[int, str] = {}
    prompt_prefix = f"{SpanAttributes.LLM_PROMPTS}."
    for key, value in attributes.items():
        if not key.startswith(prompt_prefix):
            continue
        suffix = key[len(prompt_prefix) :]
        index, _, remainder = suffix.partition(".")
        if not index.isdigit() or remainder != PromptAttributes.PROMPT_TEXT:
            continue
        buckets[int(index)] = str(value)

    return [buckets[index] for index in sorted(buckets)]


def _get_choice_texts(attributes: Mapping[str, AttributeValue]) -> List[str]:
    choices = attributes.get(SpanAttributes.LLM_CHOICES)
    parsed_choices = _maybe_parse_json(choices)
    if isinstance(parsed_choices, list):
        return [str(choice) for choice in parsed_choices]
    if isinstance(choices, Sequence) and not isinstance(choices, str):
        return [str(choice) for choice in choices]

    buckets: Dict[int, str] = {}
    choice_prefix = f"{SpanAttributes.LLM_CHOICES}."
    for key, value in attributes.items():
        if not key.startswith(choice_prefix):
            continue
        suffix = key[len(choice_prefix) :]
        index, _, remainder = suffix.partition(".")
        if not index.isdigit() or remainder != ChoiceAttributes.COMPLETION_TEXT:
            continue
        buckets[int(index)] = str(value)

    return [buckets[index] for index in sorted(buckets)]


def _get_default_output_finish_reason(attributes: Mapping[str, AttributeValue]) -> str:
    explicit_finish_reasons = _get_explicit_finish_reasons(attributes)
    if explicit_finish_reasons:
        return explicit_finish_reasons[0]

    if any(
        key.startswith(f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.")
        and MessageAttributes.MESSAGE_TOOL_CALLS in key
        for key in attributes
    ):
        return "tool_call"
    return "stop"


def _get_response_finish_reasons(
    attributes: Mapping[str, AttributeValue],
    *,
    _output_messages: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    explicit_finish_reasons = _get_explicit_finish_reasons(attributes)
    if explicit_finish_reasons:
        return explicit_finish_reasons

    output_messages = _output_messages if _output_messages is not None else _get_output_messages(
        attributes
    )
    return [
        _normalize_finish_reason(_as_optional_str(message.get("finish_reason"))) or "stop"
        for message in output_messages
        if isinstance(message, Mapping)
    ]


def _get_explicit_finish_reasons(attributes: Mapping[str, AttributeValue]) -> List[str]:
    finish_reason = attributes.get(SpanAttributes.LLM_FINISH_REASON)
    if isinstance(finish_reason, str):
        return [_normalize_finish_reason(finish_reason) or "stop"]
    if isinstance(finish_reason, Sequence) and not isinstance(finish_reason, str):
        return [
            normalized_finish_reason
            for raw_finish_reason in finish_reason
            if (normalized_finish_reason := _normalize_finish_reason(str(raw_finish_reason)))
        ]
    return []


def _normalize_finish_reason(finish_reason: Optional[str]) -> Optional[str]:
    if finish_reason is None:
        return None
    lowered = finish_reason.lower()
    if lowered in {"tool_calls", "function_call"}:
        return "tool_call"
    return lowered


def _default_finish_reason(message: Mapping[str, Any]) -> str:
    parts = message.get("parts")
    if not isinstance(parts, Sequence) or isinstance(parts, str):
        return "stop"
    if any(
        isinstance(part, Mapping)
        and _as_optional_str(part.get("type")) == GenAIMessagePartTypeValues.TOOL_CALL.value
        for part in parts
    ):
        return "tool_call"
    return "stop"


def _get_tool_definitions(attributes: Mapping[str, AttributeValue]) -> List[Dict[str, Any]]:
    root_tools = _parse_json_list(attributes.get(SpanAttributes.LLM_TOOLS))
    if root_tools:
        normalized_root_tools = [
            normalized_tool
            for tool in root_tools
            if (normalized_tool := _normalize_tool_definition(tool)) is not None
        ]
        if normalized_root_tools:
            return normalized_root_tools

    tool_buckets: Dict[int, Dict[str, Any]] = defaultdict(dict)
    for key, value in attributes.items():
        if not key.startswith(f"{SpanAttributes.LLM_TOOLS}."):
            continue
        suffix = key[len(SpanAttributes.LLM_TOOLS) + 1 :]
        index, _, remainder = suffix.partition(".")
        if not index.isdigit() or not remainder:
            continue
        tool_buckets[int(index)][remainder] = value

    tools: List[Dict[str, Any]] = []
    for index in sorted(tool_buckets):
        if normalized_tool := _normalize_tool_definition(tool_buckets[index]):
            tools.append(normalized_tool)
    return tools


def _normalize_tool_definition(tool_definition: Any) -> Optional[Dict[str, Any]]:
    if isinstance(tool_definition, str):
        parsed_definition = _maybe_parse_json(tool_definition)
        if parsed_definition is None:
            return None
        tool_definition = parsed_definition

    if not isinstance(tool_definition, Mapping):
        return None

    if ToolAttributes.TOOL_JSON_SCHEMA in tool_definition:
        return _normalize_tool_definition(tool_definition[ToolAttributes.TOOL_JSON_SCHEMA])

    raw_definition = dict(tool_definition)

    if function_definition := raw_definition.get("function"):
        if isinstance(function_definition, Mapping):
            normalized_tool: Dict[str, Any] = {
                "type": _as_optional_str(raw_definition.get("type"))
                or GenAIToolTypeValues.FUNCTION.value,
                "name": _as_optional_str(function_definition.get("name")),
            }
            if description := _as_optional_str(function_definition.get("description")):
                normalized_tool["description"] = description
            if parameters := function_definition.get("parameters"):
                normalized_tool["parameters"] = parameters
            return {key: value for key, value in normalized_tool.items() if value is not None}

    if raw_definition.get("input_schema") is not None and raw_definition.get("parameters") is None:
        raw_definition["parameters"] = raw_definition.pop("input_schema")

    if raw_definition.get("type") is None and (
        raw_definition.get("name") is not None
        or raw_definition.get("parameters") is not None
        or raw_definition.get("description") is not None
    ):
        raw_definition["type"] = GenAIToolTypeValues.FUNCTION.value

    return raw_definition


def _get_tool_call_arguments(attributes: Mapping[str, AttributeValue]) -> Optional[str]:
    if SpanAttributes.TOOL_PARAMETERS in attributes:
        return _serialize_jsonish_attribute_value(attributes[SpanAttributes.TOOL_PARAMETERS])

    flattened_tool_parameters = _unflatten_attributes(
        attributes,
        prefix=f"{SpanAttributes.TOOL_PARAMETERS}.",
    )
    if flattened_tool_parameters:
        return safe_json_dumps(flattened_tool_parameters)

    return None


def _get_tool_call_result(attributes: Mapping[str, AttributeValue]) -> Optional[str]:
    if SpanAttributes.OUTPUT_VALUE not in attributes:
        return None
    return _serialize_jsonish_attribute_value(attributes[SpanAttributes.OUTPUT_VALUE])


def _get_documents(
    attributes: Mapping[str, AttributeValue],
    prefix: str,
) -> List[Dict[str, Any]]:
    root_documents = _parse_json_list(attributes.get(prefix))
    if root_documents:
        normalized_root_documents = [
            normalized_document
            for document in root_documents
            if (normalized_document := _normalize_document(document)) is not None
        ]
        if normalized_root_documents:
            return normalized_root_documents

    buckets: Dict[int, Dict[str, Any]] = defaultdict(dict)
    for key, value in attributes.items():
        if not key.startswith(f"{prefix}."):
            continue
        suffix = key[len(prefix) + 1 :]
        index, _, remainder = suffix.partition(".")
        if not index.isdigit() or not remainder:
            continue
        buckets[int(index)][remainder] = value

    documents: List[Dict[str, Any]] = []
    for index in sorted(buckets):
        if normalized_document := _normalize_document(buckets[index]):
            documents.append(normalized_document)
    return documents


def _normalize_document(document: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(document, Mapping):
        return None

    normalized_document: Dict[str, Any] = {}

    document_id = document.get(DocumentAttributes.DOCUMENT_ID) or document.get("id")
    if document_id is not None:
        normalized_document["id"] = str(document_id)

    document_score = document.get(DocumentAttributes.DOCUMENT_SCORE) or document.get("score")
    if document_score is not None:
        normalized_document["score"] = document_score

    document_content = document.get(DocumentAttributes.DOCUMENT_CONTENT) or document.get("content")
    if document_content is not None:
        normalized_document["content"] = document_content

    metadata = document.get(DocumentAttributes.DOCUMENT_METADATA) or document.get("metadata")
    if metadata is not None:
        normalized_document["metadata"] = _jsonish_python_value(metadata)

    return normalized_document or None


def _get_retrieval_query_text(attributes: Mapping[str, AttributeValue]) -> Optional[str]:
    input_value = attributes.get(SpanAttributes.INPUT_VALUE)
    if input_value is None:
        return None

    input_mime_type = _as_optional_str(attributes.get(SpanAttributes.INPUT_MIME_TYPE))
    if input_mime_type == OpenInferenceMimeTypeValues.JSON.value:
        return None

    if isinstance(input_value, str):
        return input_value
    return str(input_value)


def _get_embedding_dimension_count(attributes: Mapping[str, AttributeValue]) -> Optional[int]:
    root_embeddings = _maybe_parse_json(attributes.get(SpanAttributes.EMBEDDING_EMBEDDINGS))
    if isinstance(root_embeddings, list):
        for embedding in root_embeddings:
            if not isinstance(embedding, Mapping):
                continue
            vector = embedding.get(EmbeddingAttributes.EMBEDDING_VECTOR) or embedding.get("vector")
            if isinstance(vector, Sequence) and not isinstance(vector, str):
                return len(vector)

    embedding_prefix = f"{SpanAttributes.EMBEDDING_EMBEDDINGS}."
    for key, value in attributes.items():
        if not key.startswith(embedding_prefix):
            continue
        if not key.endswith(f".{EmbeddingAttributes.EMBEDDING_VECTOR}"):
            continue
        if isinstance(value, Sequence) and not isinstance(value, str):
            return len(value)
        parsed_value = _maybe_parse_json(value)
        if isinstance(parsed_value, list):
            return len(parsed_value)
    return None


def _unflatten_attributes(
    attributes: Mapping[str, AttributeValue],
    *,
    prefix: str,
) -> Dict[str, Any]:
    unflattened: Dict[str, Any] = {}
    for key, value in attributes.items():
        if not key.startswith(prefix):
            continue
        path = key[len(prefix) :].split(".")
        current: Dict[str, Any] = unflattened
        for segment in path[:-1]:
            next_value = current.get(segment)
            if not isinstance(next_value, dict):
                next_value = {}
                current[segment] = next_value
            current = next_value
        current[path[-1]] = value
    return unflattened


def _parse_json_mapping(value: Any) -> Dict[str, Any]:
    parsed_value = _maybe_parse_json(value)
    if isinstance(parsed_value, Mapping):
        return dict(parsed_value)
    return {}


def _parse_json_list(value: Any) -> List[Any]:
    parsed_value = _maybe_parse_json(value)
    if isinstance(parsed_value, list):
        return parsed_value
    return []


def _maybe_parse_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _jsonish_python_value(value: Any) -> Any:
    parsed_value = _maybe_parse_json(value)
    if isinstance(parsed_value, tuple):
        return list(parsed_value)
    return parsed_value


def _serialize_jsonish_attribute_value(value: Any) -> str:
    parsed_value = _jsonish_python_value(value)
    if isinstance(parsed_value, str):
        return parsed_value
    return safe_json_dumps(parsed_value)


def _as_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _coerce_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def _coerce_string_sequence(value: Any) -> Optional[Tuple[str, ...]]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return None


_PARAMETER_MAPPINGS: Tuple[Tuple[str, Tuple[str, ...], Any], ...] = (
    (GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE, ("temperature",), _coerce_float),
    (GenAIAttributes.GEN_AI_REQUEST_TOP_P, ("top_p",), _coerce_float),
    (GenAIAttributes.GEN_AI_REQUEST_TOP_K, ("top_k",), _coerce_float),
    (
        GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS,
        ("max_tokens", "max_completion_tokens", "max_output_tokens"),
        _coerce_int,
    ),
    (GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY, ("frequency_penalty",), _coerce_float),
    (GenAIAttributes.GEN_AI_REQUEST_PRESENCE_PENALTY, ("presence_penalty",), _coerce_float),
    (GenAIAttributes.GEN_AI_REQUEST_SEED, ("seed",), _coerce_int),
    (GenAIAttributes.GEN_AI_REQUEST_STREAM, ("stream",), _coerce_bool),
    (
        GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES,
        ("stop_sequences", "stop"),
        _coerce_string_sequence,
    ),
    (
        GenAIAttributes.GEN_AI_REQUEST_CHOICE_COUNT,
        ("n", "choice_count", "candidate_count"),
        _coerce_int,
    ),
    (
        GenAIAttributes.GEN_AI_REQUEST_ENCODING_FORMATS,
        ("encoding_formats", "encoding_format"),
        _coerce_string_sequence,
    ),
)
