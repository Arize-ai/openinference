import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime
from json import JSONEncoder
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    Literal,
    Optional,
    Tuple,
    Union,
)

from opentelemetry.util.types import AttributeValue
from typing_extensions import TypeGuard

from openinference.semconv.trace import (
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceLLMProviderValues,
    OpenInferenceLLMSystemValues,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)

from ._types import (
    Message,
    OpenInferenceLLMProvider,
    OpenInferenceLLMSystem,
    OpenInferenceMimeType,
    OpenInferenceSpanKind,
    TokenCount,
    Tool,
)

pydantic: Optional[ModuleType]
try:
    import pydantic  # try to import without adding a dependency
except ImportError:
    pydantic = None

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


def get_span_kind_attributes(kind: "OpenInferenceSpanKind", /) -> Dict[str, AttributeValue]:
    normalized_kind = _normalize_openinference_span_kind(kind)
    return {
        OPENINFERENCE_SPAN_KIND: normalized_kind.value,
    }


def get_input_attributes(
    value: Any,
    *,
    mime_type: Optional[OpenInferenceMimeType] = None,
) -> Dict[str, AttributeValue]:
    normalized_mime_type: Optional[OpenInferenceMimeTypeValues] = None
    if mime_type is not None:
        normalized_mime_type = _normalize_mime_type(mime_type)
    if normalized_mime_type is OpenInferenceMimeTypeValues.TEXT:
        value = str(value)
    elif normalized_mime_type is OpenInferenceMimeTypeValues.JSON:
        if not isinstance(value, str):
            value = _json_serialize(value)
    else:
        value, normalized_mime_type = _infer_serialized_io_value_and_mime_type(value)
    attributes = {
        INPUT_VALUE: value,
    }
    if normalized_mime_type is not None:
        attributes[INPUT_MIME_TYPE] = normalized_mime_type.value
    return attributes


def get_output_attributes(
    value: Any,
    *,
    mime_type: Optional[OpenInferenceMimeType] = None,
) -> Dict[str, AttributeValue]:
    normalized_mime_type: Optional[OpenInferenceMimeTypeValues] = None
    if mime_type is not None:
        normalized_mime_type = _normalize_mime_type(mime_type)
    if normalized_mime_type is OpenInferenceMimeTypeValues.TEXT:
        value = str(value)
    elif normalized_mime_type is OpenInferenceMimeTypeValues.JSON:
        if not isinstance(value, str):
            value = _json_serialize(value)
    else:
        value, normalized_mime_type = _infer_serialized_io_value_and_mime_type(value)
    attributes = {
        OUTPUT_VALUE: value,
    }
    if normalized_mime_type is not None:
        attributes[OUTPUT_MIME_TYPE] = normalized_mime_type.value
    return attributes


def _infer_serialized_io_value_and_mime_type(
    value: Any,
) -> Tuple[Any, Optional[OpenInferenceMimeTypeValues]]:
    if isinstance(value, (str, bool, int, float)):
        return str(value), OpenInferenceMimeTypeValues.TEXT
    if isinstance(value, (Sequence, Mapping)):
        return _json_serialize(value), OpenInferenceMimeTypeValues.JSON
    if _is_dataclass_instance(value):
        return _json_serialize(value), OpenInferenceMimeTypeValues.JSON
    if pydantic is not None and isinstance(value, pydantic.BaseModel):
        return _json_serialize(value), OpenInferenceMimeTypeValues.JSON
    return str(value), OpenInferenceMimeTypeValues.TEXT


class IOValueJSONEncoder(JSONEncoder):
    def default(self, obj: Any) -> Any:
        try:
            if _is_dataclass_instance(obj):
                return asdict(obj)
            if pydantic is not None and isinstance(obj, pydantic.BaseModel):
                return obj.model_dump()
            if isinstance(obj, datetime):
                return obj.isoformat()
            return super().default(obj)
        except Exception:
            return str(obj)


def _json_serialize(obj: Any, **kwargs: Any) -> str:
    """
    Safely JSON dumps input and handles special types such as dataclasses and
    pydantic models.
    """
    return json.dumps(
        obj,
        cls=IOValueJSONEncoder,
        ensure_ascii=False,
    )


def get_tool_attributes(
    *,
    name: str,
    description: Optional[str] = None,
    parameters: Union[str, Dict[str, Any]],
) -> Dict[str, AttributeValue]:
    if isinstance(parameters, str):
        parameters_json = parameters
    elif isinstance(parameters, Mapping):
        parameters_json = _json_serialize(parameters)
    else:
        raise ValueError(f"Invalid parameters type: {type(parameters)}")
    attributes: Dict[str, AttributeValue] = {
        TOOL_NAME: name,
        TOOL_PARAMETERS: parameters_json,
    }
    if description is not None:
        attributes[TOOL_DESCRIPTION] = description
    return attributes


def _normalize_mime_type(mime_type: OpenInferenceMimeType) -> OpenInferenceMimeTypeValues:
    if isinstance(mime_type, OpenInferenceMimeTypeValues):
        return mime_type
    try:
        return OpenInferenceMimeTypeValues(mime_type)
    except ValueError:
        raise ValueError(f"Invalid mime type: {mime_type}")


def _normalize_openinference_span_kind(
    kind: "OpenInferenceSpanKind",
) -> OpenInferenceSpanKindValues:
    if isinstance(kind, OpenInferenceSpanKindValues):
        return kind
    if not kind.islower():
        raise ValueError("kind must be lowercase if provided as a string")
    try:
        return OpenInferenceSpanKindValues(kind.upper())
    except ValueError:
        raise ValueError(f"Invalid OpenInference span kind: {kind}")


def _is_dataclass_instance(obj: Any) -> TypeGuard["DataclassInstance"]:
    """
    dataclasses.is_dataclass return true for both dataclass types and instances.
    This function returns true only for instances.

    See https://github.com/python/cpython/blob/05d12eecbde1ace39826320cadf8e673d709b229/Lib/dataclasses.py#L1391
    """
    cls = type(obj)
    return hasattr(cls, "__dataclass_fields__")


def get_llm_attributes(
    *,
    provider: Optional[OpenInferenceLLMProvider] = None,
    system: Optional[OpenInferenceLLMSystem] = None,
    model_name: Optional[str] = None,
    invocation_parameters: Optional[Union[str, Dict[str, Any]]] = None,
    input_messages: Optional["Sequence[Message]"] = None,
    output_messages: Optional["Sequence[Message]"] = None,
    token_count: Optional[TokenCount] = None,
    tools: Optional["Sequence[Tool]"] = None,
) -> Dict[str, AttributeValue]:
    return {
        **dict(_llm_provider_attributes(_normalize_llm_provider(provider))),
        **dict(_llm_system_attributes(_normalize_llm_system(system))),
        **dict(_llm_model_name_attributes(model_name)),
        **dict(_llm_invocation_parameters_attributes(invocation_parameters)),
        **dict(_llm_messages_attributes(input_messages, "input")),
        **dict(_llm_messages_attributes(output_messages, "output")),
        **dict(_llm_token_count_attributes(token_count)),
        **dict(_llm_tool_attributes(tools)),
    }


def _normalize_llm_provider(
    provider: Optional[OpenInferenceLLMProvider],
) -> Optional[OpenInferenceLLMProviderValues]:
    if isinstance(provider, OpenInferenceLLMProviderValues):
        return provider
    if isinstance(provider, str):
        try:
            return OpenInferenceLLMProviderValues(provider)
        except ValueError:
            valid_providers = [
                provider.value.lower() for provider in OpenInferenceLLMProviderValues
            ]
            warnings.warn(
                f"Invalid OpenInference LLM provider: {provider}. "
                f"Valid providers include: {', '.join(map(_quote_string, valid_providers))}"
            )
            return None
    return None


def _llm_provider_attributes(
    provider: Optional[OpenInferenceLLMProviderValues],
) -> Iterator[Tuple[str, AttributeValue]]:
    if isinstance(provider, OpenInferenceLLMProviderValues):
        yield LLM_PROVIDER, provider.value


def _normalize_llm_system(
    system: Optional[OpenInferenceLLMSystem],
) -> Optional[OpenInferenceLLMSystemValues]:
    if isinstance(system, OpenInferenceLLMSystemValues):
        return system
    if isinstance(system, str):
        try:
            return OpenInferenceLLMSystemValues(system)
        except ValueError:
            valid_systems = [system.value.lower() for system in OpenInferenceLLMSystemValues]
            warnings.warn(
                f"Invalid OpenInference LLM system: {system}. "
                f"Valid systems include: {', '.join(map(_quote_string, valid_systems))}"
            )
    return None


def _llm_system_attributes(
    system: Optional[OpenInferenceLLMSystemValues],
) -> Iterator[Tuple[str, AttributeValue]]:
    if isinstance(system, OpenInferenceLLMSystemValues):
        yield LLM_SYSTEM, system.value


def _llm_model_name_attributes(model_name: Optional[str]) -> Iterator[Tuple[str, AttributeValue]]:
    if isinstance(model_name, str):
        yield LLM_MODEL_NAME, model_name


def _llm_invocation_parameters_attributes(
    invocation_parameters: Optional[Union[str, Dict[str, Any]]],
) -> Iterator[Tuple[str, AttributeValue]]:
    if isinstance(invocation_parameters, str):
        yield LLM_INVOCATION_PARAMETERS, invocation_parameters
    elif isinstance(invocation_parameters, Dict):
        yield LLM_INVOCATION_PARAMETERS, _json_serialize(invocation_parameters)


def _llm_messages_attributes(
    messages: Optional["Sequence[Message]"],
    message_type: Literal["input", "output"],
) -> Iterator[Tuple[str, AttributeValue]]:
    base_key = LLM_INPUT_MESSAGES if message_type == "input" else LLM_OUTPUT_MESSAGES
    if not isinstance(messages, Sequence):
        return
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if (role := message.get("role")) is not None:
            yield f"{base_key}.{message_index}.{MESSAGE_ROLE}", role
        if (content := message.get("content")) is not None:
            yield f"{base_key}.{message_index}.{MESSAGE_CONTENT}", content
        if isinstance(contents := message.get("contents"), Sequence):
            for content_block_index, content_block in enumerate(contents):
                if not isinstance(content_block, dict):
                    continue
                if (type := content_block.get("type")) is not None:
                    yield (
                        f"{base_key}.{message_index}.{MESSAGE_CONTENTS}.{content_block_index}.{MESSAGE_CONTENT_TYPE}",
                        type,
                    )
                if (text := content_block.get("text")) is not None:
                    yield (
                        f"{base_key}.{message_index}.{MESSAGE_CONTENTS}.{content_block_index}.{MESSAGE_CONTENT_TEXT}",
                        text,
                    )
                if isinstance(image := content_block.get("image"), dict):
                    if isinstance(url := image.get("url"), str):
                        yield (
                            f"{base_key}.{message_index}.{MESSAGE_CONTENTS}.{content_block_index}.{MESSAGE_CONTENT_IMAGE}.{IMAGE_URL}",
                            url,
                        )
        if isinstance(tool_call_id := message.get("tool_call_id"), str):
            yield f"{base_key}.{message_index}.{MESSAGE_TOOL_CALL_ID}", tool_call_id
        if isinstance(tool_calls := message.get("tool_calls"), Sequence):
            for tool_call_index, tool_call in enumerate(tool_calls):
                if not isinstance(tool_call, dict):
                    continue
                if (tool_call_id := tool_call.get("id")) is not None:
                    yield (
                        f"{base_key}.{message_index}.{MESSAGE_TOOL_CALLS}.{tool_call_index}.{TOOL_CALL_ID}",
                        tool_call_id,
                    )
                if (function := tool_call.get("function")) is not None:
                    if isinstance(function, dict):
                        if isinstance(function_name := function.get("name"), str):
                            yield (
                                f"{base_key}.{message_index}.{MESSAGE_TOOL_CALLS}.{tool_call_index}.{TOOL_CALL_FUNCTION_NAME}",
                                function_name,
                            )
                        if isinstance(function_arguments := function.get("arguments"), str):
                            yield (
                                f"{base_key}.{message_index}.{MESSAGE_TOOL_CALLS}.{tool_call_index}.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                                function_arguments,
                            )
                        elif isinstance(function_arguments, dict):
                            yield (
                                f"{base_key}.{message_index}.{MESSAGE_TOOL_CALLS}.{tool_call_index}.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                                _json_serialize(function_arguments),
                            )


def _llm_token_count_attributes(
    token_count: Optional[TokenCount],
) -> Iterator[Tuple[str, AttributeValue]]:
    if isinstance(token_count, dict):
        if (prompt := token_count.get("prompt")) is not None:
            yield LLM_TOKEN_COUNT_PROMPT, prompt
        if (completion := token_count.get("completion")) is not None:
            yield LLM_TOKEN_COUNT_COMPLETION, completion
        if (total := token_count.get("total")) is not None:
            yield LLM_TOKEN_COUNT_TOTAL, total


def _llm_tool_attributes(tools: Optional["Sequence[Tool]"]) -> Iterator[Tuple[str, AttributeValue]]:
    if not isinstance(tools, Sequence):
        return
    for tool_index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            continue
        if isinstance(tool_name := tool.get("name"), str):
            yield f"{LLM_TOOLS}.{tool_index}.{TOOL_NAME}", tool_name
        if isinstance(tool_parameters := tool.get("parameters"), str):
            yield f"{LLM_TOOLS}.{tool_index}.{TOOL_PARAMETERS}", tool_parameters
        elif isinstance(tool_parameters, dict):
            yield f"{LLM_TOOLS}.{tool_index}.{TOOL_PARAMETERS}", _json_serialize(tool_parameters)


def _quote_string(string: str) -> str:
    return f'"{string}"'


# image attributes
IMAGE_URL = ImageAttributes.IMAGE_URL


# message attributes
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_CONTENTS = MessageAttributes.MESSAGE_CONTENTS
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_TOOL_CALL_ID = MessageAttributes.MESSAGE_TOOL_CALL_ID
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS


# message content attributes
MESSAGE_CONTENT_IMAGE = MessageContentAttributes.MESSAGE_CONTENT_IMAGE
MESSAGE_CONTENT_TEXT = MessageContentAttributes.MESSAGE_CONTENT_TEXT
MESSAGE_CONTENT_TYPE = MessageContentAttributes.MESSAGE_CONTENT_TYPE


# span attributes
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_PROVIDER = SpanAttributes.LLM_PROVIDER
LLM_SYSTEM = SpanAttributes.LLM_SYSTEM
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
LLM_TOOLS = SpanAttributes.LLM_TOOLS
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
TOOL_DESCRIPTION = SpanAttributes.TOOL_DESCRIPTION
TOOL_NAME = SpanAttributes.TOOL_NAME
TOOL_PARAMETERS = SpanAttributes.TOOL_PARAMETERS


# tool call attributes
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
TOOL_CALL_ID = ToolCallAttributes.TOOL_CALL_ID
