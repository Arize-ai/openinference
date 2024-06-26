import logging
from enum import Enum
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Iterator,
    List,
    Mapping,
    Tuple,
    Type,
    TypeVar,
)

from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.openai._utils import _get_openai_version
from openinference.semconv.trace import (
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
    ToolCallAttributes,
)
from opentelemetry.util.types import AttributeValue

if TYPE_CHECKING:
    from openai.types import Completion, CreateEmbeddingResponse
    from openai.types.chat import ChatCompletion

__all__ = ("_RequestAttributesExtractor",)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _RequestAttributesExtractor:
    __slots__ = (
        "_openai",
        "_chat_completion_type",
        "_completion_type",
        "_create_embedding_response_type",
    )

    def __init__(self, openai: ModuleType) -> None:
        self._openai = openai
        self._chat_completion_type: Type["ChatCompletion"] = openai.types.chat.ChatCompletion
        self._completion_type: Type["Completion"] = openai.types.Completion
        self._create_embedding_response_type: Type["CreateEmbeddingResponse"] = (
            openai.types.CreateEmbeddingResponse
        )

    def get_attributes_from_request(
        self,
        cast_to: type,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if not isinstance(request_parameters, Mapping):
            return
        if cast_to is self._chat_completion_type:
            yield from _get_attributes_from_chat_completion_create_param(request_parameters)
        elif cast_to is self._create_embedding_response_type:
            yield from _get_attributes_from_embedding_create_param(request_parameters)
        elif cast_to is self._completion_type:
            yield from _get_attributes_from_completion_create_param(request_parameters)
        else:
            try:
                yield SpanAttributes.LLM_INVOCATION_PARAMETERS, safe_json_dumps(request_parameters)
            except Exception:
                logger.exception("Failed to serialize request options")


def _get_attributes_from_chat_completion_create_param(
    params: Mapping[str, Any],
) -> Iterator[Tuple[str, AttributeValue]]:
    # openai.types.chat.completion_create_params.CompletionCreateParamsBase
    # See https://github.com/openai/openai-python/blob/f1c7d714914e3321ca2e72839fe2d132a8646e7f/src/openai/types/chat/completion_create_params.py#L28  # noqa: E501
    if not isinstance(params, Mapping):
        return
    invocation_params = dict(params)
    invocation_params.pop("messages", None)
    invocation_params.pop("functions", None)
    invocation_params.pop("tools", None)
    yield SpanAttributes.LLM_INVOCATION_PARAMETERS, safe_json_dumps(invocation_params)
    if (input_messages := params.get("messages")) and isinstance(input_messages, Iterable):
        for index, input_message in list(enumerate(input_messages)):
            for key, value in _get_attributes_from_message_param(input_message):
                yield f"{SpanAttributes.LLM_INPUT_MESSAGES}.{index}.{key}", value


T = TypeVar("T", bound=type)


def is_iterable_of(lst: Iterable[object], tp: T) -> bool:
    return isinstance(lst, Iterable) and all(isinstance(x, tp) for x in lst)


def _get_attributes_from_message_param(
    message: Mapping[str, Any],
) -> Iterator[Tuple[str, AttributeValue]]:
    # openai.types.chat.ChatCompletionMessageParam
    # See https://github.com/openai/openai-python/blob/f1c7d714914e3321ca2e72839fe2d132a8646e7f/src/openai/types/chat/chat_completion_message_param.py#L15  # noqa: E501
    if not hasattr(message, "get"):
        return
    if role := message.get("role"):
        yield (
            MessageAttributes.MESSAGE_ROLE,
            role.value if isinstance(role, Enum) else role,
        )
    if content := message.get("content"):
        if isinstance(content, str):
            yield MessageAttributes.MESSAGE_CONTENT, content
        elif is_iterable_of(content, dict):
            for index, c in list(enumerate(content)):
                for key, value in _get_attributes_from_message_content(c):
                    yield f"{MessageAttributes.MESSAGE_CONTENTS}.{index}.{key}", value
        elif isinstance(content, List):
            # See https://github.com/openai/openai-python/blob/f1c7d714914e3321ca2e72839fe2d132a8646e7f/src/openai/types/chat/chat_completion_user_message_param.py#L14  # noqa: E501
            try:
                json_string = safe_json_dumps(content)
            except Exception:
                logger.exception("Failed to serialize message content")
            else:
                yield MessageAttributes.MESSAGE_CONTENT, json_string
    if name := message.get("name"):
        yield MessageAttributes.MESSAGE_NAME, name
    if (function_call := message.get("function_call")) and hasattr(function_call, "get"):
        # See https://github.com/openai/openai-python/blob/f1c7d714914e3321ca2e72839fe2d132a8646e7f/src/openai/types/chat/chat_completion_assistant_message_param.py#L13  # noqa: E501
        if function_name := function_call.get("name"):
            yield MessageAttributes.MESSAGE_FUNCTION_CALL_NAME, function_name
        if function_arguments := function_call.get("arguments"):
            yield (
                MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON,
                function_arguments,
            )
    if (
        _get_openai_version() >= (1, 1, 0)
        and (tool_calls := message.get("tool_calls"),)
        and isinstance(tool_calls, Iterable)
    ):
        for index, tool_call in enumerate(tool_calls):
            # See https://github.com/openai/openai-python/blob/f1c7d714914e3321ca2e72839fe2d132a8646e7f/src/openai/types/chat/chat_completion_message_tool_call_param.py#L23  # noqa: E501
            if not hasattr(tool_call, "get"):
                continue
            if (function := tool_call.get("function")) and hasattr(function, "get"):
                # See https://github.com/openai/openai-python/blob/f1c7d714914e3321ca2e72839fe2d132a8646e7f/src/openai/types/chat/chat_completion_message_tool_call_param.py#L10  # noqa: E501
                if name := function.get("name"):
                    yield (
                        f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{index}."
                        f"{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}",
                        name,
                    )
                if arguments := function.get("arguments"):
                    yield (
                        f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{index}."
                        f"{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                        arguments,
                    )


def _get_attributes_from_message_content(
    content: Mapping[str, Any],
) -> Iterator[Tuple[str, AttributeValue]]:
    content = dict(content)
    type_ = content.pop("type")
    if type_ == "text":
        yield f"{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", "text"
        if text := content.pop("text"):
            yield f"{MessageContentAttributes.MESSAGE_CONTENT_TEXT}", text
    elif type_ == "image_url":
        yield f"{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", "image"
        if image := content.pop("image_url"):
            for key, value in _get_attributes_from_image(image):
                yield f"{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}.{key}", value


def _get_attributes_from_image(
    image: Mapping[str, Any],
) -> Iterator[Tuple[str, AttributeValue]]:
    image = dict(image)
    if url := image.pop("url"):
        yield f"{ImageAttributes.IMAGE_URL}", url


def _get_attributes_from_completion_create_param(
    params: Mapping[str, Any],
) -> Iterator[Tuple[str, AttributeValue]]:
    # openai.types.completion_create_params.CompletionCreateParamsBase
    # See https://github.com/openai/openai-python/blob/f1c7d714914e3321ca2e72839fe2d132a8646e7f/src/openai/types/completion_create_params.py#L11  # noqa: E501
    if not isinstance(params, Mapping):
        return
    invocation_params = dict(params)
    invocation_params.pop("prompt", None)
    yield SpanAttributes.LLM_INVOCATION_PARAMETERS, safe_json_dumps(invocation_params)


def _get_attributes_from_embedding_create_param(
    params: Mapping[str, Any],
) -> Iterator[Tuple[str, AttributeValue]]:
    # openai.types.EmbeddingCreateParams
    # See https://github.com/openai/openai-python/blob/f1c7d714914e3321ca2e72839fe2d132a8646e7f/src/openai/types/embedding_create_params.py#L11  # noqa: E501
    if not isinstance(params, Mapping):
        return
    invocation_params = dict(params)
    invocation_params.pop("input", None)
    yield SpanAttributes.LLM_INVOCATION_PARAMETERS, safe_json_dumps(invocation_params)
