import logging
from typing import Any, Iterable, Iterator, Mapping, Tuple

from opentelemetry.util.types import AttributeValue

from openinference.instrumentation.groq._utils import _as_output_attributes, _io_value_and_type
from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
    ToolCallAttributes,
)

__all__ = ("_ResponseAttributesExtractor",)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _ResponseAttributesExtractor:
    def get_attributes(self, response: Any) -> Iterator[Tuple[str, AttributeValue]]:
        yield from _as_output_attributes(
            _io_value_and_type(response),
        )

    def get_extra_attributes(
        self,
        response: Any,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield from self._get_attributes_from_chat_completion(
            completion=response,
            request_parameters=request_parameters,
        )

    def _get_attributes_from_chat_completion(
        self,
        completion: Any,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if model := getattr(completion, "model", None):
            yield SpanAttributes.LLM_MODEL_NAME, model
        if usage := getattr(completion, "usage", None):
            yield from self._get_attributes_from_completion_usage(usage)
        if (choices := getattr(completion, "choices", None)) and isinstance(choices, Iterable):
            for choice in choices:
                if (index := getattr(choice, "index", None)) is None:
                    continue
                if message := getattr(choice, "message", None):
                    for key, value in self._get_attributes_from_chat_completion_message(message):
                        yield f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{index}.{key}", value
                # Only capture finish_reason for the first choice.
                if index == 0:
                    if (finish_reason := getattr(choice, "finish_reason", None)) is not None:
                        yield SpanAttributes.LLM_FINISH_REASON, finish_reason

    def _get_attributes_from_chat_completion_message(
        self,
        message: object,
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if role := getattr(message, "role", None):
            yield MessageAttributes.MESSAGE_ROLE, role
        content = getattr(message, "content", None)
        # Reasoning models return their reasoning beside the answer when the request
        # asks for `reasoning_format="parsed"`. It goes out as a reasoning content
        # block ahead of the text, the shape the other instrumentors use.
        if reasoning := getattr(message, "reasoning", None):
            prefix = f"{MessageAttributes.MESSAGE_CONTENTS}.0"
            yield f"{prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", "reasoning"
            yield f"{prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}", reasoning
            if content:
                prefix = f"{MessageAttributes.MESSAGE_CONTENTS}.1"
                yield f"{prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", "text"
                yield f"{prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}", content
        elif content:
            yield MessageAttributes.MESSAGE_CONTENT, content
        if function_call := getattr(message, "function_call", None):
            if name := getattr(function_call, "name", None):
                yield MessageAttributes.MESSAGE_FUNCTION_CALL_NAME, name
            if arguments := getattr(function_call, "arguments", None):
                yield MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON, arguments
        if (tool_calls := getattr(message, "tool_calls", None)) and isinstance(
            tool_calls, Iterable
        ):
            for index, tool_call in enumerate(tool_calls):
                if (tool_call_id := getattr(tool_call, "id", None)) is not None:
                    # https://github.com/groq/groq-python/blob/fa2e13b5747d18aeb478700f1e5426af2fd087a1/src/groq/types/chat/chat_completion_tool_message_param.py#L17  # noqa: E501
                    yield (
                        f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{index}."
                        f"{ToolCallAttributes.TOOL_CALL_ID}",
                        tool_call_id,
                    )
                if function := getattr(tool_call, "function", None):
                    # https://github.com/groq/groq-python/blob/fa2e13b5747d18aeb478700f1e5426af2fd087a1/src/groq/types/chat/chat_completion_message_tool_call.py#L10  # noqa: E501
                    if name := getattr(function, "name", None):
                        yield (
                            (
                                f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{index}."
                                f"{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"
                            ),
                            name,
                        )
                    if arguments := getattr(function, "arguments", None):
                        yield (
                            f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{index}."
                            f"{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                            arguments,
                        )

    def _get_attributes_from_completion_usage(
        self,
        usage: object,
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if (total_tokens := getattr(usage, "total_tokens", None)) is not None:
            yield SpanAttributes.LLM_TOKEN_COUNT_TOTAL, total_tokens
        if (prompt_tokens := getattr(usage, "prompt_tokens", None)) is not None:
            yield SpanAttributes.LLM_TOKEN_COUNT_PROMPT, prompt_tokens
        if (completion_tokens := getattr(usage, "completion_tokens", None)) is not None:
            yield SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, completion_tokens
        # https://github.com/groq/groq-python/blob/main/src/groq/types/completion_usage.py
        if (prompt_details := getattr(usage, "prompt_tokens_details", None)) is not None:
            if (cached_tokens := _field(prompt_details, "cached_tokens")) is not None:
                yield SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ, cached_tokens
        if (completion_details := getattr(usage, "completion_tokens_details", None)) is not None:
            if (reasoning_tokens := _field(completion_details, "reasoning_tokens")) is not None:
                yield SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING, reasoning_tokens


def _field(obj: Any, name: str) -> Any:
    # Older SDKs keep fields they do not know as a plain mapping under the parent model.
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)
