import logging
from typing import Any, Iterable, Iterator, Mapping, Tuple

from opentelemetry.util.types import AttributeValue

from openinference.instrumentation.portkey._utils import _as_output_attributes, _io_value_and_type
from openinference.semconv.trace import MessageAttributes, SpanAttributes

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

    def _get_attributes_from_chat_completion_message(
        self,
        message: object,
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if role := getattr(message, "role", None):
            yield MessageAttributes.MESSAGE_ROLE, role
        if content := getattr(message, "content", None):
            yield MessageAttributes.MESSAGE_CONTENT, content
        if function_call := getattr(message, "function_call", None):
            if name := getattr(function_call, "name", None):
                yield MessageAttributes.MESSAGE_FUNCTION_CALL_NAME, name
            if arguments := getattr(function_call, "arguments", None):
                yield MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON, arguments

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
