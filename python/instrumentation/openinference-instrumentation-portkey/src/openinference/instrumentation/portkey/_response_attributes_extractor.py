import logging
from typing import Any, Iterable, Iterator, Mapping, Tuple

from opentelemetry.util.types import AttributeValue

from openinference.instrumentation.portkey._utils import (
    _as_output_attributes,
    _io_value_and_type,
    get_attribute,
)
from openinference.semconv.trace import MessageAttributes, SpanAttributes, ToolCallAttributes

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
        if content := getattr(message, "content", None):
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
                    yield (
                        f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{index}."
                        f"{ToolCallAttributes.TOOL_CALL_ID}",
                        tool_call_id,
                    )
                if function := getattr(tool_call, "function", None):
                    if name := getattr(function, "name", None):
                        yield (
                            f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{index}."
                            f"{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}",
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
        total_tokens = getattr(usage, "total_tokens", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        # For Anthropic models the Portkey gateway adds Anthropic's own cache counts to usage.
        # Its Anthropic and Vertex transforms leave them out of prompt_tokens but count them in
        # total_tokens, while Bedrock counts them in both. The cache counts are sub-counts of the
        # prompt, so fold them in only when the total shows prompt_tokens left them out.
        cache_write_tokens = getattr(usage, "cache_creation_input_tokens", None)
        anthropic_cache_tokens = (getattr(usage, "cache_read_input_tokens", None) or 0) + (
            cache_write_tokens or 0
        )
        if (
            anthropic_cache_tokens
            and isinstance(prompt_tokens, int)
            and isinstance(completion_tokens, int)
            and prompt_tokens + completion_tokens + anthropic_cache_tokens == total_tokens
        ):
            prompt_tokens += anthropic_cache_tokens
        if total_tokens is not None:
            yield SpanAttributes.LLM_TOKEN_COUNT_TOTAL, total_tokens
        if prompt_tokens is not None:
            yield SpanAttributes.LLM_TOKEN_COUNT_PROMPT, prompt_tokens
        if completion_tokens is not None:
            yield SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, completion_tokens
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        if (cached_tokens := get_attribute(prompt_details, "cached_tokens")) is not None:
            yield SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ, cached_tokens
        if cache_write_tokens is not None:
            yield SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE, cache_write_tokens
        completion_details = getattr(usage, "completion_tokens_details", None)
        if (reasoning_tokens := get_attribute(completion_details, "reasoning_tokens")) is not None:
            yield SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING, reasoning_tokens
