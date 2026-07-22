import logging
from typing import Any, Iterable, Iterator, Mapping, Tuple

from openinference.semconv.trace import MessageAttributes, SpanAttributes, ToolCallAttributes
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import safe_json_dumps

__all__ = ("_ResponseAttributesExtractor",)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _ResponseAttributesExtractor:
    def get_attributes(self, response: Any) -> Iterator[Tuple[str, AttributeValue]]:
        # ``ChatResponse`` is a pydantic model; serialize the whole thing as the
        # output value, matching the JSON output of the other instrumentors.
        try:
            value = response.model_dump_json(exclude_unset=True)
        except Exception:
            yield SpanAttributes.OUTPUT_VALUE, str(response)
            return
        yield SpanAttributes.OUTPUT_VALUE, value
        yield SpanAttributes.OUTPUT_MIME_TYPE, "application/json"

    def get_extra_attributes(
        self,
        response: Any,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield from self._get_attributes_from_chat_response(response=response)

    def _get_attributes_from_chat_response(
        self,
        response: Any,
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if model := getattr(response, "model", None):
            yield SpanAttributes.LLM_MODEL_NAME, model
        yield from self._get_attributes_from_token_counts(response)
        # Ollama returns a single message rather than a list of choices.
        if message := getattr(response, "message", None):
            for key, value in self._get_attributes_from_response_message(message):
                yield f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{key}", value

    def _get_attributes_from_response_message(
        self,
        message: Any,
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if role := getattr(message, "role", None):
            yield MessageAttributes.MESSAGE_ROLE, role
        if content := getattr(message, "content", None):
            yield MessageAttributes.MESSAGE_CONTENT, content
        if (tool_calls := getattr(message, "tool_calls", None)) and isinstance(
            tool_calls, Iterable
        ):
            for index, tool_call in enumerate(tool_calls):
                if function := getattr(tool_call, "function", None):
                    if name := getattr(function, "name", None):
                        yield (
                            f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{index}."
                            f"{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}",
                            name,
                        )
                    if (arguments := getattr(function, "arguments", None)) is not None:
                        # Ollama returns arguments as a mapping, not a JSON string.
                        yield (
                            f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{index}."
                            f"{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                            arguments if isinstance(arguments, str) else safe_json_dumps(arguments),
                        )

    def _get_attributes_from_token_counts(
        self,
        response: Any,
    ) -> Iterator[Tuple[str, AttributeValue]]:
        prompt_tokens = getattr(response, "prompt_eval_count", None)
        completion_tokens = getattr(response, "eval_count", None)
        if prompt_tokens is not None:
            yield SpanAttributes.LLM_TOKEN_COUNT_PROMPT, prompt_tokens
        if completion_tokens is not None:
            yield SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, completion_tokens
        if prompt_tokens is not None and completion_tokens is not None:
            yield SpanAttributes.LLM_TOKEN_COUNT_TOTAL, prompt_tokens + completion_tokens
