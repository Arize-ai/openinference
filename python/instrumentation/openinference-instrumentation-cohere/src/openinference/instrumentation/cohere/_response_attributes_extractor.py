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
        # The chat response does not echo the model, so read it from the request.
        if isinstance(request_parameters, Mapping) and (model := request_parameters.get("model")):
            yield SpanAttributes.LLM_MODEL_NAME, model
        yield from self._get_attributes_from_usage(getattr(response, "usage", None))
        if message := getattr(response, "message", None):
            for key, value in self._get_attributes_from_response_message(message):
                yield f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{key}", value

    def _get_attributes_from_response_message(
        self,
        message: Any,
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if role := getattr(message, "role", None):
            yield MessageAttributes.MESSAGE_ROLE, role
        if (content := getattr(message, "content", None)) is not None:
            if text := _content_text(content):
                yield MessageAttributes.MESSAGE_CONTENT, text
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
                    if (arguments := getattr(function, "arguments", None)) is not None:
                        yield (
                            f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{index}."
                            f"{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                            arguments if isinstance(arguments, str) else safe_json_dumps(arguments),
                        )

    def _get_attributes_from_usage(
        self,
        usage: Any,
    ) -> Iterator[Tuple[str, AttributeValue]]:
        # Cohere nests token counts under ``usage.tokens``.
        tokens = getattr(usage, "tokens", None)
        if tokens is None:
            return
        prompt_tokens = getattr(tokens, "input_tokens", None)
        completion_tokens = getattr(tokens, "output_tokens", None)
        if prompt_tokens is not None:
            yield SpanAttributes.LLM_TOKEN_COUNT_PROMPT, int(prompt_tokens)
        if completion_tokens is not None:
            yield SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, int(completion_tokens)
        if prompt_tokens is not None and completion_tokens is not None:
            yield SpanAttributes.LLM_TOKEN_COUNT_TOTAL, int(prompt_tokens) + int(completion_tokens)


def _content_text(content: Any) -> str:
    """Collapse Cohere content (a string or a list of typed blocks) into a string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            text = item.get("text") if isinstance(item, Mapping) else getattr(item, "text", None)
            if text:
                parts.append(text)
        return "".join(parts)
    return ""
