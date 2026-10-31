import logging
from typing import Any, Iterable, Iterator, Tuple

from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import get_output_attributes
from openinference.instrumentation.ollama._utils import _as_arguments_json
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
        # ``ChatResponse`` is a pydantic model; serialize the whole thing as the
        # output value, matching the JSON output of the other instrumentors.
        yield from get_output_attributes(response).items()

    def get_extra_attributes(
        self,
        response: Any,
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
        if (done_reason := getattr(response, "done_reason", None)) is not None:
            yield SpanAttributes.LLM_FINISH_REASON, done_reason

    def _get_attributes_from_response_message(
        self,
        message: Any,
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if role := getattr(message, "role", None):
            yield MessageAttributes.MESSAGE_ROLE, role
        content = getattr(message, "content", None)
        if content:
            yield MessageAttributes.MESSAGE_CONTENT, content
        # Thinking models put their reasoning in `message.thinking`. Following
        # `spec/llm_spans.md`, it is recorded as a "reasoning" content item so the
        # message is also emitted as ordered contents (reasoning, then text) the way
        # the other instrumentors do. The flat `message.content` above stays, so
        # existing consumers keep seeing the answer.
        thinking = _as_text(getattr(message, "thinking", None))
        if thinking:
            contents_prefix = MessageAttributes.MESSAGE_CONTENTS
            yield (
                f"{contents_prefix}.0.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}",
                "reasoning",
            )
            yield (
                f"{contents_prefix}.0.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}",
                thinking,
            )
            if text := _as_text(content):
                yield (
                    f"{contents_prefix}.1.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}",
                    "text",
                )
                yield (
                    f"{contents_prefix}.1.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}",
                    text,
                )
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
                        yield (
                            f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{index}."
                            f"{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                            _as_arguments_json(arguments),
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


def _as_text(value: Any) -> str:
    """Return the value as a string, or "" when there is nothing to record."""
    return value if isinstance(value, str) else ""
