import logging
from enum import Enum
from typing import Any, Iterable, Iterator, Mapping, Tuple

from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.cohere._utils import _as_input_attributes, _io_value_and_type

__all__ = ("_RequestAttributesExtractor",)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _RequestAttributesExtractor:
    __slots__ = ()

    def get_attributes_from_request(
        self,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value
        try:
            yield from _as_input_attributes(
                _io_value_and_type(request_parameters),
            )
        except Exception:
            logger.exception(
                f"Failed to get input attributes from request parameters of "
                f"type {type(request_parameters)}"
            )

    def get_extra_attributes_from_request(
        self,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if not isinstance(request_parameters, Mapping):
            return
        invocation_params = dict(request_parameters)
        invocation_params.pop("messages", None)  # Remove LLM input messages
        invocation_params.pop("model", None)  # Captured separately as the model name

        if isinstance((tools := invocation_params.pop("tools", None)), Iterable):
            for i, tool in enumerate(tools):
                yield f"llm.tools.{i}.tool.json_schema", safe_json_dumps(tool)

        yield SpanAttributes.LLM_INVOCATION_PARAMETERS, safe_json_dumps(invocation_params)

        if (input_messages := request_parameters.get("messages")) and isinstance(
            input_messages, Iterable
        ):
            for index, input_message in reversed(list(enumerate(input_messages))):
                # Use reversed() to get the last message first. This is because OTEL has a default
                # limit of 128 attributes per span, and flattening increases the number of
                # attributes very quickly.
                for key, value in self._get_attributes_from_message_param(input_message):
                    yield f"{SpanAttributes.LLM_INPUT_MESSAGES}.{index}.{key}", value

    def _get_attributes_from_message_param(
        self,
        message: Any,
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if role := get_attribute(message, "role"):
            yield (
                MessageAttributes.MESSAGE_ROLE,
                role.value if isinstance(role, Enum) else role,
            )
        if (content := get_attribute(message, "content")) is not None:
            if text := _content_text(content):
                yield MessageAttributes.MESSAGE_CONTENT, text

        if (tool_calls := get_attribute(message, "tool_calls")) and isinstance(
            tool_calls, Iterable
        ):
            for index, tool_call in enumerate(tool_calls):
                if (tool_call_id := get_attribute(tool_call, "id")) is not None:
                    yield (
                        f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{index}."
                        f"{ToolCallAttributes.TOOL_CALL_ID}",
                        tool_call_id,
                    )
                if function := get_attribute(tool_call, "function"):
                    if name := get_attribute(function, "name"):
                        yield (
                            f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{index}."
                            f"{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}",
                            name,
                        )
                    if (arguments := get_attribute(function, "arguments")) is not None:
                        yield (
                            f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{index}."
                            f"{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                            arguments if isinstance(arguments, str) else safe_json_dumps(arguments),
                        )


def get_attribute(obj: Any, attr_name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(attr_name, default)
    return getattr(obj, attr_name, default)


def _content_text(content: Any) -> str:
    """Cohere content may be a plain string or a list of typed content blocks.

    Collapse either form into a single string, keeping only the ``text`` of each
    block so the message content attribute is always a string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if text := get_attribute(item, "text"):
                parts.append(text)
        return "".join(parts)
    return ""
