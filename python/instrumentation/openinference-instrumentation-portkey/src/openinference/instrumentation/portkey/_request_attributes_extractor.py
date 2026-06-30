import logging
from enum import Enum
from typing import Any, Iterable, Iterator, Mapping, Tuple

from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.portkey._utils import _as_input_attributes, _io_value_and_type
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)

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
        invocation_params.pop("functions", None)

        if isinstance((tools := invocation_params.pop("tools", None)), Iterable):
            for i, tool in enumerate(tools):
                yield f"llm.tools.{i}.tool.json_schema", safe_json_dumps(tool)

        yield SpanAttributes.LLM_INVOCATION_PARAMETERS, safe_json_dumps(invocation_params)

        if prompt_id := invocation_params.get("prompt_id"):
            yield SpanAttributes.PROMPT_ID, prompt_id

        if prompt_variables := invocation_params.get("variables"):
            yield SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES, safe_json_dumps(prompt_variables)

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
        message: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if role := get_attribute(message, "role"):
            yield (
                MessageAttributes.MESSAGE_ROLE,
                role.value if isinstance(role, Enum) else role,
            )
        if content := get_attribute(message, "content"):
            yield (
                MessageAttributes.MESSAGE_CONTENT,
                content,
            )
        if name := get_attribute(message, "name"):
            yield MessageAttributes.MESSAGE_NAME, name

        if tool_call_id := get_attribute(message, "tool_call_id"):
            yield MessageAttributes.MESSAGE_TOOL_CALL_ID, tool_call_id

        # Deprecated by Groq
        if function_call := get_attribute(message, "function_call"):
            if function_name := get_attribute(function_call, "name"):
                yield MessageAttributes.MESSAGE_FUNCTION_CALL_NAME, function_name
            if function_arguments := get_attribute(function_call, "arguments"):
                yield (
                    MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON,
                    function_arguments,
                )

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
                    if arguments := get_attribute(function, "arguments"):
                        yield (
                            f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{index}."
                            f"{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                            arguments,
                        )


def get_attribute(obj: Any, attr_name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(attr_name, default)
    return getattr(obj, attr_name, default)
