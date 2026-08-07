import logging
from enum import Enum
from typing import Any, Iterable, Iterator, Mapping, Sequence, Tuple

from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import get_input_attributes, safe_json_dumps
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolAttributes,
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
            yield from get_input_attributes(request_parameters).items()
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
        # Literal value of OpenInferenceLLMProviderValues.OLLAMA, inlined so the
        # package works against semconv releases that predate the enum member.
        yield SpanAttributes.LLM_PROVIDER, "ollama"
        invocation_params = dict(request_parameters)
        invocation_params.pop("messages", None)  # Captured separately as input messages
        if model := invocation_params.pop("model", None):
            # Capture the model on the request side too, so it is present on
            # spans for calls that error out before a response arrives.
            yield SpanAttributes.LLM_MODEL_NAME, model

        # Only iterate materialized sequences: a caller-supplied iterator would
        # be exhausted here before the real request is sent.
        if isinstance((tools := invocation_params.pop("tools", None)), Sequence) and not isinstance(
            tools, (str, bytes)
        ):
            for i, tool in enumerate(tools):
                yield (
                    f"{SpanAttributes.LLM_TOOLS}.{i}.{ToolAttributes.TOOL_JSON_SCHEMA}",
                    _tool_json_schema(tool),
                )

        yield SpanAttributes.LLM_INVOCATION_PARAMETERS, safe_json_dumps(invocation_params)

        if (
            (input_messages := request_parameters.get("messages"))
            and isinstance(input_messages, Sequence)
            and not isinstance(input_messages, (str, bytes))
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
        if content := get_attribute(message, "content"):
            yield MessageAttributes.MESSAGE_CONTENT, content
        # Ollama uses `tool_name` on tool-result messages to identify the tool.
        if tool_name := get_attribute(message, "tool_name"):
            yield MessageAttributes.MESSAGE_NAME, tool_name

        if (tool_calls := get_attribute(message, "tool_calls")) and isinstance(
            tool_calls, Iterable
        ):
            for index, tool_call in enumerate(tool_calls):
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
                            _as_arguments_json(arguments),
                        )


def _tool_json_schema(tool: Any) -> str:
    if callable(tool) and not isinstance(tool, Mapping):
        # ollama >= 0.4 accepts plain Python functions as tools and converts
        # them to Tool schemas internally; mirror that conversion so the span
        # records the schema instead of the function's repr.
        try:
            from ollama._utils import convert_function_to_tool

            tool = convert_function_to_tool(tool)
        except Exception:
            logger.exception(f"Failed to convert callable tool {tool!r} to a schema")
    if hasattr(tool, "model_dump"):
        try:
            return safe_json_dumps(tool.model_dump(exclude_none=True))
        except Exception:
            logger.exception(f"Failed to dump tool of type {type(tool)}")
    return safe_json_dumps(tool)


def get_attribute(obj: Any, attr_name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(attr_name, default)
    return getattr(obj, attr_name, default)


def _as_arguments_json(arguments: Any) -> str:
    # Ollama returns tool-call arguments as a mapping, unlike the OpenAI-style
    # JSON string. Serialize mappings so the attribute is always a JSON string.
    if isinstance(arguments, str):
        return arguments
    return safe_json_dumps(arguments)
