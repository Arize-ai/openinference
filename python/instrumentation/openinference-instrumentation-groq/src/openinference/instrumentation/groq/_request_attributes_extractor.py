import logging
from enum import Enum
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Tuple, TypeVar

from opentelemetry.util.types import AttributeValue

from groq.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from groq.types.chat.chat_completion_message_tool_call import Function
from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.groq._utils import _as_input_attributes, _io_value_and_type
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
        if not hasattr(message, "get"):
            if isinstance(message, ChatCompletionMessage):
                message = self._cast_chat_completion_to_mapping(message)
            else:
                return
        if role := message.get("role"):
            yield (
                MessageAttributes.MESSAGE_ROLE,
                role.value if isinstance(role, Enum) else role,
            )

        if content := message.get("content"):
            yield (
                MessageAttributes.MESSAGE_CONTENT,
                content,
            )

        if name := message.get("name"):
            yield MessageAttributes.MESSAGE_NAME, name

        if tool_call_id := message.get("tool_call_id"):
            yield MessageAttributes.MESSAGE_TOOL_CALL_ID, tool_call_id

        # Deprecated by Groq
        if (function_call := message.get("function_call")) and hasattr(function_call, "get"):
            if function_name := function_call.get("name"):
                yield MessageAttributes.MESSAGE_FUNCTION_CALL_NAME, function_name
            if function_arguments := function_call.get("arguments"):
                yield (
                    MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON,
                    function_arguments,
                )

        if (tool_calls := message.get("tool_calls"),) and isinstance(tool_calls, Iterable):
            for index, tool_call in enumerate(tool_calls):
                if not hasattr(tool_call, "get"):
                    continue
                if (tool_call_id := tool_call.get("id")) is not None:
                    yield (
                        f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{index}."
                        f"{ToolCallAttributes.TOOL_CALL_ID}",
                        tool_call_id,
                    )
                if (function := tool_call.get("function")) and hasattr(function, "get"):
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

    def _cast_chat_completion_to_mapping(self, message: ChatCompletionMessage) -> Mapping[str, Any]:
        try:
            casted_message = dict(message)
            if (tool_calls := casted_message.get("tool_calls")) and isinstance(
                tool_calls, Iterable
            ):
                casted_tool_calls: List[Dict[str, Any]] = []
                for tool_call in tool_calls:
                    if isinstance(tool_call, ChatCompletionMessageToolCall):
                        tool_call_dict = dict(tool_call)

                        if (function := tool_call_dict.get("function")) and isinstance(
                            function, Function
                        ):
                            tool_call_dict["function"] = dict(function)

                        casted_tool_calls.append(tool_call_dict)
                    else:
                        logger.debug(f"Skipping tool_call of unexpected type: {type(tool_call)}")

                casted_message["tool_calls"] = casted_tool_calls

            return casted_message

        except Exception as e:
            logger.exception(
                f"Failed to convert ChatCompletionMessage to mapping for {message}: {e}"
            )
            return {}


T = TypeVar("T", bound=type)


def is_iterable_of(lst: Iterable[object], tp: T) -> bool:
    return isinstance(lst, Iterable) and all(isinstance(x, tp) for x in lst)
