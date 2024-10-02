import logging
from enum import Enum
from typing import (
    Any,
    Iterable,
    Iterator,
    List,
    Mapping,
    Tuple,
    TypeVar,
)

from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import safe_json_dumps
from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
)
from openinference.instrumentation.groq._utils import _as_input_attributes, _io_value_and_type

from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
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
        invocation_params.pop("messages", None)
        invocation_params.pop("functions", None)
        invocation_params.pop("tools", None)
        yield SpanAttributes.LLM_INVOCATION_PARAMETERS, safe_json_dumps(invocation_params)
        if (input_messages := request_parameters.get("messages")) and isinstance(input_messages, Iterable):
            # Use reversed() to get the last message first. This is because OTEL has a default limit of
            # 128 attributes per span, and flattening increases the number of attributes very quickly.
            for index, input_message in reversed(list(enumerate(input_messages))):
                if role := input_message.get("role"):
                    yield f"{SpanAttributes.LLM_INPUT_MESSAGES}.{index}.{MessageAttributes.MESSAGE_ROLE}", role
                if content := input_message.get("content"):
                    yield f"{SpanAttributes.LLM_INPUT_MESSAGES}.{index}.{MessageAttributes.MESSAGE_CONTENT}", content


T = TypeVar("T", bound=type)


def is_iterable_of(lst: Iterable[object], tp: T) -> bool:
    return isinstance(lst, Iterable) and all(isinstance(x, tp) for x in lst)
