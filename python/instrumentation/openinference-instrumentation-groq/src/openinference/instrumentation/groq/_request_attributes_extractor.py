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
    ImageAttributes,
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
        yield from self._get_attributes_from_request_parameters(request_parameters)

    def _get_attributes_from_request_parameters(
        self,
        params: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if not isinstance(params, Mapping):
            return
        invocation_params = dict(params)
        yield SpanAttributes.LLM_INVOCATION_PARAMETERS, safe_json_dumps(invocation_params)

    def _get_attributes_from_chat_completion(
            self,
            params: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if not isinstance(params, Mapping):
            return
        invocation_params = dict(params)
        invocation_params.pop("messages", None)
        invocation_params.pop("functions", None)
        if isinstance((tools := invocation_params.pop("tools", None)), Iterable):
            for i, tool in enumerate(tools):
                yield f"llm.tools.{i}.tool.json_schema", safe_json_dumps(tool)
        yield SpanAttributes.LLM_INVOCATION_PARAMETERS, safe_json_dumps(invocation_params)

        if (input_messages := params.get("messages")) and isinstance(input_messages, Iterable):
            for index, input_message in list(enumerate(input_messages)):
                for key, value in self._get_attributes_from_message_param(input_message):
                    yield f"{SpanAttributes.LLM_INPUT_MESSAGES}.{index}.{key}", value

    def _get_attributes_from_message_content(
        self,
        content: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        content = dict(content)
        type_ = content.pop("type")
        if type_ == "text":
            yield f"{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", "text"
            if text := content.pop("text"):
                yield f"{MessageContentAttributes.MESSAGE_CONTENT_TEXT}", text
        elif type_ == "image_url":
            yield f"{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", "image"
            if image := content.pop("image_url"):
                for key, value in self._get_attributes_from_image(image):
                    yield f"{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}.{key}", value

    def _get_attributes_from_image(
        self,
        image: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        image = dict(image)
        if url := image.pop("url"):
            yield f"{ImageAttributes.IMAGE_URL}", url


T = TypeVar("T", bound=type)


def is_iterable_of(lst: Iterable[object], tp: T) -> bool:
    return isinstance(lst, Iterable) and all(isinstance(x, tp) for x in lst)
