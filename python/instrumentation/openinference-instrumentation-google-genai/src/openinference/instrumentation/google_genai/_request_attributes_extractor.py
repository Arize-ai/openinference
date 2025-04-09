import logging
from enum import Enum
from typing import Any, Iterable, Iterator, Mapping, Tuple, TypeVar

from google.genai.types import Content, Part
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.google_genai._utils import (
    _as_input_attributes,
    _io_value_and_type,
)
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
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
        invocation_params.pop("contents", None)  # Remove LLM input contents
        yield SpanAttributes.LLM_INVOCATION_PARAMETERS, safe_json_dumps(invocation_params)

        if input_contents := request_parameters.get("contents"):
            if isinstance(input_contents, list):
                for index, input_content in reversed(list(enumerate(input_contents))):
                    # Use reversed() to get the last message first. This is because OTEL has a default
                    # limit of 128 attributes per span, and flattening increases the number of
                    # attributes very quickly.
                    for attr, value in self._get_attributes_from_message_param(input_content):
                        yield f"{SpanAttributes.LLM_INPUT_MESSAGES}.{index}.{attr}", value
            else:
                for attr, value in self._get_attributes_from_message_param(input_contents):
                    # Default to index 0 for a single message
                    yield f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{attr}", value

    def _get_attributes_from_message_param(
        self,
        input_contents: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        # https://github.com/googleapis/python-genai/blob/6e55222895a6639d41e54202e3d9a963609a391f/google/genai/models.py#L3890 # noqa: E501
        if isinstance(input_contents, str):
            # When provided a string, the GenAI SDK ingests it as a UserContent object with role "user"
            # https://googleapis.github.io/python-genai/index.html#provide-a-string
            yield (MessageAttributes.MESSAGE_CONTENT, input_contents)
            yield (MessageAttributes.MESSAGE_ROLE, "user")
        elif isinstance(input_contents, Content):
            yield from self._get_attributes_from_content(input_contents)
        elif isinstance(input_contents, Part):
            yield from self._get_attributes_from_part(input_contents)
        else:
            # TODO: Implement for File, PIL_Image
            logger.exception(f"Unexpected input contents type: {type(input_contents)}")

    def _get_attributes_from_content(
        self, content: Content
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if role := get_attribute(content, "role"):
            yield (
                MessageAttributes.MESSAGE_ROLE,
                role.value if isinstance(role, Enum) else role,
            )
        else:
            yield (
                MessageAttributes.MESSAGE_ROLE,
                "user",
            )

        # Flatten parts into a single message content
        if parts := get_attribute(content, "parts"):
            yield from self._flatten_parts(parts)

    def _flatten_parts(self, parts: list[Part]) -> Iterator[Tuple[str, AttributeValue]]:
        text_values = []
        for part in parts:
            attr, value = self._get_attributes_from_part(part)
            if isinstance(value, str):
                text_values.append(value)
            else:
                # TODO: Handle other types of parts
                yield attr, value

        if text_values:
            yield (MessageAttributes.MESSAGE_CONTENT, "\n\n".join(text_values))

    def _get_attributes_from_part(self, part: Part) -> Iterator[Tuple[str, AttributeValue]]:
        # https://github.com/googleapis/python-genai/blob/main/google/genai/types.py#L566
        if isinstance(part, Part):
            if text := get_attribute(part, "text"):
                yield (
                    MessageAttributes.MESSAGE_CONTENT,
                    text,
                )
            else:
                logger.exception("Other field types of parts are not supported yet")


T = TypeVar("T", bound=type)


def is_iterable_of(lst: Iterable[object], tp: T) -> bool:
    return isinstance(lst, Iterable) and all(isinstance(x, tp) for x in lst)


def get_attribute(obj: Any, attr_name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(attr_name, default)
    return getattr(obj, attr_name, default)
