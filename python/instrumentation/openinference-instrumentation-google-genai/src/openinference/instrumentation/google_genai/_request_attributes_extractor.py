import logging
from enum import Enum
from typing import Any, Iterable, Iterator, Mapping, Tuple, TypeVar

from google.genai.types import Content, Part, UserContent
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation.google_genai._utils import (
    _as_input_attributes,
    _io_value_and_type,
)
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceLLMProviderValues,
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
        yield SpanAttributes.LLM_PROVIDER, OpenInferenceLLMProviderValues.GOOGLE.value
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
        # Start an index for the messages since we want to start with system instruction
        input_messages_index = 0
        if not isinstance(request_parameters, Mapping):
            return

        request_params_dict = dict(request_parameters)
        request_params_dict.pop("contents", None)  # Remove LLM input contents
        if config := request_params_dict.get("config", None):
            # Config is a pydantic object, so we need to convert it to a JSON string
            yield (
                SpanAttributes.LLM_INVOCATION_PARAMETERS,
                config.model_dump_json(exclude_none=True),
            )
            # We push the system instruction to the first message for replay and consistency
            system_instruction = getattr(config, "system_instruction", None)
            if system_instruction:
                yield (
                    f"{SpanAttributes.LLM_INPUT_MESSAGES}.{input_messages_index}.{MessageAttributes.MESSAGE_CONTENT}",
                    system_instruction,
                )
                yield (
                    f"{SpanAttributes.LLM_INPUT_MESSAGES}.{input_messages_index}.{MessageAttributes.MESSAGE_ROLE}",
                    "system",
                )
                input_messages_index += 1

        if input_contents := request_parameters.get("contents"):
            if isinstance(input_contents, list):
                for input_content in input_contents:
                    for attr, value in self._get_attributes_from_message_param(input_content):
                        yield (
                            f"{SpanAttributes.LLM_INPUT_MESSAGES}.{input_messages_index}.{attr}",
                            value,
                        )
                    # Move on to the next message
                    input_messages_index += 1
            else:
                for attr, value in self._get_attributes_from_message_param(input_contents):
                    # Default to index 0 for a single message
                    yield (
                        f"{SpanAttributes.LLM_INPUT_MESSAGES}.{input_messages_index}.{attr}",
                        value,
                    )

    def _get_attributes_from_message_param(
        self,
        input_contents: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        # https://github.com/googleapis/python-genai/blob/6e55222895a6639d41e54202e3d9a963609a391f/google/genai/models.py#L3890 # noqa: E501
        if isinstance(input_contents, str):
            # When provided a string, the GenAI SDK ingests it as
            # a UserContent object with role "user"
            # https://googleapis.github.io/python-genai/index.html#provide-a-string
            yield (MessageAttributes.MESSAGE_CONTENT, input_contents)
            yield (MessageAttributes.MESSAGE_ROLE, "user")
        elif isinstance(input_contents, Content) or isinstance(input_contents, UserContent):
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
            for attr, value in self._get_attributes_from_part(part):
                if isinstance(value, str):
                    text_values.append(value)
            else:
                # TODO: Handle other types of parts
                logger.debug(f"Non-text part encountered: {part}")
        if text_values:
            yield (MessageAttributes.MESSAGE_CONTENT, "\n\n".join(text_values))

    def _get_attributes_from_part(self, part: Part) -> Iterator[Tuple[str, AttributeValue]]:
        # https://github.com/googleapis/python-genai/blob/main/google/genai/types.py#L566
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
