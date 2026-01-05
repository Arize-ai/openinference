import logging
from typing import Any, Iterator, Mapping, Tuple

from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.google_genai._utils import (
    _as_input_attributes,
    _get_attributes_from_message_param,
    _get_tools_from_config,
    _io_value_and_type,
    _serialize_config_safely,
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

        # Extract tools as high-priority attributes (avoid 128 attribute limit dropping them)
        if isinstance(request_parameters, Mapping):
            if config := request_parameters.get("config", None):
                yield from _get_tools_from_config(config)

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
            # config can either be a TypedDict or a pydantic object so we need to handle both cases
            if isinstance(config, dict):
                config_json = safe_json_dumps(config)
            else:
                config_json = _serialize_config_safely(config)
            yield (
                SpanAttributes.LLM_INVOCATION_PARAMETERS,
                config_json,
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

            # Tools are now extracted in get_attributes_from_request for higher priority

        if input_contents := request_parameters.get("contents"):
            if isinstance(input_contents, list):
                for input_content in input_contents:
                    for attr, value in _get_attributes_from_message_param(input_content):
                        yield (
                            f"{SpanAttributes.LLM_INPUT_MESSAGES}.{input_messages_index}.{attr}",
                            value,
                        )
                    # Move on to the next message
                    input_messages_index += 1
            else:
                for attr, value in _get_attributes_from_message_param(input_contents):
                    # Default to index 0 for a single message
                    yield (
                        f"{SpanAttributes.LLM_INPUT_MESSAGES}.{input_messages_index}.{attr}",
                        value,
                    )
