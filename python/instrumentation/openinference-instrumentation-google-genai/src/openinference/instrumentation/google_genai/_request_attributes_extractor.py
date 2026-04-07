import logging
from enum import Enum
from typing import (
    Any,
    Iterator,
    Mapping,
)

from google.genai._transformers import t_contents
from google.genai.types import (
    Content,
    GenerateContentConfig,
    Part,
)
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.google_genai._utils import (
    _get_attributes_from_inline_data,
)
from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceLLMProviderValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)

__all__ = ("_RequestAttributesExtractor",)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _RequestAttributesExtractor:
    def get_attributes_from_request(
        self,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[tuple[str, AttributeValue]]:
        yield SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value
        yield SpanAttributes.LLM_PROVIDER, OpenInferenceLLMProviderValues.GOOGLE.value
        if model := request_parameters.get("model"):
            if isinstance(model, str):
                yield SpanAttributes.LLM_MODEL_NAME, model
        # input_value, tools, and invocation_parameters are extracted via
        # ContextVar in the wrappers after the SDK call transforms them.

        # Start an index for the messages since we want to start with system instruction
        input_messages_index = 0
        if not isinstance(request_parameters, Mapping):
            return

        if config := request_parameters.get("config", None):
            # Normalize config to a GenerateContentConfig for direct attribute access
            try:
                if isinstance(config, dict):
                    config = GenerateContentConfig.model_validate(config)
            except Exception:
                logger.exception("Failed to normalize config")
                config = None

            # System instruction as the first message
            if config and (system_instruction := config.system_instruction):
                try:
                    for content in t_contents(system_instruction):
                        for attr, value in self._get_attributes_from_content(content):
                            yield (
                                f"{SpanAttributes.LLM_INPUT_MESSAGES}.{input_messages_index}.{attr}",
                                value,
                            )
                        # Override role to "system"
                        yield (
                            f"{SpanAttributes.LLM_INPUT_MESSAGES}.{input_messages_index}.{MessageAttributes.MESSAGE_ROLE}",
                            "system",
                        )
                        input_messages_index += 1
                except Exception:
                    logger.exception("Failed to normalize system instruction")

        if input_contents := request_parameters.get("contents"):
            try:
                for content in t_contents(input_contents):
                    for attr, value in self._get_attributes_from_content(content):
                        yield (
                            f"{SpanAttributes.LLM_INPUT_MESSAGES}.{input_messages_index}.{attr}",
                            value,
                        )
                    input_messages_index += 1
            except Exception:
                logger.exception(
                    f"Failed to normalize input contents of type {type(input_contents)}"
                )

    def _get_attributes_from_content(
        self, content: Content
    ) -> Iterator[tuple[str, AttributeValue]]:
        if role := content.role:
            yield (
                MessageAttributes.MESSAGE_ROLE,
                role.value if isinstance(role, Enum) else role,
            )
        else:
            yield (MessageAttributes.MESSAGE_ROLE, "user")
        if parts := content.parts:
            yield from self._flatten_parts(parts)

    def _flatten_parts(self, parts: list[Part]) -> Iterator[tuple[str, AttributeValue]]:
        content_index = 0
        tool_call_index = 0
        for part in parts:
            increment_content_index = False
            if (text := part.text) is not None:
                if len(parts) == 1:
                    yield (MessageAttributes.MESSAGE_CONTENT, text)
                else:
                    prefix = f"{MessageAttributes.MESSAGE_CONTENTS}.{content_index}"
                    yield (
                        f"{prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}",
                        "text",
                    )
                    yield (
                        f"{prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}",
                        text,
                    )
                    increment_content_index = True
            elif function_call := part.function_call:
                tc = f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{tool_call_index}"
                if name := function_call.name:
                    yield (f"{tc}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}", name)
                if args := function_call.args:
                    yield (
                        f"{tc}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                        safe_json_dumps(args),
                    )
                if id_ := function_call.id:
                    yield (f"{tc}.{ToolCallAttributes.TOOL_CALL_ID}", id_)
                tool_call_index += 1
            elif function_response := part.function_response:
                if response := function_response.response:
                    if len(parts) == 1:
                        yield (MessageAttributes.MESSAGE_CONTENT, safe_json_dumps(response))
                    else:
                        prefix = f"{MessageAttributes.MESSAGE_CONTENTS}.{content_index}"
                        yield (f"{prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", "text")
                        yield (
                            f"{prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}",
                            safe_json_dumps(response),
                        )
                        increment_content_index = True
                if id_ := function_response.id:
                    yield (MessageAttributes.MESSAGE_TOOL_CALL_ID, id_)
            if inline_data := part.inline_data:
                inline_attributes = dict(
                    _get_attributes_from_inline_data(inline_data, content_index)
                )
                if inline_attributes:
                    for key, value in inline_attributes.items():
                        yield key, value
                    increment_content_index = True
            else:
                logger.debug("Unsupported part type: %s", type(part))
            if increment_content_index:
                content_index += 1
