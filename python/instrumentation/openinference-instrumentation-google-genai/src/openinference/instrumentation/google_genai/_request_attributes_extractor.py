import base64
import logging
from enum import Enum
from typing import (
    Any,
    Iterator,
    Mapping,
    Optional,
)

from google.genai._transformers import t_contents
from google.genai.types import (
    Content,
    FunctionResponse,
    GenerateContentConfig,
    Part,
)
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.google_genai._utils import (
    _get_attributes_from_file_data,
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
                logger.warning("Failed to normalize config")
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
                    logger.warning("Failed to normalize system instruction")

        if input_contents := request_parameters.get("contents"):
            try:
                for content in t_contents(input_contents):
                    for message_attributes in self._iter_messages_from_content(content):
                        for attr, value in message_attributes:
                            yield (
                                f"{SpanAttributes.LLM_INPUT_MESSAGES}.{input_messages_index}.{attr}",
                                value,
                            )
                        input_messages_index += 1
            except Exception:
                logger.warning(f"Failed to normalize input contents of type {type(input_contents)}")

    def _get_attributes_from_content(
        self, content: Content
    ) -> Iterator[tuple[str, AttributeValue]]:
        yield from self._get_attributes_from_message(
            content.role, list(content.parts) if content.parts else []
        )

    def _iter_messages_from_content(
        self, content: Content
    ) -> Iterator[Iterator[tuple[str, AttributeValue]]]:
        parts = list(content.parts or [])
        function_response_parts = [part for part in parts if part.function_response is not None]
        if not function_response_parts:
            yield self._get_attributes_from_message(content.role, parts)
            return
        # Each function response becomes its own tool message, so every id
        # survives and each result can be matched back to its call.
        other_parts = [part for part in parts if part.function_response is None]
        if other_parts:
            yield self._get_attributes_from_message(content.role, other_parts)
        for part in function_response_parts:
            yield self._get_attributes_from_function_response_part(part.function_response)

    def _get_attributes_from_message(
        self, role: Optional[str], parts: list[Part]
    ) -> Iterator[tuple[str, AttributeValue]]:
        if role:
            yield (
                MessageAttributes.MESSAGE_ROLE,
                role.value if isinstance(role, Enum) else role,
            )
        else:
            yield (MessageAttributes.MESSAGE_ROLE, "user")
        if parts:
            yield from self._flatten_parts(parts)

    def _get_attributes_from_function_response_part(
        self, function_response: FunctionResponse
    ) -> Iterator[tuple[str, AttributeValue]]:
        yield (MessageAttributes.MESSAGE_ROLE, "tool")
        if name := function_response.name:
            yield (MessageAttributes.MESSAGE_NAME, name)
        if response := function_response.response:
            yield (MessageAttributes.MESSAGE_CONTENT, safe_json_dumps(response))
        if id_ := function_response.id:
            yield (MessageAttributes.MESSAGE_TOOL_CALL_ID, id_)

    def _flatten_parts(self, parts: list[Part]) -> Iterator[tuple[str, AttributeValue]]:
        content_index = 0
        tool_call_index = 0
        for part in parts:
            increment_content_index = False
            thought: Optional[bool] = part.thought
            thought_signature: Optional[bytes] = part.thought_signature
            if thought:
                text = part.text
                if text or thought_signature is not None:
                    prefix = f"{MessageAttributes.MESSAGE_CONTENTS}.{content_index}"
                    yield (
                        f"{prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}",
                        "reasoning",
                    )
                    if text:
                        yield (
                            f"{prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}",
                            text,
                        )
                    if thought_signature is not None:
                        yield (
                            f"{prefix}.{MessageContentAttributes.MESSAGE_CONTENT_SIGNATURE}",
                            base64.b64encode(thought_signature).decode(),
                        )
                    increment_content_index = True
            elif (text := part.text) is not None:
                if thought_signature is not None:
                    # Force indexed format so we can attach the signature
                    prefix = f"{MessageAttributes.MESSAGE_CONTENTS}.{content_index}"
                    yield (f"{prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", "text")
                    yield (f"{prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}", text)
                    yield (
                        f"{prefix}.{MessageContentAttributes.MESSAGE_CONTENT_SIGNATURE}",
                        base64.b64encode(thought_signature).decode(),
                    )
                    increment_content_index = True
                elif len(parts) == 1:
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
                if thought_signature is not None:
                    yield (
                        f"{tc}.{ToolCallAttributes.TOOL_CALL_REASONING_SIGNATURE}",
                        base64.b64encode(thought_signature).decode(),
                    )
                tool_call_index += 1
            if inline_data := part.inline_data:
                inline_attributes = dict(
                    _get_attributes_from_inline_data(inline_data, content_index)
                )
                if inline_attributes:
                    for key, value in inline_attributes.items():
                        yield key, value
                    increment_content_index = True
            if file_data := part.file_data:
                file_attributes = dict(_get_attributes_from_file_data(file_data, content_index))
                if file_attributes:
                    for key, value in file_attributes.items():
                        yield key, value
                    increment_content_index = True
            if increment_content_index:
                content_index += 1
