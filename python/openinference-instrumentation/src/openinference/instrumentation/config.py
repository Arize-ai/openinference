import os
from dataclasses import dataclass, fields
from types import TracebackType
from typing import (
    Any,
    Callable,
    Optional,
    Type,
    Union,
    get_args,
)

from opentelemetry.context import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    attach,
    detach,
    set_value,
)
from opentelemetry.util.types import AttributeValue

from openinference.semconv.trace import (
    EmbeddingAttributes,
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
)

from .logging import logger


class suppress_tracing:
    """
    Context manager to pause OpenTelemetry instrumentation.

    Examples:
        with suppress_tracing():
            # No tracing will occur within this block
            ...
    """

    def __enter__(self) -> "suppress_tracing":
        self._token = attach(set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
        return self

    def __aenter__(self) -> "suppress_tracing":
        self._token = attach(set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        detach(self._token)

    def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        detach(self._token)


# Replacement value for hidden/redacted sensitive data
REDACTED_VALUE = "__REDACTED__"

# Default values for trace configuration, keyed by field names in TraceConfig
_TRACE_CONFIG_DEFAULTS = {
    "hide_llm_invocation_parameters": False,
    "hide_inputs": False,
    "hide_outputs": False,
    "hide_input_messages": False,
    "hide_output_messages": False,
    "hide_input_images": False,
    "hide_input_text": False,
    "hide_output_text": False,
    "hide_embeddings_vectors": False,
    "hide_embeddings_text": False,
    "hide_prompts": False,
    "base64_image_max_length": 32_000,  # 32KB
}


@dataclass(frozen=True)
class TraceConfig:
    """
    TraceConfig helps you modify the observability level of your tracing.
    For instance, you may want to keep sensitive information from being logged
    for security reasons, or you may want to limit the size of the base64
    encoded images to reduce payloads.

    For those attributes not passed, this object tries to read from designated
    environment variables (computed as f"OPENINFERENCE_{field_name.upper()}") and,
    if not found, falls back to default values that maximize observability.
    """

    hide_llm_invocation_parameters: Optional[bool] = None
    """Removes llm.invocation_parameters attribute entirely from spans"""
    hide_inputs: Optional[bool] = None
    """Replaces input.value with REDACTED_VALUE and removes input.mime_type"""
    hide_outputs: Optional[bool] = None
    """Replaces output.value with REDACTED_VALUE and removes output.mime_type"""
    hide_input_messages: Optional[bool] = None
    """Removes all llm.input_messages attributes entirely from spans"""
    hide_output_messages: Optional[bool] = None
    """Removes all llm.output_messages attributes entirely from spans"""
    hide_input_images: Optional[bool] = None
    """Removes image URLs from llm.input_messages message content blocks"""
    hide_input_text: Optional[bool] = None
    """Replaces text content in llm.input_messages with REDACTED_VALUE"""
    hide_output_text: Optional[bool] = None
    """Replaces text content in llm.output_messages with REDACTED_VALUE"""
    hide_embeddings_vectors: Optional[bool] = None
    """Replaces embedding.embeddings.*.embedding.vector values with REDACTED_VALUE"""
    hide_embeddings_text: Optional[bool] = None
    """Replaces embedding.embeddings.*.embedding.text values with REDACTED_VALUE"""
    hide_prompts: Optional[bool] = None
    """Replaces llm.prompts values with REDACTED_VALUE"""
    base64_image_max_length: Optional[int] = None
    """Truncates base64-encoded images to this length, replacing excess with REDACTED_VALUE"""

    def __post_init__(self) -> None:
        for f in fields(self):
            expected_type = get_args(f.type)[0]
            self._parse_value(
                f.name,
                expected_type,
                f"OPENINFERENCE_{f.name.upper()}",
                _TRACE_CONFIG_DEFAULTS[f.name],
            )

    def mask(
        self,
        key: str,
        value: Union[AttributeValue, Callable[[], AttributeValue]],
    ) -> Optional[AttributeValue]:
        if self.hide_llm_invocation_parameters and key == SpanAttributes.LLM_INVOCATION_PARAMETERS:
            return None
        elif self.hide_prompts and key == SpanAttributes.LLM_PROMPTS:
            value = REDACTED_VALUE
        elif self.hide_inputs and key == SpanAttributes.INPUT_VALUE:
            value = REDACTED_VALUE
        elif self.hide_inputs and key == SpanAttributes.INPUT_MIME_TYPE:
            return None
        elif self.hide_outputs and key == SpanAttributes.OUTPUT_VALUE:
            value = REDACTED_VALUE
        elif self.hide_outputs and key == SpanAttributes.OUTPUT_MIME_TYPE:
            return None
        elif (
            self.hide_inputs or self.hide_input_messages
        ) and SpanAttributes.LLM_INPUT_MESSAGES in key:
            return None
        elif (
            self.hide_outputs or self.hide_output_messages
        ) and SpanAttributes.LLM_OUTPUT_MESSAGES in key:
            return None
        elif (
            self.hide_input_text
            and SpanAttributes.LLM_INPUT_MESSAGES in key
            and MessageAttributes.MESSAGE_CONTENT in key
            and MessageAttributes.MESSAGE_CONTENTS not in key
        ):
            value = REDACTED_VALUE
        elif (
            self.hide_output_text
            and SpanAttributes.LLM_OUTPUT_MESSAGES in key
            and MessageAttributes.MESSAGE_CONTENT in key
            and MessageAttributes.MESSAGE_CONTENTS not in key
        ):
            value = REDACTED_VALUE
        elif (
            self.hide_input_text
            and SpanAttributes.LLM_INPUT_MESSAGES in key
            and MessageContentAttributes.MESSAGE_CONTENT_TEXT in key
        ):
            value = REDACTED_VALUE
        elif (
            self.hide_output_text
            and SpanAttributes.LLM_OUTPUT_MESSAGES in key
            and MessageContentAttributes.MESSAGE_CONTENT_TEXT in key
        ):
            value = REDACTED_VALUE
        elif (
            self.hide_input_images
            and SpanAttributes.LLM_INPUT_MESSAGES in key
            and MessageContentAttributes.MESSAGE_CONTENT_IMAGE in key
        ):
            return None
        elif (
            is_base64_url(value)  # type:ignore
            and len(value) > self.base64_image_max_length  # type:ignore
            and SpanAttributes.LLM_INPUT_MESSAGES in key
            and MessageContentAttributes.MESSAGE_CONTENT_IMAGE in key
            and key.endswith(ImageAttributes.IMAGE_URL)
        ):
            value = REDACTED_VALUE
        elif (
            self.hide_embeddings_vectors
            and SpanAttributes.EMBEDDING_EMBEDDINGS in key
            and EmbeddingAttributes.EMBEDDING_VECTOR in key
        ):
            value = REDACTED_VALUE
        elif (
            self.hide_embeddings_text
            and SpanAttributes.EMBEDDING_EMBEDDINGS in key
            and EmbeddingAttributes.EMBEDDING_TEXT in key
        ):
            value = REDACTED_VALUE
        return value() if callable(value) else value

    def _parse_value(
        self,
        field_name: str,
        expected_type: Any,
        env_var: str,
        default_value: Any,
    ) -> None:
        type_name = expected_type.__name__
        init_value = getattr(self, field_name, None)
        if init_value is None:
            env_value = os.getenv(env_var)
            if env_value is None:
                object.__setattr__(self, field_name, default_value)
            else:
                try:
                    env_value = self._cast_value(env_value, expected_type)
                    object.__setattr__(self, field_name, env_value)
                except Exception:
                    logger.warning(
                        f"Could not parse '{env_value}' to {type_name} "
                        f"for the environment variable '{env_var}'. "
                        f"Using default value instead: {default_value}."
                    )
                    object.__setattr__(self, field_name, default_value)
        else:
            if not isinstance(init_value, expected_type):
                raise TypeError(
                    f"The field {field_name} must be of type '{type_name}' "
                    f"but '{type(init_value).__name__}' was found."
                )

    def _cast_value(
        self,
        value: Any,
        cast_to: Any,
    ) -> Any:
        if cast_to is bool:
            if isinstance(value, str) and value.lower() == "true":
                return True
            if isinstance(value, str) and value.lower() == "false":
                return False
            raise
        else:
            return cast_to(value)


def is_base64_url(url: str) -> bool:
    if not isinstance(url, str):
        return False
    return url.startswith("data:image/") and "base64" in url
