import os
from dataclasses import dataclass
from typing import Any, Optional

from opentelemetry.context import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    attach,
    detach,
    set_value,
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

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        detach(self._token)

    def __aexit__(self, exc_type, exc_value, traceback) -> None:
        detach(self._token)


OPENINFERENCE_HIDE_INPUTS = "OPENINFERENCE_HIDE_INPUTS"
# Hides input value & messages
OPENINFERENCE_HIDE_OUTPUTS = "OPENINFERENCE_HIDE_OUTPUTS"
# Hides output value & messages
OPENINFERENCE_HIDE_INPUT_MESSAGES = "OPENINFERENCE_HIDE_INPUT_MESSAGES"
# Hides all input messages
OPENINFERENCE_HIDE_OUTPUT_MESSAGES = "OPENINFERENCE_HIDE_OUTPUT_MESSAGES"
# Hides all output messages
OPENINFERENCE_HIDE_INPUT_IMAGES = "OPENINFERENCE_HIDE_INPUT_IMAGES"
# Hides images from input messages
OPENINFERENCE_HIDE_INPUT_TEXT = "OPENINFERENCE_HIDE_INPUT_TEXT"
# Hides text from input messages
OPENINFERENCE_HIDE_OUTPUT_TEXT = "OPENINFERENCE_HIDE_OUTPUT_TEXT"
# Hides text from output messages
OPENINFERENCE_HIDE_EMBEDDING_VECTORS = "OPENINFERENCE_HIDE_EMBEDDING_VECTORS"
# Hides returned embedding vectors
OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH = "OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH"
# Limits characters of a base64 encoding of an image
REDACTED_VALUE = "__REDACTED__"
# When a value is hidden, it will be replaced by this redacted value

DEFAULT_HIDE_INPUTS = False
DEFAULT_HIDE_OUTPUTS = False

DEFAULT_HIDE_INPUT_MESSAGES = False
DEFAULT_HIDE_OUTPUT_MESSAGES = False

DEFAULT_HIDE_INPUT_IMAGES = False
DEFAULT_HIDE_INPUT_TEXT = False
DEFAULT_HIDE_OUTPUT_TEXT = False

DEFAULT_HIDE_EMBEDDING_VECTORS = False
DEFAULT_BASE64_IMAGE_MAX_LENGTH = 32_000


@dataclass(frozen=True)
class TraceConfig:
    hide_inputs: Optional[bool] = None
    hide_outputs: Optional[bool] = None
    hide_input_messages: Optional[bool] = None
    hide_output_messages: Optional[bool] = None
    hide_input_images: Optional[bool] = None
    hide_input_text: Optional[bool] = None
    hide_output_text: Optional[bool] = None
    hide_embedding_vectors: Optional[bool] = None
    base64_image_max_length: Optional[int] = None

    def __post_init__(self) -> None:
        self._parse_value(
            "hide_inputs",
            bool,
            OPENINFERENCE_HIDE_INPUTS,
            DEFAULT_HIDE_INPUTS,
        )
        self._parse_value(
            "hide_outputs",
            bool,
            OPENINFERENCE_HIDE_OUTPUTS,
            DEFAULT_HIDE_OUTPUTS,
        )
        self._parse_value(
            "hide_input_messages",
            bool,
            OPENINFERENCE_HIDE_INPUT_MESSAGES,
            DEFAULT_HIDE_INPUT_MESSAGES,
        )
        self._parse_value(
            "hide_output_messages",
            bool,
            OPENINFERENCE_HIDE_OUTPUT_MESSAGES,
            DEFAULT_HIDE_OUTPUT_MESSAGES,
        )
        self._parse_value(
            "hide_input_images",
            bool,
            OPENINFERENCE_HIDE_INPUT_IMAGES,
            DEFAULT_HIDE_INPUT_IMAGES,
        )
        self._parse_value(
            "hide_input_text",
            bool,
            OPENINFERENCE_HIDE_INPUT_TEXT,
            DEFAULT_HIDE_INPUT_TEXT,
        )
        self._parse_value(
            "hide_output_text",
            bool,
            OPENINFERENCE_HIDE_OUTPUT_TEXT,
            DEFAULT_HIDE_OUTPUT_TEXT,
        )
        self._parse_value(
            "hide_embedding_vectors",
            bool,
            OPENINFERENCE_HIDE_EMBEDDING_VECTORS,
            DEFAULT_HIDE_EMBEDDING_VECTORS,
        )
        self._parse_value(
            "base64_image_max_length",
            int,
            OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH,
            DEFAULT_BASE64_IMAGE_MAX_LENGTH,
        )

    def _parse_value(self, field_name: str, cast_to: Any, env_var: str, default_value: Any) -> None:
        init_value = object.__getattribute__(self, field_name)
        if init_value is None:
            env_value = os.getenv(env_var)
            if env_value is None:
                object.__setattr__(self, field_name, default_value)
            else:
                try:
                    env_value = cast_to(env_value)
                    object.__setattr__(self, field_name, env_value)
                except Exception:
                    logger.warning(
                        f"Could not parse '{env_value}' to {cast_to.__name__} "
                        f"for the environment variable '{env_var}'. "
                        f"Using default value instead: {default_value}."
                    )
                    object.__setattr__(self, field_name, default_value)
        else:
            if not isinstance(init_value, cast_to):
                raise TypeError(
                    f"The field {field_name} must be of type "
                    f"{cast_to.__name__} but '{type(init_value)}' was found."
                )
