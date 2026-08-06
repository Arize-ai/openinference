import inspect
import os
from dataclasses import dataclass, field, fields
from types import TracebackType
from typing import (
    Any,
    Callable,
    Dict,
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

from ._blob_upload import (
    BlobUploader,
    decode_base64_data_uri_to_blob,
    is_valid_reference_uri,
    load_blob_uploader,
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


OPENINFERENCE_HIDE_LLM_INVOCATION_PARAMETERS = "OPENINFERENCE_HIDE_LLM_INVOCATION_PARAMETERS"
OPENINFERENCE_HIDE_LLM_TOOLS = "OPENINFERENCE_HIDE_LLM_TOOLS"
# Hides the tool definitions advertised to the LLM
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
# Deprecated: use OPENINFERENCE_HIDE_EMBEDDINGS_VECTORS instead
OPENINFERENCE_HIDE_EMBEDDINGS_VECTORS = "OPENINFERENCE_HIDE_EMBEDDINGS_VECTORS"
# Hides embedding vectors
OPENINFERENCE_HIDE_EMBEDDINGS_TEXT = "OPENINFERENCE_HIDE_EMBEDDINGS_TEXT"
# Hides embedding text
OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH = "OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH"
# Limits characters of a base64 encoding of an image
OPENINFERENCE_BLOB_UPLOADER = "OPENINFERENCE_BLOB_UPLOADER"
# Names a BlobUploader registered under the
# "openinference_blob_uploader" entry-point group; base64 images exceeding
# base64_image_max_length are handed to it and the attribute records the
# returned URI
OPENINFERENCE_HIDE_PROMPTS = "OPENINFERENCE_HIDE_PROMPTS"
# Hides LLM prompts (completions API)
OPENINFERENCE_HIDE_CHOICES = "OPENINFERENCE_HIDE_CHOICES"
# Hides LLM choices (completions API outputs)
OPENINFERENCE_ENABLE_GENAI_SEMCONV = "OPENINFERENCE_ENABLE_GENAI_SEMCONV"
# Emits OTel GenAI semantic conventions alongside OpenInference attributes
REDACTED_VALUE = "__REDACTED__"
# When a value is hidden, it will be replaced by this redacted value

DEFAULT_HIDE_LLM_INVOCATION_PARAMETERS = False
DEFAULT_HIDE_LLM_TOOLS = False
DEFAULT_HIDE_PROMPTS = False
DEFAULT_HIDE_CHOICES = False
DEFAULT_ENABLE_GENAI_SEMCONV = False
DEFAULT_HIDE_INPUTS = False
DEFAULT_HIDE_OUTPUTS = False

DEFAULT_HIDE_INPUT_MESSAGES = False
DEFAULT_HIDE_OUTPUT_MESSAGES = False

DEFAULT_HIDE_INPUT_IMAGES = False
DEFAULT_HIDE_INPUT_TEXT = False
DEFAULT_HIDE_OUTPUT_TEXT = False

DEFAULT_HIDE_EMBEDDING_VECTORS = False  # Deprecated
DEFAULT_HIDE_EMBEDDINGS_VECTORS = False
DEFAULT_HIDE_EMBEDDINGS_TEXT = False
DEFAULT_BASE64_IMAGE_MAX_LENGTH = 32_000


@dataclass(frozen=True)
class TraceConfig:
    """
    TraceConfig helps you modify the observability level of your tracing.
    For instance, you may want to keep sensitive information from being logged
    for security reasons, or you may want to limit the size of the base64
    encoded images to reduce payloads.

    For those attributes not passed, this object tries to read from designated
    environment variables and, if not found, has default values that maximize
    observability.
    """

    hide_llm_invocation_parameters: Optional[bool] = field(
        default=None,
        metadata={
            "env_var": OPENINFERENCE_HIDE_LLM_INVOCATION_PARAMETERS,
            "default_value": DEFAULT_HIDE_LLM_INVOCATION_PARAMETERS,
        },
    )
    hide_llm_tools: Optional[bool] = field(
        default=None,
        metadata={
            "env_var": OPENINFERENCE_HIDE_LLM_TOOLS,
            "default_value": DEFAULT_HIDE_LLM_TOOLS,
        },
    )
    """Hides the tool definitions advertised to the LLM"""
    hide_inputs: Optional[bool] = field(
        default=None,
        metadata={
            "env_var": OPENINFERENCE_HIDE_INPUTS,
            "default_value": DEFAULT_HIDE_INPUTS,
        },
    )
    """Hides input value & messages"""
    hide_outputs: Optional[bool] = field(
        default=None,
        metadata={
            "env_var": OPENINFERENCE_HIDE_OUTPUTS,
            "default_value": DEFAULT_HIDE_OUTPUTS,
        },
    )
    """Hides output value & messages"""
    hide_input_messages: Optional[bool] = field(
        default=None,
        metadata={
            "env_var": OPENINFERENCE_HIDE_INPUT_MESSAGES,
            "default_value": DEFAULT_HIDE_INPUT_MESSAGES,
        },
    )
    """Hides all input messages"""
    hide_output_messages: Optional[bool] = field(
        default=None,
        metadata={
            "env_var": OPENINFERENCE_HIDE_OUTPUT_MESSAGES,
            "default_value": DEFAULT_HIDE_OUTPUT_MESSAGES,
        },
    )
    """Hides all output messages"""
    hide_input_images: Optional[bool] = field(
        default=None,
        metadata={
            "env_var": OPENINFERENCE_HIDE_INPUT_IMAGES,
            "default_value": DEFAULT_HIDE_INPUT_IMAGES,
        },
    )
    """Hides images from input messages"""
    hide_input_text: Optional[bool] = field(
        default=None,
        metadata={
            "env_var": OPENINFERENCE_HIDE_INPUT_TEXT,
            "default_value": DEFAULT_HIDE_INPUT_TEXT,
        },
    )
    """Hides text from input messages"""
    hide_output_text: Optional[bool] = field(
        default=None,
        metadata={
            "env_var": OPENINFERENCE_HIDE_OUTPUT_TEXT,
            "default_value": DEFAULT_HIDE_OUTPUT_TEXT,
        },
    )
    """Hides text from output messages"""
    hide_embedding_vectors: Optional[bool] = field(
        default=None,
        metadata={
            "env_var": OPENINFERENCE_HIDE_EMBEDDING_VECTORS,
            "default_value": DEFAULT_HIDE_EMBEDDING_VECTORS,
        },
    )
    """Deprecated: use hide_embeddings_vectors instead"""
    hide_embeddings_vectors: Optional[bool] = field(
        default=None,
        metadata={
            "env_var": OPENINFERENCE_HIDE_EMBEDDINGS_VECTORS,
            "default_value": DEFAULT_HIDE_EMBEDDINGS_VECTORS,
        },
    )
    """Hides embedding vectors"""
    hide_embeddings_text: Optional[bool] = field(
        default=None,
        metadata={
            "env_var": OPENINFERENCE_HIDE_EMBEDDINGS_TEXT,
            "default_value": DEFAULT_HIDE_EMBEDDINGS_TEXT,
        },
    )
    """Hides embedding text"""
    hide_prompts: Optional[bool] = field(
        default=None,
        metadata={
            "env_var": OPENINFERENCE_HIDE_PROMPTS,
            "default_value": DEFAULT_HIDE_PROMPTS,
        },
    )
    """Hides LLM prompts (completions API)"""
    hide_choices: Optional[bool] = field(
        default=None,
        metadata={
            "env_var": OPENINFERENCE_HIDE_CHOICES,
            "default_value": DEFAULT_HIDE_CHOICES,
        },
    )
    """Hides LLM choices (completions API outputs)"""
    enable_genai_semconv: Optional[bool] = field(
        default=None,
        metadata={
            "env_var": OPENINFERENCE_ENABLE_GENAI_SEMCONV,
            "default_value": DEFAULT_ENABLE_GENAI_SEMCONV,
        },
    )
    """Emits OTel GenAI semantic conventions alongside OpenInference attributes"""
    base64_image_max_length: Optional[int] = field(
        default=None,
        metadata={
            "env_var": OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH,
            "default_value": DEFAULT_BASE64_IMAGE_MAX_LENGTH,
        },
    )
    """Limits characters of a base64 encoding of an image"""
    blob_uploader: Optional[BlobUploader] = field(
        default=None,
        metadata={"env_var": None, "default_value": None},
    )
    """Uploads base64 images exceeding base64_image_max_length
    to external storage and records the destination URI in the span attribute
    instead of redacting. OpenInference ships no implementation — pass any
    object satisfying the BlobUploader protocol, or set the
    OPENINFERENCE_BLOB_UPLOADER environment variable to the name of an
    uploader registered under the "openinference_blob_uploader" entry-point
    group."""

    def __post_init__(self) -> None:
        for f in fields(self):
            if f.metadata.get("env_var") is None:
                continue
            expected_type = get_args(f.type)[0]
            # Optional is Union[T,NoneType]. get_args()returns (T, NoneType).
            # We collect the first type
            self._parse_value(
                f.name,
                expected_type,
                f.metadata["env_var"],
                f.metadata["default_value"],
            )
        if self.blob_uploader is None:
            self._init_blob_uploader_from_env()
        else:
            self._validate_blob_uploader()

    def _init_blob_uploader_from_env(self) -> None:
        name = os.getenv(OPENINFERENCE_BLOB_UPLOADER)
        if not name:
            return
        if (uploader := load_blob_uploader(name)) is not None:
            object.__setattr__(self, "blob_uploader", uploader)

    def _validate_blob_uploader(self) -> None:
        uploader = self.blob_uploader
        if isinstance(uploader, str):
            raise TypeError(
                f"blob_uploader must be a BlobUploader instance, not the string {uploader!r}. "
                "To select a registered uploader by name, set the "
                f"{OPENINFERENCE_BLOB_UPLOADER} environment variable, or pass "
                f"load_blob_uploader({uploader!r})."
            )
        if inspect.isclass(uploader):
            raise TypeError(
                f"blob_uploader must be a BlobUploader instance, not the class "
                f"{uploader.__name__}; pass an instance, e.g. "
                f"blob_uploader={uploader.__name__}(...)."
            )
        if not isinstance(uploader, BlobUploader):
            raise TypeError(
                "blob_uploader must satisfy the BlobUploader protocol — "
                "upload(blob) -> Optional[str] and shutdown(timeout_sec) — got "
                f"{type(uploader).__name__}."
            )

    def mask(
        self,
        key: str,
        value: Union[AttributeValue, Callable[[], AttributeValue]],
        *,
        externalize: bool = True,
    ) -> Optional[AttributeValue]:
        if self.hide_llm_invocation_parameters and key == SpanAttributes.LLM_INVOCATION_PARAMETERS:
            return None
        elif (self.hide_inputs or self.hide_llm_tools) and SpanAttributes.LLM_TOOLS in key:
            return None
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
        elif (self.hide_inputs or self.hide_prompts) and SpanAttributes.LLM_PROMPTS in key:
            value = REDACTED_VALUE
        elif (
            self.hide_outputs or self.hide_output_messages
        ) and SpanAttributes.LLM_OUTPUT_MESSAGES in key:
            return None
        elif (self.hide_outputs or self.hide_choices) and SpanAttributes.LLM_CHOICES in key:
            value = REDACTED_VALUE
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
            (SpanAttributes.LLM_INPUT_MESSAGES in key or SpanAttributes.LLM_OUTPUT_MESSAGES in key)
            and MessageContentAttributes.MESSAGE_CONTENT_IMAGE in key
            and key.endswith(ImageAttributes.IMAGE_URL)
        ):
            # Resolve lazy values before the size check so an oversized image
            # cannot bypass the budget by arriving as a callable.
            value = value() if callable(value) else value
            if (
                is_base64_url(value)  # type:ignore
                and len(value) > self.base64_image_max_length  # type:ignore
            ):
                value = (
                    self._externalize_or_redact(key, value)  # type:ignore
                    if externalize
                    else REDACTED_VALUE
                )
        elif (
            (self.hide_embedding_vectors or self.hide_embeddings_vectors)
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

    def _externalize_or_redact(self, key: str, value: str) -> AttributeValue:
        """
        Uploads an oversized base64 data URI to external storage and returns
        the destination URI, falling back to redaction when no uploader is
        configured, the upload cannot be accepted, or the returned reference
        is not a valid absolute URI.
        """
        if self.blob_uploader is not None:
            try:
                # The base64 decode and sha256 digest run synchronously on the
                # instrumented call path before the uploader can refuse the
                # blob. Payload size is effectively bounded by provider
                # request limits today; if that stops holding, a pre-decode
                # ceiling on the base64 string length belongs here, as a
                # separate knob from the inline-vs-offload budget.
                if blob := decode_base64_data_uri_to_blob(value, attribute_key=key):
                    if uri := self.blob_uploader.upload(blob):
                        if is_valid_reference_uri(uri):
                            return uri
                        logger.warning(
                            f"Blob uploader {type(self.blob_uploader).__name__} returned "
                            f"{uri!r} for attribute '{key}', which is not a valid absolute "
                            "URI (a scheme such as https:// or s3:// is required). "
                            "Falling back to redaction."
                        )
            except Exception:
                logger.exception(f"Failed to externalize media for attribute '{key}'.")
        return REDACTED_VALUE

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


_MASK_EXTERNALIZE_SUPPORT: Dict[type, bool] = {}


def _mask_supports_externalize(config_type: Type[TraceConfig]) -> bool:
    if config_type not in _MASK_EXTERNALIZE_SUPPORT:
        try:
            supported = "externalize" in inspect.signature(config_type.mask).parameters
        except (TypeError, ValueError):
            supported = False
        _MASK_EXTERNALIZE_SUPPORT[config_type] = supported
    return _MASK_EXTERNALIZE_SUPPORT[config_type]


def mask_without_externalization(
    config: TraceConfig,
    key: str,
    value: Union[AttributeValue, Callable[[], AttributeValue]],
) -> Optional[AttributeValue]:
    """
    Applies ``config.mask`` as a pure view — no blob-upload side effects.
    Falls back to the two-argument form for TraceConfig subclasses that
    override ``mask`` without the ``externalize`` keyword.
    """
    if _mask_supports_externalize(type(config)):
        return config.mask(key, value, externalize=False)
    return config.mask(key, value)


def is_base64_url(url: str) -> bool:
    if not isinstance(url, str):
        return False
    return url.startswith("data:image/") and "base64" in url
