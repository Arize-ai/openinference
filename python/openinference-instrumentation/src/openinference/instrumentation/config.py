import os
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from secrets import randbits
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
    get_args,
)

import wrapt
from opentelemetry.context import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    Context,
    attach,
    detach,
    set_value,
)
from opentelemetry.sdk.trace import IdGenerator
from opentelemetry.trace import (
    INVALID_SPAN_ID,
    INVALID_TRACE_ID,
    Link,
    Span,
    SpanKind,
    Tracer,
    use_span,
)
from opentelemetry.util.types import Attributes, AttributeValue
from typing_extensions import TypeAlias

from openinference.semconv.trace import (
    EmbeddingAttributes,
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

from .logging import logger

OpenInferenceMimeType: TypeAlias = Union[
    Literal["application/json", "text/plain"],
    OpenInferenceMimeTypeValues,
]
OpenInferenceSpanKind: TypeAlias = Union[
    Literal[
        "agent",
        "chain",
        "embedding",
        "evaluator",
        "guardrail",
        "llm",
        "reranker",
        "retriever",
        "tool",
        "unknown",
    ],
    OpenInferenceSpanKindValues,
]


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


OPENINFERENCE_HIDE_LLM_INVOCATION_PARAMETERS = "OPENINFERENCE_HIDE_LLM_INVOCATION_PARAMETERS"
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
# Hides embedding vectors
OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH = "OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH"
# Limits characters of a base64 encoding of an image
REDACTED_VALUE = "__REDACTED__"
# When a value is hidden, it will be replaced by this redacted value

DEFAULT_HIDE_LLM_INVOCATION_PARAMETERS = False
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
    """Hides embedding vectors"""
    base64_image_max_length: Optional[int] = field(
        default=None,
        metadata={
            "env_var": OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH,
            "default_value": DEFAULT_BASE64_IMAGE_MAX_LENGTH,
        },
    )
    """Limits characters of a base64 encoding of an image"""

    def __post_init__(self) -> None:
        for f in fields(self):
            expected_type = get_args(f.type)[0]
            # Optional is Union[T,NoneType]. get_args()returns (T, NoneType).
            # We collect the first type
            self._parse_value(
                f.name,
                expected_type,
                f.metadata["env_var"],
                f.metadata["default_value"],
            )

    def mask(
        self,
        key: str,
        value: Union[AttributeValue, Callable[[], AttributeValue]],
    ) -> Optional[AttributeValue]:
        if self.hide_llm_invocation_parameters and key == SpanAttributes.LLM_INVOCATION_PARAMETERS:
            return
        elif self.hide_inputs and key == SpanAttributes.INPUT_VALUE:
            value = REDACTED_VALUE
        elif self.hide_inputs and key == SpanAttributes.INPUT_MIME_TYPE:
            return
        elif self.hide_outputs and key == SpanAttributes.OUTPUT_VALUE:
            value = REDACTED_VALUE
        elif self.hide_outputs and key == SpanAttributes.OUTPUT_MIME_TYPE:
            return
        elif (
            self.hide_inputs or self.hide_input_messages
        ) and SpanAttributes.LLM_INPUT_MESSAGES in key:
            return
        elif (
            self.hide_outputs or self.hide_output_messages
        ) and SpanAttributes.LLM_OUTPUT_MESSAGES in key:
            return
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
            return
        elif (
            is_base64_url(value)  # type:ignore
            and len(value) > self.base64_image_max_length  # type:ignore
            and SpanAttributes.LLM_INPUT_MESSAGES in key
            and MessageContentAttributes.MESSAGE_CONTENT_IMAGE in key
            and key.endswith(ImageAttributes.IMAGE_URL)
        ):
            value = REDACTED_VALUE
        elif (
            self.hide_embedding_vectors
            and SpanAttributes.EMBEDDING_EMBEDDINGS in key
            and EmbeddingAttributes.EMBEDDING_VECTOR in key
        ):
            return
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
    ) -> None:
        if cast_to is bool:
            if isinstance(value, str) and value.lower() == "true":
                return True
            if isinstance(value, str) and value.lower() == "false":
                return False
            raise
        else:
            return cast_to(value)


_IMPORTANT_ATTRIBUTES = [
    SpanAttributes.OPENINFERENCE_SPAN_KIND,
]


def get_openinference_span_kind(kind: OpenInferenceSpanKind) -> Dict[str, AttributeValue]:
    normalized_kind = _normalize_openinference_span_kind(kind)
    return {
        OPENINFERENCE_SPAN_KIND: normalized_kind.value,
    }


def get_input_value_and_mime_type(
    value: str,
    mime_type: OpenInferenceMimeType = OpenInferenceMimeTypeValues.TEXT,
) -> Dict[str, AttributeValue]:
    mime_type = _normalize_mime_type(mime_type)
    return {
        INPUT_VALUE: value,
        INPUT_MIME_TYPE: mime_type.value,
    }


def get_output_value_and_mime_type(
    value: Union[int, float, bool, str],
    mime_type: OpenInferenceMimeType = OpenInferenceMimeTypeValues.TEXT,
) -> Dict[str, AttributeValue]:
    mime_type = _normalize_mime_type(mime_type)
    return {
        OUTPUT_VALUE: value,
        OUTPUT_MIME_TYPE: mime_type.value,
    }


class OpenInferenceSpan(wrapt.ObjectProxy):  # type: ignore[misc]
    def __init__(self, wrapped: Span, config: TraceConfig) -> None:
        super().__init__(wrapped)
        self._self_config = config
        self._self_important_attributes: Dict[str, AttributeValue] = {}

    def set_attributes(self, attributes: Dict[str, AttributeValue]) -> None:
        for k, v in attributes.items():
            self.set_attribute(k, v)

    def set_attribute(
        self,
        key: str,
        value: Union[AttributeValue, Callable[[], AttributeValue]],
    ) -> None:
        value = self._self_config.mask(key, value)
        if value is not None:
            if key in _IMPORTANT_ATTRIBUTES:
                self._self_important_attributes[key] = value
            else:
                span = cast(Span, self.__wrapped__)
                span.set_attribute(key, value)

    def end(self, end_time: Optional[int] = None) -> None:
        span = cast(Span, self.__wrapped__)
        for k, v in reversed(self._self_important_attributes.items()):
            span.set_attribute(k, v)
        span.end(end_time)

    def set_input(
        self,
        value: Any,
        mime_type: OpenInferenceMimeType = OpenInferenceMimeTypeValues.TEXT,
    ) -> None:
        self.set_attributes(get_input_value_and_mime_type(value, mime_type))

    def set_output(
        self,
        value: Any,
        mime_type: OpenInferenceMimeType = OpenInferenceMimeTypeValues.TEXT,
    ) -> None:
        self.set_attributes(get_output_value_and_mime_type(value, mime_type))


class ChainSpan(OpenInferenceSpan):
    def __init__(self, wrapped: Span, config: TraceConfig) -> None:
        super().__init__(wrapped, config)
        self.__wrapped__.set_attributes(get_openinference_span_kind(CHAIN))


class _IdGenerator(IdGenerator):
    """
    An IdGenerator that uses a different source of randomness to
    avoid being affected by seeds set by user application.
    """

    def generate_span_id(self) -> int:
        while (span_id := randbits(64)) == INVALID_SPAN_ID:
            continue
        return span_id

    def generate_trace_id(self) -> int:
        while (trace_id := randbits(128)) == INVALID_TRACE_ID:
            continue
        return trace_id


class OITracer(wrapt.ObjectProxy):  # type: ignore[misc]
    def __init__(self, wrapped: Tracer, config: TraceConfig) -> None:
        super().__init__(wrapped)
        self._self_config = config
        self._self_id_generator = _IdGenerator()

    @property
    def id_generator(self) -> IdGenerator:
        return self._self_id_generator

    @contextmanager
    def start_as_current_span(
        self,
        name: str,
        context: Optional[Context] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Attributes = None,
        links: Optional[Sequence[Link]] = (),
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
        *,
        openinference_span_kind: Optional[OpenInferenceSpanKind] = None,
    ) -> Iterator[OpenInferenceSpan]:
        span = self.start_span(
            name=name,
            openinference_span_kind=openinference_span_kind,
            context=context,
            kind=kind,
            attributes=attributes,
            links=links,
            start_time=start_time,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
        )
        with use_span(
            span,
            end_on_exit=end_on_exit,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
        ) as span:
            yield span

    def start_span(
        self,
        name: str,
        context: Optional[Context] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Attributes = None,
        links: Optional[Sequence[Link]] = (),
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        *,
        openinference_span_kind: Optional[OpenInferenceSpanKind] = None,
    ) -> OpenInferenceSpan:
        tracer = cast(Tracer, self.__wrapped__)
        span = tracer.__class__.start_span(
            self,
            name=name,
            context=context,
            kind=kind,
            attributes=None,
            links=links,
            start_time=start_time,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
        )
        span_wrapper_cls = OpenInferenceSpan
        if openinference_span_kind is not None:
            normalized_span_kind = _normalize_openinference_span_kind(openinference_span_kind)
            span_wrapper_cls = _get_span_wrapper_cls(normalized_span_kind)
        span = span_wrapper_cls(span, config=self._self_config)
        if attributes:
            span.set_attributes(attributes)
        return span


def is_base64_url(url: str) -> bool:
    if not isinstance(url, str):
        return False
    return url.startswith("data:image/") and "base64" in url


def _normalize_mime_type(mime_type: OpenInferenceMimeType) -> OpenInferenceMimeTypeValues:
    if isinstance(mime_type, OpenInferenceMimeTypeValues):
        return mime_type
    try:
        return OpenInferenceMimeTypeValues(mime_type)
    except ValueError:
        raise ValueError(f"Invalid mime type: {mime_type}")


def _normalize_openinference_span_kind(kind: OpenInferenceSpanKind) -> OpenInferenceSpanKindValues:
    if isinstance(kind, OpenInferenceSpanKindValues):
        return kind
    try:
        return OpenInferenceSpanKindValues(kind.upper())
    except ValueError:
        raise ValueError(f"Invalid OpenInference span kind: {kind}")


def _get_span_wrapper_cls(kind: OpenInferenceSpanKindValues) -> Type[OpenInferenceSpan]:
    if kind is OpenInferenceSpanKindValues.CHAIN:
        return ChainSpan
    raise NotImplementedError(f"Span kind {kind.value} is not yet supported")


# span attributes
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE

# mime types
JSON = OpenInferenceMimeTypeValues.JSON.value
TEXT = OpenInferenceMimeTypeValues.TEXT.value

# span kinds
CHAIN = OpenInferenceSpanKindValues.CHAIN.value
