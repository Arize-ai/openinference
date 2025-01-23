import asyncio
import collections.abc
import inspect
import json
import os
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from json import JSONEncoder
from secrets import randbits
from types import ModuleType, TracebackType
from typing import (  #  type: ignore[attr-defined]
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    _TypedDictMeta,
    cast,
    get_args,
    get_origin,
)

import wrapt  # type: ignore[import-untyped]
from opentelemetry.context import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    Context,
    attach,
    detach,
    get_value,
    set_value,
)
from opentelemetry.sdk.trace import TracerProvider as OTelTracerProvider
from opentelemetry.sdk.trace.id_generator import IdGenerator
from opentelemetry.trace import (
    INVALID_SPAN,
    INVALID_SPAN_ID,
    INVALID_TRACE_ID,
    Link,
    Span,
    SpanKind,
    Status,
    StatusCode,
    Tracer,
    use_span,
)
from opentelemetry.util.types import Attributes, AttributeValue
from typing_extensions import ParamSpec, TypeAlias, TypeGuard, _AnnotatedAlias, overload

from openinference.semconv.trace import (
    EmbeddingAttributes,
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

from .context_attributes import get_attributes_from_context
from .logging import logger

if TYPE_CHECKING:
    from _typeshed import DataclassInstance
pydantic: Optional[ModuleType]
try:
    import pydantic  # try to import without adding a dependency
except ImportError:
    pydantic = None


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
ParametersType = ParamSpec("ParametersType")
ReturnType = TypeVar("ReturnType")


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
            self.hide_embedding_vectors
            and SpanAttributes.EMBEDDING_EMBEDDINGS in key
            and EmbeddingAttributes.EMBEDDING_VECTOR in key
        ):
            return None
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


_IMPORTANT_ATTRIBUTES = [
    SpanAttributes.OPENINFERENCE_SPAN_KIND,
]


def get_span_kind_attributes(kind: OpenInferenceSpanKind, /) -> Dict[str, AttributeValue]:
    normalized_kind = _normalize_openinference_span_kind(kind)
    return {
        OPENINFERENCE_SPAN_KIND: normalized_kind.value,
    }


def get_input_attributes(
    value: Any,
    *,
    mime_type: Optional[OpenInferenceMimeType] = None,
) -> Dict[str, AttributeValue]:
    normalized_mime_type: Optional[OpenInferenceMimeTypeValues] = None
    if mime_type is not None:
        normalized_mime_type = _normalize_mime_type(mime_type)
    if normalized_mime_type is OpenInferenceMimeTypeValues.TEXT:
        value = str(value)
    elif normalized_mime_type is OpenInferenceMimeTypeValues.JSON:
        if not isinstance(value, str):
            value = _json_serialize(value)
    else:
        value, normalized_mime_type = _infer_serialized_io_value_and_mime_type(value)
    attributes = {
        INPUT_VALUE: value,
    }
    if normalized_mime_type is not None:
        attributes[INPUT_MIME_TYPE] = normalized_mime_type.value
    return attributes


def get_output_attributes(
    value: Any,
    *,
    mime_type: Optional[OpenInferenceMimeType] = None,
) -> Dict[str, AttributeValue]:
    normalized_mime_type: Optional[OpenInferenceMimeTypeValues] = None
    if mime_type is not None:
        normalized_mime_type = _normalize_mime_type(mime_type)
    if normalized_mime_type is OpenInferenceMimeTypeValues.TEXT:
        value = str(value)
    elif normalized_mime_type is OpenInferenceMimeTypeValues.JSON:
        if not isinstance(value, str):
            value = _json_serialize(value)
    else:
        value, normalized_mime_type = _infer_serialized_io_value_and_mime_type(value)
    attributes = {
        OUTPUT_VALUE: value,
    }
    if normalized_mime_type is not None:
        attributes[OUTPUT_MIME_TYPE] = normalized_mime_type.value
    return attributes


def _infer_serialized_io_value_and_mime_type(
    value: Any,
) -> Tuple[Any, Optional[OpenInferenceMimeTypeValues]]:
    if isinstance(value, str):
        return value, OpenInferenceMimeTypeValues.TEXT
    if isinstance(value, (bool, int, float)):
        return value, None
    if isinstance(value, Sequence):
        for element_type in (str, bool, int, float):
            if all(isinstance(element, element_type) for element in value):
                return value, None
        return _json_serialize(value), OpenInferenceMimeTypeValues.JSON
    if isinstance(value, Mapping):
        return _json_serialize(value), OpenInferenceMimeTypeValues.JSON
    if _is_dataclass_instance(value):
        return _json_serialize(value), OpenInferenceMimeTypeValues.JSON
    if pydantic is not None and isinstance(value, pydantic.BaseModel):
        return _json_serialize(value), OpenInferenceMimeTypeValues.JSON
    return str(value), OpenInferenceMimeTypeValues.TEXT


class IOValueJSONEncoder(JSONEncoder):
    def default(self, obj: Any) -> Any:
        try:
            if _is_dataclass_instance(obj):
                return asdict(obj)
            if pydantic is not None and isinstance(obj, pydantic.BaseModel):
                return obj.model_dump()
            if isinstance(obj, datetime):
                return obj.isoformat()
            return super().default(obj)
        except Exception:
            return str(obj)


def _json_serialize(obj: Any, **kwargs: Any) -> str:
    """
    Safely JSON dumps input and handles special types such as dataclasses and
    pydantic models.
    """
    return json.dumps(
        obj,
        cls=IOValueJSONEncoder,
        ensure_ascii=False,
    )


def get_tool_attributes(
    *,
    name: str,
    description: Optional[str] = None,
    parameters: Union[str, Dict[str, Any]],
) -> Dict[str, AttributeValue]:
    if isinstance(parameters, str):
        parameters_json = parameters
    elif isinstance(parameters, Mapping):
        parameters_json = _json_serialize(parameters)
    else:
        raise ValueError(f"Invalid parameters type: {type(parameters)}")
    attributes: Dict[str, AttributeValue] = {
        TOOL_NAME: name,
        TOOL_PARAMETERS: parameters_json,
    }
    if description is not None:
        attributes[TOOL_DESCRIPTION] = description
    return attributes


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
        masked_value = self._self_config.mask(key, value)
        if masked_value is not None:
            if key in _IMPORTANT_ATTRIBUTES:
                self._self_important_attributes[key] = masked_value
            else:
                span = cast(Span, self.__wrapped__)
                span.set_attribute(key, masked_value)

    def end(self, end_time: Optional[int] = None) -> None:
        span = cast(Span, self.__wrapped__)
        for k, v in reversed(self._self_important_attributes.items()):
            span.set_attribute(k, v)
        span.end(end_time)

    def set_input(
        self,
        value: Any,
        *,
        mime_type: Optional[OpenInferenceMimeType] = None,
    ) -> None:
        if OPENINFERENCE_SPAN_KIND not in self._self_important_attributes:
            raise ValueError("Cannot set input attributes on a non-OpenInference span")
        self.set_attributes(get_input_attributes(value, mime_type=mime_type))

    def set_output(
        self,
        value: Any,
        *,
        mime_type: Optional[OpenInferenceMimeType] = None,
    ) -> None:
        if OPENINFERENCE_SPAN_KIND not in self._self_important_attributes:
            raise ValueError("Cannot set output attributes on a non-OpenInference span")
        self.set_attributes(get_output_attributes(value, mime_type=mime_type))

    def set_tool(
        self,
        *,
        name: str,
        description: Optional[str] = None,
        parameters: Union[str, Dict[str, Any]],
    ) -> None:
        if self._self_important_attributes.get(OPENINFERENCE_SPAN_KIND) != TOOL:
            raise ValueError("Cannot set tool attributes on a non-tool span")
        self.set_attributes(
            get_tool_attributes(
                name=name,
                description=description,
                parameters=parameters,
            )
        )


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
        links: Optional["Sequence[Link]"] = (),
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
        ) as current_span:
            yield cast(OpenInferenceSpan, current_span)

    def start_span(
        self,
        name: str,
        context: Optional[Context] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Attributes = None,
        links: Optional["Sequence[Link]"] = (),
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        *,
        openinference_span_kind: Optional[OpenInferenceSpanKind] = None,
    ) -> OpenInferenceSpan:
        span: Span
        if get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            span = INVALID_SPAN
        else:
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
        span = OpenInferenceSpan(span, config=self._self_config)
        if attributes:
            span.set_attributes(dict(attributes))
        if openinference_span_kind is not None:
            span.set_attributes(get_span_kind_attributes(openinference_span_kind))
        span.set_attributes(dict(get_attributes_from_context()))
        return span

    @overload  # for @tracer.agent usage (no parameters)
    def agent(
        self,
        wrapped_function: Callable[ParametersType, ReturnType],
        /,
        *,
        name: None = None,
    ) -> Callable[ParametersType, ReturnType]: ...

    @overload  # for @tracer.agent(name="name") usage (with parameters)
    def agent(
        self,
        wrapped_function: None = None,
        /,
        *,
        name: Optional[str] = None,
    ) -> Callable[[Callable[ParametersType, ReturnType]], Callable[ParametersType, ReturnType]]: ...

    def agent(
        self,
        wrapped_function: Optional[Callable[ParametersType, ReturnType]] = None,
        /,
        *,
        name: Optional[str] = None,
    ) -> Union[
        Callable[ParametersType, ReturnType],
        Callable[[Callable[ParametersType, ReturnType]], Callable[ParametersType, ReturnType]],
    ]:
        return self._chain(
            wrapped_function,
            kind=OpenInferenceSpanKindValues.AGENT,  # chains and agents differ only in span kind
            name=name,
        )

    @overload  # for @tracer.chain usage (no parameters)
    def chain(
        self,
        wrapped_function: Callable[ParametersType, ReturnType],
        /,
        *,
        name: None = None,
    ) -> Callable[ParametersType, ReturnType]: ...

    @overload  # for @tracer.chain(name="name") usage (with parameters)
    def chain(
        self,
        wrapped_function: None = None,
        /,
        *,
        name: Optional[str] = None,
    ) -> Callable[[Callable[ParametersType, ReturnType]], Callable[ParametersType, ReturnType]]: ...

    def chain(
        self,
        wrapped_function: Optional[Callable[ParametersType, ReturnType]] = None,
        /,
        *,
        name: Optional[str] = None,
    ) -> Union[
        Callable[ParametersType, ReturnType],
        Callable[[Callable[ParametersType, ReturnType]], Callable[ParametersType, ReturnType]],
    ]:
        return self._chain(wrapped_function, kind=OpenInferenceSpanKindValues.CHAIN, name=name)

    def _chain(
        self,
        wrapped_function: Optional[Callable[ParametersType, ReturnType]] = None,
        /,
        *,
        kind: OpenInferenceSpanKindValues,
        name: Optional[str] = None,
    ) -> Union[
        Callable[ParametersType, ReturnType],
        Callable[[Callable[ParametersType, ReturnType]], Callable[ParametersType, ReturnType]],
    ]:
        @wrapt.decorator  # type: ignore[misc]
        def sync_wrapper(
            wrapped: Callable[ParametersType, ReturnType],
            instance: Any,
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
        ) -> ReturnType:
            tracer = self
            with _chain_context(
                tracer=tracer,
                name=name,
                kind=kind,
                wrapped=wrapped,
                instance=instance,
                args=args,
                kwargs=kwargs,
            ) as chain_context:
                output = wrapped(*args, **kwargs)
                return chain_context.process_output(output)

        @wrapt.decorator  #  type: ignore[misc]
        async def async_wrapper(
            wrapped: Callable[ParametersType, Awaitable[ReturnType]],
            instance: Any,
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
        ) -> ReturnType:
            tracer = self
            with _chain_context(
                tracer=tracer,
                name=name,
                kind=kind,
                wrapped=wrapped,
                instance=instance,
                args=args,
                kwargs=kwargs,
            ) as chain_context:
                output = await wrapped(*args, **kwargs)
                return chain_context.process_output(output)

        if wrapped_function is not None:
            if asyncio.iscoroutinefunction(wrapped_function):
                return async_wrapper(wrapped_function)  # type: ignore[no-any-return]
            return sync_wrapper(wrapped_function)  # type: ignore[no-any-return]
        return lambda f: async_wrapper(f) if asyncio.iscoroutinefunction(f) else sync_wrapper(f)

    @overload  # for @tracer.tool usage (no parameters)
    def tool(
        self,
        wrapped_function: Callable[ParametersType, ReturnType],
        /,
        *,
        name: None = None,
        description: Optional[str] = None,
        parameters: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Callable[ParametersType, ReturnType]: ...

    @overload  # for @tracer.tool(name="name") usage (with parameters)
    def tool(
        self,
        wrapped_function: None = None,
        /,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Callable[[Callable[ParametersType, ReturnType]], Callable[ParametersType, ReturnType]]: ...

    def tool(
        self,
        wrapped_function: Optional[Callable[ParametersType, ReturnType]] = None,
        /,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Union[
        Callable[ParametersType, ReturnType],
        Callable[[Callable[ParametersType, ReturnType]], Callable[ParametersType, ReturnType]],
    ]:
        @wrapt.decorator  # type: ignore[misc]
        def sync_wrapper(
            wrapped: Callable[ParametersType, ReturnType],
            instance: Any,
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
        ) -> ReturnType:
            tracer = self
            with _tool_context(
                tracer=tracer,
                name=name,
                description=description,
                parameters=parameters,
                wrapped=wrapped,
                instance=instance,
                args=args,
                kwargs=kwargs,
            ) as tool_context:
                output = wrapped(*args, **kwargs)
                return tool_context.process_output(output)

        @wrapt.decorator  #  type: ignore[misc]
        async def async_wrapper(
            wrapped: Callable[ParametersType, Awaitable[ReturnType]],
            instance: Any,
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
        ) -> ReturnType:
            tracer = self
            with _tool_context(
                tracer=tracer,
                name=name,
                description=description,
                parameters=parameters,
                wrapped=wrapped,
                instance=instance,
                args=args,
                kwargs=kwargs,
            ) as tool_context:
                output = await wrapped(*args, **kwargs)
                return tool_context.process_output(output)

        if wrapped_function is not None:
            if asyncio.iscoroutinefunction(wrapped_function):
                return async_wrapper(wrapped_function)  # type: ignore[no-any-return]
            return sync_wrapper(wrapped_function)  # type: ignore[no-any-return]
        return lambda f: async_wrapper(f) if asyncio.iscoroutinefunction(f) else sync_wrapper(f)


class TracerProvider(OTelTracerProvider):
    def __init__(
        self,
        *args: Any,
        config: Optional[TraceConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._oi_trace_config = config or TraceConfig()

    def get_tracer(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> OITracer:
        tracer = super().get_tracer(*args, **kwargs)
        return OITracer(tracer, config=self._oi_trace_config)


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
    if not kind.islower():
        raise ValueError("kind must be lowercase if provided as a string")
    try:
        return OpenInferenceSpanKindValues(kind.upper())
    except ValueError:
        raise ValueError(f"Invalid OpenInference span kind: {kind}")


def _is_dataclass_instance(obj: Any) -> TypeGuard["DataclassInstance"]:
    """
    dataclasses.is_dataclass return true for both dataclass types and instances.
    This function returns true only for instances.

    See https://github.com/python/cpython/blob/05d12eecbde1ace39826320cadf8e673d709b229/Lib/dataclasses.py#L1391
    """
    cls = type(obj)
    return hasattr(cls, "__dataclass_fields__")


class _ChainContext:
    def __init__(self, span: OpenInferenceSpan) -> None:
        self._span = span

    def process_output(self, output: ReturnType) -> ReturnType:
        attributes = getattr(self._span, "attributes", {})
        has_output = OUTPUT_VALUE in attributes
        if not has_output:
            self._span.set_output(value=output)
        return output


@contextmanager
def _chain_context(
    *,
    tracer: "OITracer",
    name: Optional[str],
    kind: OpenInferenceSpanKindValues,
    wrapped: Callable[ParametersType, ReturnType],
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Iterator[_ChainContext]:
    span_name = name or _infer_span_name(instance=instance, callable=wrapped)
    bound_args = inspect.signature(wrapped).bind(*args, **kwargs)
    bound_args.apply_defaults()
    arguments = bound_args.arguments

    if len(arguments) == 1:
        argument = next(iter(arguments.values()))
        input_attributes = get_input_attributes(argument)
    else:
        input_attributes = get_input_attributes(arguments)

    with tracer.start_as_current_span(
        span_name,
        openinference_span_kind=kind,
        attributes=input_attributes,
    ) as span:
        context = _ChainContext(span=span)
        yield context
        span.set_status(Status(StatusCode.OK))


class _ToolContext:
    def __init__(self, span: OpenInferenceSpan) -> None:
        self._span = span

    def process_output(self, output: ReturnType) -> ReturnType:
        attributes = getattr(self._span, "attributes", {})
        has_output = OUTPUT_VALUE in attributes
        if not has_output:
            self._span.set_output(value=output)
        return output


@contextmanager
def _tool_context(
    *,
    tracer: "OITracer",
    name: Optional[str],
    description: Optional[str],
    parameters: Optional[Union[str, Dict[str, Any]]],
    wrapped: Callable[ParametersType, ReturnType],
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Iterator[_ToolContext]:
    tool_name = name or _infer_span_name(instance=instance, callable=wrapped)
    bound_args = inspect.signature(wrapped).bind(*args, **kwargs)
    bound_args.apply_defaults()
    arguments = bound_args.arguments
    input_attributes = get_input_attributes(arguments)
    tool_description = description or _infer_tool_description(wrapped)
    tool_parameters = parameters or _infer_tool_parameters(
        callable=wrapped,
        tool_name=tool_name,
        tool_description=tool_description,
    )
    tool_attributes = get_tool_attributes(
        name=tool_name,
        description=tool_description,
        parameters=tool_parameters,
    )
    with tracer.start_as_current_span(
        tool_name,
        openinference_span_kind=OpenInferenceSpanKindValues.TOOL,
        attributes={
            **input_attributes,
            **tool_attributes,
        },
    ) as span:
        context = _ToolContext(span=span)
        yield context
        span.set_status(Status(StatusCode.OK))


def _infer_span_name(*, instance: Any, callable: Callable[..., Any]) -> str:
    """
    Makes a best-effort attempt to infer a span name from the bound instance
    (e.g., self or cls) and the callable (the function or method being wrapped).
    Handles functions, methods, and class methods.
    """

    if inspect.ismethod(callable):
        is_class_method = isinstance(instance, type)
        if is_class_method:
            class_name = instance.__name__
        else:  # regular method
            class_name = instance.__class__.__name__
        method_name = callable.__name__
        return f"{class_name}.{method_name}"
    function_name = callable.__name__
    return function_name


def _infer_tool_description(callable: Callable[..., Any]) -> Optional[str]:
    """
    Infers a tool description from the callable's docstring if one exists.
    """
    docstring = callable.__doc__
    if docstring is not None and (stripped_docstring := docstring.strip()):
        return stripped_docstring
    return None


def _infer_tool_parameters(
    *,
    callable: Callable[..., Any],
    tool_name: str,
    tool_description: Optional[str],
) -> Dict[str, Any]:
    json_schema: Dict[str, Any] = {"type": "object", "title": tool_name}
    if tool_description:
        json_schema["description"] = tool_description
    properties = {}
    required_properties = []
    signature = inspect.signature(callable)
    for param_name, param in signature.parameters.items():
        property_data = {}
        default_value = param.default
        has_default = default_value is not inspect.Parameter.empty
        if has_default:
            property_data["default"] = default_value
        if not has_default:
            required_properties.append(param_name)
        annotation = param.annotation
        property_data.update(_get_jsonschema_type(annotation))
        metadata = getattr(annotation, "__metadata__", None)
        description: Optional[str] = None
        if metadata and isinstance(first_metadata := metadata[0], str):
            description = first_metadata
        if description:
            property_data["description"] = description
        properties[param_name] = property_data

    json_schema["properties"] = properties
    if required_properties:
        json_schema["required"] = required_properties
    return json_schema


def _get_jsonschema_type(annotation_type: type) -> Dict[str, Any]:
    if isinstance(annotation_type, _AnnotatedAlias):
        annotation_type = annotation_type.__args__[0]
        return _get_jsonschema_type(annotation_type)
    if annotation_type is type(None) or annotation_type is None:
        return {"type": "null"}
    if annotation_type is str:
        return {"type": "string"}
    if annotation_type is int:
        return {"type": "integer"}
    if annotation_type is float:
        return {"type": "number"}
    if annotation_type is bool:
        return {"type": "boolean"}
    if annotation_type is datetime:
        return {
            "type": "string",
            "format": "date-time",
        }
    annotation_type_origin = get_origin(annotation_type)
    annotation_type_args = get_args(annotation_type)
    is_union_type = annotation_type_origin is Union
    if is_union_type:
        jsonschema_types = []
        for type_ in annotation_type_args:
            jsonschema_types.append(_get_jsonschema_type(type_))
        return {"anyOf": jsonschema_types}
    is_literal_type = annotation_type_origin is Literal
    if is_literal_type:
        enum_values = list(annotation_type_args)
        unique_enum_types = dict.fromkeys(type(value) for value in enum_values)
        jsonschema_types = [_get_jsonschema_type(value_type) for value_type in unique_enum_types]
        result = {}
        if len(jsonschema_types) == 1:
            result.update(jsonschema_types[0])
        elif len(jsonschema_types) > 1:
            result["anyOf"] = jsonschema_types
        result["enum"] = enum_values
        return result
    is_list_type = (
        annotation_type_origin is list or annotation_type_origin is collections.abc.Sequence
    )
    if is_list_type:
        result = {"type": "array"}
        if len(annotation_type_args) == 1:
            list_item_type = annotation_type_args[0]
            result["items"] = _get_jsonschema_type(list_item_type)
        return result
    is_tuple_type = annotation_type_origin is tuple
    if is_tuple_type:
        result = {"type": "array"}
        if len(annotation_type_args) == 2 and annotation_type_args[-1] is Ellipsis:
            item_type = annotation_type_args[0]
            result["items"] = _get_jsonschema_type(item_type)
        elif annotation_type_args:
            items = []
            for arg_type in annotation_type_args:
                item_schema = _get_jsonschema_type(arg_type)
                items.append(item_schema)
            result["items"] = items
            result["minItems"] = len(annotation_type_args)
            result["maxItems"] = len(annotation_type_args)
        return result
    is_dict_type = (
        annotation_type_origin is dict or annotation_type_origin is collections.abc.Mapping
    )
    if is_dict_type:
        result = {"type": "object"}
        if len(annotation_type_args) == 2:
            # jsonschema requires that the keys in object type are strings, so
            # we ignore the key type
            _, value_type = annotation_type_args
            result["additionalProperties"] = _get_jsonschema_type(value_type)
        return result
    is_typed_dict_type = isinstance(annotation_type, _TypedDictMeta)
    if is_typed_dict_type:
        result = {"type": "object"}
        properties = {}
        for field_name, field_type in annotation_type.__annotations__.items():
            properties[field_name] = _get_jsonschema_type(field_type)
        result["properties"] = properties
        return result
    if (
        pydantic is not None
        and isinstance(annotation_type, type)
        and issubclass(annotation_type, pydantic.BaseModel)
    ):
        return annotation_type.schema()  # type: ignore[no-any-return,attr-defined]
    return {}


# span kinds
TOOL = OpenInferenceSpanKindValues.TOOL.value

# span attributes
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
TOOL_DESCRIPTION = SpanAttributes.TOOL_DESCRIPTION
TOOL_NAME = SpanAttributes.TOOL_NAME
TOOL_PARAMETERS = SpanAttributes.TOOL_PARAMETERS
