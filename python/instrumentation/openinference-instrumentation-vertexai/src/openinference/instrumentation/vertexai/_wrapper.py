import base64
import inspect
import json
import logging
from functools import singledispatch
from itertools import chain
from typing import (
    Any,
    AsyncIterable,
    Awaitable,
    Callable,
    ContextManager,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import proto
import wrapt
from google.cloud import aiplatform_v1 as v1  # Stable
from google.cloud import aiplatform_v1beta1 as v1beta1  # Supports latest preview features
from openinference.instrumentation import get_attributes_from_context, safe_json_dumps
from openinference.instrumentation.vertexai import _instrumentation_status
from openinference.instrumentation.vertexai._accumulator import (
    _IndexedAccumulator,
    _KeyValuesAccumulator,
    _StringAccumulator,
)
from openinference.instrumentation.vertexai._proxy import _proxy
from openinference.semconv.trace import (
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace import Span, Status, StatusCode, use_span
from opentelemetry.util.types import AttributeValue
from typing_extensions import TypeAlias

__all__ = ("_Wrapper",)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_AnyT = TypeVar("_AnyT")

# https://cloud.google.com/vertex-ai/docs/reference#versions
# - v1: Stable
# - v1beta1: Supports latest preview features
Content: TypeAlias = Union[v1.Content, v1beta1.Content]
Candidate: TypeAlias = Union[v1.Candidate, v1beta1.Candidate]
Part: TypeAlias = Union[v1.Part, v1beta1.Part]
UsageMetadata: TypeAlias = Union[
    v1.GenerateContentResponse.UsageMetadata,
    v1beta1.GenerateContentResponse.UsageMetadata,
]
GenerateContentRequest: TypeAlias = Union[
    v1.GenerateContentRequest,
    v1beta1.GenerateContentRequest,
]
GenerateContentResponse: TypeAlias = Union[
    v1.GenerateContentResponse,
    v1beta1.GenerateContentResponse,
]

_PREDICTION_SERVICE_REQUESTS = sorted(
    {
        name
        for module in (
            # https://cloud.google.com/vertex-ai/docs/reference#versions
            # - v1: Stable
            # - v1beta1: Supports latest preview features
            v1.types.prediction_service,
            v1beta1.types.prediction_service,
        )
        for name, _ in inspect.getmembers(
            module,
            lambda cls: inspect.isclass(cls)
            and issubclass(cls, proto.Message)
            and cls.__name__.endswith("Request"),
        )
    }
)


class _Wrapper:
    _status = _instrumentation_status
    """
    We can't track all the functions that have been monkey-patched. Instead, all
    monkey-patched functions are inactivated or activated by setting the global
    variable `_instrumentation_status._IS_INSTRUMENTED` to False or True, respectively.
    """

    def __init__(self, tracer: trace_api.Tracer, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tracer = tracer

    @wrapt.decorator  # type: ignore[misc]
    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if (
            not (request := _get_prediction_service_request(chain(args, kwargs.values())))
            or context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY)
            or not self._status._IS_INSTRUMENTED
        ):
            return wrapped(*args, **kwargs)
        span_name = inspect.stack()[1].function
        attributes = dict(get_attributes_from_context())
        span = self._tracer.start_span(name=span_name, attributes=attributes)
        span.set_attribute(OPENINFERENCE_SPAN_KIND, LLM)
        if isinstance(request, proto.Message):
            span.set_attribute(INPUT_VALUE, safe_json_dumps(request.__class__.to_dict(request)))
            span.set_attribute(INPUT_MIME_TYPE, JSON)
        else:
            span.set_attribute(INPUT_VALUE, str(request))
        try:
            _update_span(request, span)
        except BaseException:
            pass
        try:
            with use_span(span, False, False, False):
                result = wrapped(*args, **kwargs)
        except BaseException as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}"))
            span.end()
            raise
        if isinstance(result, Awaitable):
            return _proxy(result, _CallbackForAwaitable(request, span), _use_span(span))
        if isinstance(result, (Iterable, AsyncIterable)):
            return _proxy(result, _CallbackForIterable(request, span), _use_span(span))
        _finish(span, result)
        return result


class _CallbackForAwaitable:
    """
    Callback function for the proxy object when wrapping an Awaitable. This is how we get notified
    when the Awaitable returns: the proxy will invoke this callback with the returned value of the
    Awaitable or when an exception is raised.
    """

    def __init__(self, request: proto.Message, span: Span) -> None:
        self._request = request
        self._span = span

    def __call__(self, obj: _AnyT) -> _AnyT:
        request, span = self._request, self._span
        if isinstance(obj, (Iterable, AsyncIterable)):
            obj = cast(_AnyT, _proxy(obj, _CallbackForIterable(request, span), _use_span(span)))
        else:
            _finish(span, obj)
        return obj


class _CallbackForIterable:
    """
    Callback function for the proxy object when wrapping an Iterable. This is how we get notified
    when the Iterable is being iterated on: the proxy will invoke this callback with the returned
    value of each iteration, as well as when the iteration is done or when an exception is raised.
    """

    def __init__(self, request: proto.Message, span: Span) -> None:
        self._span = span
        self._accumulator = _get_response_accumulator(request)

    def __call__(self, obj: _AnyT) -> _AnyT:
        span = self._span
        if isinstance(obj, proto.Message):
            self._accumulator.accumulate(obj)
        elif isinstance(obj, (StopIteration, StopAsyncIteration)):
            try:
                result = self._accumulator.result
            except BaseException as exc:
                logger.exception(str(exc))
                result = None
            _finish(span, result)
        elif isinstance(obj, BaseException):
            _finish(span, obj)
        return cast(_AnyT, obj)


_IncrementType = TypeVar("_IncrementType", contravariant=True)
_ResultType = TypeVar("_ResultType", covariant=True)


class _ResponseAccumulator(Protocol[_IncrementType, _ResultType]):
    @property
    def result(self) -> Optional[_ResultType]: ...

    def accumulate(self, _: _IncrementType) -> None: ...


def _get_response_accumulator(
    request: proto.Message,
) -> _ResponseAccumulator[proto.Message, proto.Message]:
    if isinstance(request, v1.GenerateContentRequest):
        return _GenerateContentResponseAccumulator(v1.GenerateContentResponse)
    if isinstance(request, v1beta1.GenerateContentRequest):
        return _GenerateContentResponseAccumulator(v1beta1.GenerateContentResponse)
    return _NoOpResponseAccumulator()


class _NoOpResponseAccumulator:
    def accumulate(self, _: Any) -> None: ...

    @property
    def result(self) -> None:
        return None


_GenerateContentResponseT = TypeVar(
    "_GenerateContentResponseT",
    v1.GenerateContentResponse,
    v1beta1.GenerateContentResponse,
)


class _GenerateContentResponseAccumulator:
    _kv: _KeyValuesAccumulator

    def __init__(self, cls: Type[_GenerateContentResponseT]) -> None:
        self._cls: Union[
            Type[v1.GenerateContentResponse],
            Type[v1beta1.GenerateContentResponse],
        ] = cls
        self._is_null = True
        self._cached_result: Optional[_GenerateContentResponseT] = None
        self._kv = _KeyValuesAccumulator(
            candidates=_IndexedAccumulator(
                lambda: _KeyValuesAccumulator(
                    # FIXME: It's unclear how to accumulate `safety_ratings` if there are several.
                    content=_KeyValuesAccumulator(
                        # FIXME: Because `Part` doesn't have `index`, it's unclear what should
                        # happen during streaming if there are multiple separate parts. The current
                        # setup just merges all the parts.
                        parts=_IndexedAccumulator(
                            lambda: _KeyValuesAccumulator(
                                # FIXME: It's unclear whether we should have an accumulator for
                                # `function_call`.
                                text=_StringAccumulator(),
                            )
                        ),
                    ),
                ),
            ),
        )

    def accumulate(self, increment: _GenerateContentResponseT) -> None:
        self._is_null = False
        self._cached_result = None
        kv = increment.__class__.to_dict(increment)
        self._kv += kv

    @property
    def result(self) -> Optional[_GenerateContentResponseT]:
        if self._is_null:
            return None
        if self._cached_result is None:
            obj = dict(self._kv)
            # Note that directly calling `cls(obj)` would fail to handle nested
            # `google.protobuf.struct_pb2.Struct`, e.g. inside `tool.FunctionCall`.
            # See https://github.com/googleapis/proto-plus-python/issues/424
            result = self._cls.from_json(json.dumps(obj))
            self._cached_result = result
        return self._cached_result


def _use_span(span: Span) -> Callable[[], ContextManager[Span]]:
    # The `use_span` context manager can't be entered more than once. It would err here:
    # https://github.com/open-telemetry/opentelemetry-python/blob/b1e99c1555721f818e578d7457587693e767e182/opentelemetry-api/src/opentelemetry/util/_decorator.py#L56  # noqa E501
    # So we need a factory.
    return lambda: cast(ContextManager[Span], use_span(span, False, False, False))


def _finish(span: Span, result: Any) -> None:
    if isinstance(result, BaseException):
        span.record_exception(result)
        span.set_status(Status(StatusCode.ERROR, f"{type(result).__name__}: {result}"))
        span.end()
        return
    if isinstance(result, proto.Message):
        span.set_attribute(OUTPUT_VALUE, safe_json_dumps(result.__class__.to_dict(result)))
        span.set_attribute(OUTPUT_MIME_TYPE, JSON)
    elif result is not None:
        span.set_attribute(OUTPUT_VALUE, str(result))
    try:
        _update_span(result, span)
    except BaseException as exc:
        logger.exception(str(exc))
    span.set_status(Status(StatusCode.OK))
    span.end()


def _get_prediction_service_request(args: Iterable[Any]) -> Optional[proto.Message]:
    for arg in args:
        if type(arg).__name__ in _PREDICTION_SERVICE_REQUESTS:
            return arg
    return None


@singledispatch
def _update_span(obj: Any, span: Span) -> None: ...


@_update_span.register(v1.GenerateContentRequest)
@_update_span.register(v1beta1.GenerateContentRequest)
def _(req: GenerateContentRequest, span: Span) -> None:
    span.set_attribute(LLM_MODEL_NAME, req.model)
    span.set_attribute(
        LLM_INVOCATION_PARAMETERS,
        safe_json_dumps(
            (
                v1.GenerationConfig.to_dict(req.generation_config)
                if isinstance(req.generation_config, v1.GenerationConfig)
                else v1beta1.GenerationConfig.to_dict(req.generation_config)
            )
        ),
    )
    msg_idx = -1
    if (system_instruction := req.system_instruction).parts:
        msg_idx += 1
        prefix = f"{LLM_INPUT_MESSAGES}.{msg_idx}."
        for k, v in _parse_content(cast(Content, system_instruction), prefix, "system"):
            span.set_attribute(k, v)
    for content in cast(Iterable[Content], req.contents):
        msg_idx += 1
        prefix = f"{LLM_INPUT_MESSAGES}.{msg_idx}."
        for k, v in _parse_content(content, prefix):
            span.set_attribute(k, v)


@_update_span.register(v1.GenerateContentResponse)
@_update_span.register(v1beta1.GenerateContentResponse)
def _(resp: GenerateContentResponse, span: Span) -> None:
    for k, v in _parse_usage_metadata(resp.usage_metadata):
        span.set_attribute(k, v)
    for candidate in cast(Iterable[Candidate], resp.candidates):
        prefix = f"{LLM_OUTPUT_MESSAGES}.{candidate.index}."
        for k, v in _parse_content(cast(Content, candidate.content), prefix):
            span.set_attribute(k, v)


def stop_on_exception(it: Callable[..., Iterator[_AnyT]]) -> Callable[..., Iterator[_AnyT]]:
    def _(*args: Any, **kwargs: Any) -> Iterator[_AnyT]:
        try:
            yield from it(*args, **kwargs)
        except Exception as exc:
            logger.exception(str(exc))

    return _


@stop_on_exception
def _parse_usage_metadata(
    usage_metadata: UsageMetadata,
) -> Iterator[Tuple[str, AttributeValue]]:
    if prompt_token_count := usage_metadata.prompt_token_count:
        yield LLM_TOKEN_COUNT_PROMPT, prompt_token_count
    if candidates_token_count := usage_metadata.candidates_token_count:
        yield LLM_TOKEN_COUNT_COMPLETION, candidates_token_count
    if total_token_count := usage_metadata.total_token_count:
        yield LLM_TOKEN_COUNT_TOTAL, total_token_count


@stop_on_exception
def _parse_content(
    content: Content,
    prefix: str = "",
    role_override: Optional[str] = None,
) -> Iterator[Tuple[str, AttributeValue]]:
    """
    Extract semantic convention span attributes as key-value pairs.

    Args:
        content: a `Content` proto.Message
        prefix: optional prefix for the keys, useful when flattening a list,
            e.g. "llm.input_messages.0.", "llm.input_messages.1.", etc.
        role_override: useful for overriding the role attribute, e.g. returning
            "system" even though the raw value is "user".

    Returns:
        Iterator: semantic convention key-value pairs for span attributes.
    """
    yield f"{prefix}{MESSAGE_ROLE}", role_override or _role(content.role)
    parts = cast(Iterable[Part], content.parts)
    for part in parts:
        if part.function_response.name:
            # FIXME: It's unclear whether multiple `function_response` can
            # coexist, but currently we can retain only one.
            yield f"{prefix}{MESSAGE_ROLE}", "tool"
            yield f"{prefix}{MESSAGE_NAME}", part.function_response.name
            cls = part.function_response.__class__
            # Maybe there's an easier way to do this.
            function_response = cls.to_dict(part.function_response)
            yield (
                f"{prefix}{MESSAGE_CONTENT}",
                safe_json_dumps(function_response.get("response") or {}),
            )
    yield from _parse_parts(parts, prefix)
    yield from _parse_tool_calls(parts, prefix)


@stop_on_exception
def _parse_tool_calls(
    parts: Iterable[Part],
    prefix: str = "",
) -> Iterator[Tuple[str, AttributeValue]]:
    """
    Extract semantic convention span attributes as key-value pairs.

    Args:
        parts: an iterable of the `Part` proto.Message
        prefix: optional prefix for the keys, useful when flattening a list,
            e.g. "llm.input_messages.0.", "llm.input_messages.1.", etc.

    Returns:
        Iterator: semantic convention key-value pairs for span attributes.
    """
    idx = -1
    for part in parts:
        if part.function_call.name:
            idx += 1
            inner_prefix = f"{prefix}{MESSAGE_TOOL_CALLS}.{idx}."
            yield f"{inner_prefix}{TOOL_CALL_FUNCTION_NAME}", part.function_call.name
            cls = part.function_call.__class__
            # Maybe there's an easier way to do this.
            function_call = cls.to_dict(part.function_call)
            yield (
                f"{inner_prefix}{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                safe_json_dumps(function_call.get("args") or {}),
            )


@stop_on_exception
def _parse_parts(
    parts: Iterable[Part],
    prefix: str = "",
) -> Iterator[Tuple[str, AttributeValue]]:
    """
    Extract semantic convention span attributes as key-value pairs.

    Args:
        parts: an iterable of the `Part` proto.Message
        prefix: optional prefix for the keys, useful when flattening a list,
            e.g. "llm.input_messages.0.", "llm.input_messages.1.", etc.

    Returns:
        Iterator: semantic convention key-value pairs for span attributes.
    """
    for j, part in enumerate(parts):
        inner_prefix = f"{prefix}{MESSAGE_CONTENTS}.{j}."
        for k, v in _parse_part(part, inner_prefix):
            yield k, v


@stop_on_exception
def _parse_part(
    part: Part,
    prefix: str = "",
) -> Iterator[Tuple[str, AttributeValue]]:
    """
    Extract semantic convention span attributes as key-value pairs.

    Args:
        part: a `Part` proto.Message
        prefix: optional prefix for the keys, useful when flattening a list,
            e.g. "llm.input_messages.0.", "llm.input_messages.1.", etc.

    Returns:
        Iterator: semantic convention key-value pairs for span attributes.
    """
    if part.text:
        yield f"{prefix}{MESSAGE_CONTENT_TEXT}", part.text
    elif part.inline_data.mime_type.startswith("image"):
        yield (
            f"{prefix}{MESSAGE_CONTENT_IMAGE}.{IMAGE_URL}",
            f"data:{part.inline_data.mime_type};"
            f"base64,{base64.b64encode(part.inline_data.data).decode()}",
        )
    elif part.file_data.mime_type.startswith("image"):
        yield f"{prefix}{MESSAGE_CONTENT_IMAGE}.{IMAGE_URL}", part.file_data.file_uri


def _role(role: str) -> str:
    if role == "model":
        return "assistant"
    return role


IMAGE_URL = ImageAttributes.IMAGE_URL
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
JSON = OpenInferenceMimeTypeValues.JSON.value
LLM = OpenInferenceSpanKindValues.LLM.value
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_CONTENTS = MessageAttributes.MESSAGE_CONTENTS
MESSAGE_CONTENT_IMAGE = MessageContentAttributes.MESSAGE_CONTENT_IMAGE
MESSAGE_CONTENT_TEXT = MessageContentAttributes.MESSAGE_CONTENT_TEXT
MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON = MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON
MESSAGE_FUNCTION_CALL_NAME = MessageAttributes.MESSAGE_FUNCTION_CALL_NAME
MESSAGE_NAME = MessageAttributes.MESSAGE_NAME
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
