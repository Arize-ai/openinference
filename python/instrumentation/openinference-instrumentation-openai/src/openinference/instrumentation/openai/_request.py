import logging
from abc import ABC
from contextlib import contextmanager
from types import MappingProxyType
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Hashable,
    Iterator,
    Mapping,
    Optional,
    Tuple,
)

from openinference.instrumentation.openai._extra_attributes_from_request import (
    _get_extra_attributes_from_request,
)
from openinference.instrumentation.openai._extra_attributes_from_response import (
    _get_extra_attributes_from_response,
)
from openinference.instrumentation.openai._response_accumulator import (
    _ChatCompletionAccumulator,
    _CompletionAccumulator,
)
from openinference.instrumentation.openai._stream import (
    _ResponseAccumulator,
    _Stream,
)
from openinference.instrumentation.openai._utils import (
    _as_input_attributes,
    _as_output_attributes,
    _finish_tracing,
    _io_value_and_type,
)
from openinference.instrumentation.openai._with_span import _WithSpan
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace import INVALID_SPAN
from opentelemetry.util.types import AttributeValue

from openai import AsyncStream, Stream
from openai.types import Completion, CreateEmbeddingResponse
from openai.types.chat import ChatCompletion

__all__ = (
    "_Request",
    "_AsyncRequest",
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _WithTracer(ABC):
    __slots__ = (
        "_tracer",
        "_include_extra_attributes",
    )

    def __init__(
        self,
        tracer: trace_api.Tracer,
        include_extra_attributes: bool = True,
    ) -> None:
        self._tracer = tracer
        self._include_extra_attributes = include_extra_attributes

    @contextmanager
    def _start_as_current_span(
        self,
        span_name: str,
        cast_to: type,
        request_options: Mapping[str, Any],
    ) -> Iterator[_WithSpan]:
        span_kind = (
            OpenInferenceSpanKindValues.EMBEDDING
            if cast_to is CreateEmbeddingResponse else OpenInferenceSpanKindValues.LLM
        )
        attributes: Dict[str, AttributeValue] = {SpanAttributes.OPENINFERENCE_SPAN_KIND: span_kind}
        try:
            attributes.update(_as_input_attributes(_io_value_and_type(request_options)))
        except Exception:
            logger.exception(
                f"Failed to get input attributes from request options of "
                f"type {type(request_options)}"
            )
        # Secondary attributes should be added after input and output to ensure
        # that input and output are not dropped if there are too many attributes.
        try:
            extra_attributes = (
                dict(_get_extra_attributes_from_request(cast_to, request_options))
                if self._include_extra_attributes
                else {}
            )
        except Exception:
            logger.exception(
                f"Failed to get extra attributes from request options of "
                f"type {type(request_options)}"
            )
            extra_attributes = {}
        try:
            span = self._tracer.start_span(span_name, attributes=attributes)
        except Exception:
            logger.exception("Failed to start span")
            span = INVALID_SPAN
        with trace_api.use_span(
            span,
            end_on_exit=False,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            yield _WithSpan(span, extra_attributes)


class _Request(_WithTracer):
    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        try:
            cast_to, request_options = _parse_request_args(args)
            # E.g. cast_to = openai.types.chat.ChatCompletion => span_name = "ChatCompletion"
            span_name: str = cast_to.__name__.split(".")[-1]
        except Exception:
            logger.exception("Failed to parse request args")
            return wrapped(*args, **kwargs)
        with self._start_as_current_span(
            span_name=span_name,
            cast_to=cast_to,
            request_options=request_options,
        ) as with_span:
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                status_code = trace_api.StatusCode.ERROR
                with_span.record_exception(exception)
                with_span.finish_tracing(status_code=status_code)
                raise
            try:
                response = _finalize_response(
                    response=response,
                    with_span=with_span,
                    cast_to=cast_to,
                    request_options=request_options,
                    include_extra_attributes=self._include_extra_attributes,
                )
            except Exception:
                logger.exception(f"Failed to finalize response of type {type(response)}")
                with_span.finish_tracing(status_code=None)
        return response


class _AsyncRequest(_WithTracer):
    async def __call__(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)
        try:
            cast_to, request_options = _parse_request_args(args)
            # E.g. cast_to = openai.types.chat.ChatCompletion => span_name = "ChatCompletion"
            span_name: str = cast_to.__name__.split(".")[-1]
        except Exception:
            logger.exception("Failed to parse request args")
            return await wrapped(*args, **kwargs)
        with self._start_as_current_span(
            span_name=span_name,
            cast_to=cast_to,
            request_options=request_options,
        ) as with_span:
            try:
                response = await wrapped(*args, **kwargs)
            except Exception as exception:
                status_code = trace_api.StatusCode.ERROR
                with_span.record_exception(exception)
                with_span.finish_tracing(status_code=status_code)
                raise
            try:
                response = _finalize_response(
                    response=response,
                    with_span=with_span,
                    cast_to=cast_to,
                    request_options=request_options,
                    include_extra_attributes=self._include_extra_attributes,
                )
            except Exception:
                logger.exception(f"Failed to finalize response of type {type(response)}")
                with_span.finish_tracing(status_code=None)
        return response


def _parse_request_args(args: Tuple[type, Any]) -> Tuple[type, Mapping[str, Any]]:
    # We don't use `signature(request).bind()` because `request` could have been monkey-patched
    # (incorrectly) by others and the signature at runtime may not match the original.
    # The targeted signature of `request` is here:
    # https://github.com/openai/openai-python/blob/f1c7d714914e3321ca2e72839fe2d132a8646e7f/src/openai/_base_client.py#L846-L847  # noqa: E501
    cast_to: type = args[0]
    options: Mapping[str, Any] = (
        json_data
        # See https://github.com/openai/openai-python/blob/f1c7d714914e3321ca2e72839fe2d132a8646e7f/src/openai/_models.py#L427  # noqa: E501
        if hasattr(args[1], "json_data") and isinstance(json_data := args[1].json_data, Mapping)
        else {}
    )
    # FIXME: Because request options is just a Mapping, it can contain any value as long as it
    # serializes correctly in an HTTP request body. For example, Enum values may be present if a
    # third-party library puts them there. Enums can turn into their intended string values via
    # `json.dumps` when the final HTTP request body is serialized, but can pose problems when we
    # try to extract attributes. However, this round-trip seems expensive, so we opted to treat
    # only the Enums that we know about: e.g. message role sometimes can be an Enum, so we will
    # convert it only when it's encountered.
    # try:
    #     options = json.loads(json.dumps(options))
    # except Exception:
    #     pass
    return cast_to, options


def _finalize_response(
    response: Any,
    with_span: _WithSpan,
    cast_to: type,
    request_options: Mapping[str, Any],
    include_extra_attributes: bool = True,
) -> Any:
    """Monkey-patch the response object to trace the stream, or finish tracing if the response is
    not a stream.
    """
    if hasattr(response, "parse") and callable(response.parse):
        # `.request()` may be called under `.with_raw_response` and it's necessary to call
        # `.parse()` to get back the usual response types.
        # E.g. see https://github.com/openai/openai-python/blob/f1c7d714914e3321ca2e72839fe2d132a8646e7f/src/openai/_base_client.py#L518  # noqa: E501
        try:
            response.parse()
        except Exception:
            logger.exception(f"Failed to parse response of type {type(response)}")
    if (
        isinstance(response, (Stream, AsyncStream))
        or hasattr(
            # FIXME: Ideally we should not rely on a private attribute (but it may be impossible).
            # The assumption here is that calling `.parse()` stores the stream object in `._parsed`
            # and calling `.parse()` again will not overwrite the monkey-patched version.
            # See https://github.com/openai/openai-python/blob/f1c7d714914e3321ca2e72839fe2d132a8646e7f/src/openai/_response.py#L65  # noqa: E501
            response,
            "_parsed",
        )
        # Note that we must have called `.parse()` beforehand, otherwise `._parsed` is None.
        and isinstance(response._parsed, (Stream, AsyncStream))
    ):
        # For streaming, we need (optional) accumulators to process each chunk iteration.
        response_accumulator = _ResponseAccumulators.find(cast_to)
        if hasattr(response, "_parsed") and isinstance(
            parsed := response._parsed, (Stream, AsyncStream)
        ):
            # Monkey-patch a private attribute assumed to be caching the output of `.parse()`.
            response._parsed = _Stream(
                stream=parsed,
                with_span=with_span,
                response_accumulator=response_accumulator,
                include_extra_attributes=include_extra_attributes,
            )
            return response
        return _Stream(
            stream=response,
            with_span=with_span,
            response_accumulator=response_accumulator,
            include_extra_attributes=include_extra_attributes,
        )
    _finish_tracing(
        status_code=trace_api.StatusCode.OK,
        with_span=with_span,
        has_attributes=_ResponseAttributes(
            response=response,
            request_options=request_options,
            include_extra_attributes=include_extra_attributes,
        ),
    )
    return response


class _ResponseAttributes:
    __slots__ = (
        "_request_options",
        "_response",
        "_include_extra_attributes",
    )

    def __init__(
        self,
        response: Any,
        request_options: Mapping[str, Any],
        include_extra_attributes: bool = True,
    ) -> None:
        if hasattr(response, "parse") and callable(response.parse):
            # E.g. see https://github.com/openai/openai-python/blob/f1c7d714914e3321ca2e72839fe2d132a8646e7f/src/openai/_base_client.py#L518  # noqa: E501
            try:
                response = response.parse()
            except Exception:
                logger.exception(f"Failed to parse response of type {type(response)}")
        self._request_options = request_options
        self._response = response
        self._include_extra_attributes = include_extra_attributes

    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        yield from _as_output_attributes(_io_value_and_type(self._response))

    def get_extra_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        if self._include_extra_attributes:
            yield from _get_extra_attributes_from_response(
                self._response,
                self._request_options,
            )


class _Accumulators(ABC):
    _mapping: Mapping[type, type]

    def __init_subclass__(cls, mapping: Mapping[type, type], **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._mapping = mapping

    @classmethod
    def find(cls, cast_to: type) -> Optional[_ResponseAccumulator]:
        if not isinstance(cast_to, Hashable):
            # `cast_to` may not be hashable
            # See https://github.com/openai/openai-python/blob/f1c7d714914e3321ca2e72839fe2d132a8646e7f/src/openai/_response.py#L172  # noqa: E501
            return None
        try:
            factory = cls._mapping.get(cast_to)
        except Exception:
            logger.exception(f"Failed to get factory for {cast_to}")
            return None
        return factory() if factory else None


class _ResponseAccumulators(
    _Accumulators,
    ABC,
    mapping=MappingProxyType(
        {
            ChatCompletion: _ChatCompletionAccumulator,
            Completion: _CompletionAccumulator,
        }
    ),
):
    ...
