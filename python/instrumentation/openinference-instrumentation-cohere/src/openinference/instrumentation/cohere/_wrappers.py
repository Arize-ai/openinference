import logging
from abc import ABC
from contextlib import contextmanager
from inspect import Signature, signature
from typing import Any, Callable, Dict, FrozenSet, Iterable, Iterator, Mapping, Optional, Tuple

import opentelemetry.context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.trace import INVALID_SPAN
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import (
    OITracer,
    get_attributes_from_context,
    safe_json_dumps,
)
from openinference.instrumentation.cohere._request_attributes_extractor import (
    _RequestAttributesExtractor,
)
from openinference.instrumentation.cohere._response_attributes_extractor import (
    _ResponseAttributesExtractor,
)
from openinference.instrumentation.cohere._stream import _Stream
from openinference.instrumentation.cohere._with_span import _WithSpan

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _WithTracer(ABC):
    """Base class for wrappers that need a tracer."""

    def __init__(self, tracer: OITracer, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tracer = tracer
        self._request_extractor = _RequestAttributesExtractor()
        self._response_extractor = _ResponseAttributesExtractor()

    @contextmanager
    def _start_as_current_span(
        self,
        span_name: str,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[_WithSpan]:
        # Because OTEL has a default limit of 128 attributes, we split our
        # attributes into two tiers, where "extra_attributes" are added first to
        # ensure that the most important "attributes" are added last and are not
        # dropped.
        span: trace_api.Span
        try:
            span = self._tracer.start_span(
                name=span_name,
                attributes=dict(
                    self._request_extractor.get_extra_attributes_from_request(request_parameters)
                ),
            )
        except Exception:
            logger.exception(f"Failed to start span {span_name}")
            span = INVALID_SPAN
        with trace_api.use_span(
            span,
            end_on_exit=False,
            record_exception=False,
            set_status_on_exception=False,
        ) as current_span:
            yield _WithSpan(
                span=current_span,
                context_attributes=dict(get_attributes_from_context()),
                extra_attributes=dict(
                    self._request_extractor.get_attributes_from_request(request_parameters)
                ),
            )

    def _record_failure(self, span: _WithSpan, exception: BaseException) -> None:
        span.record_exception(exception)
        span.finish_tracing(
            status=trace_api.Status(
                status_code=trace_api.StatusCode.ERROR,
                description=f"{type(exception).__name__}: {exception}",
            )
        )

    def _finalize_response(self, span: _WithSpan, response: Any) -> None:
        try:
            _finish_tracing(
                status=trace_api.Status(status_code=trace_api.StatusCode.OK),
                with_span=span,
                attributes=self._response_extractor.get_attributes(response=response),
                extra_attributes=self._response_extractor.get_extra_attributes(response=response),
            )
        except Exception:
            logger.exception(f"Failed to finalize response of type {type(response)}")
            span.finish_tracing()


_EXCLUDED_REQUEST_PARAMETERS: FrozenSet[str] = frozenset(
    {
        # Transport configuration rather than an LLM parameter, and its
        # ``additional_headers`` routinely carries credentials.
        "request_options",
    }
)


def _parse_args(
    signature: Signature,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Map the call's arguments to parameter names, keeping only values the caller set.

    Cohere uses an ``OMIT`` sentinel (``Ellipsis``) as the default for unset chat
    parameters, so both ``None`` and ``Ellipsis`` are filtered out.
    """
    arguments: Mapping[str, Any]
    try:
        arguments = signature.bind(*args, **kwargs).arguments
    except TypeError:
        # The installed SDK's signature no longer matches the call (e.g. a newer
        # cohere release); fall back to the keyword arguments so the call still
        # gets traced and the SDK raises its own error.
        arguments = kwargs
    request_data: Dict[str, Any] = {}
    for key, value in arguments.items():
        if value is None or value is ... or key in _EXCLUDED_REQUEST_PARAMETERS:
            continue
        try:
            # Ensure the value is JSON-encodable, so that a single unserializable
            # argument cannot later fail attribute extraction and drop the span.
            safe_json_dumps(value)
            request_data[key] = value
        except Exception:
            request_data[key] = str(value)
    return request_data


class _ChatWrapper(_WithTracer):
    """Wraps ``cohere.V2Client.chat`` to trace synchronous chat calls."""

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        request_parameters = _parse_args(signature(wrapped), *args, **kwargs)
        with self._start_as_current_span("ClientV2.chat", request_parameters) as span:
            try:
                response = wrapped(*args, **kwargs)
            except BaseException as exception:
                self._record_failure(span, exception)
                raise
            self._finalize_response(span, response)
        return response


class _AsyncChatWrapper(_WithTracer):
    """Wraps ``cohere.AsyncV2Client.chat`` to trace asynchronous chat calls."""

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        request_parameters = _parse_args(signature(wrapped), *args, **kwargs)
        with self._start_as_current_span("AsyncClientV2.chat", request_parameters) as span:
            try:
                response = await wrapped(*args, **kwargs)
            except BaseException as exception:
                self._record_failure(span, exception)
                raise
            self._finalize_response(span, response)
        return response


class _ChatStreamWrapper(_WithTracer):
    """Wraps ``cohere.V2Client.chat_stream`` to trace synchronous streamed chat calls."""

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        request_parameters = _parse_args(signature(wrapped), *args, **kwargs)
        # The span is not finished here: the response arrives as the returned
        # iterator is consumed, so _Stream ends the span when it is exhausted.
        with self._start_as_current_span("ClientV2.chat_stream", request_parameters) as span:
            try:
                stream = wrapped(*args, **kwargs)
            except BaseException as exception:
                self._record_failure(span, exception)
                raise
            return _Stream(stream, span, self._response_extractor)


class _AsyncChatStreamWrapper(_WithTracer):
    """Wraps ``cohere.AsyncV2Client.chat_stream`` to trace streamed chat calls.

    ``chat_stream`` is an async *generator* function, so calling it returns an async
    iterator rather than a coroutine; there is nothing to await here.
    """

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        request_parameters = _parse_args(signature(wrapped), *args, **kwargs)
        with self._start_as_current_span("AsyncClientV2.chat_stream", request_parameters) as span:
            try:
                stream = wrapped(*args, **kwargs)
            except BaseException as exception:
                self._record_failure(span, exception)
                raise
            return _Stream(stream, span, self._response_extractor)

class _EmbedWrapper(_WithTracer):
    """Wraps ``cohere.V2Client.embed`` to trace synchronous embedding calls."""

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        request_parameters = _parse_args(signature(wrapped), *args, **kwargs)
        with self._start_as_current_span("ClientV2.embed", request_parameters) as span:
            try:
                response = wrapped(*args, **kwargs)
            except BaseException as exception:
                self._record_failure(span, exception)
                raise
            self._finalize_response(span, response)
        return response


class _AsyncEmbedWrapper(_WithTracer):
    """Wraps ``cohere.AsyncV2Client.embed`` to trace asynchronous embedding calls."""

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        request_parameters = _parse_args(signature(wrapped), *args, **kwargs)
        with self._start_as_current_span("AsyncClientV2.embed", request_parameters) as span:
            try:
                response = await wrapped(*args, **kwargs)
            except BaseException as exception:
                self._record_failure(span, exception)
                raise
            self._finalize_response(span, response)
        return response


def _finish_tracing(
    with_span: _WithSpan,
    attributes: Iterable[Tuple[str, AttributeValue]],
    extra_attributes: Iterable[Tuple[str, AttributeValue]],
    status: Optional[trace_api.Status] = None,
) -> None:
    attributes_dict: Optional[Dict[str, AttributeValue]] = None
    try:
        attributes_dict = dict(attributes)
    except Exception:
        logger.exception("Failed to get attributes")
    extra_attributes_dict: Optional[Dict[str, AttributeValue]] = None
    try:
        extra_attributes_dict = dict(extra_attributes)
    except Exception:
        logger.exception("Failed to get extra attributes")
    with_span.finish_tracing(
        status=status,
        attributes=attributes_dict,
        extra_attributes=extra_attributes_dict,
    )
