import logging
from contextlib import contextmanager
from inspect import Signature, signature
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional, Tuple

import opentelemetry.context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.trace import INVALID_SPAN
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation.together._request_attributes_extractor import (
    _RequestAttributesExtractor,
)
from openinference.instrumentation.together._response_attributes_extractor import (
    _ResponseAttributesExtractor,
)
from openinference.instrumentation.together._stream import _Stream
from openinference.instrumentation.together._utils import _finish_tracing
from openinference.instrumentation.together._with_span import _WithSpan
from together import AsyncStream, NotGiven, Omit, Stream

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _WithTracer:
    """Base class for the sync and async ``create`` wrappers."""

    def __init__(self, tracer: trace_api.Tracer, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tracer = tracer
        self._request_extractor = _RequestAttributesExtractor()
        self._response_extractor = _ResponseAttributesExtractor()
        self._signature: Optional[Signature] = None

    def _parse_request(
        self,
        wrapped: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        # Each wrapper instance wraps exactly one method, so the signature can
        # be computed once and reused across calls.
        if self._signature is None:
            self._signature = signature(wrapped)
        bound_signature = self._signature.bind(*args, **kwargs)
        bound_signature.apply_defaults()
        return {
            key: value
            for key, value in bound_signature.arguments.items()
            if value is not None and not isinstance(value, (Omit, NotGiven))
        }

    @contextmanager
    def _start_as_current_span(
        self,
        span_name: str,
        attributes: Iterable[Tuple[str, AttributeValue]],
        extra_attributes: Iterable[Tuple[str, AttributeValue]],
    ) -> Iterator[_WithSpan]:
        # Because OTEL has a default limit of 128 attributes, we split our
        # attributes into two tiers, where "extra_attributes" are added first to
        # ensure that the most important "attributes" are added last and are not
        # dropped. Context attributes (session, user, metadata, tags) are
        # injected by OITracer.start_span.
        try:
            span = self._tracer.start_span(name=span_name, attributes=dict(extra_attributes))
        except Exception:
            span = INVALID_SPAN
        with trace_api.use_span(
            span,
            end_on_exit=False,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            yield _WithSpan(span=span, extra_attributes=dict(attributes))

    def _finalize_error(self, span: _WithSpan, exception: Exception) -> None:
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


class _CompletionsWrapper(_WithTracer):
    """Wraps ``CompletionsResource.create`` to trace synchronous chat calls."""

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        request_parameters = self._parse_request(wrapped, args, kwargs)
        with self._start_as_current_span(
            span_name="Completions",
            attributes=self._request_extractor.get_attributes_from_request(request_parameters),
            extra_attributes=self._request_extractor.get_extra_attributes_from_request(
                request_parameters
            ),
        ) as span:
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                self._finalize_error(span, exception)
                raise
            if isinstance(response, (Stream, AsyncStream)):
                # The span is finished by the stream wrapper once the stream
                # is fully consumed (or its context manager exits).
                return _Stream(response, span, self._response_extractor)
            self._finalize_response(span, response)
        return response


class _AsyncCompletionsWrapper(_WithTracer):
    """Wraps ``AsyncCompletionsResource.create`` to trace asynchronous chat calls."""

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        request_parameters = self._parse_request(wrapped, args, kwargs)
        with self._start_as_current_span(
            span_name="AsyncCompletions",
            attributes=self._request_extractor.get_attributes_from_request(request_parameters),
            extra_attributes=self._request_extractor.get_extra_attributes_from_request(
                request_parameters
            ),
        ) as span:
            try:
                response = await wrapped(*args, **kwargs)
            except Exception as exception:
                self._finalize_error(span, exception)
                raise
            if isinstance(response, (Stream, AsyncStream)):
                # The span is finished by the stream wrapper once the stream
                # is fully consumed (or its context manager exits).
                return _Stream(response, span, self._response_extractor)
            self._finalize_response(span, response)
        return response
