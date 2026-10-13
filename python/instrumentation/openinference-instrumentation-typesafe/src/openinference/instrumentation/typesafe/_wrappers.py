import logging
from contextlib import AbstractContextManager
from inspect import Signature, signature
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from opentelemetry import context as context_api
from opentelemetry.trace import Span, Status, StatusCode
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import OITracer
from openinference.instrumentation.typesafe._attributes import (
    get_request_attributes,
    get_response_attributes,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _finish(span: Span, response: Any) -> None:
    """Records the response attributes and marks the span ``OK``."""
    # Sampled-out spans discard every attribute, so skip the serialization entirely.
    if span.is_recording():
        try:
            span.set_attributes(get_response_attributes(response))
        except Exception:
            logger.exception("Failed to extract response attributes")
    span.set_status(Status(StatusCode.OK))


def _fail(span: Span, exception: BaseException) -> None:
    """Records the exception and marks the span ``ERROR``."""
    span.record_exception(exception)
    span.set_status(Status(StatusCode.ERROR, f"{type(exception).__name__}: {exception}"))


class _WithTracer:
    """Base class for the sync and async ``system_one`` wrappers."""

    def __init__(self, tracer: OITracer, span_name: str) -> None:
        self._tracer = tracer
        self._span_name = span_name
        self._signature: Optional[Signature] = None

    def _start_span(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> AbstractContextManager[Span]:
        """Returns the span context manager for one ``system_one`` call."""
        return self._tracer.start_as_current_span(
            self._span_name,
            attributes=self._request_attributes(wrapped, instance, args, kwargs),
            record_exception=False,
            set_status_on_exception=False,
        )

    def _request_attributes(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Dict[str, AttributeValue]:
        """Returns the request-side attributes, or an empty mapping if extraction fails."""
        try:
            # Each wrapper instance wraps exactly one method, so the signature is
            # computed once and reused across calls.
            if self._signature is None:
                self._signature = signature(wrapped)
            bound = self._signature.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = bound.arguments
            model = arguments["model"]
            if model is None:
                # Fall back to the client default (constructor arg or TYPESAFE_DEFAULT_MODEL).
                model = getattr(getattr(instance, "_config", None), "default_model", None)
            return get_request_attributes(
                state=arguments["state"],
                questions=arguments["questions"],
                model=model,
                extra_body=arguments["extra_body"],
            )
        except Exception:
            logger.exception("Failed to extract request attributes")
            return {}


class _SystemOneWrapper(_WithTracer):
    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        with self._start_span(wrapped, instance, args, kwargs) as span:
            try:
                response = wrapped(*args, **kwargs)
            except BaseException as exception:
                _fail(span, exception)
                raise
            _finish(span, response)
            return response


class _AsyncSystemOneWrapper(_WithTracer):
    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)
        with self._start_span(wrapped, instance, args, kwargs) as span:
            try:
                response = await wrapped(*args, **kwargs)
            except BaseException as exception:
                _fail(span, exception)
                raise
            _finish(span, response)
            return response
