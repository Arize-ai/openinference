import logging
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Tuple,
)

from openinference.instrumentation.mistralai._utils import _finish_tracing
from openinference.instrumentation.mistralai._with_span import _WithSpan
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace import INVALID_SPAN
from opentelemetry.util.types import AttributeValue

__all__ = (
    "_SyncChatWrapper",
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())



class _SyncChatWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

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
            request_parameters = _parse_sync_chat_args(args)
            span_name = "MistralClient.chat"
        except Exception:
            logger.exception("Failed to parse request args")
            return wrapped(*args, **kwargs)
        with self._start_as_current_span(
            span_name=span_name,
            attributes=(),
            extra_attributes=(),
        ) as with_span:
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                with_span.record_exception(exception)
                status = trace_api.Status(
                    status_code=trace_api.StatusCode.ERROR,
                    # Follow the format in OTEL SDK for description, see:
                    # https://github.com/open-telemetry/opentelemetry-python/blob/2b9dcfc5d853d1c10176937a6bcaade54cda1a31/opentelemetry-api/src/opentelemetry/trace/__init__.py#L588  # noqa E501
                    description=f"{type(exception).__name__}: {exception}",
                )
                with_span.finish_tracing(status=status)
                raise
            try:
                _finish_tracing(
                    status=trace_api.Status(status_code=trace_api.StatusCode.OK),
                    with_span=with_span,
                    has_attributes=_ResponseAttributes(
                        request_parameters=request_parameters,
                        response=response,
                    ),
                )
            except Exception:
                logger.exception(f"Failed to finish tracing for response of type {type(response)}")
                with_span.finish_tracing()
        return response

    @contextmanager
    def _start_as_current_span(
        self,
        span_name: str,
        attributes: Iterable[Tuple[str, AttributeValue]],
        extra_attributes: Iterable[Tuple[str, AttributeValue]],
    ) -> Iterator[_WithSpan]:
        # Because OTEL has a default limit of 128 attributes, we split our attributes into
        # two tiers, where the addition of "extra_attributes" is deferred until the end
        # and only after the "attributes" are added.
        try:
            span = self._tracer.start_span(name=span_name, attributes=dict(attributes))
        except Exception:
            logger.exception("Failed to start span")
            span = INVALID_SPAN
        with trace_api.use_span(
            span,
            end_on_exit=False,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            yield _WithSpan(span=span, extra_attributes=dict(extra_attributes))


class _ResponseAttributes:
    __slots__ = (
        "_response",
        "_request_parameters",
    )

    def __init__(
        self,
        response: Any,
        request_parameters: Mapping[str, Any],
    ) -> None:
        self._response = response
        self._request_parameters = request_parameters

    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        yield from ()

    def get_extra_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        yield from ()


def _parse_sync_chat_args(args: Tuple[type, Any]) -> Dict[str, Any]:
    return {}
