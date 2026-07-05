import logging
from abc import ABC
from contextlib import contextmanager
from inspect import Signature, signature
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Tuple

import opentelemetry.context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.trace import INVALID_SPAN
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import get_attributes_from_context, safe_json_dumps
from openinference.instrumentation.ollama._request_attributes_extractor import (
    _RequestAttributesExtractor,
)
from openinference.instrumentation.ollama._response_attributes_extractor import (
    _ResponseAttributesExtractor,
)
from openinference.instrumentation.ollama._utils import _finish_tracing
from openinference.instrumentation.ollama._with_span import _WithSpan

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _WithTracer(ABC):
    """Base class for wrappers that need a tracer."""

    def __init__(self, tracer: trace_api.Tracer, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tracer = tracer

    @contextmanager
    def _start_as_current_span(
        self,
        span_name: str,
        attributes: Iterable[Tuple[str, AttributeValue]],
        context_attributes: Iterable[Tuple[str, AttributeValue]],
        extra_attributes: Iterable[Tuple[str, AttributeValue]],
    ) -> Iterator[_WithSpan]:
        # Because OTEL has a default limit of 128 attributes, we split our
        # attributes into two tiers, where "extra_attributes" are added first to
        # ensure that the most important "attributes" are added last and are not
        # dropped.
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
            yield _WithSpan(
                span=span,
                context_attributes=dict(context_attributes),
                extra_attributes=dict(attributes),
            )


def _parse_args(
    signature: Signature,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    bound_signature = signature.bind(*args, **kwargs)
    bound_signature.apply_defaults()
    request_data: Dict[str, Any] = {}
    for key, value in bound_signature.arguments.items():
        if value is None:
            continue
        try:
            # ensure the value is JSON-serializable
            safe_json_dumps(value)
            request_data[key] = value
        except Exception:
            request_data[key] = str(value)
    return request_data


class _ChatWrapper(_WithTracer):
    """Wraps ``ollama.Client.chat`` to trace synchronous chat calls."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._request_extractor = _RequestAttributesExtractor()
        self._response_extractor = _ResponseAttributesExtractor()

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
        with self._start_as_current_span(
            span_name="chat",
            attributes=self._request_extractor.get_attributes_from_request(request_parameters),
            context_attributes=get_attributes_from_context(),
            extra_attributes=self._request_extractor.get_extra_attributes_from_request(
                request_parameters
            ),
        ) as span:
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.record_exception(exception)
                span.finish_tracing(
                    status=trace_api.Status(
                        status_code=trace_api.StatusCode.ERROR,
                        description=f"{type(exception).__name__}: {exception}",
                    )
                )
                raise
            try:
                _finish_tracing(
                    status=trace_api.Status(status_code=trace_api.StatusCode.OK),
                    with_span=span,
                    attributes=self._response_extractor.get_attributes(response=response),
                    extra_attributes=self._response_extractor.get_extra_attributes(
                        response=response, request_parameters=request_parameters
                    ),
                )
            except Exception:
                logger.exception(f"Failed to finalize response of type {type(response)}")
                span.finish_tracing()
        return response


class _AsyncChatWrapper(_WithTracer):
    """Wraps ``ollama.AsyncClient.chat`` to trace asynchronous chat calls."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._request_extractor = _RequestAttributesExtractor()
        self._response_extractor = _ResponseAttributesExtractor()

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
        with self._start_as_current_span(
            span_name="async_chat",
            attributes=self._request_extractor.get_attributes_from_request(request_parameters),
            context_attributes=get_attributes_from_context(),
            extra_attributes=self._request_extractor.get_extra_attributes_from_request(
                request_parameters
            ),
        ) as span:
            try:
                response = await wrapped(*args, **kwargs)
            except Exception as exception:
                span.record_exception(exception)
                span.finish_tracing(
                    status=trace_api.Status(
                        status_code=trace_api.StatusCode.ERROR,
                        description=f"{type(exception).__name__}: {exception}",
                    )
                )
                raise
            try:
                _finish_tracing(
                    status=trace_api.Status(status_code=trace_api.StatusCode.OK),
                    with_span=span,
                    attributes=self._response_extractor.get_attributes(response=response),
                    extra_attributes=self._response_extractor.get_extra_attributes(
                        response=response, request_parameters=request_parameters
                    ),
                )
            except Exception:
                logger.exception(f"Failed to finalize response of type {type(response)}")
                span.finish_tracing()
        return response
