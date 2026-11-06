import logging
from abc import ABC
from contextlib import contextmanager
from enum import Enum
from inspect import Signature, signature
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import opentelemetry.context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.trace import INVALID_SPAN
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import get_attributes_from_context
from openinference.instrumentation.portkey._request_attributes_extractor import (
    _RequestAttributesExtractor,
)
from openinference.instrumentation.portkey._response_attributes_extractor import (
    _ResponseAttributesExtractor,
)
from openinference.instrumentation.portkey._stream import _Stream
from openinference.instrumentation.portkey._utils import (
    _finish_tracing,
    _materialize_content_iterables,
)
from openinference.instrumentation.portkey._with_span import _WithSpan

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_UNSET_SENTINEL_NAMES = frozenset({"NotGiven", "Omit"})


def _is_set(value: Any) -> bool:
    return value is not None and type(value).__name__ not in _UNSET_SENTINEL_NAMES


def _is_stream(response: Any) -> bool:
    # portkey_ai 1.x streams through a chunk generator, 2.x returns openai's Stream
    return hasattr(response, "__next__") or hasattr(response, "__anext__")


def _flatten(mapping: Mapping[str, Any]) -> Iterator[Tuple[str, AttributeValue]]:
    for key, value in mapping.items():
        if value is None:
            continue
        if isinstance(value, Mapping):
            for sub_key, sub_value in _flatten(value):
                yield f"{key}.{sub_key}", sub_value
        elif isinstance(value, List) and any(isinstance(item, Mapping) for item in value):
            for index, sub_mapping in enumerate(value):
                for sub_key, sub_value in _flatten(sub_mapping):
                    yield f"{key}.{index}.{sub_key}", sub_value
        else:
            if isinstance(value, Enum):
                value = value.value
            yield key, value


class _WithTracer(ABC):
    """Base class for wrappers that use a tracer."""

    def __init__(self, tracer: trace_api.Tracer, *args: Any, **kwargs: Any) -> None:
        self._tracer = tracer
        super().__init__(*args, **kwargs)

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
    bound_args = signature.bind(*args, **kwargs)
    request_data: Dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if name not in bound_args.arguments:
            continue
        value = bound_args.arguments[name]
        if parameter.kind is parameter.VAR_KEYWORD and isinstance(value, Mapping):
            for extra_key, extra_value in value.items():
                if _is_set(extra_value):
                    request_data[extra_key] = extra_value
        elif _is_set(value):
            request_data[name] = value
    return request_data


class _CompletionsWrapper(_WithTracer):
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

        materialize_error: Optional[Exception] = None
        try:
            kwargs = _materialize_content_iterables(kwargs)
        except Exception as exception:
            # Reading a generator runs user code; record its failure on the span below.
            materialize_error = exception

        # Prepare invocation parameters by merging args and kwargs
        invocation_parameters = {**kwargs}
        for arg in args:
            if isinstance(arg, dict):
                invocation_parameters.update(arg)

        request_parameters = _parse_args(signature(wrapped), *args, **kwargs)
        span_name = "Completions"
        with self._start_as_current_span(
            span_name=span_name,
            attributes=self._request_extractor.get_attributes_from_request(request_parameters),
            context_attributes=get_attributes_from_context(),
            extra_attributes=self._request_extractor.get_extra_attributes_from_request(
                request_parameters
            ),
        ) as span:
            try:
                if materialize_error is not None:
                    raise materialize_error
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.record_exception(exception)
                status = trace_api.Status(
                    status_code=trace_api.StatusCode.ERROR,
                    description=f"{type(exception).__name__}: {exception}",
                )
                span.finish_tracing(status=status)
                raise
            if _is_stream(response):
                # The span is finished by the stream wrapper once the stream
                # is fully consumed (or its context manager exits).
                return _Stream(response, span, self._response_extractor, request_parameters)
            try:
                _finish_tracing(
                    with_span=span,
                    attributes=self._response_extractor.get_attributes(response),
                    extra_attributes=self._response_extractor.get_extra_attributes(
                        response=response,
                        request_parameters=request_parameters,
                    ),
                    status=trace_api.Status(status_code=trace_api.StatusCode.OK),
                )
            except Exception:
                logger.exception(f"Failed to finalize response of type {type(response)}")
                span.finish_tracing()
            return response


class _AsyncCompletionsWrapper(_WithTracer):
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

        materialize_error: Optional[Exception] = None
        try:
            kwargs = _materialize_content_iterables(kwargs)
        except Exception as exception:
            # Reading a generator runs user code; record its failure on the span below.
            materialize_error = exception

        # Prepare invocation parameters by merging args and kwargs
        invocation_parameters = {**kwargs}
        for arg in args:
            if isinstance(arg, dict):
                invocation_parameters.update(arg)

        request_parameters = _parse_args(signature(wrapped), *args, **kwargs)
        span_name = "AsyncCompletions"
        with self._start_as_current_span(
            span_name=span_name,
            attributes=self._request_extractor.get_attributes_from_request(request_parameters),
            context_attributes=get_attributes_from_context(),
            extra_attributes=self._request_extractor.get_extra_attributes_from_request(
                request_parameters
            ),
        ) as span:
            try:
                if materialize_error is not None:
                    raise materialize_error
                response = await wrapped(*args, **kwargs)
            except Exception as exception:
                span.record_exception(exception)
                status = trace_api.Status(
                    status_code=trace_api.StatusCode.ERROR,
                    description=f"{type(exception).__name__}: {exception}",
                )
                span.finish_tracing(status=status)
                raise
            if _is_stream(response):
                # The span is finished by the stream wrapper once the stream
                # is fully consumed (or its context manager exits).
                return _Stream(response, span, self._response_extractor, request_parameters)
            try:
                _finish_tracing(
                    with_span=span,
                    attributes=self._response_extractor.get_attributes(response),
                    extra_attributes=self._response_extractor.get_extra_attributes(
                        response=response,
                        request_parameters=request_parameters,
                    ),
                    status=trace_api.Status(status_code=trace_api.StatusCode.OK),
                )
            except Exception:
                logger.exception(f"Failed to finalize response of type {type(response)}")
                span.finish_tracing()
            return response
