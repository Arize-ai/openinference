import json
import logging
from abc import ABC
from contextlib import contextmanager
from enum import Enum
from inspect import Signature, signature
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Tuple

import opentelemetry.context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.trace import INVALID_SPAN
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import get_attributes_from_context, safe_json_dumps
from openinference.semconv.trace import (
    EmbeddingAttributes,
    MessageAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

from openinference.instrumentation.portkey._response_attributes_extractor import _ResponseAttributesExtractor
from openinference.instrumentation.portkey._request_attributes_extractor import _RequestAttributesExtractor

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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
    ) -> Iterator[Any]:
        # Because OTEL has a default limit of 128 attributes, we split our
        # attributes into two tiers, where "extra_attributes" are added first to
        # ensure that the most important "attributes" are added last and are not
        # dropped.
        span = self._tracer.start_span(span_name)
        if span is INVALID_SPAN:
            yield None
            return

        try:
            # Add extra attributes first
            for key, value in extra_attributes:
                span.set_attribute(key, value)

            # Add context attributes
            for key, value in context_attributes:
                span.set_attribute(key, value)

            # Add main attributes last
            for key, value in attributes:
                span.set_attribute(key, value)

            # Set span kind
            span.set_attribute(
                SpanAttributes.SPAN_KIND,
                OpenInferenceSpanKindValues.LLM.value,
            )

            # Set as current span
            token = context_api.attach(context_api.set_span_in_context(span))
            yield span
        finally:
            context_api.detach(token)
            span.end()


def _parse_args(
    signature: Signature,
    *args: Tuple[Any],
    **kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    """Parse args and kwargs into a dictionary of parameter names to values."""
    bound_args = signature.bind(*args, **kwargs)
    bound_args.apply_defaults()
    return dict(bound_args.arguments)


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
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.record_exception(exception)
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                raise
            span.set_status(trace_api.Status(trace_api.StatusCode.OK))
            span.set_attributes(self._response_extractor.get_attributes(response))
            return response
