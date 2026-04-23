import json
import logging
from abc import ABC
from contextlib import contextmanager
from enum import Enum
from inspect import Signature, signature
from itertools import chain
from typing import Any, Callable, Iterable, Iterator, List, Mapping

import opentelemetry.context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.trace import INVALID_SPAN
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import get_attributes_from_context, safe_json_dumps
from openinference.instrumentation.google_genai import cache_attributes
from openinference.instrumentation.google_genai._context import (
    CapturedRequestScope,
    get_embedding_invocation_parameters,
    get_input_attributes,
    get_llm_invocation_parameters,
    get_tool_attributes,
)
from openinference.instrumentation.google_genai._interactions_stream import _InteractionsStream
from openinference.instrumentation.google_genai._request_attributes_extractor import (
    _RequestAttributesExtractor,
)
from openinference.instrumentation.google_genai._response_attributes_extractor import (
    _ResponseAttributesExtractor,
)
from openinference.instrumentation.google_genai._stream import _Stream
from openinference.instrumentation.google_genai._utils import _finish_tracing
from openinference.instrumentation.google_genai._with_span import _WithSpan
from openinference.instrumentation.google_genai.interactions_attributes import (
    get_attributes_from_request,
    get_attributes_from_response,
)
from openinference.semconv.trace import (
    EmbeddingAttributes,
    MessageAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _set_captured_llm_attributes(span: _WithSpan) -> None:
    """Set input_value, tools, and LLM invocation_parameters from captured SDK request."""
    try:
        span.set_attributes(dict(get_input_attributes()))
        span.set_attributes(dict(get_tool_attributes()))
        if invocation_params := get_llm_invocation_parameters():
            span.set_attributes({SpanAttributes.LLM_INVOCATION_PARAMETERS: invocation_params})
    except Exception:
        logger.exception("Failed to set captured request attributes")


def _set_captured_embedding_attributes(span: _WithSpan) -> None:
    """Set input_value, embedding invocation_parameters, and embedding text
    from captured SDK request."""
    from openinference.instrumentation.google_genai._embedding_attributes_extractor import (
        _EmbeddingRequestAttributesExtractor,
    )

    try:
        span.set_attributes(dict(get_input_attributes()))
        if invocation_params := get_embedding_invocation_parameters():
            span.set_attributes({SpanAttributes.EMBEDDING_INVOCATION_PARAMETERS: invocation_params})
        span.set_attributes(
            dict(_EmbeddingRequestAttributesExtractor.get_embedding_text_attributes())
        )
    except Exception:
        logger.exception("Failed to set captured embedding attributes")


def _flatten(mapping: Mapping[str, Any]) -> Iterator[tuple[str, AttributeValue]]:
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
    """
    Base class for wrappers that need a tracer.
    """

    def __init__(self, tracer: trace_api.Tracer, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tracer = tracer

    @contextmanager
    def _start_as_current_span(
        self,
        span_name: str,
        attributes: Iterable[tuple[str, AttributeValue]],
    ) -> Iterator[_WithSpan]:
        # Because OTEL has a default limit of 128 attributes, we split our
        # attributes into two tiers, where "extra_attributes" are added first to
        # ensure that the most important "attributes" are added last and are not
        # dropped.
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
            yield _WithSpan(span=span)


def _parse_args(
    signature: Signature,
    instance: Any,
    *args: tuple[Any],
    **kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    # When another library wraps a method using `functools.wraps` along with `wrapt`,
    # Python inspect.signature follows `__wrapped__` to the original unbound method
    # and reports ``self`` as a required positional parameter — but wrapt has already
    # consumed the instance and hands us only the user-supplied args. Detect this by trying
    # to bind as-is and, on failure, retry with `instance` prepended.
    # See issue #2995.
    params = signature.parameters
    needs_instance = params and next(iter(params.values())).name in ("self", "cls")
    if needs_instance:
        bound_signature = signature.bind(instance, *args, **kwargs)
        first_param = next(iter(signature.parameters.values()), None)
        if first_param and first_param.name in ("self", "cls"):
            bound_signature.arguments.pop(first_param.name, None)
    else:
        bound_signature = signature.bind(*args, **kwargs)
    bound_signature.apply_defaults()
    bound_arguments = bound_signature.arguments  # Defaults empty to NOT_GIVEN
    request_data: dict[str, Any] = {}
    for key, value in bound_arguments.items():
        try:
            if value is not None:
                try:
                    # ensure the value is JSON-serializable
                    safe_json_dumps(value)
                    request_data[key] = value
                except json.JSONDecodeError:
                    request_data[key] = str(value)
        except Exception:
            request_data[key] = str(value)
    return request_data


class _SyncEmbedContentWrapper(_WithTracer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        from openinference.instrumentation.google_genai._embedding_attributes_extractor import (
            _EmbeddingRequestAttributesExtractor,
            _EmbeddingResponseAttributesExtractor,
        )

        self._request_extractor = _EmbeddingRequestAttributesExtractor()
        self._response_extractor = _EmbeddingResponseAttributesExtractor()

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        request_parameters = _parse_args(signature(wrapped), instance, *args, **kwargs)
        span_name = "EmbedContent"
        with self._start_as_current_span(
            span_name=span_name,
            attributes=chain(
                get_attributes_from_context(),
                self._request_extractor.get_attributes_from_request(request_parameters),
            ),
        ) as span:
            with CapturedRequestScope():
                try:
                    response = wrapped(*args, **kwargs)
                except Exception as exception:
                    _set_captured_embedding_attributes(span)
                    span.record_exception(exception)
                    status = trace_api.Status(
                        status_code=trace_api.StatusCode.ERROR,
                        description=f"{type(exception).__name__}: {exception}",
                    )
                    span.finish_tracing(status=status)
                    raise
                _set_captured_embedding_attributes(span)
            try:
                _finish_tracing(
                    status=trace_api.Status(status_code=trace_api.StatusCode.OK),
                    with_span=span,
                    attributes=self._response_extractor.get_attributes(
                        response=response,
                        request_parameters=request_parameters,
                    ),
                )
            except Exception:
                logger.exception(f"Failed to finalize response of type {type(response)}")
                span.finish_tracing()
        return response


class _AsyncEmbedContentWrapper(_WithTracer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        from openinference.instrumentation.google_genai._embedding_attributes_extractor import (
            _EmbeddingRequestAttributesExtractor,
            _EmbeddingResponseAttributesExtractor,
        )

        self._request_extractor = _EmbeddingRequestAttributesExtractor()
        self._response_extractor = _EmbeddingResponseAttributesExtractor()

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        request_parameters = _parse_args(signature(wrapped), instance, *args, **kwargs)
        span_name = "AsyncEmbedContent"
        with self._start_as_current_span(
            span_name=span_name,
            attributes=chain(
                get_attributes_from_context(),
                self._request_extractor.get_attributes_from_request(request_parameters),
            ),
        ) as span:
            with CapturedRequestScope():
                try:
                    response = await wrapped(*args, **kwargs)
                except Exception as exception:
                    _set_captured_embedding_attributes(span)
                    span.record_exception(exception)
                    status = trace_api.Status(
                        status_code=trace_api.StatusCode.ERROR,
                        description=f"{type(exception).__name__}: {exception}",
                    )
                    span.finish_tracing(status=status)
                    raise
                _set_captured_embedding_attributes(span)
            try:
                _finish_tracing(
                    status=trace_api.Status(status_code=trace_api.StatusCode.OK),
                    with_span=span,
                    attributes=self._response_extractor.get_attributes(
                        response=response,
                        request_parameters=request_parameters,
                    ),
                )
            except Exception:
                logger.exception(f"Failed to finalize response of type {type(response)}")
                span.finish_tracing()
        return response


class _SyncGenerateContent(_WithTracer):
    """
    Wrapper for the pipeline processing
    Captures all calls to the pipeline
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._request_extractor = _RequestAttributesExtractor()
        self._response_extractor = _ResponseAttributesExtractor()

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        request_parameters = _parse_args(signature(wrapped), instance, *args, **kwargs)
        span_name = "GenerateContent"
        with self._start_as_current_span(
            span_name=span_name,
            attributes=chain(
                get_attributes_from_context(),
                self._request_extractor.get_attributes_from_request(request_parameters),
            ),
        ) as span:
            with CapturedRequestScope():
                try:
                    response = wrapped(*args, **kwargs)
                except Exception as exception:
                    _set_captured_llm_attributes(span)
                    span.record_exception(exception)
                    status = trace_api.Status(
                        status_code=trace_api.StatusCode.ERROR,
                        description=f"{type(exception).__name__}: {exception}",
                    )
                    span.finish_tracing(status=status)
                    raise
                _set_captured_llm_attributes(span)
            try:
                _finish_tracing(
                    status=trace_api.Status(status_code=trace_api.StatusCode.OK),
                    with_span=span,
                    attributes=self._response_extractor.get_attributes(
                        response=response,
                        request_parameters=request_parameters,
                    ),
                )
            except Exception:
                logger.exception(f"Failed to finalize response of type {type(response)}")
                span.finish_tracing()
        return response


class _SyncCreateInteractionWrapper(_WithTracer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        request_parameters = _parse_args(signature(wrapped), instance, *args, **kwargs)
        span_name = "InteractionsResource.create"
        with self._start_as_current_span(
            span_name=span_name,
            attributes=chain(
                get_attributes_from_context(),
                get_attributes_from_request(request_parameters),
            ),
        ) as span:
            try:
                response = wrapped(*args, **kwargs)
                if request_parameters.get("stream", False):
                    return _InteractionsStream(
                        stream=response,
                        with_span=span,
                        request_parameters=request_parameters,
                    )
                span.set_attributes(get_attributes_from_response(request_parameters, response))
                status = trace_api.Status(status_code=trace_api.StatusCode.OK)
                span.finish_tracing(status=status)
            except Exception as exception:
                span.record_exception(exception)
                status = trace_api.Status(
                    status_code=trace_api.StatusCode.ERROR,
                    description=f"{type(exception).__name__}: {exception}",
                )
                span.finish_tracing(status=status)
                raise
        return response


class _SyncGenerateContentStream(_WithTracer):
    """
    Wrapper for the pipeline processing
    Captures all calls to the pipeline
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._request_extractor = _RequestAttributesExtractor()

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        request_parameters = _parse_args(signature(wrapped), instance, *args, **kwargs)
        span_name = "GenerateContentStream"
        with self._start_as_current_span(
            span_name=span_name,
            attributes=chain(
                get_attributes_from_context(),
                self._request_extractor.get_attributes_from_request(request_parameters),
            ),
        ) as span:
            request_scope = CapturedRequestScope()
            request_scope.__enter__()
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                _set_captured_llm_attributes(span)
                request_scope.__exit__(None, None, None)
                span.record_exception(exception)
                status = trace_api.Status(
                    status_code=trace_api.StatusCode.ERROR,
                    description=f"{type(exception).__name__}: {exception}",
                )
                span.finish_tracing(status=status)
                raise
            try:
                return _Stream(
                    stream=response,
                    with_span=span,
                    request_scope=request_scope,
                )
            except Exception:
                request_scope.__exit__(None, None, None)
                logger.exception(f"Failed to finalize response of type {type(response)}")
                span.finish_tracing()
                return response


class _AsyncGenerateContentWrapper(_WithTracer):
    """
    Wrapper for the pipeline processing
    Captures all calls to the pipeline
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._request_extractor = _RequestAttributesExtractor()
        self._response_extractor = _ResponseAttributesExtractor()

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        request_parameters = _parse_args(signature(wrapped), instance, *args, **kwargs)
        span_name = "AsyncGenerateContent"
        with self._start_as_current_span(
            span_name=span_name,
            attributes=chain(
                get_attributes_from_context(),
                self._request_extractor.get_attributes_from_request(request_parameters),
            ),
        ) as span:
            with CapturedRequestScope():
                try:
                    response = await wrapped(*args, **kwargs)
                except Exception as exception:
                    _set_captured_llm_attributes(span)
                    span.record_exception(exception)
                    status = trace_api.Status(
                        status_code=trace_api.StatusCode.ERROR,
                        description=f"{type(exception).__name__}: {exception}",
                    )
                    span.finish_tracing(status=status)
                    raise
                _set_captured_llm_attributes(span)
            try:
                _finish_tracing(
                    status=trace_api.Status(status_code=trace_api.StatusCode.OK),
                    with_span=span,
                    attributes=self._response_extractor.get_attributes(
                        response=response,
                        request_parameters=request_parameters,
                    ),
                )
            except Exception:
                logger.exception(f"Failed to finalize response of type {type(response)}")
                span.finish_tracing()
        return response


class _AsyncGenerateContentStream(_WithTracer):
    """
    Wrapper for the pipeline processing
    Captures all calls to the pipeline
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._request_extractor = _RequestAttributesExtractor()

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        request_parameters = _parse_args(signature(wrapped), instance, *args, **kwargs)
        span_name = "AsyncGenerateContentStream"
        with self._start_as_current_span(
            span_name=span_name,
            attributes=chain(
                get_attributes_from_context(),
                self._request_extractor.get_attributes_from_request(request_parameters),
            ),
        ) as span:
            request_scope = CapturedRequestScope()
            request_scope.__enter__()
            try:
                response = await wrapped(*args, **kwargs)
            except Exception as exception:
                _set_captured_llm_attributes(span)
                request_scope.__exit__(None, None, None)
                span.record_exception(exception)
                status = trace_api.Status(
                    status_code=trace_api.StatusCode.ERROR,
                    description=f"{type(exception).__name__}: {exception}",
                )
                span.finish_tracing(status=status)
                raise
            try:
                return _Stream(
                    stream=response,
                    with_span=span,
                    request_scope=request_scope,
                )
            except Exception:
                request_scope.__exit__(None, None, None)
                logger.exception(f"Failed to finalize response of type {type(response)}")
                span.finish_tracing()
                return response


class _AsyncCreateInteractionWrapper(_WithTracer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)
        request_parameters = _parse_args(signature(wrapped), instance, *args, **kwargs)
        span_name = "AsyncInteractionsResource.create"
        with self._start_as_current_span(
            span_name=span_name,
            attributes=chain(
                get_attributes_from_context(),
                get_attributes_from_request(request_parameters),
            ),
        ) as span:
            try:
                response = await wrapped(*args, **kwargs)
                if request_parameters.get("stream", False):
                    return _InteractionsStream(
                        stream=response,
                        with_span=span,
                        request_parameters=request_parameters,
                    )
                span.set_attributes(get_attributes_from_response(request_parameters, response))
                status = trace_api.Status(status_code=trace_api.StatusCode.OK)
                span.finish_tracing(status=status)
            except Exception as exception:
                span.record_exception(exception)
                status = trace_api.Status(
                    status_code=trace_api.StatusCode.ERROR,
                    description=f"{type(exception).__name__}: {exception}",
                )
                span.finish_tracing(status=status)
                raise
        return response


class _SyncCreateCachesWrapper(_WithTracer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        request_parameters = _parse_args(signature(wrapped), instance, *args, **kwargs)
        span_name = "Caches.create"
        status = trace_api.Status(status_code=trace_api.StatusCode.OK)
        with self._start_as_current_span(
            span_name=span_name,
            attributes=chain(
                get_attributes_from_context(),
                cache_attributes.get_attributes_from_request(request_parameters),
            ),
        ) as span:
            try:
                response = wrapped(*args, **kwargs)
            except BaseException as exception:
                span.record_exception(exception)
                status = trace_api.Status(
                    status_code=trace_api.StatusCode.ERROR,
                    description=f"{type(exception).__name__}: {exception}",
                )
                raise
            else:
                span.set_attributes(dict(cache_attributes.get_attributes_from_response(response)))
            finally:
                span.finish_tracing(status=status)
        return response


class _AsyncCreateCachesWrapper(_WithTracer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)
        request_parameters = _parse_args(signature(wrapped), instance, *args, **kwargs)
        span_name = "AsyncCaches.create"
        status = trace_api.Status(status_code=trace_api.StatusCode.OK)
        with self._start_as_current_span(
            span_name=span_name,
            attributes=chain(
                get_attributes_from_context(),
                cache_attributes.get_attributes_from_request(request_parameters),
            ),
        ) as span:
            try:
                response = await wrapped(*args, **kwargs)
            except BaseException as exception:
                span.record_exception(exception)
                status = trace_api.Status(
                    status_code=trace_api.StatusCode.ERROR,
                    description=f"{type(exception).__name__}: {exception}",
                )
                raise
            else:
                span.set_attributes(dict(cache_attributes.get_attributes_from_response(response)))
            finally:
                span.finish_tracing(status=status)
        return response


CHAIN = OpenInferenceSpanKindValues.CHAIN.value
RETRIEVER = OpenInferenceSpanKindValues.RETRIEVER.value
EMBEDDING = OpenInferenceSpanKindValues.EMBEDDING.value
LLM = OpenInferenceSpanKindValues.LLM.value
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
EMBEDDING_VECTOR = EmbeddingAttributes.EMBEDDING_VECTOR
EMBEDDING_TEXT = EmbeddingAttributes.EMBEDDING_TEXT
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
