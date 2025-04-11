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
from openinference.instrumentation.google_genai._request_attributes_extractor import (
    _RequestAttributesExtractor,
)
from openinference.instrumentation.google_genai._response_attributes_extractor import (
    _ResponseAttributesExtractor,
)
from openinference.instrumentation.google_genai._utils import _finish_tracing
from openinference.instrumentation.google_genai._with_span import _WithSpan
from openinference.semconv.trace import (
    EmbeddingAttributes,
    MessageAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

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
            logger.exception("Failed to start span")
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


# class _WithGoogleGenAI(ABC):
#     __slots__ = (
#         "_request_attributes_extractor",
#         "_response_attributes_extractor",
#     )

#     def __init__(self, *args: Any, **kwargs: Any) -> None:
#         super().__init__(*args, **kwargs)
#         self._request_attributes_extractor = _RequestAttributesExtractor()
#         self._response_attributes_extractor = _ResponseAttributesExtractor()

#     def _get_span_kind(self) -> str:
#         return OpenInferenceSpanKindValues.LLM.value

#     def _get_attributes_from_request(
#         self,
#         request_parameters: Dict[str, Any],
#     ) -> Iterator[Tuple[str, AttributeValue]]:
#         yield SpanAttributes.OPENINFERENCE_SPAN_KIND, self._get_span_kind()
#         try:
#             yield from _as_input_attributes(_io_value_and_type(request_parameters))
#         except Exception:
#             logger.exception(
#                 f"Failed to get input attributes from request parameters of "
#                 f"type {type(request_parameters)}"
#             )

#     def _get_extra_attributes_from_request(
#         self,
#         request_parameters: Mapping[str, Any],
#     ) -> Iterator[Tuple[str, AttributeValue]]:
#         try:
#             yield from self._request_attributes_extractor.get_attributes_from_request(
#                 request_parameters=request_parameters,
#             )
#         except Exception:
#             logger.exception(
#                 f"Failed to get extra attributes from request options of "
#                 f"type {type(request_parameters)}"
#             )

#     def _parse_args(
#         self,
#         signature: Signature,
#         *args: Tuple[Any],
#         **kwargs: Mapping[str, Any],
#     ) -> Dict[str, Any]:
#         """
#         Serialize parameters to JSON.
#         """
#         bound_signature = signature.bind(*args, **kwargs)
#         bound_signature.apply_defaults()
#         bound_arguments = bound_signature.arguments
#         request_data: Dict[str, Any] = {}
#         for key, value in bound_arguments.items():
#             try:
#                 if value is not None:
#                     try:
#                         # ensure the value is JSON-serializable
#                         safe_json_dumps(value)
#                         request_data[key] = value
#                     except json.JSONDecodeError:
#                         request_data[key] = str(value)
#             except Exception:
#                 request_data[key] = str(value)
#         return request_data

#     def _finalize_response(
#         self,
#         response: Any,
#         with_span: _WithSpan,
#         request_parameters: Mapping[str, Any],
#     ) -> Any:
#         """
#         Monkey-patch the response object to trace the stream, or finish tracing if the response is
#         not a stream.
#         """
#         from mistralai.models.chatcompletionresponse import ChatCompletionResponse
#         from mistralai.models.completionevent import CompletionEvent

#         if not isinstance(response, ChatCompletionResponse):  # assume it's a stream
#             response_accumulator = _ChatCompletionAccumulator(
#                 request_parameters=request_parameters,
#                 chat_completion_type=CompletionEvent,
#                 response_attributes_extractor=_StreamResponseAttributesExtractor(),
#             )
#             # we need to run this check first because in python 3.9 iterators are
#             # considered coroutines
#             if isinstance(response, Iterable):
#                 return _Stream(
#                     stream=response,  # type: ignore
#                     with_span=with_span,
#                     response_accumulator=response_accumulator,
#                 )
#             elif asyncio.iscoroutine(response):
#                 return _AsyncStream(
#                     stream=response,
#                     with_span=with_span,
#                     response_accumulator=response_accumulator,
#                 ).stream_async_with_accumulator()
#             else:
#                 raise TypeError("Response must be either a coroutine or an iterable")
#         _finish_tracing(
#             status=trace_api.Status(status_code=trace_api.StatusCode.OK),
#             with_span=with_span,
#             has_attributes=_ResponseAttributes(
#                 request_parameters=request_parameters,
#                 response=response,
#                 response_attributes_extractor=self._response_attributes_extractor,
#             ),
#         )
#         return response


def _parse_args(
    signature: Signature,
    *args: Tuple[Any],
    **kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    bound_signature = signature.bind(*args, **kwargs)
    bound_signature.apply_defaults()
    bound_arguments = bound_signature.arguments  # Defaults empty to NOT_GIVEN
    request_data: Dict[str, Any] = {}
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
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        # Prepare invocation parameters by merging args and kwargs
        invocation_parameters = {}
        for arg in args:
            if arg and isinstance(arg, dict):
                invocation_parameters.update(arg)
        invocation_parameters.update(kwargs)
        request_parameters = _parse_args(signature(wrapped), *args, **kwargs)
        span_name = "GenerateContent"
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
                status = trace_api.Status(
                    status_code=trace_api.StatusCode.ERROR,
                    description=f"{type(exception).__name__}: {exception}",
                )
                span.finish_tracing(status=status)
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
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        # Prepare invocation parameters by merging args and kwargs
        invocation_parameters = {}
        for arg in args:
            if arg and isinstance(arg, dict):
                invocation_parameters.update(arg)
        invocation_parameters.update(kwargs)
        request_parameters = _parse_args(signature(wrapped), *args, **kwargs)

        span_name = "AsyncGenerateContent"
        with self._start_as_current_span(
            span_name=span_name,
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
                status = trace_api.Status(
                    status_code=trace_api.StatusCode.ERROR,
                    description=f"{type(exception).__name__}: {exception}",
                )
                span.finish_tracing(status=status)
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


CHAIN = OpenInferenceSpanKindValues.CHAIN
RETRIEVER = OpenInferenceSpanKindValues.RETRIEVER
EMBEDDING = OpenInferenceSpanKindValues.EMBEDDING
LLM = OpenInferenceSpanKindValues.LLM
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
EMBEDDING_VECTOR = EmbeddingAttributes.EMBEDDING_VECTOR
EMBEDDING_TEXT = EmbeddingAttributes.EMBEDDING_TEXT
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
