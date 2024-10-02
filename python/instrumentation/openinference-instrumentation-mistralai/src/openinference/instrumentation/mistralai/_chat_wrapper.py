import asyncio
import json
import logging
import warnings
from abc import ABC
from contextlib import contextmanager
from inspect import Signature, signature
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Tuple,
)

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace import INVALID_SPAN
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import get_attributes_from_context, safe_json_dumps
from openinference.instrumentation.mistralai._request_attributes_extractor import (
    _RequestAttributesExtractor,
)
from openinference.instrumentation.mistralai._response_accumulator import _ChatCompletionAccumulator
from openinference.instrumentation.mistralai._response_attributes_extractor import (
    _ResponseAttributesExtractor,
    _StreamResponseAttributesExtractor,
)
from openinference.instrumentation.mistralai._stream import _AsyncStream, _Stream
from openinference.instrumentation.mistralai._utils import (
    _as_input_attributes,
    _finish_tracing,
    _io_value_and_type,
)
from openinference.instrumentation.mistralai._with_span import _WithSpan
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

if TYPE_CHECKING:
    from mistralai import Mistral

__all__ = ("_SyncChatWrapper",)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _WithTracer(ABC):
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


class _WithMistralAI(ABC):
    __slots__ = (
        "_request_attributes_extractor",
        "_response_attributes_extractor",
    )

    def __init__(self, mistralai: ModuleType, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._request_attributes_extractor = _RequestAttributesExtractor(mistralai)
        self._response_attributes_extractor = _ResponseAttributesExtractor()

    def _get_span_kind(self) -> str:
        return OpenInferenceSpanKindValues.LLM.value

    def _get_attributes_from_request(
        self,
        request_parameters: Dict[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield SpanAttributes.OPENINFERENCE_SPAN_KIND, self._get_span_kind()
        try:
            yield from _as_input_attributes(_io_value_and_type(request_parameters))
        except Exception:
            logger.exception(
                f"Failed to get input attributes from request parameters of "
                f"type {type(request_parameters)}"
            )

    def _get_extra_attributes_from_request(
        self,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        try:
            yield from self._request_attributes_extractor.get_attributes_from_request(
                request_parameters=request_parameters,
            )
        except Exception:
            logger.exception(
                f"Failed to get extra attributes from request options of "
                f"type {type(request_parameters)}"
            )

    def _parse_args(
        self,
        signature: Signature,
        mistral_client: "Mistral",
        *args: Tuple[Any],
        **kwargs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Serialize parameters to JSON.

        Based off of https://github.com/mistralai/client-python/blob/80c7951bad83338641d5e89684f841ce1cac938f/src/mistralai/client_base.py#L76
        """
        bound_signature = signature.bind(*args, **kwargs)
        bound_signature.apply_defaults()
        bound_arguments = bound_signature.arguments
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

    def _finalize_response(
        self,
        response: Any,
        with_span: _WithSpan,
        request_parameters: Mapping[str, Any],
    ) -> Any:
        """
        Monkey-patch the response object to trace the stream, or finish tracing if the response is
        not a stream.
        """
        from mistralai.models.chatcompletionresponse import ChatCompletionResponse
        from mistralai.models.completionevent import CompletionEvent

        if not isinstance(response, ChatCompletionResponse):  # assume it's a stream
            response_accumulator = _ChatCompletionAccumulator(
                request_parameters=request_parameters,
                chat_completion_type=CompletionEvent,
                response_attributes_extractor=_StreamResponseAttributesExtractor(),
            )
            # we need to run this check first because in python 3.9 iterators are
            # considered coroutines
            if isinstance(response, Iterable):
                return _Stream(
                    stream=response,  # type: ignore
                    with_span=with_span,
                    response_accumulator=response_accumulator,
                )
            elif asyncio.iscoroutine(response):
                return _AsyncStream(
                    stream=response,
                    with_span=with_span,
                    response_accumulator=response_accumulator,
                ).stream_async_with_accumulator()
            else:
                raise TypeError("Response must be either a coroutine or an iterable")
        _finish_tracing(
            status=trace_api.Status(status_code=trace_api.StatusCode.OK),
            with_span=with_span,
            has_attributes=_ResponseAttributes(
                request_parameters=request_parameters,
                response=response,
                response_attributes_extractor=self._response_attributes_extractor,
            ),
        )
        return response


class _SyncChatWrapper(_WithTracer, _WithMistralAI):
    def __init__(self, span_name: str, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._span_name = span_name

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: "Mistral",
        args: Tuple[Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        try:
            request_parameters = self._parse_args(signature(wrapped), instance, *args, **kwargs)
        except Exception:
            logger.exception("Failed to parse request args")
            return wrapped(*args, **kwargs)
        with self._start_as_current_span(
            span_name=self._span_name,
            attributes=self._get_attributes_from_request(request_parameters),
            context_attributes=get_attributes_from_context(),
            extra_attributes=self._get_extra_attributes_from_request(request_parameters),
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
                response = self._finalize_response(
                    response=response,
                    with_span=with_span,
                    request_parameters=request_parameters,
                )
            except Exception:
                logger.exception(f"Failed to finish tracing for response of type {type(response)}")
                with_span.finish_tracing()
        return response


class _AsyncChatWrapper(_WithTracer, _WithMistralAI):
    def __init__(self, span_name: str, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._span_name = span_name

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: "Mistral",
        args: Tuple[Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)
        try:
            request_parameters = self._parse_args(signature(wrapped), instance, *args, **kwargs)
        except Exception:
            logger.exception("Failed to parse request args")
            return await wrapped(*args, **kwargs)
        with self._start_as_current_span(
            span_name=self._span_name,
            attributes=self._get_attributes_from_request(request_parameters),
            context_attributes=get_attributes_from_context(),
            extra_attributes=self._get_extra_attributes_from_request(request_parameters),
        ) as with_span:
            try:
                response = await wrapped(*args, **kwargs)
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
                response = self._finalize_response(
                    response=response,
                    with_span=with_span,
                    request_parameters=request_parameters,
                )
            except Exception:
                logger.exception(f"Failed to finish tracing for response of type {type(response)}")
                with_span.finish_tracing()
        return response


class _AsyncStreamChatWrapper(_WithTracer, _WithMistralAI):
    def __init__(self, span_name: str, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._span_name = span_name

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: "Mistral",
        args: Tuple[Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        try:
            request_parameters = self._parse_args(signature(wrapped), instance, *args, **kwargs)
        except Exception:
            logger.exception("Failed to parse request args")
            return wrapped(*args, **kwargs)
        with self._start_as_current_span(
            span_name=self._span_name,
            attributes=self._get_attributes_from_request(request_parameters),
            context_attributes=get_attributes_from_context(),
            extra_attributes=self._get_extra_attributes_from_request(request_parameters),
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
                response = self._finalize_response(
                    response=response,
                    with_span=with_span,
                    request_parameters=request_parameters,
                )
            except Exception:
                logger.exception(f"Failed to finish tracing for response of type {type(response)}")
                with_span.finish_tracing()
        return response


class _ResponseAttributes:
    __slots__ = (
        "_response",
        "_request_parameters",
        "_response_attributes_extractor",
    )

    def __init__(
        self,
        response: Any,
        request_parameters: Mapping[str, Any],
        response_attributes_extractor: _ResponseAttributesExtractor,
    ) -> None:
        self._response = response
        self._request_parameters = request_parameters
        self._response_attributes_extractor = response_attributes_extractor

    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        if hasattr(self._response, "model_dump_json") and callable(self._response.model_dump_json):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    value = self._response.model_dump_json(exclude_unset=True)
                assert isinstance(value, str)
                yield SpanAttributes.OUTPUT_VALUE, value
                yield SpanAttributes.OUTPUT_MIME_TYPE, OpenInferenceMimeTypeValues.JSON.value
            except Exception:
                logger.exception("Failed to get model dump json")

    def get_extra_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        yield from self._response_attributes_extractor.get_attributes_from_response(
            response=self._response,
            request_parameters=self._request_parameters,
        )
