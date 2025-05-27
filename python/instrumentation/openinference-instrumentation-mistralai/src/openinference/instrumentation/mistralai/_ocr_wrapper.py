import logging
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
from openinference.instrumentation.mistralai._response_attributes_extractor import (
    _OCRResponseAttributesExtractor,
)
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

__all__ = (
    "_SyncOCRWrapper",
    "_AsyncOCRWrapper",
)

# Define OCR span kind since it's not in openinference-semantic-conventions yet
_OCR_SPAN_KIND = OpenInferenceSpanKindValues.LLM.value

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _OCRResponseAttributes:
    __slots__ = (
        "_response",
        "_request_parameters",
        "_response_attributes_extractor",
    )

    def __init__(
        self,
        response: Any,
        request_parameters: Mapping[str, Any],
        response_attributes_extractor: _OCRResponseAttributesExtractor,
    ) -> None:
        self._response = response
        self._request_parameters = request_parameters
        self._response_attributes_extractor = response_attributes_extractor

    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        if hasattr(self._response, "model_dump_json") and callable(self._response.model_dump_json):
            try:
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


class _WithTracer:
    def __init__(self, tracer: trace_api.Tracer) -> None:
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


class _WithMistralAI:
    __slots__ = (
        "_request_attributes_extractor",
        "_response_attributes_extractor",
    )

    def __init__(self, mistralai: ModuleType) -> None:
        self._request_attributes_extractor = _RequestAttributesExtractor(mistralai)
        self._response_attributes_extractor = _OCRResponseAttributesExtractor()

    def _get_span_kind(self) -> str:
        return _OCR_SPAN_KIND

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
            yield from self._request_attributes_extractor.get_attributes_from_ocr_request(
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
                    except Exception:
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
        Finish tracing for the OCR response.
        """
        try:
            _finish_tracing(
                status=trace_api.Status(status_code=trace_api.StatusCode.OK),
                with_span=with_span,
                has_attributes=_OCRResponseAttributes(
                    request_parameters=request_parameters,
                    response=response,
                    response_attributes_extractor=self._response_attributes_extractor,
                ),
            )
        except Exception:
            logger.exception(f"Failed to finish tracing for response of type {type(response)}")
            with_span.finish_tracing()
        return response


class _SyncOCRWrapper(_WithTracer, _WithMistralAI):
    def __init__(self, span_name: str, tracer: trace_api.Tracer, mistralai: ModuleType):
        _WithTracer.__init__(self, tracer)
        _WithMistralAI.__init__(self, mistralai)
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
            extra_attributes=self._get_extra_attributes_from_request(
                request_parameters
            ),  # redundant under the current span type of LLM
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
            return self._finalize_response(
                response=response,
                with_span=with_span,
                request_parameters=request_parameters,
            )


class _AsyncOCRWrapper(_WithTracer, _WithMistralAI):
    def __init__(self, span_name: str, tracer: trace_api.Tracer, mistralai: ModuleType):
        _WithTracer.__init__(self, tracer)
        _WithMistralAI.__init__(self, mistralai)
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
            return self._finalize_response(
                response=response,
                with_span=with_span,
                request_parameters=request_parameters,
            )
