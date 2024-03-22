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

from openinference.instrumentation.mistralai._request_attributes_extractor import (
    _RequestAttributesExtractor,
)
from openinference.instrumentation.mistralai._response_attributes_extractor import (
    _ResponseAttributesExtractor,
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
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace import INVALID_SPAN
from opentelemetry.util.types import AttributeValue

if TYPE_CHECKING:
    from mistralai.client import MistralClient

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
            yield _WithSpan(span=span, extra_attributes=dict(attributes))


class _WithMistralAI(ABC):
    __slots__ = (
        "_mistral_client",
        "_request_attributes_extractor",
        "_response_attributes_extractor",
    )

    def __init__(self, mistralai: ModuleType, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._mistral_client = mistralai.client.MistralClient()
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
        mistral_client: "MistralClient",
        *args: Tuple[Any],
        **kwargs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Serialize parameters to JSON.

        Based off of https://github.com/mistralai/client-python/blob/80c7951bad83338641d5e89684f841ce1cac938f/src/mistralai/client_base.py#L76
        """
        bound_arguments = signature.bind(*args, **kwargs).arguments
        request_data: Dict[str, Any] = {}
        for key, value in bound_arguments.items():
            try:
                if key == "messages":
                    request_data[key] = mistral_client._parse_messages(value)
                elif key == "tools":
                    request_data[key] = mistral_client._parse_tools(value)
                elif key == "tool_choice":
                    request_data[key] = mistral_client._parse_tool_choice(value)
                elif key == "response_format" and value is not None:
                    request_data[key] = mistral_client._parse_response_format(value)
                elif value is not None:
                    try:
                        # ensure the value is JSON-serializable
                        json.dumps(value)
                        request_data[key] = value
                    except json.JSONDecodeError:
                        request_data[key] = str(value)
            except Exception:
                request_data[key] = str(value)
        return request_data


class _SyncChatWrapper(_WithTracer, _WithMistralAI):
    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        try:
            request_parameters = self._parse_args(
                signature(wrapped), self._mistral_client, *args, **kwargs
            )
            span_name = "MistralClient.chat"
        except Exception:
            logger.exception("Failed to parse request args")
            return wrapped(*args, **kwargs)
        with self._start_as_current_span(
            span_name=span_name,
            attributes=self._get_attributes_from_request(request_parameters),
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
                _finish_tracing(
                    status=trace_api.Status(status_code=trace_api.StatusCode.OK),
                    with_span=with_span,
                    has_attributes=_ResponseAttributes(
                        request_parameters=request_parameters,
                        response=response,
                        response_attributes_extractor=self._response_attributes_extractor,
                    ),
                )
            except Exception:
                logger.exception(f"Failed to finish tracing for response of type {type(response)}")
                with_span.finish_tracing()
        return response


class _AsyncChatWrapper(_WithTracer, _WithMistralAI):
    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)
        try:
            request_parameters = self._parse_args(
                signature(wrapped), self._mistral_client, *args, **kwargs
            )
            span_name = "MistralAsyncClient.chat"
        except Exception:
            logger.exception("Failed to parse request args")
            return await wrapped(*args, **kwargs)
        with self._start_as_current_span(
            span_name=span_name,
            attributes=self._get_attributes_from_request(request_parameters),
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
                _finish_tracing(
                    status=trace_api.Status(status_code=trace_api.StatusCode.OK),
                    with_span=with_span,
                    has_attributes=_ResponseAttributes(
                        request_parameters=request_parameters,
                        response=response,
                        response_attributes_extractor=self._response_attributes_extractor,
                    ),
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
        )
