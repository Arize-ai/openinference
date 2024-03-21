import json
import logging
from contextlib import contextmanager
from inspect import Signature, signature
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

from openinference.instrumentation.mistralai._response_attributes_extractor import (
    _ResponseAttributesExtractor,
)
from openinference.instrumentation.mistralai._utils import _finish_tracing
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


class _SyncChatWrapper:
    __slots__ = (
        "_tracer",
        "_mistral_client",
        "_response_attributes_extractor",
    )

    def __init__(self, tracer: trace_api.Tracer, mistral_client: "MistralClient") -> None:
        self._tracer = tracer
        self._mistral_client = mistral_client
        self._response_attributes_extractor = _ResponseAttributesExtractor()

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
            request_parameters = self._parse_args(signature(wrapped), *args, **kwargs)
            span_name = "MistralClient.chat"
        except Exception:
            logger.exception("Failed to parse request args")
            return wrapped(*args, **kwargs)
        with self._start_as_current_span(
            span_name=span_name,
            attributes=self._get_attributes_from_request(request_parameters),
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
                        response_attributes_extractor=self._response_attributes_extractor,
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

    def _get_span_kind(self) -> str:
        return OpenInferenceSpanKindValues.LLM.value

    def _get_attributes_from_request(
        self,
        request_parameters: Dict[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield SpanAttributes.OPENINFERENCE_SPAN_KIND, self._get_span_kind()
        try:
            yield SpanAttributes.INPUT_MIME_TYPE, OpenInferenceMimeTypeValues.JSON.value
            yield SpanAttributes.INPUT_VALUE, json.dumps(request_parameters)
            invocation_parameters = {
                param_key: param_value
                for param_key, param_value in request_parameters.items()
                if param_key != "messages"
            }
            yield SpanAttributes.LLM_INVOCATION_PARAMETERS, json.dumps(invocation_parameters)
            yield SpanAttributes.LLM_INPUT_MESSAGES, json.dumps(request_parameters.get("messages"))
        except Exception:
            logger.exception("Failed to get input attributes from request parameters.")

    def _parse_args(
        self, signature: Signature, *args: Tuple[Any], **kwargs: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """
        Invokes the private _make_chat_request method on MistralClient, which
        serializes the request parameters to JSON
        """
        bound_arguments = signature.bind(*args, **kwargs).arguments
        return self._mistral_client._make_chat_request(**bound_arguments)


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
        yield from ()

    def get_extra_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        yield from self._response_attributes_extractor.get_attributes_from_response(
            response=self._response,
        )
