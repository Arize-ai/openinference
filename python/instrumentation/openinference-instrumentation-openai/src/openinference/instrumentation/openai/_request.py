import logging
from abc import ABC
from contextlib import contextmanager
from itertools import chain
from types import ModuleType
from typing import Any, Awaitable, Callable, Iterable, Iterator, Mapping, Tuple

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace import INVALID_SPAN
from opentelemetry.util.types import AttributeValue
from typing_extensions import TypeAlias

from openinference.instrumentation import get_attributes_from_context
from openinference.instrumentation.openai._request_attributes_extractor import (
    _RequestAttributesExtractor,
)
from openinference.instrumentation.openai._response_accumulator import (
    _ChatCompletionAccumulator,
    _CompletionAccumulator,
    _ResponsesAccumulator,
)
from openinference.instrumentation.openai._response_attributes_extractor import (
    _ResponseAttributesExtractor,
)
from openinference.instrumentation.openai._stream import _ResponseAccumulator, _Stream
from openinference.instrumentation.openai._utils import (
    _as_input_attributes,
    _as_output_attributes,
    _finish_tracing,
    _io_value_and_type,
)
from openinference.instrumentation.openai._with_span import _WithSpan
from openinference.semconv.trace import (
    OpenInferenceLLMProviderValues,
    OpenInferenceLLMSystemValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

__all__ = (
    "_Request",
    "_AsyncRequest",
)


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
            yield _WithSpan(
                span=span,
                context_attributes=dict(context_attributes),
                extra_attributes=dict(extra_attributes),
            )


_RequestParameters: TypeAlias = Mapping[str, Any]


class _WithOpenAI(ABC):
    __slots__ = (
        "_openai",
        "_stream_types",
        "_request_attributes_extractor",
        "_response_attributes_extractor",
        "_response_accumulator_factories",
    )

    def __init__(self, openai: ModuleType, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._openai = openai
        self._stream_types = (openai.Stream, openai.AsyncStream)
        self._request_attributes_extractor = _RequestAttributesExtractor(openai=openai)
        self._response_attributes_extractor = _ResponseAttributesExtractor(openai=openai)

        def responses_accumulator(request_parameters: _RequestParameters) -> Any:
            return _ResponsesAccumulator(
                request_parameters=request_parameters,
                chat_completion_type=openai.types.responses.response.Response,
                response_attributes_extractor=self._response_attributes_extractor,
            )

        self._response_accumulator_factories: Mapping[
            type, Callable[[_RequestParameters], _ResponseAccumulator]
        ] = {
            openai.types.Completion: lambda request_parameters: _CompletionAccumulator(
                request_parameters=request_parameters,
                completion_type=openai.types.Completion,
                response_attributes_extractor=self._response_attributes_extractor,
            ),
            openai.types.chat.ChatCompletion: lambda request_parameters: _ChatCompletionAccumulator(
                request_parameters=request_parameters,
                chat_completion_type=openai.types.chat.ChatCompletion,
                response_attributes_extractor=self._response_attributes_extractor,
            ),
            openai.types.responses.response.Response: responses_accumulator,
        }

    def _get_span_kind(self, cast_to: type) -> str:
        return (
            OpenInferenceSpanKindValues.EMBEDDING.value
            if cast_to is self._openai.types.CreateEmbeddingResponse
            else OpenInferenceSpanKindValues.LLM.value
        )

    def _get_attributes_from_instance(self, instance: Any) -> Iterator[Tuple[str, AttributeValue]]:
        if (
            not (base_url := getattr(instance, "base_url", None))
            or not (host := getattr(base_url, "host", None))
            or not isinstance(host, str)
        ):
            return
        if host.endswith("api.openai.com"):
            yield SpanAttributes.LLM_PROVIDER, OpenInferenceLLMProviderValues.OPENAI.value
        elif host.endswith("openai.azure.com"):
            yield SpanAttributes.LLM_PROVIDER, OpenInferenceLLMProviderValues.AZURE.value
        elif host.endswith("googleapis.com"):
            yield SpanAttributes.LLM_PROVIDER, OpenInferenceLLMProviderValues.GOOGLE.value

    def _get_attributes_from_request(
        self,
        cast_to: type,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield SpanAttributes.OPENINFERENCE_SPAN_KIND, self._get_span_kind(cast_to=cast_to)
        yield SpanAttributes.LLM_SYSTEM, OpenInferenceLLMSystemValues.OPENAI.value
        try:
            yield from _as_input_attributes(_io_value_and_type(request_parameters))
        except Exception:
            logger.exception(
                f"Failed to get input attributes from request parameters of "
                f"type {type(request_parameters)}"
            )

    def _get_extra_attributes_from_request(
        self,
        cast_to: type,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        # Secondary attributes should be added after input and output to ensure
        # that input and output are not dropped if there are too many attributes.
        try:
            yield from self._request_attributes_extractor.get_attributes_from_request(
                cast_to=cast_to,
                request_parameters=request_parameters,
            )
        except Exception:
            logger.exception(
                f"Failed to get extra attributes from request options of "
                f"type {type(request_parameters)}"
            )

    def _is_streaming(self, response: Any) -> bool:
        return isinstance(response, self._stream_types)

    def _finalize_response(
        self,
        response: Any,
        with_span: _WithSpan,
        cast_to: type,
        request_parameters: Mapping[str, Any],
    ) -> Any:
        """
        Monkey-patch the response object to trace the stream, or finish tracing if the response is
        not a stream.
        """

        if hasattr(response, "parse") and callable(response.parse):
            # `.request()` may be called under `.with_raw_response` and it's necessary to call
            # `.parse()` to get back the usual response types.
            # E.g. see https://github.com/openai/openai-python/blob/f1c7d714914e3321ca2e72839fe2d132a8646e7f/src/openai/_base_client.py#L518  # noqa: E501
            try:
                response.parse()
            except Exception:
                logger.exception(f"Failed to parse response of type {type(response)}")
        if (
            self._is_streaming(response)
            or hasattr(
                # FIXME: Ideally we shouldn't rely on a private attribute (but it may be
                # impossible). The assumption here is that calling `.parse()` stores the
                # stream object in `._parsed` and calling `.parse()` again will not
                # overwrite the monkey-patched version.
                # See https://github.com/openai/openai-python/blob/f1c7d714914e3321ca2e72839fe2d132a8646e7f/src/openai/_response.py#L65  # noqa: E501
                response,
                "_parsed",
            )
            # Note that we must have called `.parse()` beforehand, otherwise `._parsed` is None.
            and self._is_streaming(response._parsed)
            or hasattr(response, "_parsed_by_type")
            and hasattr(response._parsed_by_type, "get")
            and self._is_streaming(response._parsed_by_type.get(cast_to))
        ):
            # For streaming, we need an (optional) accumulator to process each chunk iteration.
            try:
                response_accumulator_factory = self._response_accumulator_factories.get(cast_to)
                response_accumulator = (
                    response_accumulator_factory(request_parameters)
                    if response_accumulator_factory
                    else None
                )
            except Exception:
                # Note that cast_to may not be hashable.
                logger.exception(f"Failed to get response accumulator for {cast_to}")
                response_accumulator = None
            if hasattr(response, "_parsed") and self._is_streaming(parsed := response._parsed):
                # Monkey-patch a private attribute assumed to be caching the output of `.parse()`.
                response._parsed = _Stream(
                    stream=parsed,
                    with_span=with_span,
                    response_accumulator=response_accumulator,
                )
                return response
            if (
                hasattr(response, "_parsed_by_type")
                and isinstance(response._parsed_by_type, dict)
                and self._is_streaming(parsed := response._parsed_by_type.get(cast_to))
            ):
                # New in openai v1.8.0. Streaming with .with_raw_response now returns
                # LegacyAPIResponse and caching is done differently.
                # See https://github.com/openai/openai-python/blob/d231d1fa783967c1d3a1db3ba1b52647fff148ac/src/openai/_legacy_response.py#L112-L113  # noqa: E501
                response._parsed_by_type[cast_to] = _Stream(
                    stream=parsed,
                    with_span=with_span,
                    response_accumulator=response_accumulator,
                )
                return response
            return _Stream(
                stream=response,
                with_span=with_span,
                response_accumulator=response_accumulator,
            )
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


class _Request(_WithTracer, _WithOpenAI):
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
            cast_to, request_parameters = _parse_request_args(args)
            # E.g. cast_to = openai.types.chat.ChatCompletion => span_name = "ChatCompletion"
            span_name: str = cast_to.__name__.split(".")[-1]
        except Exception:
            logger.exception("Failed to parse request args")
            return wrapped(*args, **kwargs)
        with self._start_as_current_span(
            span_name=span_name,
            attributes=chain(
                self._get_attributes_from_instance(instance),
                self._get_attributes_from_request(
                    cast_to=cast_to,
                    request_parameters=request_parameters,
                ),
            ),
            context_attributes=get_attributes_from_context(),
            extra_attributes=self._get_extra_attributes_from_request(
                cast_to=cast_to,
                request_parameters=request_parameters,
            ),
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
                    cast_to=cast_to,
                    request_parameters=request_parameters,
                )
            except Exception:
                logger.exception(f"Failed to finalize response of type {type(response)}")
                with_span.finish_tracing()
        return response


class _AsyncRequest(_WithTracer, _WithOpenAI):
    async def __call__(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)
        try:
            cast_to, request_parameters = _parse_request_args(args)
            # E.g. cast_to = openai.types.chat.ChatCompletion => span_name = "ChatCompletion"
            span_name: str = cast_to.__name__.split(".")[-1]
        except Exception:
            logger.exception("Failed to parse request args")
            return await wrapped(*args, **kwargs)
        with self._start_as_current_span(
            span_name=span_name,
            attributes=chain(
                self._get_attributes_from_instance(instance),
                self._get_attributes_from_request(
                    cast_to=cast_to,
                    request_parameters=request_parameters,
                ),
            ),
            context_attributes=get_attributes_from_context(),
            extra_attributes=self._get_extra_attributes_from_request(
                cast_to=cast_to,
                request_parameters=request_parameters,
            ),
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
                    cast_to=cast_to,
                    request_parameters=request_parameters,
                )
            except Exception:
                logger.exception(f"Failed to finalize response of type {type(response)}")
                with_span.finish_tracing()
        return response


def _parse_request_args(args: Tuple[type, Any]) -> Tuple[type, Mapping[str, Any]]:
    # We don't use `signature(request).bind()` because `request` could have been monkey-patched
    # (incorrectly) by others and the signature at runtime may not match the original.
    # The targeted signature of `request` is here:
    # https://github.com/openai/openai-python/blob/f1c7d714914e3321ca2e72839fe2d132a8646e7f/src/openai/_base_client.py#L846-L847  # noqa: E501
    cast_to: type = args[0]
    request_parameters: Mapping[str, Any] = (
        json_data
        # See https://github.com/openai/openai-python/blob/f1c7d714914e3321ca2e72839fe2d132a8646e7f/src/openai/_models.py#L427  # noqa: E501
        if hasattr(args[1], "json_data") and isinstance(json_data := args[1].json_data, Mapping)
        else {}
    )
    # FIXME: Because request parameters is just a Mapping, it can contain any value as long as it
    # serializes correctly in an HTTP request body. For example, Enum values may be present if a
    # third-party library puts them there. Enums can turn into their intended string values via
    # `json.dumps` when the final HTTP request body is serialized, but can pose problems when we
    # try to extract attributes. However, this round-trip seems expensive, so we opted to treat
    # only the Enums that we know about: e.g. message role sometimes can be an Enum, so we will
    # convert it only when it's encountered.
    # try:
    #     request_parameters = json.loads(json.dumps(request_parameters))
    # except Exception:
    #     pass
    return cast_to, request_parameters


class _ResponseAttributes:
    __slots__ = ("_response", "_request_parameters", "_response_attributes_extractor")

    def __init__(
        self,
        response: Any,
        request_parameters: Mapping[str, Any],
        response_attributes_extractor: _ResponseAttributesExtractor,
    ) -> None:
        if hasattr(response, "parse") and callable(response.parse):
            # E.g. see https://github.com/openai/openai-python/blob/f1c7d714914e3321ca2e72839fe2d132a8646e7f/src/openai/_base_client.py#L518  # noqa: E501
            try:
                response = response.parse()
            except Exception:
                logger.exception(f"Failed to parse response of type {type(response)}")
        self._request_parameters = request_parameters
        self._response = response
        self._response_attributes_extractor = response_attributes_extractor

    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        yield from _as_output_attributes(
            _io_value_and_type(self._response),
        )

    def get_extra_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        yield from self._response_attributes_extractor.get_attributes_from_response(
            response=self._response,
            request_parameters=self._request_parameters,
        )
