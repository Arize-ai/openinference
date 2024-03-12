import io
import json
from functools import wraps
from importlib import import_module
from inspect import signature
from typing import IO, Any, Callable, Collection, Dict, Optional, Tuple, TypeVar, cast

from botocore.client import BaseClient
from botocore.response import StreamingBody
from openinference.instrumentation.bedrock.package import _instruments
from openinference.instrumentation.bedrock.version import __version__
from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from opentelemetry.trace import Tracer
from opentelemetry.util.types import AttributeValue
from wrapt import wrap_function_wrapper

ClientCreator = TypeVar("ClientCreator", bound=Callable[..., BaseClient])

_MODULE = "botocore.client"


class InstrumentedClient(BaseClient):  # type: ignore
    """
    Proxy class representing an instrumented boto client.
    """

    invoke_model: Callable[..., Any]
    _unwrapped_invoke_model: Callable[..., Any]


class BufferedStreamingBody(StreamingBody):  # type: ignore
    _raw_stream: IO[bytes]

    def __init__(self, raw_stream: IO[bytes], content_length: int) -> None:
        super().__init__(raw_stream, content_length)
        self._buffer: Optional[io.IOBase] = None

    def read(self, amt: Optional[int] = None) -> bytes:
        if self._buffer is None:
            self._buffer = io.BytesIO(self._raw_stream.read())

        output: bytes = self._buffer.read(amt)
        return output

    def reset(self) -> None:
        # Reset the buffer to enable reading the stream again
        if self._buffer is not None:
            self._buffer.seek(0)


def _client_creation_wrapper(tracer: Tracer) -> Callable[[ClientCreator], ClientCreator]:
    def _client_wrapper(
        wrapped: ClientCreator,
        instance: Optional[Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> BaseClient:
        """Instruments boto client creation."""
        client = wrapped(*args, **kwargs)
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return client

        call_signature = signature(wrapped)
        bound_arguments = call_signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()

        if bound_arguments.arguments.get("service_name") == "bedrock-runtime":
            client = cast(InstrumentedClient, client)
            client._unwrapped_invoke_model = client.invoke_model
            client.invoke_model = _model_invocation_wrapper(tracer)(client)
        return client

    return _client_wrapper  # type: ignore


def _model_invocation_wrapper(tracer: Tracer) -> Callable[[InstrumentedClient], Callable[..., Any]]:
    def _invocation_wrapper(wrapped_client: InstrumentedClient) -> Callable[..., Any]:
        """Instruments a bedrock client's `invoke_model` method."""

        @wraps(wrapped_client.invoke_model)
        def instrumented_response(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
                return wrapped_client._unwrapped_invoke_model(*args, **kwargs)  # type: ignore

            with tracer.start_as_current_span("bedrock.invoke_model") as span:
                span.set_attribute(
                    SpanAttributes.OPENINFERENCE_SPAN_KIND,
                    OpenInferenceSpanKindValues.LLM.value,
                )
                response = wrapped_client._unwrapped_invoke_model(*args, **kwargs)
                response["body"] = BufferedStreamingBody(
                    response["body"]._raw_stream, response["body"]._content_length
                )
                if raw_request_body := kwargs.get("body"):
                    request_body = json.loads(raw_request_body)
                response_body = json.loads(response.get("body").read())
                response["body"].reset()

                prompt = request_body.pop("prompt")
                invocation_parameters = json.dumps(request_body)

                _set_span_attribute(span, SpanAttributes.INPUT_VALUE, prompt)
                _set_span_attribute(
                    span, SpanAttributes.LLM_INVOCATION_PARAMETERS, invocation_parameters
                )

                if metadata := response.get("ResponseMetadata"):
                    if headers := metadata.get("HTTPHeaders"):
                        if input_token_count := headers.get("x-amzn-bedrock-input-token-count"):
                            input_token_count = int(input_token_count)
                            _set_span_attribute(
                                span, SpanAttributes.LLM_TOKEN_COUNT_PROMPT, input_token_count
                            )
                        if response_token_count := headers.get("x-amzn-bedrock-output-token-count"):
                            response_token_count = int(response_token_count)
                            _set_span_attribute(
                                span,
                                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
                                response_token_count,
                            )
                        if total_token_count := (
                            input_token_count + response_token_count
                            if input_token_count and response_token_count
                            else None
                        ):
                            _set_span_attribute(
                                span, SpanAttributes.LLM_TOKEN_COUNT_TOTAL, total_token_count
                            )

                if model_id := kwargs.get("modelId"):
                    _set_span_attribute(span, SpanAttributes.LLM_MODEL_NAME, model_id)

                    if isinstance(model_id, str):
                        (vendor, *_) = model_id.split(".")

                    if vendor == "ai21":
                        content = str(response_body.get("completions"))
                    elif vendor == "anthropic":
                        content = str(response_body.get("completion"))
                    elif vendor == "cohere":
                        content = str(response_body.get("generations"))
                    else:
                        content = ""

                    if content:
                        _set_span_attribute(span, SpanAttributes.OUTPUT_VALUE, content)

                return response  # type: ignore

        return instrumented_response

    return _invocation_wrapper


class BedrockInstrumentor(BaseInstrumentor):  # type: ignore
    __slots__ = ("_original_client_creator",)

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        tracer = trace_api.get_tracer(__name__, __version__, tracer_provider)

        boto = import_module(_MODULE)
        self._original_client_creator = boto.ClientCreator.create_client

        wrap_function_wrapper(
            module=_MODULE,
            name="ClientCreator.create_client",
            wrapper=_client_creation_wrapper(tracer=tracer),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        boto = import_module(_MODULE)
        boto.ClientCreator.create_client = self._original_client_creator
        self._original_client_creator = None


def _set_span_attribute(span: trace_api.Span, name: str, value: AttributeValue) -> None:
    if value is not None and value != "":
        span.set_attribute(name, value)
