import io
import json
from functools import wraps
from importlib import import_module
from inspect import signature
from typing import Any, Callable, Collection, Dict, Optional, Tuple, TypeVar

from botocore.response import StreamingBody
from openinference.semconv.trace import MessageAttributes, SpanAttributes
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
)
from opentelemetry.trace import Tracer
from opentelemetry.util.types import AttributeValue
from wrapt import wrap_function_wrapper

CallableType = TypeVar("CallableType", bound=Callable[..., Any])

_MODULE = "botocore.client"
__version__ = "0.1.0"
_instruments = ("boto3 >= 1.28.57",)


def _set_span_attribute(span: trace_api.Span, name: str, value: AttributeValue) -> None:
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


class BufferedStreamingBody(StreamingBody):
    def __init__(self, raw_stream: io.IOBase, content_length: int) -> None:
        super().__init__(raw_stream, content_length)
        self._buffer: Optional[io.IOBase] = None

    def read(self, amt: Optional[int] = None) -> bytes:
        if self._buffer is None:
            self._buffer = io.BytesIO(self._raw_stream.read())

        return self._buffer.read(amt)

    def reset(self) -> None:
        # Reset the buffer to enable reading the stream again
        if self._buffer is not None:
            self._buffer.seek(0)


def _client_creation_wrapper(tracer: Tracer) -> Callable[[CallableType], CallableType]:
    def _client_wrapper(
        wrapped: CallableType,
        instance: Optional[Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> CallableType:
        @wraps(wrapped)
        def create_instrumented_client(*args: Any, **kwargs: Any) -> Any:
            """Instruments and calls every function defined in TO_WRAP."""
            client = wrapped(*args, **kwargs)
            if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
                return client

            call_signature = signature(wrapped)
            bound_arguments = call_signature.bind(*args, **kwargs)
            bound_arguments.apply_defaults()

            if bound_arguments.arguments.get("service_name") == "bedrock-runtime":
                client._unwrapped_invoke_model = client.invoke_model
                client.invoke_model = _model_invocation_wrapper(tracer)(client)
                return client
            return client

        return create_instrumented_client(*args, **kwargs)

    return _client_wrapper  # type: ignore


def _model_invocation_wrapper(tracer: Tracer) -> Callable[[Any], CallableType]:
    def _invocation_wrapper(wrapped_client: Any) -> CallableType:
        """Instruments a bedrock client's `invoke_model` method."""

        @wraps(wrapped_client)
        def instrumented_response(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            with tracer.start_as_current_span("bedrock.invoke_model") as span:
                response = wrapped_client._unwrapped_invoke_model(*args, **kwargs)
                response["body"] = BufferedStreamingBody(
                    response["body"]._raw_stream, response["body"]._content_length
                )
                if raw_request_body := kwargs.get("body"):
                    request_body = json.loads(raw_request_body)
                response_body = json.loads(response.get("body").read())
                response["body"].reset()

                prompt = request_body.pop("prompt")
                invocation_parameters = request_body

                _set_span_attribute(span, SpanAttributes.LLM_PROMPTS, prompt)
                _set_span_attribute(
                    span, SpanAttributes.LLM_INVOCATION_PARAMETERS, invocation_parameters
                )

                if model_id := kwargs.get("modelId"):
                    _set_span_attribute(span, SpanAttributes.LLM_MODEL_NAME, model_id)

                    if isinstance(model_id, str):
                        (vendor, _) = model_id.split(".")

                    if vendor == "ai21":
                        content = str(response_body.get("completions"))
                        _set_span_attribute(
                            span,
                            MessageAttributes.MESSAGE_CONTENT,
                            content,
                        )
                    elif vendor == "anthropic":
                        content = str(response_body.get("completion"))
                        _set_span_attribute(span, MessageAttributes.MESSAGE_CONTENT, content)
                    elif vendor == "cohere":
                        content = str(response_body.get("generations"))
                        _set_span_attribute(
                            span,
                            MessageAttributes.MESSAGE_CONTENT,
                            content,
                        )

                return response

        return instrumented_response  # type: ignore

    return _invocation_wrapper


class BotoInstrumentor(BaseInstrumentor):
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
