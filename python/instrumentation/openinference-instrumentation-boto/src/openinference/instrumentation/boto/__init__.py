import io
import json
from functools import wraps
from inspect import signature
from typing import Any, Collection

from botocore.response import StreamingBody
from openinference.semconv.trace import MessageAttributes, SpanAttributes
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
)
from opentelemetry.trace import SpanKind
from wrapt import wrap_function_wrapper

_MODULE = "botocore.client"
__version__ = "0.1.0"
_instruments = ("boto3 >= 1.28.57",)


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


class BufferedStreamingBody(StreamingBody):
    def __init__(self, raw_stream, content_length):
        super().__init__(raw_stream, content_length)
        self._buffer = None

    def read(self, amt=None):
        if self._buffer is None:
            self._buffer = io.BytesIO(self._raw_stream.read())

        return self._buffer.read(amt)

    def reset(self):
        # Reset the buffer to enable reading the stream again
        if self._buffer is not None:
            self._buffer.seek(0)


def _client_creation_wrapper(tracer):
    def _client_wrapper(wrapped, instance, args, kwargs):
        @wraps(wrapped)
        def create_instrumented_client(*args, **kwargs):
            """Instruments and calls every function defined in TO_WRAP."""
            client = wrapped(*args, **kwargs)
            if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
                return client

            call_signature = signature(wrapped)
            bound_arguments = call_signature.bind(*args, **kwargs)
            bound_arguments.apply_defaults()

            if bound_arguments.arguments.get("service_name") == "bedrock-runtime":
                client.invoke_model = _model_invocation_wrapper(tracer)(client.invoke_model)
                return client
            return client

        return create_instrumented_client(*args, **kwargs)

    return _client_wrapper


def _model_invocation_wrapper(tracer):
    def _invocation_wrapper(wrapped):
        """Instruments a bedrock client's `invoke_model` method."""

        @wraps(wrapped)
        def instrumented_response(*args, **kwargs):
            with tracer.start_as_current_span("bedrock.completion", kind=SpanKind.CLIENT) as span:
                response = wrapped(*args, **kwargs)
                response["body"] = BufferedStreamingBody(
                    response["body"]._raw_stream, response["body"]._content_length
                )
                request_body = json.loads(kwargs.get("body"))
                response_body = json.loads(response.get("body").read())
                response["body"].reset()

                prompt = request_body.pop("prompt")
                invocation_parameters = request_body

                _set_span_attribute(span, SpanAttributes.LLM_PROMPTS, prompt)
                _set_span_attribute(
                    span, SpanAttributes.LLM_INVOCATION_PARAMETERS, invocation_parameters
                )

                model_id = kwargs.get("modelId")
                (vendor, model) = model_id.split(".")
                _set_span_attribute(span, SpanAttributes.LLM_MODEL_NAME, model_id)

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

        return instrumented_response

    return _invocation_wrapper


class BotoInstrumentor(BaseInstrumentor):
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        tracer = trace_api.get_tracer(__name__, __version__, tracer_provider)

        wrap_function_wrapper(
            module=_MODULE,
            name="ClientCreator.create_client",
            wrapper=_client_creation_wrapper(tracer=tracer),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        pass
