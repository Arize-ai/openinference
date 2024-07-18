import io
import json
from functools import wraps
from importlib import import_module
from inspect import signature
from typing import IO, Any, Callable, Collection, Dict, Optional, Tuple, TypeVar, cast

from botocore.client import BaseClient
from botocore.response import StreamingBody
from openinference.instrumentation import get_attributes_from_context, safe_json_dumps
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
_BASE_MODULE = "botocore"
_MINIMUM_CONVERSE_BOTOCORE_VERSION = "1.34.116"


class InstrumentedClient(BaseClient):  # type: ignore
    """
    Proxy class representing an instrumented boto client.
    """

    invoke_model: Callable[..., Any]
    _unwrapped_invoke_model: Callable[..., Any]

    converse: Callable[..., Any]
    _unwrapped_converse: Callable[..., Any]


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


def _client_creation_wrapper(
    tracer: Tracer, module_version: str
) -> Callable[[ClientCreator], ClientCreator]:
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

            if module_version >= _MINIMUM_CONVERSE_BOTOCORE_VERSION:
                client._unwrapped_converse = client.converse
                client.converse = _model_converse_wrapper(tracer)(client)
        return client

    return _client_wrapper  # type: ignore


def _model_invocation_wrapper(tracer: Tracer) -> Callable[[InstrumentedClient], Callable[..., Any]]:
    def _invocation_wrapper(wrapped_client: InstrumentedClient) -> Callable[..., Any]:
        """Instruments a bedrock client's `invoke_model` or `converse` method."""

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
                invocation_parameters = safe_json_dumps(request_body)
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
                    elif vendor == "meta":
                        content = str(response_body.get("generation"))
                    else:
                        content = ""

                    if content:
                        _set_span_attribute(span, SpanAttributes.OUTPUT_VALUE, content)

                span.set_attributes(dict(get_attributes_from_context()))
                return response  # type: ignore

        return instrumented_response

    return _invocation_wrapper


def _model_converse_wrapper(tracer: Tracer) -> Callable[[InstrumentedClient], Callable[..., Any]]:
    def _converse_wrapper(wrapped_client: InstrumentedClient) -> Callable[..., Any]:
        """Instruments a bedrock client's `converse` method."""

        @wraps(wrapped_client.converse)
        def instrumented_response(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
                return wrapped_client._unwrapped_converse(*args, **kwargs)  # type: ignore

            with tracer.start_as_current_span("bedrock.converse") as span:
                span.set_attribute(
                    SpanAttributes.OPENINFERENCE_SPAN_KIND,
                    OpenInferenceSpanKindValues.LLM.value,
                )

                if model_id := kwargs.get("modelId"):
                    _set_span_attribute(span, SpanAttributes.LLM_MODEL_NAME, model_id)

                if inference_config := kwargs.get("inferenceConfig"):
                    invocation_parameters = safe_json_dumps(inference_config)
                    _set_span_attribute(
                        span, SpanAttributes.LLM_INVOCATION_PARAMETERS, invocation_parameters
                    )

                aggregated_messages = []
                if system_prompts := kwargs.get("system"):
                    aggregated_messages.append(
                        {
                            "role": "system",
                            "content": [
                                {
                                    "text": " ".join(
                                        prompt.get("text", "") for prompt in system_prompts
                                    )
                                }
                            ],
                        }
                    )

                aggregated_messages.extend(kwargs.get("messages", []))
                for idx, request_msg in enumerate(aggregated_messages):
                    # Currently only supports text-based data
                    # Future implementation can be extended to handle different media types
                    if (
                        isinstance(request_msg, dict)
                        and (request_msg_role := str(request_msg.get("role")))
                        and (request_msg_content := request_msg.get("content"))
                    ):
                        request_msg_prompt = "\n".join(
                            content_input.get("text", "")  # type: ignore
                            for content_input in request_msg_content
                        )
                        if idx == len(aggregated_messages) - 1:
                            _set_span_attribute(
                                span, SpanAttributes.INPUT_VALUE, request_msg_prompt
                            )

                        span_prefix = f"{SpanAttributes.LLM_INPUT_MESSAGES}.{idx}"
                        _set_span_attribute(span, f"{span_prefix}.message.role", request_msg_role)
                        _set_span_attribute(
                            span, f"{span_prefix}.message.content", request_msg_prompt
                        )

                response = wrapped_client._unwrapped_converse(*args, **kwargs)
                if (
                    (response_message := response.get("output", {}).get("message"))
                    and (response_role := response_message.get("role"))
                    and (response_content := response_message.get("content", []))
                ):
                    # Currently only supports text-based data
                    response_text = "\n".join(
                        content_input.get("text", "") for content_input in response_content
                    )
                    _set_span_attribute(span, SpanAttributes.OUTPUT_VALUE, response_text)

                    span_prefix = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0"
                    _set_span_attribute(span, f"{span_prefix}.message.role", response_role)
                    _set_span_attribute(span, f"{span_prefix}.message.content", response_text)

                if usage := response.get("usage"):
                    if input_token_count := usage.get("inputTokens"):
                        _set_span_attribute(
                            span, SpanAttributes.LLM_TOKEN_COUNT_PROMPT, input_token_count
                        )
                    if response_token_count := usage.get("outputTokens"):
                        _set_span_attribute(
                            span,
                            SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
                            response_token_count,
                        )
                    if total_token_count := usage.get("totalTokens"):
                        _set_span_attribute(
                            span, SpanAttributes.LLM_TOKEN_COUNT_TOTAL, total_token_count
                        )

                span.set_attributes(dict(get_attributes_from_context()))
            return response  # type: ignore

        return instrumented_response

    return _converse_wrapper


class BedrockInstrumentor(BaseInstrumentor):  # type: ignore
    __slots__ = ("_original_client_creator",)

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        tracer = trace_api.get_tracer(__name__, __version__, tracer_provider)

        boto = import_module(_MODULE)
        botocore = import_module(_BASE_MODULE)
        self._original_client_creator = boto.ClientCreator.create_client

        wrap_function_wrapper(
            module=_MODULE,
            name="ClientCreator.create_client",
            wrapper=_client_creation_wrapper(tracer=tracer, module_version=botocore.__version__),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        boto = import_module(_MODULE)
        boto.ClientCreator.create_client = self._original_client_creator
        self._original_client_creator = None


def _set_span_attribute(span: trace_api.Span, name: str, value: AttributeValue) -> None:
    if value is not None and value != "":
        span.set_attribute(name, value)
