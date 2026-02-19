"""
OpenInference instrumentation for AWS Bedrock (boto3 and aioboto3).

This module patches botocore's ClientCreator and aiobotocore's AioClientCreator so that
Bedrock clients (bedrock-runtime and bedrock-agent-runtime) are instrumented with
OpenTelemetry spans. Sync and async clients are handled separately: sync uses
_model_invocation_wrapper and BufferedStreamingBody for invoke_model; async uses
_async_model_invocation_wrapper and _LazyAsyncInvokeModelBody so that the response
body is read only when the caller awaits read(), and span attributes are set at that time.

Streaming APIs (invoke_model_with_response_stream, converse_stream, invoke_agent, etc.)
use wrappers that start a span and wrap the event stream so the span is ended when
the stream is fully consumed.

Request/response body types (from botocore/aiobotocore service shapes): InvokeModel
request body is the API ``body`` payload (shape blob), so kwargs["body"] may be str or
bytes; json.loads() accepts both in Python 3.6+. Response body is botocore's
StreamingBody (sync read) or aiobotocore's StreamingBody (async read) respectively.

Edge cases and limitations:
- Suppression: All wrappers check _SUPPRESS_INSTRUMENTATION_KEY first and call through
  unwrapped without creating spans.
- Errors: API or parse failures set span status to ERROR, record_exception when we
  handle the exception, and end the span. Sync invoke_model uses a context manager
  that also records and sets status on any uncaught exception.
- Async invoke_model: Span is completed on first read() of the response body, or in
  __del__ if the body is never read (best-effort).
- Streaming: Span ends when the event stream is fully consumed or on first exception;
  partial consumption then discard leaves the span ended by the stream wrapper.
- Double consumption of the same stream is not supported; the underlying stream is
  single-read.
- Missing response body: If InvokeModel returns without a body we raise ValueError
  after setting span status to ERROR.
"""

import base64
import io
import json
import logging
from enum import Enum
from functools import wraps
from importlib import import_module
from inspect import signature
from typing import (
    IO,
    Any,
    Callable,
    Collection,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    cast,
)

from botocore.client import BaseClient
from botocore.response import StreamingBody
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from opentelemetry.trace import Status, StatusCode, Tracer
from opentelemetry.util.types import AttributeValue
from wrapt import wrap_function_wrapper

from openinference.instrumentation import (
    OITracer,
    TraceConfig,
    get_attributes_from_context,
    safe_json_dumps,
)
from openinference.instrumentation.bedrock._rag_wrappers import (
    _retrieve_and_generate_wrapper,
    _retrieve_wrapper,
)
from openinference.instrumentation.bedrock._wrappers import (
    _ConverseStream,
    _InvokeAgentWithResponseStream,
    _InvokeModelWithResponseStream,
    _RetrieveAndGenerateStream,
)
from openinference.instrumentation.bedrock.package import _instruments
from openinference.instrumentation.bedrock.utils import _extract_invoke_model_attributes
from openinference.instrumentation.bedrock.utils.anthropic import (
    _attributes as anthropic_attributes,
)
from openinference.instrumentation.bedrock.version import __version__
from openinference.semconv.trace import (
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

ClientCreator = TypeVar("ClientCreator", bound=Callable[..., BaseClient])

_MODULE = "botocore.client"
_AIO_MODULE = "aiobotocore.client"
_BASE_MODULE = "botocore"
_AIO_BASE_MODULE = "aiobotocore"
# Converse / ConverseStream require this botocore version or newer.
_MINIMUM_CONVERSE_BOTOCORE_VERSION = "1.34.116"

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# -----------------------------------------------------------------------------
# Instrumented client and response body helpers
# -----------------------------------------------------------------------------


class InstrumentedClient(BaseClient):  # type: ignore
    """
    Type stub for an instrumented Bedrock client.

    After _instrument_client() runs, the client's Bedrock API methods are replaced
    with wrapped versions that create spans and set OpenInference attributes. The
    original methods are stored as _unwrapped_* so the wrappers can call through.
    """

    invoke_model: Callable[..., Any]
    _unwrapped_invoke_model: Callable[..., Any]

    converse: Callable[..., Any]
    _unwrapped_converse: Callable[..., Any]

    invoke_agent: Callable[..., Any]
    _unwrapped_invoke_agent: Callable[..., Any]

    invoke_inline_agent: Callable[..., Any]
    _unwrapped_invoke_inline_agent: Callable[..., Any]

    retrieve: Callable[..., Any]
    _unwrapped_retrieve: Callable[..., Any]

    retrieve_and_generate: Callable[..., Any]
    _unwrapped_retrieve_and_generate: Callable[..., Any]

    retrieve_and_generate_stream: Callable[..., Any]
    _unwrapped_retrieve_and_generate_stream: Callable[..., Any]


class BufferedStreamingBody(StreamingBody):  # type: ignore
    """
    Sync streaming body that buffers the full response on first read(), so we can
    parse it for span attributes and then reset() to allow the caller to read again.

    Used by the sync invoke_model wrapper: read once for attribution, reset(), then
    return the same body to the caller. Expects botocore's StreamingBody interface
    (_raw_stream, _content_length).
    """

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
        """Rewind the buffer so the body can be read again by the caller."""
        if self._buffer is not None:
            self._buffer.seek(0)


class _LazyAsyncInvokeModelBody:
    """
    Async response body that defers reading and span completion until first read().

    real_stream is aiobotocore's StreamingBody (async read()). On first read() we
    await it, parse JSON for span attributes, end the span, and cache the bytes.

    - On first read(): reads the full body from the real stream, parses JSON, sets
      response attributes on the span, ends the span, and returns (optionally
      sliced) bytes. Result is cached.
    - On later read(): returns from cache so the caller can read in chunks if desired.
    - If the body is never read (e.g. response discarded), __del__ ends the span
      as a best-effort fallback so we do not leave spans open indefinitely.
    """

    def __init__(
        self,
        real_stream: Any,
        span: trace_api.Span,
        kwargs: Dict[str, Any],
        response: Dict[str, Any],
        is_claude_message_api: bool,
    ) -> None:
        self._real_stream = real_stream
        self._span = span
        self._kwargs = kwargs
        self._response = response
        self._is_claude_message_api = is_claude_message_api
        self._cached: Optional[bytes] = None
        self._span_ended = False

    async def read(self, amt: Optional[int] = None) -> bytes:
        # Already read and attributed: serve from cache.
        if self._cached is not None:
            out: bytes = self._cached if amt is None else self._cached[:amt]
            return out
        # First read: pull full body, set span attributes, end span, then return.
        body_bytes = cast(bytes, await self._real_stream.read())
        self._cached = body_bytes
        try:
            # json.loads() accepts bytes in Python 3.6+ (decodes as UTF-8).
            response_body = json.loads(body_bytes)
            if self._is_claude_message_api:
                anthropic_attributes.set_response_attributes(self._span, response_body)
            else:
                _extract_invoke_model_attributes.set_response_attributes(
                    self._span, self._kwargs, response_body, self._response
                )
            self._span.set_attributes(dict(get_attributes_from_context()))
            self._span.set_status(Status(StatusCode.OK))
        except Exception as e:
            self._span.record_exception(e)
            self._span.set_status(Status(StatusCode.ERROR))
            raise
        finally:
            if not self._span_ended:
                self._span.end()
                self._span_ended = True
        return body_bytes if amt is None else body_bytes[:amt]

    def __del__(self) -> None:
        """Best-effort: end the span if the body was never read."""
        if not self._span_ended:
            try:
                self._span.end()
            except Exception:
                pass


# -----------------------------------------------------------------------------
# Client instrumentation
# -----------------------------------------------------------------------------


def _instrument_client(
    client: Any, bound_arguments: Any, tracer: Tracer, module_version: str, is_async: bool
) -> BaseClient:
    """
    Attach tracing wrappers to a Bedrock client's API methods.

    Dispatches by service_name (bedrock-agent-runtime vs bedrock-runtime) and
    for bedrock-runtime uses is_async to choose sync vs async invoke_model wrapper.
    """
    # --- bedrock-agent-runtime: agents and RAG ---
    if bound_arguments.arguments.get("service_name") == "bedrock-agent-runtime":
        client = cast(InstrumentedClient, client)

        client._unwrapped_invoke_agent = client.invoke_agent
        client.invoke_agent = _InvokeAgentWithResponseStream(tracer)(client.invoke_agent)

        client._unwrapped_invoke_inline_agent = client.invoke_inline_agent
        client.invoke_inline_agent = _InvokeAgentWithResponseStream(tracer)(
            client.invoke_inline_agent
        )

        client._unwrapped_retrieve = client.retrieve
        client.retrieve = _retrieve_wrapper(tracer)(client)

        client._unwrapped_retrieve_and_generate = client.retrieve_and_generate
        client.retrieve_and_generate = _retrieve_and_generate_wrapper(tracer)(client)

        client._unwrapped_retrieve_and_generate_stream = client.retrieve_and_generate_stream
        client.retrieve_and_generate_stream = _RetrieveAndGenerateStream(tracer)(
            client.retrieve_and_generate_stream
        )

    # --- bedrock-runtime: invoke_model, streaming, converse ---
    if bound_arguments.arguments.get("service_name") == "bedrock-runtime":
        client = cast(InstrumentedClient, client)

        # Non-streaming invoke_model: sync uses botocore StreamingBody (sync read),
        # async uses aiobotocore StreamingBody (async read). We choose wrapper accordingly.
        client._unwrapped_invoke_model = client.invoke_model
        if is_async:
            client.invoke_model = _async_model_invocation_wrapper(tracer)(client)
        else:
            client.invoke_model = _model_invocation_wrapper(tracer)(client)

        # Streaming invoke_model: same wrapper handles sync and async (see _wrappers).
        client._unwrapped_invoke_model_with_response_stream = (
            client.invoke_model_with_response_stream
        )
        client.invoke_model_with_response_stream = _InvokeModelWithResponseStream(tracer)(
            client.invoke_model_with_response_stream
        )

        # Converse / ConverseStream only if botocore is new enough.
        if module_version >= _MINIMUM_CONVERSE_BOTOCORE_VERSION:
            client._unwrapped_converse = client.converse
            client.converse = _model_converse_wrapper(tracer)(client)
            client._unwrapped_converse_stream = client.converse_stream
            client.converse_stream = _ConverseStream(tracer)(client.converse_stream)

    return client


def _sync_client_creation_wrapper(tracer: Tracer, module_version: str) -> Any:
    """Wraps botocore ClientCreator.create_client so new sync clients are instrumented."""

    def _client_wrapper(
        wrapped: ClientCreator,
        instance: Optional[Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> BaseClient:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        call_signature = signature(wrapped)
        bound_arguments = call_signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        client = wrapped(*args, **kwargs)
        return _instrument_client(client, bound_arguments, tracer, module_version, is_async=False)

    return _client_wrapper


def _async_client_creation_wrapper(tracer: Tracer, module_version: str) -> Any:
    """
    Wraps aiobotocore AioClientCreator.create_client so new async clients are instrumented.

    Returns a coroutine; the session's _create_client awaits it and receives the
    instrumented client (same pattern as unwrapped create_client).
    """

    def _client_wrapper(
        wrapped: ClientCreator,
        instance: Optional[Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        call_signature = signature(wrapped)
        bound_arguments = call_signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()

        async def async_wrapper() -> BaseClient:
            async_client = await wrapped(*args, **kwargs)
            return _instrument_client(
                async_client, bound_arguments, tracer, module_version, is_async=True
            )

        return async_wrapper()

    return _client_wrapper


# -----------------------------------------------------------------------------
# invoke_model wrappers (non-streaming)
# -----------------------------------------------------------------------------


def _model_invocation_wrapper(tracer: Tracer) -> Callable[[InstrumentedClient], Callable[..., Any]]:
    """
    Wraps sync invoke_model: one span per call; body is buffered, read once for
    attribution, then reset so the caller can read it again.
    """

    def _invocation_wrapper(wrapped_client: InstrumentedClient) -> Callable[..., Any]:
        @wraps(wrapped_client.invoke_model)
        def instrumented_response(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
                return wrapped_client._unwrapped_invoke_model(*args, **kwargs)  # type: ignore

            with tracer.start_as_current_span("bedrock.invoke_model") as span:
                # kwargs["body"] is InvokeModelRequest payload (blob: str or bytes).
                request_body = json.loads(kwargs["body"])
                model_id = str(kwargs.get("modelId"))
                is_claude_message_api = _extract_invoke_model_attributes.is_claude_message_api(
                    model_id
                )
                if is_claude_message_api:
                    anthropic_attributes.set_input_attributes(span, request_body, model_id)
                else:
                    _extract_invoke_model_attributes.set_input_attributes(span, request_body)

                try:
                    response = wrapped_client._unwrapped_invoke_model(*args, **kwargs)
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR))
                    span.end()
                    raise e

                body = response.get("body")
                if body is None:
                    span.set_status(Status(StatusCode.ERROR))
                    span.end()
                    raise ValueError("InvokeModel response missing 'body'")
                # Botocore response body has _raw_stream and _content_length.
                response["body"] = BufferedStreamingBody(body._raw_stream, body._content_length)
                response_body = json.loads(response.get("body").read())
                response["body"].reset()

                if is_claude_message_api:
                    anthropic_attributes.set_response_attributes(span, response_body)
                else:
                    _extract_invoke_model_attributes.set_response_attributes(
                        span, kwargs, response_body, response
                    )
                span.set_attributes(dict(get_attributes_from_context()))
                span.set_status(Status(StatusCode.OK))
                return response  # type: ignore

        return instrumented_response

    return _invocation_wrapper


def _async_model_invocation_wrapper(
    tracer: Tracer,
) -> Callable[[InstrumentedClient], Callable[..., Any]]:
    """
    Wraps async invoke_model: span is left open (end_on_exit=False) and the
    response body is replaced with _LazyAsyncInvokeModelBody so the span is
    completed on first read() of the body.
    """

    def _invocation_wrapper(wrapped_client: InstrumentedClient) -> Callable[..., Any]:
        @wraps(wrapped_client.invoke_model)
        async def instrumented_response(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
                return await wrapped_client._unwrapped_invoke_model(*args, **kwargs)  # type: ignore

            with tracer.start_as_current_span(
                "bedrock.invoke_model",
                end_on_exit=False,
            ) as span:
                # kwargs["body"] is InvokeModelRequest payload (blob: str or bytes).
                request_body = json.loads(kwargs["body"])
                model_id = str(kwargs.get("modelId"))
                is_claude_message_api = _extract_invoke_model_attributes.is_claude_message_api(
                    model_id
                )
                if is_claude_message_api:
                    anthropic_attributes.set_input_attributes(span, request_body, model_id)
                else:
                    _extract_invoke_model_attributes.set_input_attributes(span, request_body)
                try:
                    response = await wrapped_client._unwrapped_invoke_model(*args, **kwargs)
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR))
                    span.end()
                    raise e
                body = response.get("body")
                if body is None:
                    span.set_status(Status(StatusCode.ERROR))
                    span.end()
                    raise ValueError("InvokeModel response missing 'body'")
                # Lazy body: on first read() we consume the stream, set response attrs, end span.
                response["body"] = _LazyAsyncInvokeModelBody(
                    body, span, kwargs, response, is_claude_message_api
                )
                return response  # type: ignore

        return instrumented_response

    return _invocation_wrapper


# -----------------------------------------------------------------------------
# converse wrapper
# -----------------------------------------------------------------------------


def _model_converse_wrapper(tracer: Tracer) -> Callable[[InstrumentedClient], Callable[..., Any]]:
    """
    Wraps bedrock-runtime converse: one span per call; sets input/output and
    token usage. Chooses sync or async implementation based on client type.
    """

    def _converse_wrapper(wrapped_client: InstrumentedClient) -> Callable[..., Any]:
        def _process_converse(
            span: trace_api.Span,
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
            response: Dict[str, Any],
        ) -> Dict[str, Any]:
            """Set output message, role, content and token usage on span from response."""
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
            return response

        def _set_input_attributes(span: trace_api.Span, kwargs: Dict[str, Any]) -> None:
            """Set input attributes on the span."""
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

            aggregated_messages: List[Any] = []
            if system_prompts := kwargs.get("system"):
                aggregated_messages.append(
                    {
                        "role": "system",
                        "content": [
                            {"text": " ".join(prompt.get("text", "") for prompt in system_prompts)}
                        ],
                    }
                )

            aggregated_messages.extend(kwargs.get("messages", []))
            for idx, msg in enumerate(aggregated_messages):
                if not isinstance(msg, dict):
                    # Only dictionaries supported for now
                    continue
                for key, value in _get_attributes_from_message_param(msg):
                    _set_span_attribute(
                        span,
                        f"{SpanAttributes.LLM_INPUT_MESSAGES}.{idx}.{key}",
                        value,
                    )
            last_message = aggregated_messages[-1] if aggregated_messages else None
            if isinstance(last_message, dict) and (
                request_msg_content := last_message.get("content")
            ):
                request_msg_prompt = "\n".join(
                    content_input.get("text", "") for content_input in request_msg_content
                ).strip("\n")
                _set_span_attribute(span, SpanAttributes.INPUT_VALUE, request_msg_prompt)

        # Use async wrapper when client is async (aiobotocore).
        if hasattr(wrapped_client, "__aenter__"):

            @wraps(wrapped_client.converse)
            async def async_instrumented_response(*args: Any, **kwargs: Any) -> Dict[str, Any]:
                if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
                    return await wrapped_client._unwrapped_converse(*args, **kwargs)  # type: ignore

                with tracer.start_as_current_span("bedrock.converse") as span:
                    _set_input_attributes(span, kwargs)
                    response = await wrapped_client._unwrapped_converse(*args, **kwargs)
                    return _process_converse(span, args, kwargs, response)

            return async_instrumented_response
        else:

            @wraps(wrapped_client.converse)
            def sync_instrumented_response(*args: Any, **kwargs: Any) -> Dict[str, Any]:
                if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
                    return wrapped_client._unwrapped_converse(*args, **kwargs)  # type: ignore

                with tracer.start_as_current_span("bedrock.converse") as span:
                    _set_input_attributes(span, kwargs)
                    response = wrapped_client._unwrapped_converse(*args, **kwargs)
                    return _process_converse(span, args, kwargs, response)

            return sync_instrumented_response

    return _converse_wrapper


# -----------------------------------------------------------------------------
# Instrumentor
# -----------------------------------------------------------------------------


class BedrockInstrumentor(BaseInstrumentor):  # type: ignore
    """
    OpenTelemetry instrumentor for AWS Bedrock (boto3 and aiobotocore).

    Patches ClientCreator.create_client (sync) and AioClientCreator.create_client
    (async) so that newly created clients have their Bedrock API methods wrapped.
    Sync and async clients are instrumented independently; aiobotocore is optional.
    """

    __slots__ = (
        "_tracer",
        "_original_client_creator",
        "_original_aio_client_creator",
    )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        if not (config := kwargs.get("config")):
            config = TraceConfig()
        else:
            assert isinstance(config, TraceConfig)
        self._tracer = OITracer(
            trace_api.get_tracer(__name__, __version__, tracer_provider),
            config=config,
        )

        # Patch sync client creation (boto3).
        boto = import_module(_MODULE)
        botocore = import_module(_BASE_MODULE)
        self._original_client_creator = boto.ClientCreator.create_client
        wrap_function_wrapper(
            module=_MODULE,
            name="ClientCreator.create_client",
            wrapper=_sync_client_creation_wrapper(
                tracer=self._tracer,
                module_version=botocore.__version__,
            ),
        )
        # Patch async client creation (aiobotocore) if available.
        try:
            aioboto = import_module(_AIO_MODULE)
            aiobotocore = import_module(_AIO_BASE_MODULE)
            self._original_aio_client_creator = aioboto.AioClientCreator.create_client
            wrap_function_wrapper(
                module=_AIO_MODULE,
                name="AioClientCreator.create_client",
                wrapper=_async_client_creation_wrapper(
                    tracer=self._tracer,
                    module_version=aiobotocore.__version__,
                ),
            )
        except ImportError:
            pass

    def _uninstrument(self, **kwargs: Any) -> None:
        """Restore original create_client implementations."""
        boto = import_module(_MODULE)
        boto.ClientCreator.create_client = self._original_client_creator
        self._original_client_creator = None
        try:
            aioboto = import_module(_AIO_MODULE)
            aioboto.AioClientCreator.create_client = self._original_aio_client_creator
            self._original_aio_client_creator = None
        except ImportError:
            pass


# -----------------------------------------------------------------------------
# Span and message attribute helpers (converse / message APIs)
# -----------------------------------------------------------------------------


def _set_span_attribute(span: trace_api.Span, name: str, value: AttributeValue) -> None:
    """Set a span attribute only if value is non-empty."""
    if value is not None and value != "":
        span.set_attribute(name, value)


def _get_attributes_from_message_param(
    message: Dict[str, Any],
) -> Iterator[Tuple[str, AttributeValue]]:
    """Yield (attr_name, value) for role and content from a converse message dict."""
    if not hasattr(message, "get"):
        return
    if role := message.get("role"):
        yield (
            MessageAttributes.MESSAGE_ROLE,
            role.value if isinstance(role, Enum) else role,
        )

    if content := message.get("content"):
        if isinstance(content, str):
            yield MessageAttributes.MESSAGE_CONTENT, content
        elif is_iterable_of(content, dict):
            for index, c in list(enumerate(content)):
                for key, value in _get_attributes_from_message_content(c):
                    yield f"{MessageAttributes.MESSAGE_CONTENTS}.{index}.{key}", value
        elif isinstance(content, List):
            # See https://github.com/openai/openai-python/blob/f1c7d714914e3321ca2e72839fe2d132a8646e7f/src/openai/types/chat/chat_completion_user_message_param.py#L14  # noqa: E501
            try:
                value = safe_json_dumps(content)
            except Exception:
                logger.exception("Failed to serialize message content")
            yield MessageAttributes.MESSAGE_CONTENT, value


def _get_attributes_from_message_content(
    content: Dict[str, Any],
) -> Iterator[Tuple[str, AttributeValue]]:
    """Yield (attr_name, value) for a single content block (text or image)."""
    content = dict(content)
    if text := content.get("text"):
        yield f"{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", "text"
        yield f"{MessageContentAttributes.MESSAGE_CONTENT_TEXT}", text
    if image := content.get("image"):
        yield f"{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", "image"
        for key, value in _get_attributes_from_image(image):
            yield f"{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}.{key}", value


def _get_attributes_from_image(
    image: Dict[str, Any],
) -> Iterator[Tuple[str, AttributeValue]]:
    """Yield (attr_name, value) for image content (e.g. base64-encoded bytes)."""
    if (source := image.get("source")) and (img_bytes := source.get("bytes")):
        base64_img = base64.b64encode(img_bytes).decode("utf-8")
        yield (
            f"{ImageAttributes.IMAGE_URL}",
            f"data:image/jpeg;base64,{base64_img}",
        )


T = TypeVar("T", bound=type)


def is_iterable_of(lst: Iterable[object], tp: T) -> bool:
    """Return True if lst is an iterable whose elements are all of type tp."""
    return isinstance(lst, Iterable) and all(isinstance(x, tp) for x in lst)
