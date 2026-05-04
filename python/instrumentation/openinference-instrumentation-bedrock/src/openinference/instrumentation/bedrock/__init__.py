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

import io
import json
import logging
from functools import wraps
from importlib import import_module
from inspect import signature
from typing import (
    IO,
    Any,
    Callable,
    Collection,
    Dict,
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
)
from openinference.instrumentation.bedrock._converse_attributes import (
    get_attributes_from_request_data as _get_converse_request_attributes,
)
from openinference.instrumentation.bedrock._converse_attributes import (
    get_attributes_from_response_data as _get_converse_response_attributes,
)
from openinference.instrumentation.bedrock._rag_wrappers import (
    _retrieve_and_generate_wrapper,
    _retrieve_wrapper,
)
from openinference.instrumentation.bedrock._wrappers import (
    _apply_guardrail_wrapper,
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

    apply_guardrail: Callable[..., Any]
    _unwrapped_apply_guardrail: Callable[..., Any]


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

    Limitation: only implements read() — does not subclass aiobotocore's StreamingBody,
    so isinstance checks and methods beyond read() (iter_lines, iter_chunks, __aiter__,
    etc.) are not available. Callers using only read() are unaffected.
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
        # Stream read is a legitimate failure (network/service error): record and propagate.
        try:
            body_bytes = cast(bytes, await self._real_stream.read())
        except Exception as e:
            try:
                self._span.record_exception(e)
                self._span.set_status(Status(StatusCode.ERROR))
            finally:
                if not self._span_ended:
                    self._span.end()
                    self._span_ended = True
            raise
        self._cached = body_bytes
        # Instrumentation failures (JSON parsing, attribute extraction) are internal errors.
        # Log and continue so the caller always receives their response bytes, mirroring the
        # sync _model_invocation_wrapper which uses logger.warning for the same errors.
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
        except Exception:
            logger.warning("Failed to extract response attributes", exc_info=True)
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
        client._unwrapped_apply_guardrail = client.apply_guardrail
        client.apply_guardrail = _apply_guardrail_wrapper(tracer)(client)

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
                is_claude_message_api = False
                try:
                    # kwargs["body"] is InvokeModelRequest payload (blob: str or bytes).
                    if "body" in kwargs:
                        request_body = json.loads(kwargs["body"])
                        model_id = str(kwargs.get("modelId"))
                        is_claude_message_api = (
                            _extract_invoke_model_attributes.is_claude_message_api(model_id)
                        )
                        if is_claude_message_api:
                            anthropic_attributes.set_input_attributes(span, request_body, model_id)
                        else:
                            _extract_invoke_model_attributes.set_input_attributes(
                                span, request_body
                            )
                except Exception:
                    logger.warning("Failed to extract input attributes", exc_info=True)

                response = wrapped_client._unwrapped_invoke_model(*args, **kwargs)

                try:
                    body = response.get("body")
                    if body is not None:
                        # Botocore response body has _raw_stream and _content_length.
                        response["body"] = BufferedStreamingBody(
                            body._raw_stream, body._content_length
                        )
                        try:
                            response_body = json.loads(response.get("body").read())
                        finally:
                            response["body"].reset()

                        if is_claude_message_api:
                            anthropic_attributes.set_response_attributes(span, response_body)
                        else:
                            _extract_invoke_model_attributes.set_response_attributes(
                                span, kwargs, response_body, response
                            )
                except Exception:
                    logger.warning("Failed to extract response attributes", exc_info=True)

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
                is_claude_message_api = False
                try:
                    # kwargs["body"] is InvokeModelRequest payload (blob: str or bytes).
                    if "body" in kwargs:
                        request_body = json.loads(kwargs["body"])
                        model_id = str(kwargs.get("modelId"))
                        is_claude_message_api = (
                            _extract_invoke_model_attributes.is_claude_message_api(model_id)
                        )
                        if is_claude_message_api:
                            anthropic_attributes.set_input_attributes(span, request_body, model_id)
                        else:
                            _extract_invoke_model_attributes.set_input_attributes(
                                span, request_body
                            )
                except Exception:
                    logger.warning("Failed to extract input attributes", exc_info=True)
                try:
                    response = await wrapped_client._unwrapped_invoke_model(*args, **kwargs)
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.end()
                    raise
                body = response.get("body")
                if body is None:
                    logger.warning(
                        "InvokeModel response missing 'body'; span ended without attributes"
                    )
                    span.end()
                    return response  # type: ignore
                # Lazy body: on first read() we consume the stream, set response attrs, end span.
                response["body"] = _LazyAsyncInvokeModelBody(
                    body, span, kwargs, response, is_claude_message_api
                )
                return response  # type: ignore

        return instrumented_response

    return _invocation_wrapper


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
            """Set output message and token usage on span from response."""
            try:
                span.set_attributes(_get_converse_response_attributes(kwargs, response))  # type: ignore[arg-type]
            except Exception:
                logger.warning("Failed to set response attributes on span", exc_info=True)
            span.set_attributes(dict(get_attributes_from_context()))
            span.set_status(Status(StatusCode.OK))
            return response

        # Use async wrapper when client is async (aiobotocore).
        if hasattr(wrapped_client, "__aenter__"):

            @wraps(wrapped_client.converse)
            async def async_instrumented_response(*args: Any, **kwargs: Any) -> Dict[str, Any]:
                if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
                    return await wrapped_client._unwrapped_converse(*args, **kwargs)  # type: ignore

                with tracer.start_as_current_span("bedrock.converse") as span:
                    try:
                        span.set_attributes(_get_converse_request_attributes(kwargs))  # type: ignore[arg-type]
                    except Exception:
                        logger.warning("Failed to set request attributes on span", exc_info=True)
                    try:
                        response = await wrapped_client._unwrapped_converse(*args, **kwargs)
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.end()
                        raise
                    return _process_converse(span, args, kwargs, response)

            return async_instrumented_response
        else:

            @wraps(wrapped_client.converse)
            def sync_instrumented_response(*args: Any, **kwargs: Any) -> Dict[str, Any]:
                if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
                    return wrapped_client._unwrapped_converse(*args, **kwargs)  # type: ignore

                with tracer.start_as_current_span("bedrock.converse") as span:
                    try:
                        span.set_attributes(_get_converse_request_attributes(kwargs))  # type: ignore[arg-type]
                    except Exception:
                        logger.warning("Failed to set request attributes on span", exc_info=True)
                    try:
                        response = wrapped_client._unwrapped_converse(*args, **kwargs)
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.end()
                        raise
                    return _process_converse(span, args, kwargs, response)

            return sync_instrumented_response

    return _converse_wrapper


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
            _MODULE,
            "ClientCreator.create_client",
            _sync_client_creation_wrapper(
                tracer=self._tracer,
                module_version=botocore.__version__,
            ),
        )
        # Patch async client creation (aiobotocore) if available.
        try:
            aioboto = import_module(_AIO_MODULE)
            self._original_aio_client_creator = aioboto.AioClientCreator.create_client
            wrap_function_wrapper(
                _AIO_MODULE,
                "AioClientCreator.create_client",
                _async_client_creation_wrapper(
                    tracer=self._tracer,
                    # Converse check uses botocore version (same as sync);
                    # aiobotocore wraps botocore.
                    module_version=botocore.__version__,
                ),
            )
        except ImportError:
            # aiobotocore not installed; initialize slot so _uninstrument can always read it.
            self._original_aio_client_creator = None

    def _uninstrument(self, **kwargs: Any) -> None:
        """Restore original create_client implementations."""
        boto = import_module(_MODULE)
        boto.ClientCreator.create_client = self._original_client_creator
        self._original_client_creator = None
        try:
            aioboto = import_module(_AIO_MODULE)
            # Only restore if we actually patched it; _original_aio_client_creator is None when
            # aiobotocore was unavailable at _instrument time.
            if self._original_aio_client_creator is not None:
                aioboto.AioClientCreator.create_client = self._original_aio_client_creator
                self._original_aio_client_creator = None
        except ImportError:
            pass


def _set_span_attribute(span: trace_api.Span, name: str, value: AttributeValue) -> None:
    """Set a span attribute only if value is non-empty."""
    if value is not None and value != "":
        span.set_attribute(name, value)
