from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable, Collection, Tuple, cast

from opentelemetry import context, propagate
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import Status, StatusCode
from wrapt import ObjectProxy, register_post_import_hook, wrap_function_wrapper

from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.mcp.package import _instruments
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes


class MCPInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumenter for MCP.
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        # Instrument high-level MCP client operations
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.client.session",
                "ClientSession.call_tool",
                self._wrap_call_tool,
            ),
            "mcp.client.session",
        )

        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.client.session",
                "ClientSession.get_prompt",
                self._wrap_get_prompt,
            ),
            "mcp.client.session",
        )

        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.client.session",
                "ClientSession.read_resource",
                self._wrap_read_resource,
            ),
            "mcp.client.session",
        )

        # Existing transport-level instrumentation for context propagation
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.client.streamable_http",
                "streamablehttp_client",
                self._wrap_transport_with_callback,
            ),
            "mcp.client.streamable_http",
        )

        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.server.streamable_http",
                "StreamableHTTPServerTransport.connect",
                self._wrap_plain_transport,
            ),
            "mcp.server.streamable_http",
        )

        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.client.sse", "sse_client", self._wrap_plain_transport
            ),
            "mcp.client.sse",
        )
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.server.sse", "SseServerTransport.connect_sse", self._wrap_plain_transport
            ),
            "mcp.server.sse",
        )
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.client.stdio", "stdio_client", self._wrap_plain_transport
            ),
            "mcp.client.stdio",
        )
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.server.stdio", "stdio_server", self._wrap_plain_transport
            ),
            "mcp.server.stdio",
        )
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.server.lowlevel.server",
                "Server._handle_request",
                self._wrap_handle_request,
            ),
            "mcp.server.lowlevel.server",
        )

        # While we prefer to instrument the lowest level primitive, the transports above, it doesn't
        # mean context will be propagated to handlers automatically. Notably, the MCP SDK passes
        # server messages to a handler with a separate stream in between, losing context. We go
        # ahead and instrument this second stream just to propagate context so transports can still
        # be used independently while also supporting the major usage of the MCP SDK. Notably, this
        # may be a reasonable generic instrumentation for anyio itself to allow its streams to
        # propagate context broadly.
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.server.session", "ServerSession.__init__", self._base_session_init_wrapper
            ),
            "mcp.server.session",
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        unwrap("mcp.client.session", "ClientSession.call_tool")
        unwrap("mcp.client.session", "ClientSession.get_prompt")
        unwrap("mcp.client.session", "ClientSession.read_resource")
        unwrap("mcp.client.stdio", "stdio_client")
        unwrap("mcp.server.stdio", "stdio_server")
        unwrap("mcp.client.session", "ClientSession.call_tool")

    @asynccontextmanager
    async def _wrap_transport_with_callback(
        self, wrapped: Callable[..., Any], instance: Any, args: Any, kwargs: Any
    ) -> AsyncGenerator[Tuple["InstrumentedStreamReader", "InstrumentedStreamWriter", Any], None]:
        async with wrapped(*args, **kwargs) as (read_stream, write_stream, get_session_id_callback):
            yield (
                InstrumentedStreamReader(read_stream),
                InstrumentedStreamWriter(write_stream),
                get_session_id_callback,
            )

    @asynccontextmanager
    async def _wrap_plain_transport(
        self, wrapped: Callable[..., Any], instance: Any, args: Any, kwargs: Any
    ) -> AsyncGenerator[Tuple["InstrumentedStreamReader", "InstrumentedStreamWriter"], None]:
        async with wrapped(*args, **kwargs) as (read_stream, write_stream):
            yield InstrumentedStreamReader(read_stream), InstrumentedStreamWriter(write_stream)

    async def _wrap_handle_request(
        self, wrapped: Callable[..., Any], instance: Any, args: Any, kwargs: Any
    ) -> Any:
        token = None
        try:
            # Message has been deserialized, we need to extract the traceparent
            _meta = {"traceparent": args[1].params.meta.traceparent}
            ctx = propagate.extract(_meta)
            token = context.attach(ctx)
        finally:
            res = await wrapped(*args, **kwargs)
            if token:
                context.detach(token)
            return res

    def _base_session_init_wrapper(
        self, wrapped: Callable[..., None], instance: Any, args: Any, kwargs: Any
    ) -> None:
        wrapped(*args, **kwargs)
        reader = getattr(instance, "_incoming_message_stream_reader", None)
        writer = getattr(instance, "_incoming_message_stream_writer", None)
        if reader and writer:
            setattr(
                instance, "_incoming_message_stream_reader", ContextAttachingStreamReader(reader)
            )
            setattr(instance, "_incoming_message_stream_writer", ContextSavingStreamWriter(writer))

    async def _wrap_call_tool(
        self, wrapped: Callable[..., Any], instance: Any, args: Any, kwargs: Any
    ) -> Any:
        """Wrap MCP call_tool operation with tracing."""
        # Extract arguments
        name = kwargs.get("name", "unknown")
        arguments = kwargs.get("arguments")

        tracer = trace_api.get_tracer(__name__)
        with tracer.start_as_current_span(
            f"mcp.call_tool.{name}",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL.value,
                SpanAttributes.TOOL_NAME: name,
                SpanAttributes.SESSION_ID: instance._request_id,
                SpanAttributes.INPUT_VALUE: safe_json_dumps(kwargs),
                SpanAttributes.INPUT_MIME_TYPE: "application/json",
            },
        ) as span:
            # Add input attributes
            if arguments:
                span.set_attribute(SpanAttributes.TOOL_PARAMETERS, safe_json_dumps(arguments))

            try:
                # Call the original method
                result = await wrapped(*args, **kwargs)

                # Add output attributes
                # TODO: handle content types
                if hasattr(result, "content") and result.content:
                    span.set_attribute(SpanAttributes.OUTPUT_VALUE, safe_json_dumps(result.content))
                    span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "application/json")

                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    # TODO: update once OpenInference support for prompts grows
    async def _wrap_get_prompt(
        self, wrapped: Callable[..., Any], instance: Any, args: Any, kwargs: Any
    ) -> Any:
        """Wrap MCP get_prompt operation with tracing."""
        # Extract arguments
        name = kwargs.get("name", "unknown")

        tracer = trace_api.get_tracer(__name__)
        with tracer.start_as_current_span(
            f"mcp.get_prompt.{name}",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.UNKNOWN.value,
                SpanAttributes.PROMPT_ID: name,
                SpanAttributes.SESSION_ID: instance._request_id,
                SpanAttributes.INPUT_VALUE: safe_json_dumps(kwargs),
                SpanAttributes.INPUT_MIME_TYPE: "application/json",
            },
        ) as span:
            try:
                # Call the original method
                result = await wrapped(*args, **kwargs)

                # Add output attributes
                if hasattr(result, "messages") and result.messages:
                    span.set_attribute(
                        SpanAttributes.OUTPUT_VALUE,
                        safe_json_dumps(
                            {"description": result.description, "messages": result.messages}
                        ),
                    )
                    span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "application/json")

                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    # TODO: further e2e testing once Phoenix error #7687 is resolved
    async def _wrap_read_resource(
        self, wrapped: Callable[..., Any], instance: Any, args: Any, kwargs: Any
    ) -> Any:
        """Wrap MCP read_resource operation with tracing."""
        # Extract arguments
        uri = args[0]
        tracer = trace_api.get_tracer(__name__)
        with tracer.start_as_current_span(
            f"mcp.read_resource.{uri}",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.RETRIEVER.value,
                SpanAttributes.SESSION_ID: instance._request_id,
                SpanAttributes.INPUT_VALUE: safe_json_dumps({"uri": uri}),
                SpanAttributes.INPUT_MIME_TYPE: "application/json",
            },
        ) as span:
            try:
                # Call the original method
                result = await wrapped(*args, **kwargs)
                # Add output attributes
                if hasattr(result, "contents") and result.contents:
                    for i, content in enumerate(result.contents):
                        if content.text:
                            span.set_attribute(
                                f"retrieval.documents.{i}.document.content", content.text
                            )
                            span.set_attribute(
                                f"retrieval.documents.{i}.document.metadata",
                                safe_json_dumps({"type": content.__class__.__name__}),
                            )
                        elif content.blob:
                            span.set_attribute(
                                f"retrieval.documents.{i}.document.content", content.blob
                            )
                            span.set_attribute(
                                f"retrieval.documents.{i}.document.metadata",
                                safe_json_dumps({"type": content.__class__.__name__}),
                            )
                        else:
                            # fail silently for now
                            print(f"Unknown document type: {type(content)}")
                    serialized_contents = [content.model_dump() for content in result.contents]
                    span.set_attribute(
                        SpanAttributes.OUTPUT_VALUE,
                        safe_json_dumps({"contents": serialized_contents}),
                    )
                    span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "application/json")

                span.set_status(Status(StatusCode.OK))
                return result

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise


class InstrumentedStreamReader(ObjectProxy):  # type: ignore
    # ObjectProxy missing context manager - https://github.com/GrahamDumpleton/wrapt/issues/73
    async def __aenter__(self) -> Any:
        return await self.__wrapped__.__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
        return await self.__wrapped__.__aexit__(exc_type, exc_value, traceback)

    async def __aiter__(self) -> AsyncGenerator[Any, None]:
        from mcp.shared.message import SessionMessage
        from mcp.types import JSONRPCRequest

        async for item in self.__wrapped__:
            session_message = cast(SessionMessage, item)
            request = session_message.message.root

            if not isinstance(request, JSONRPCRequest):
                yield item
                continue

            if request.params:
                meta = request.params.get("_meta")
                if meta:
                    ctx = propagate.extract(meta)
                    restore = context.attach(ctx)
                    try:
                        yield item
                        continue
                    finally:
                        context.detach(restore)
            yield item


class InstrumentedStreamWriter(ObjectProxy):  # type: ignore
    # ObjectProxy missing context manager - https://github.com/GrahamDumpleton/wrapt/issues/73
    async def __aenter__(self) -> Any:
        return await self.__wrapped__.__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
        return await self.__wrapped__.__aexit__(exc_type, exc_value, traceback)

    async def send(self, item: Any) -> Any:
        from mcp.shared.message import SessionMessage
        from mcp.types import JSONRPCRequest

        session_message = cast(SessionMessage, item)
        request = session_message.message.root
        if not isinstance(request, JSONRPCRequest):
            return await self.__wrapped__.send(item)
        meta = None
        if not request.params:
            request.params = {}
        meta = request.params.setdefault("_meta", {})
        propagate.get_global_textmap().inject(meta)
        return await self.__wrapped__.send(item)


@dataclass(slots=True, frozen=True)
class ItemWithContext:
    item: Any
    ctx: context.Context


class ContextSavingStreamWriter(ObjectProxy):  # type: ignore
    # ObjectProxy missing context manager - https://github.com/GrahamDumpleton/wrapt/issues/73
    async def __aenter__(self) -> Any:
        return await self.__wrapped__.__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
        return await self.__wrapped__.__aexit__(exc_type, exc_value, traceback)

    async def send(self, item: Any) -> Any:
        ctx = context.get_current()
        return await self.__wrapped__.send(ItemWithContext(item, ctx))


class ContextAttachingStreamReader(ObjectProxy):  # type: ignore
    # ObjectProxy missing context manager - https://github.com/GrahamDumpleton/wrapt/issues/73
    async def __aenter__(self) -> Any:
        return await self.__wrapped__.__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
        return await self.__wrapped__.__aexit__(exc_type, exc_value, traceback)

    async def __aiter__(self) -> AsyncGenerator[Any, None]:
        async for item in self.__wrapped__:
            item_with_context = cast(ItemWithContext, item)
            restore = context.attach(item_with_context.ctx)
            try:
                yield item_with_context.item
            finally:
                context.detach(restore)
