from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable, Collection, Tuple, cast

from opentelemetry import context, propagate
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from opentelemetry.instrumentation.utils import unwrap
from wrapt import ObjectProxy, register_post_import_hook, wrap_function_wrapper

from openinference.instrumentation.mcp.package import _instruments


class MCPInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumenter for MCP.
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.client.sse", "sse_client", self._transport_wrapper
            ),
            "mcp.client.sse",
        )
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.server.sse", "SseServerTransport.connect_sse", self._transport_wrapper
            ),
            "mcp.server.sse",
        )
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.client.stdio", "stdio_client", self._transport_wrapper
            ),
            "mcp.client.stdio",
        )
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.server.stdio", "stdio_server", self._transport_wrapper
            ),
            "mcp.server.stdio",
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
        unwrap("mcp.client.stdio", "stdio_client")
        unwrap("mcp.server.stdio", "stdio_server")

    @asynccontextmanager
    async def _transport_wrapper(
        self, wrapped: Callable[..., Any], instance: Any, args: Any, kwargs: Any
    ) -> AsyncGenerator[Tuple["InstrumentedStreamReader", "InstrumentedStreamWriter"], None]:
        async with wrapped(*args, **kwargs) as (read_stream, write_stream):
            yield InstrumentedStreamReader(read_stream), InstrumentedStreamWriter(write_stream)

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
