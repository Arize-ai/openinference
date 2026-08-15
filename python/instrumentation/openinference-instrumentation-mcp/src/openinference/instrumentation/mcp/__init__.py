import contextvars
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable, Collection, Tuple

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
        # Bump the generation counter so that any post-import hooks registered in a
        # previous instrument() call become no-ops if they fire after uninstrument().
        self._self_generation: int = getattr(self, "_self_generation", 0) + 1
        generation = self._self_generation

        def _guarded(module: str, target: str, wrapper: Any) -> Any:
            def hook(_: Any) -> None:
                if getattr(self, "_self_generation", 0) == generation:
                    wrap_function_wrapper(module, target, wrapper)

            return hook

        register_post_import_hook(
            _guarded(
                "mcp.client.streamable_http",
                "streamable_http_client",
                self._wrap_transport_with_callback,
            ),
            "mcp.client.streamable_http",
        )

        register_post_import_hook(
            _guarded(
                "mcp.server.streamable_http",
                "StreamableHTTPServerTransport.connect",
                self._wrap_plain_transport,
            ),
            "mcp.server.streamable_http",
        )

        register_post_import_hook(
            _guarded("mcp.client.sse", "sse_client", self._wrap_plain_transport),
            "mcp.client.sse",
        )
        register_post_import_hook(
            _guarded(
                "mcp.server.sse", "SseServerTransport.connect_sse", self._wrap_plain_transport
            ),
            "mcp.server.sse",
        )
        register_post_import_hook(
            _guarded("mcp.client.stdio", "stdio_client", self._wrap_plain_transport),
            "mcp.client.stdio",
        )
        register_post_import_hook(
            _guarded("mcp.server.stdio", "stdio_server", self._wrap_plain_transport),
            "mcp.server.stdio",
        )

        # While we prefer to instrument the lowest level primitive, the transports above, it
        # doesn't mean context will be propagated to handlers automatically. Notably, the MCP SDK
        # passes server messages to a handler with a separate stream in between, losing context. We
        # go ahead and instrument this second stream just to propagate context so transports can
        # still be used independently while also supporting the major usage of the MCP SDK.
        # Notably, this may be a reasonable generic instrumentation for anyio itself to allow its
        # streams to propagate context broadly.
        register_post_import_hook(
            _guarded(
                "mcp.server.session", "ServerSession.__init__", self._base_session_init_wrapper
            ),
            "mcp.server.session",
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        # Invalidate any pending deferred post-import hooks registered by this instrumentor
        # so they become no-ops if a not-yet-imported module is loaded later.
        self._self_generation = getattr(self, "_self_generation", 0) + 1

        # Unwrap top-level transport functions (module -> function).
        for module_name, func_name in (
            ("mcp.client.streamable_http", "streamable_http_client"),
            ("mcp.client.sse", "sse_client"),
            ("mcp.client.stdio", "stdio_client"),
            ("mcp.server.stdio", "stdio_server"),
        ):
            module = sys.modules.get(module_name)
            if module is not None:
                unwrap(module, func_name)

        # Unwrap class methods (module -> class -> method).
        for module_name, class_name, method_name in (
            ("mcp.server.streamable_http", "StreamableHTTPServerTransport", "connect"),
            ("mcp.server.sse", "SseServerTransport", "connect_sse"),
            ("mcp.server.session", "ServerSession", "__init__"),
        ):
            module = sys.modules.get(module_name)
            if module is not None:
                cls = getattr(module, class_name, None)
                if cls is not None:
                    unwrap(cls, method_name)

    @asynccontextmanager
    async def _wrap_transport_with_callback(
        self, wrapped: Callable[..., Any], instance: Any, args: Any, kwargs: Any
    ) -> AsyncGenerator[Tuple[Any, ...], None]:
        async with wrapped(*args, **kwargs) as streams:
            read_stream, write_stream, *extra = streams
            yield (
                InstrumentedStreamReader(read_stream),  # type: ignore[no-untyped-call,unused-ignore]
                InstrumentedStreamWriter(write_stream),  # type: ignore[no-untyped-call,unused-ignore]
                *extra,
            )

    @asynccontextmanager
    async def _wrap_plain_transport(
        self, wrapped: Callable[..., Any], instance: Any, args: Any, kwargs: Any
    ) -> AsyncGenerator[Tuple["InstrumentedStreamReader", "InstrumentedStreamWriter"], None]:
        async with wrapped(*args, **kwargs) as (read_stream, write_stream):
            yield InstrumentedStreamReader(read_stream), InstrumentedStreamWriter(write_stream)  # type: ignore[no-untyped-call,unused-ignore]

    def _base_session_init_wrapper(
        self, wrapped: Callable[..., None], instance: Any, args: Any, kwargs: Any
    ) -> None:
        wrapped(*args, **kwargs)
        reader = getattr(instance, "_incoming_message_stream_reader", None)
        writer = getattr(instance, "_incoming_message_stream_writer", None)
        if reader and writer:
            setattr(
                instance,
                "_incoming_message_stream_reader",
                ContextAttachingStreamReader(reader),  # type: ignore[no-untyped-call,unused-ignore]
            )
            setattr(instance, "_incoming_message_stream_writer", ContextSavingStreamWriter(writer))  # type: ignore[no-untyped-call,unused-ignore]


class InstrumentedStreamReader(ObjectProxy):  # type: ignore[misc,name-defined,type-arg,unused-ignore]
    @property
    def last_context(self) -> contextvars.Context | None:
        """Context snapshot used by the MCP 2.0 dispatcher for spawned handlers."""
        return getattr(self, "_self_last_context", None)

    # ObjectProxy missing context manager - https://github.com/GrahamDumpleton/wrapt/issues/73
    async def __aenter__(self) -> Any:
        return await self.__wrapped__.__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
        return await self.__wrapped__.__aexit__(exc_type, exc_value, traceback)

    async def __aiter__(self) -> AsyncGenerator[Any, None]:
        from mcp.shared.message import SessionMessage
        from mcp.types import JSONRPCRequest

        async for item in self.__wrapped__:
            self._self_last_context = contextvars.copy_context()

            # Handle exceptions and other non-SessionMessage items
            # MCP can pass ValidationError or other exceptions through the stream
            if not isinstance(item, SessionMessage):
                yield item
                continue

            # mcp < 2.0 wraps the JSON-RPC payload in a pydantic RootModel exposed as
            # `.root`; mcp >= 2.0 stores the union member directly on `.message`.
            message = item.message
            request = getattr(message, "root", message)

            if not isinstance(request, JSONRPCRequest):
                yield item
                continue

            if request.params:
                meta = request.params.get("_meta")
                if meta:
                    ctx = propagate.extract(meta)
                    restore = context.attach(ctx)
                    try:
                        self._self_last_context = contextvars.copy_context()
                        yield item
                        continue
                    finally:
                        context.detach(restore)
            yield item


class InstrumentedStreamWriter(ObjectProxy):  # type: ignore[misc,name-defined,type-arg,unused-ignore]
    # ObjectProxy missing context manager - https://github.com/GrahamDumpleton/wrapt/issues/73
    async def __aenter__(self) -> Any:
        return await self.__wrapped__.__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
        return await self.__wrapped__.__aexit__(exc_type, exc_value, traceback)

    async def send(self, item: Any) -> Any:
        from mcp.shared.message import SessionMessage
        from mcp.types import JSONRPCRequest

        # Handle exceptions and other non-SessionMessage items
        # MCP can pass ValidationError or other exceptions through the stream
        if not isinstance(item, SessionMessage):
            return await self.__wrapped__.send(item)

        # mcp < 2.0 wraps the JSON-RPC payload in a pydantic RootModel exposed as
        # `.root`; mcp >= 2.0 stores the union member directly on `.message`.
        message = item.message
        request = getattr(message, "root", message)
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


class ContextSavingStreamWriter(ObjectProxy):  # type: ignore[misc,name-defined,type-arg,unused-ignore]
    # ObjectProxy missing context manager - https://github.com/GrahamDumpleton/wrapt/issues/73
    async def __aenter__(self) -> Any:
        return await self.__wrapped__.__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
        return await self.__wrapped__.__aexit__(exc_type, exc_value, traceback)

    async def send(self, item: Any) -> Any:
        ctx = context.get_current()
        return await self.__wrapped__.send(ItemWithContext(item, ctx))


class ContextAttachingStreamReader(ObjectProxy):  # type: ignore[misc,name-defined,type-arg,unused-ignore]
    # ObjectProxy missing context manager - https://github.com/GrahamDumpleton/wrapt/issues/73
    async def __aenter__(self) -> Any:
        return await self.__wrapped__.__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
        return await self.__wrapped__.__aexit__(exc_type, exc_value, traceback)

    async def __aiter__(self) -> AsyncGenerator[Any, None]:
        async for item in self.__wrapped__:
            # Handle items that are not ItemWithContext (e.g., exceptions)
            if not isinstance(item, ItemWithContext):
                yield item
                continue

            restore = context.attach(item.ctx)
            try:
                yield item.item
            finally:
                context.detach(restore)
