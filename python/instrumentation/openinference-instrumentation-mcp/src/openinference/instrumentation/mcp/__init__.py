import contextvars
import logging
import sys
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable, Collection, Tuple

from opentelemetry import context, propagate
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from opentelemetry.instrumentation.utils import unwrap
from wrapt import ObjectProxy, register_post_import_hook, wrap_function_wrapper

from openinference.instrumentation.mcp.package import _instruments

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# The (module, target, wrapper method) triples wrapped by MCPInstrumentor. Class methods
# are expressed as dotted "Class.method" targets. This single table drives _instrument,
# _uninstrument, and the uninstrument test so the three cannot drift apart.
#
# While we prefer to instrument the lowest level primitive, the transports, it doesn't
# mean context will be propagated to handlers automatically. Notably, the MCP SDK passes
# server messages to a handler with a separate stream in between, losing context. We go
# ahead and instrument this second stream (ServerSession.__init__ below) just to
# propagate context so transports can still be used independently while also supporting
# the major usage of the MCP SDK. Notably, this may be a reasonable generic
# instrumentation for anyio itself to allow its streams to propagate context broadly.
_WRAP_TARGETS: Tuple[Tuple[str, str, str], ...] = (
    ("mcp.client.streamable_http", "streamable_http_client", "_wrap_transport_with_callback"),
    (
        "mcp.server.streamable_http",
        "StreamableHTTPServerTransport.connect",
        "_wrap_plain_transport",
    ),
    ("mcp.client.sse", "sse_client", "_wrap_plain_transport"),
    ("mcp.server.sse", "SseServerTransport.connect_sse", "_wrap_plain_transport"),
    ("mcp.client.stdio", "stdio_client", "_wrap_plain_transport"),
    ("mcp.server.stdio", "stdio_server", "_wrap_plain_transport"),
    ("mcp.server.session", "ServerSession.__init__", "_base_session_init_wrapper"),
)


class MCPInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumenter for MCP.
    """

    # Generation counter guarding deferred post-import hooks. BaseInstrumentor is a
    # singleton, so this state is shared by all MCPInstrumentor() constructions. A
    # boolean would not suffice: instrument() -> uninstrument() -> instrument() leaves
    # multiple pending hooks for a not-yet-imported module, and all of them would wrap
    # (double-wrapping) when the module is finally imported; with the counter, only the
    # hooks from the latest instrument() call match and wrap exactly once.
    _generation: int = 0
    # Serializes generation checks/bumps against hook firing so a deferred hook cannot
    # wrap a module concurrently with (or after) an uninstrument() call.
    _lock = threading.Lock()

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        # Bump the generation counter so that any post-import hooks registered in a
        # previous instrument() call become no-ops if they fire after uninstrument().
        with self._lock:
            MCPInstrumentor._generation += 1
            generation = MCPInstrumentor._generation

        def _guarded(target: str, wrapper: Callable[..., Any]) -> Callable[[Any], None]:
            def hook(module: Any) -> None:
                with self._lock:
                    if MCPInstrumentor._generation != generation:
                        return
                    try:
                        wrap_function_wrapper(module, target, wrapper)
                    except Exception:
                        # This hook runs inside the user's import statement; never let
                        # instrumentation failures raise into user code.
                        logger.exception(
                            "Failed to instrument %s.%s",
                            getattr(module, "__name__", module),
                            target,
                        )

            return hook

        for module_name, target, wrapper_name in _WRAP_TARGETS:
            register_post_import_hook(_guarded(target, getattr(self, wrapper_name)), module_name)

    def _uninstrument(self, **kwargs: Any) -> None:
        with self._lock:
            # Invalidate any pending deferred post-import hooks registered by this
            # instrumentor so they become no-ops if a not-yet-imported module is loaded
            # later.
            MCPInstrumentor._generation += 1

            for module_name, target, _ in _WRAP_TARGETS:
                # Only unwrap modules that are already imported; unwrap()'s dotted-path
                # form would import them as a side effect.
                obj: Any = sys.modules.get(module_name)
                if obj is None:
                    continue
                prefix, _sep, attr = target.rpartition(".")
                if prefix:  # resolve the class for "Class.method" targets
                    obj = getattr(obj, prefix, None)
                    if obj is None:
                        continue
                unwrap(obj, attr)

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
