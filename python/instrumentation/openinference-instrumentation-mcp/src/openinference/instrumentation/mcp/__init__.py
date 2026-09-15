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


@asynccontextmanager
async def _wrap_transport_with_callback(
    wrapped: Callable[..., Any], instance: Any, args: Any, kwargs: Any
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
    wrapped: Callable[..., Any], instance: Any, args: Any, kwargs: Any
) -> AsyncGenerator[Tuple["InstrumentedStreamReader", "InstrumentedStreamWriter"], None]:
    async with wrapped(*args, **kwargs) as (read_stream, write_stream):
        yield InstrumentedStreamReader(read_stream), InstrumentedStreamWriter(write_stream)  # type: ignore[no-untyped-call,unused-ignore]


def _base_session_init_wrapper(
    wrapped: Callable[..., None], instance: Any, args: Any, kwargs: Any
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


# The (module, target, wrapper) triples wrapped by MCPInstrumentor; class methods are
# dotted "Class.method" targets. This table drives both _instrument and _uninstrument.
_WRAP_TARGETS: Tuple[Tuple[str, str, Callable[..., Any]], ...] = (
    ("mcp.client.streamable_http", "streamable_http_client", _wrap_transport_with_callback),
    (
        "mcp.server.streamable_http",
        "StreamableHTTPServerTransport.connect",
        _wrap_plain_transport,
    ),
    ("mcp.client.sse", "sse_client", _wrap_plain_transport),
    ("mcp.server.sse", "SseServerTransport.connect_sse", _wrap_plain_transport),
    ("mcp.client.stdio", "stdio_client", _wrap_plain_transport),
    ("mcp.server.stdio", "stdio_server", _wrap_plain_transport),
    # Instrumenting the transports alone does not propagate context to handlers: the MCP
    # SDK passes server messages to handlers through a separate internal stream, losing
    # context. Wrapping ServerSession.__init__ instruments that stream as well.
    ("mcp.server.session", "ServerSession.__init__", _base_session_init_wrapper),
)


def _resolve_target(module: Any, target: str) -> Tuple[Any, str]:
    """Resolves a dotted target to the object holding the wrapped attribute."""
    prefix, _, attr = target.rpartition(".")
    owner = getattr(module, prefix, None) if prefix else module
    return owner, attr


def _register_guarded_hook(
    module_name: str, target: str, wrapper: Callable[..., Any], generation: int
) -> None:
    def hook(module: Any) -> None:
        with MCPInstrumentor._lock:
            if MCPInstrumentor._generation != generation:
                return  # uninstrument() has run since this hook was registered
            try:
                wrap_function_wrapper(module, target, wrapper)
            except Exception:
                # Runs inside the user's import statement; never raise into user code.
                logger.exception("Failed to instrument %s.%s", module_name, target)

    register_post_import_hook(hook, module_name)


class MCPInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumenter for MCP.
    """

    # Deferred post-import hooks capture this counter and wrap only while their value is
    # current; uninstrument() bumps it to neutralize every hook registered so far. A
    # counter (not a boolean) so stale hooks cannot double-wrap after re-instrumenting.
    _generation: int = 0
    _lock = threading.Lock()

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        with MCPInstrumentor._lock:
            generation = MCPInstrumentor._generation
        for module_name, target, wrapper in _WRAP_TARGETS:
            _register_guarded_hook(module_name, target, wrapper, generation)

    def _uninstrument(self, **kwargs: Any) -> None:
        with MCPInstrumentor._lock:
            MCPInstrumentor._generation += 1
            for module_name, target, _ in _WRAP_TARGETS:
                # Modules never imported were never wrapped; avoid importing them here.
                module = sys.modules.get(module_name)
                if module is not None:
                    unwrap(*_resolve_target(module, target))


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
