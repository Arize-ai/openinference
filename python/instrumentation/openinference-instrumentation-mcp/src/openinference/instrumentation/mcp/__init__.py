from typing import Any, Awaitable, Callable, Collection, TypeVar, cast

from opentelemetry import context, propagate
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from opentelemetry.instrumentation.utils import unwrap
from wrapt import register_post_import_hook, wrap_function_wrapper

from openinference.instrumentation.mcp.package import _instruments

T = TypeVar("T")


class MCPInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumenter for MCP.
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        register_post_import_hook(self._patch, "mcp")

    def _patch(self, module: Any) -> None:
        wrap_function_wrapper(
            "mcp.client.session",
            "ClientSession.send_request",
            self._client_request_wrapper,
        )
        wrap_function_wrapper(
            "mcp.server.lowlevel.server",
            "Server._handle_request",
            self._server_request_wrapper,
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        unwrap("mcp.client.session.ClientSession", "send_request")
        unwrap("mcp.server.lowlevel.server", "_handle_request")

    def _client_request_wrapper(
        self, wrapped: Callable[..., T], instance: Any, args: Any, kwargs: Any
    ) -> T:
        from mcp.types import JSONRPCMessage, Request, RequestParams

        message = cast(JSONRPCMessage, args[0])
        request = cast(Request[RequestParams, Any], message.root)
        if not request.params:
            request.params = RequestParams()
        if not request.params.meta:
            request.params.meta = RequestParams.Meta()
        propagate.get_global_textmap().inject(request.params.meta.__pydantic_extra__)
        return wrapped(*args, **kwargs)

    async def _server_request_wrapper(
        self, wrapped: Callable[..., Awaitable[T]], instance: Any, args: Any, kwargs: Any
    ) -> T:
        from mcp.types import Request, RequestParams

        request = cast(Request[RequestParams, Any], args[1])
        if hasattr(request, "params") and hasattr(request.params, "meta"):
            meta = request.params.meta
            if meta and hasattr(meta, "__pydantic_extra__"):
                ctx = propagate.extract(meta.__pydantic_extra__)
                restore = context.attach(ctx)
                try:
                    return await wrapped(*args, **kwargs)
                finally:
                    context.detach(restore)
        return await wrapped(*args, **kwargs)
