import logging
from importlib import import_module
from typing import Any, Collection

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from wrapt import wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.cohere.version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_instruments = ("cohere >= 5.13.0",)


class CohereInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """An instrumentor for the Cohere Python client (v2 chat API)."""

    __slots__ = ("_original_chat", "_original_async_chat", "_tracer")

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        from cohere.v2.client import AsyncV2Client, V2Client
        from openinference.instrumentation.cohere._wrappers import (
            _AsyncChatWrapper,
            _ChatWrapper,
        )

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

        # ``cohere.ClientV2`` / ``AsyncClientV2`` inherit ``chat`` from these base
        # classes, so wrapping the base methods covers both client entry points.
        self._original_chat = V2Client.chat
        wrap_function_wrapper(
            "cohere.v2.client",
            "V2Client.chat",
            _ChatWrapper(tracer=self._tracer),
        )

        self._original_async_chat = AsyncV2Client.chat
        wrap_function_wrapper(
            "cohere.v2.client",
            "AsyncV2Client.chat",
            _AsyncChatWrapper(tracer=self._tracer),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        cohere_module = import_module("cohere.v2.client")
        if getattr(self, "_original_chat", None) is not None:
            cohere_module.V2Client.chat = self._original_chat
        if getattr(self, "_original_async_chat", None) is not None:
            cohere_module.AsyncV2Client.chat = self._original_async_chat
