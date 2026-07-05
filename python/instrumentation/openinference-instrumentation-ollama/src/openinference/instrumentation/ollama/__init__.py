import logging
from importlib import import_module
from typing import Any, Collection

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from wrapt import wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.ollama.version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_instruments = ("ollama >= 0.4.0",)


class OllamaInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """An instrumentor for the Ollama Python client."""

    __slots__ = ("_original_chat", "_original_async_chat", "_original_module_chat", "_tracer")

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        from ollama._client import AsyncClient, Client
        from openinference.instrumentation.ollama._wrappers import (
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

        # The module-level ``ollama.chat`` / ``ollama.generate`` helpers delegate
        # to a shared ``Client`` instance, so wrapping the class methods covers
        # the module-level helpers, ``Client``, and ``AsyncClient`` alike.
        self._original_chat = Client.chat
        wrap_function_wrapper(
            "ollama._client",
            "Client.chat",
            _ChatWrapper(tracer=self._tracer),
        )

        self._original_async_chat = AsyncClient.chat
        wrap_function_wrapper(
            "ollama._client",
            "AsyncClient.chat",
            _AsyncChatWrapper(tracer=self._tracer),
        )

        # ``ollama.chat`` is bound to the shared module-level client at import
        # time, so it captured the unwrapped method. Re-bind it to the now-wrapped
        # method so the module-level helper is traced too.
        import ollama

        self._original_module_chat = ollama.chat
        ollama.chat = ollama._client.chat

    def _uninstrument(self, **kwargs: Any) -> None:
        import ollama

        ollama_module = import_module("ollama._client")
        if getattr(self, "_original_chat", None) is not None:
            ollama_module.Client.chat = self._original_chat
        if getattr(self, "_original_async_chat", None) is not None:
            ollama_module.AsyncClient.chat = self._original_async_chat
        if getattr(self, "_original_module_chat", None) is not None:
            ollama.chat = self._original_module_chat
