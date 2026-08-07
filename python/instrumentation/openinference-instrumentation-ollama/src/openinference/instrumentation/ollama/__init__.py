"""OpenInference instrumentation for the Ollama Python client.

Usage::

    from openinference.instrumentation.ollama import OllamaInstrumentor

    OllamaInstrumentor().instrument(tracer_provider=tracer_provider)

Chat calls made through ``ollama.chat``, ``ollama.Client.chat``, and
``ollama.AsyncClient.chat`` (including ``stream=True``) are exported as
OpenInference LLM spans. Call ``instrument()`` before making chat calls;
references captured earlier (e.g. ``from ollama import chat`` at import
time) keep the unwrapped method and are not traced.
"""

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
    """An instrumentor for the Ollama Python client.

    ``instrument()`` accepts two optional keyword arguments: ``tracer_provider``
    (an OpenTelemetry ``TracerProvider``) and ``config`` (an OpenInference
    ``TraceConfig`` for masking sensitive data). It wraps ``Client.chat`` and
    ``AsyncClient.chat`` (covering the module-level ``ollama.chat`` helper) so
    chat calls are exported as OpenInference LLM spans; ``uninstrument()``
    restores the original methods.
    """

    __slots__ = (
        "_original_chat",
        "_original_async_chat",
        "_original_module_chat",
        "_tracer",
    )

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

        # Wrap the class methods so every ``Client``/``AsyncClient`` instance
        # is covered. Note: references captured before instrumentation (e.g.
        # ``from ollama import chat`` at import time) keep the unwrapped method
        # and cannot be traced retroactively.
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

        # ``ollama.chat`` is bound to the package-level client at import time,
        # so it captured the unwrapped method. Re-bind it through the now-wrapped
        # class method so the module-level helper is traced too. Guarded because
        # it relies on ollama's private package-level ``_client`` instance.
        import ollama

        self._original_module_chat = getattr(ollama, "chat", None)
        wrapped_module_chat = getattr(getattr(ollama, "_client", None), "chat", None)
        if self._original_module_chat is not None and callable(wrapped_module_chat):
            ollama.chat = wrapped_module_chat
        else:
            self._original_module_chat = None

    def _uninstrument(self, **kwargs: Any) -> None:
        import ollama

        ollama_module = import_module("ollama._client")
        # Restore the saved originals (the repo-wide convention): this
        # guarantees the OpenInference wrapper is removed even if another
        # library wrapped the method afterwards. Layered unwinding cannot be
        # done safely with wrapt, so any wrappers installed on top are dropped
        # along with ours.
        if getattr(self, "_original_chat", None) is not None:
            ollama_module.Client.chat = self._original_chat
        if getattr(self, "_original_async_chat", None) is not None:
            ollama_module.AsyncClient.chat = self._original_async_chat
        if (original_module_chat := getattr(self, "_original_module_chat", None)) is not None:
            ollama.chat = original_module_chat
