import logging
from importlib import import_module
from typing import Any, Collection

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from wrapt import wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.together.version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_instruments = ("together >= 1.3.0",)


class TogetherInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """An instrumentor for the Together AI Python client (chat completions)."""

    __slots__ = ("_original_create", "_original_async_create", "_tracer")

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        from openinference.instrumentation.together._wrappers import (
            _AsyncCompletionsWrapper,
            _CompletionsWrapper,
        )
        from together.resources.chat.completions import (
            AsyncCompletionsResource,
            CompletionsResource,
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

        self._original_create = CompletionsResource.create
        wrap_function_wrapper(
            "together.resources.chat.completions",
            "CompletionsResource.create",
            _CompletionsWrapper(tracer=self._tracer),
        )

        self._original_async_create = AsyncCompletionsResource.create
        wrap_function_wrapper(
            "together.resources.chat.completions",
            "AsyncCompletionsResource.create",
            _AsyncCompletionsWrapper(tracer=self._tracer),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        together_module = import_module("together.resources.chat.completions")
        if getattr(self, "_original_create", None) is not None:
            together_module.CompletionsResource.create = self._original_create
        if getattr(self, "_original_async_create", None) is not None:
            together_module.AsyncCompletionsResource.create = self._original_async_create
