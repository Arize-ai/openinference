import logging
from importlib import import_module
from typing import Any, Collection

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from portkey_ai.api_resources.apis.chat_complete import (
    AsyncCompletions,
    Completions,
)
from wrapt import wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.portkey._wrappers import (
    _AsyncCompletionsWrapper,
    _CompletionsWrapper,
)
from openinference.instrumentation.portkey.version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_instruments = ("portkey_ai >= 0.1.0",)


class PortkeyInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """An instrumentor for the Portkey AI framework."""

    __slots__ = ("_original_completions_create", "_original_async_completions_create", "_tracer")

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
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

        self._original_completions_create = Completions.create
        wrap_function_wrapper(
            module="portkey_ai.api_resources.apis.chat_complete",
            name="Completions.create",
            wrapper=_CompletionsWrapper(tracer=self._tracer),
        )
        wrap_function_wrapper(
            module="portkey_ai.api_resources.apis.generation",
            name="Completions.create",
            wrapper=_CompletionsWrapper(tracer=self._tracer),
        )

        self._original_async_completions_create = AsyncCompletions.create
        wrap_function_wrapper(
            module="portkey_ai.api_resources.apis.chat_complete",
            name="AsyncCompletions.create",
            wrapper=_AsyncCompletionsWrapper(tracer=self._tracer),
        )
        wrap_function_wrapper(
            module="portkey_ai.api_resources.apis.generation",
            name="AsyncCompletions.create",
            wrapper=_AsyncCompletionsWrapper(tracer=self._tracer),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        portkey_module = import_module("portkey_ai.api_resources.apis.chat_complete")
        if self._original_completions_create is not None:
            portkey_module.Completions.create = self._original_completions_create

        if self._original_async_completions_create is not None:
            portkey_module.AsyncCompletions.create = self._original_async_completions_create
