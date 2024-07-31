import logging
from importlib import import_module
from typing import Any, Collection

from _wrappers import _BothCompletionsWrapper, _CompletionsWrapper  #CHANGE
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import (  # type: ignore[attr-defined]
    BaseInstrumentor,
)
from opentelemetry.trace import get_tracer
from version import __version__  #CHANGE
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)

_instruments = ("groq >= 0.9.0",)


class GroqInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """An instrumentor for the Groq framework."""

    __slots__ = ("_original_completions_create", "_original_async_completions_create")
    groq_module = import_module("groq.resources.chat.completions")


    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        tracer = get_tracer(__name__, __version__, tracer_provider)
        groq_module = import_module("groq.resources.chat.completions")

        self._original_completions_create = groq_module.Completions.create
        wrap_function_wrapper(
            module="groq.resources.chat.completions",
            name="Completions.create",
            wrapper=_CompletionsWrapper(tracer=tracer),
        )

        self._original_async_completions_create = groq_module.AsyncCompletions.create
        wrap_function_wrapper(
            module="groq.resources.chat.completions",
            name="AsyncCompletions.create",
            wrapper=_BothCompletionsWrapper(tracer=tracer),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        groq_module = import_module("groq.resources.chat.completions")
        if self._original_completions_create is not None:
            groq_module.AsyncCompletions.create = self._original_completions_create
        if self._original_completions_create is not None:
            groq_module.AsyncCompletions.create = self._original_completions_create
