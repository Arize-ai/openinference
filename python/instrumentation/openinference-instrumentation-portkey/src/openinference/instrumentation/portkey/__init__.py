import logging
from importlib import import_module
from typing import Any, Collection

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
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

    __slots__ = (
        "_original_completions_create",
        "_original_async_completions_create",
        "_original_prompt_completions_create",
        "_original_async_prompt_completions_create",
        "_tracer",
    )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        from portkey_ai.api_resources.apis.chat_complete import AsyncCompletions, Completions
        from portkey_ai.api_resources.apis.generation import (
            AsyncCompletions as AsyncPromptCompletions,
            Completions as PromptCompletions,
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

        self._original_completions_create = Completions.create
        wrap_function_wrapper(
            "portkey_ai.api_resources.apis.chat_complete",
            "Completions.create",
            _CompletionsWrapper(tracer=self._tracer),
        )
        self._original_prompt_completions_create = PromptCompletions.create
        wrap_function_wrapper(
            "portkey_ai.api_resources.apis.generation",
            "Completions.create",
            _CompletionsWrapper(tracer=self._tracer),
        )

        self._original_async_completions_create = AsyncCompletions.create
        wrap_function_wrapper(
            "portkey_ai.api_resources.apis.chat_complete",
            "AsyncCompletions.create",
            _AsyncCompletionsWrapper(tracer=self._tracer),
        )
        self._original_async_prompt_completions_create = AsyncPromptCompletions.create
        wrap_function_wrapper(
            "portkey_ai.api_resources.apis.generation",
            "AsyncCompletions.create",
            _AsyncCompletionsWrapper(tracer=self._tracer),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        chat_complete_module = import_module("portkey_ai.api_resources.apis.chat_complete")
        generation_module = import_module("portkey_ai.api_resources.apis.generation")
        if self._original_completions_create is not None:
            chat_complete_module.Completions.create = self._original_completions_create

        if self._original_async_completions_create is not None:
            chat_complete_module.AsyncCompletions.create = self._original_async_completions_create

        if self._original_prompt_completions_create is not None:
            generation_module.Completions.create = self._original_prompt_completions_create

        if self._original_async_prompt_completions_create is not None:
            generation_module.AsyncCompletions.create = (
                self._original_async_prompt_completions_create
            )
