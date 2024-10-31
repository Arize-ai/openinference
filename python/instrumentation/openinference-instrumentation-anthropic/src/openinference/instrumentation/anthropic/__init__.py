import logging
from typing import Any, Collection

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import (  # type: ignore[attr-defined]
    BaseInstrumentor,
)
from wrapt import wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.anthropic._wrappers import (
    _AsyncCompletionsWrapper,
    _AsyncMessagesWrapper,
    _CompletionsWrapper,
    _MessagesWrapper,
)
from openinference.instrumentation.anthropic.version import __version__

logger = logging.getLogger(__name__)

_instruments = ("anthropic >= 0.30.0",)


class AnthropicInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """An instrumentor for the Anthropic framework."""

    __slots__ = (
        "_original_completions_create",
        "_original_async_completions_create",
        "_original_messages_create",
        "_original_async_messages_create",
        "_instruments",
        "_tracer",
    )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        from anthropic.resources.completions import AsyncCompletions, Completions
        from anthropic.resources.messages import AsyncMessages, Messages

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
            module="anthropic.resources.completions",
            name="Completions.create",
            wrapper=_CompletionsWrapper(tracer=self._tracer),
        )

        self._original_async_completions_create = AsyncCompletions.create
        wrap_function_wrapper(
            module="anthropic.resources.completions",
            name="AsyncCompletions.create",
            wrapper=_AsyncCompletionsWrapper(tracer=self._tracer),
        )

        self._original_messages_create = Messages.create
        wrap_function_wrapper(
            module="anthropic.resources.messages",
            name="Messages.create",
            wrapper=_MessagesWrapper(tracer=self._tracer),
        )

        self._original_async_messages_create = AsyncMessages.create
        wrap_function_wrapper(
            module="anthropic.resources.messages",
            name="AsyncMessages.create",
            wrapper=_AsyncMessagesWrapper(tracer=self._tracer),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        from anthropic.resources.completions import AsyncCompletions, Completions
        from anthropic.resources.messages import AsyncMessages, Messages

        if self._original_completions_create is not None:
            Completions.create = self._original_completions_create  # type: ignore[method-assign]
        if self._original_async_completions_create is not None:
            AsyncCompletions.create = self._original_async_completions_create  # type: ignore[method-assign]

        if self._original_messages_create is not None:
            Messages.create = self._original_messages_create  # type: ignore[method-assign]
        if self._original_async_messages_create is not None:
            AsyncMessages.create = self._original_async_messages_create  # type: ignore[method-assign]
