import logging
from typing import Any, Collection

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import (  # type: ignore[attr-defined]
    BaseInstrumentor,
)
from wrapt.patches import wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.anthropic._wrappers import (
    _AsyncCompletionsWrapper,
    _AsyncMessagesStreamWrapper,
    _AsyncMessageStreamManager,
    _AsyncMessagesWrapper,
    _AsyncTransformWrapper,
    _BetaAsyncMessageStreamManager,
    _BetaMessageStreamManager,
    _CompletionsWrapper,
    _MessagesStreamWrapper,
    _MessageStreamManager,
    _MessagesWrapper,
    _TransformWrapper,
)
from openinference.instrumentation.anthropic.version import __version__

logger = logging.getLogger(__name__)

_instruments = ("anthropic >= 0.84.0",)


class AnthropicInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """An instrumentor for the Anthropic framework."""

    __slots__ = (
        "_original_completions_create",
        "_original_async_completions_create",
        "_original_messages_create",
        "_original_async_messages_create",
        "_original_messages_stream",
        "_original_async_messages_stream",
        "_original_messages_parse",
        "_original_async_messages_parse",
        "_original_beta_messages_create",
        "_original_async_beta_messages_create",
        "_original_beta_messages_stream",
        "_original_async_beta_messages_stream",
        "_original_beta_messages_parse",
        "_original_async_beta_messages_parse",
        "_original_transform",
        "_original_async_transform",
        "_instruments",
        "_tracer",
    )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        from anthropic.resources.beta.messages import AsyncMessages as AsyncBetaMessages
        from anthropic.resources.beta.messages import Messages as BetaMessages
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
        wrap_function_wrapper(  # type: ignore[no-untyped-call]
            "anthropic.resources.completions",
            "Completions.create",
            _CompletionsWrapper(
                tracer=self._tracer,  # type: ignore[arg-type]
                span_name="completions.create",
            ),
        )

        self._original_async_completions_create = AsyncCompletions.create
        wrap_function_wrapper(  # type: ignore[no-untyped-call]
            "anthropic.resources.completions",
            "AsyncCompletions.create",
            _AsyncCompletionsWrapper(
                tracer=self._tracer,  # type: ignore[arg-type]
                span_name="completions.create",
            ),
        )

        self._original_messages_create = Messages.create
        wrap_function_wrapper(  # type: ignore[no-untyped-call]
            "anthropic.resources.messages",
            "Messages.create",
            _MessagesWrapper(
                tracer=self._tracer,  # type: ignore[arg-type]
                span_name="messages.create",
            ),
        )

        self._original_async_messages_create = AsyncMessages.create
        wrap_function_wrapper(  # type: ignore[no-untyped-call]
            "anthropic.resources.messages",
            "AsyncMessages.create",
            _AsyncMessagesWrapper(
                tracer=self._tracer,  # type: ignore[arg-type]
                span_name="messages.create",
            ),
        )

        self._original_messages_stream = Messages.stream
        wrap_function_wrapper(  # type: ignore[no-untyped-call]
            "anthropic.resources.messages",
            "Messages.stream",
            _MessagesStreamWrapper(
                tracer=self._tracer,  # type: ignore[arg-type]
                span_name="messages.stream",
                manager_class=_MessageStreamManager,
            ),
        )

        self._original_async_messages_stream = AsyncMessages.stream
        wrap_function_wrapper(  # type: ignore[no-untyped-call]
            "anthropic.resources.messages",
            "AsyncMessages.stream",
            _AsyncMessagesStreamWrapper(
                tracer=self._tracer,  # type: ignore[arg-type]
                span_name="messages.stream",
                manager_class=_AsyncMessageStreamManager,
            ),
        )

        self._original_messages_parse = Messages.parse
        wrap_function_wrapper(  # type: ignore[no-untyped-call]
            "anthropic.resources.messages",
            "Messages.parse",
            _MessagesWrapper(
                tracer=self._tracer,  # type: ignore[arg-type]
                span_name="messages.parse",
            ),
        )

        self._original_async_messages_parse = AsyncMessages.parse
        wrap_function_wrapper(  # type: ignore[no-untyped-call]
            "anthropic.resources.messages",
            "AsyncMessages.parse",
            _AsyncMessagesWrapper(
                tracer=self._tracer,  # type: ignore[arg-type]
                span_name="messages.parse",
            ),
        )

        self._original_beta_messages_create = BetaMessages.create
        wrap_function_wrapper(  # type: ignore[no-untyped-call]
            "anthropic.resources.beta.messages",
            "Messages.create",
            _MessagesWrapper(
                tracer=self._tracer,  # type: ignore[arg-type]
                span_name="beta.messages.create",
            ),
        )

        self._original_async_beta_messages_create = AsyncBetaMessages.create
        wrap_function_wrapper(  # type: ignore[no-untyped-call]
            "anthropic.resources.beta.messages",
            "AsyncMessages.create",
            _AsyncMessagesWrapper(
                tracer=self._tracer,  # type: ignore[arg-type]
                span_name="beta.messages.create",
            ),
        )

        self._original_beta_messages_stream = BetaMessages.stream
        wrap_function_wrapper(  # type: ignore[no-untyped-call]
            "anthropic.resources.beta.messages",
            "Messages.stream",
            _MessagesStreamWrapper(
                tracer=self._tracer,  # type: ignore[arg-type]
                span_name="beta.messages.stream",
                manager_class=_BetaMessageStreamManager,  # type: ignore[arg-type]
            ),
        )

        self._original_async_beta_messages_stream = AsyncBetaMessages.stream
        wrap_function_wrapper(  # type: ignore[no-untyped-call]
            "anthropic.resources.beta.messages",
            "AsyncMessages.stream",
            _AsyncMessagesStreamWrapper(
                tracer=self._tracer,  # type: ignore[arg-type]
                span_name="beta.messages.stream",
                manager_class=_BetaAsyncMessageStreamManager,  # type: ignore[arg-type]
            ),
        )

        self._original_beta_messages_parse = BetaMessages.parse
        wrap_function_wrapper(  # type: ignore[no-untyped-call]
            "anthropic.resources.beta.messages",
            "Messages.parse",
            _MessagesWrapper(
                tracer=self._tracer,  # type: ignore[arg-type]
                span_name="beta.messages.parse",
            ),
        )

        self._original_async_beta_messages_parse = AsyncBetaMessages.parse
        wrap_function_wrapper(  # type: ignore[no-untyped-call]
            "anthropic.resources.beta.messages",
            "AsyncMessages.parse",
            _AsyncMessagesWrapper(
                tracer=self._tracer,  # type: ignore[arg-type]
                span_name="beta.messages.parse",
            ),
        )

        import anthropic._utils._transform as _transform_module

        self._original_transform = _transform_module.transform
        wrap_function_wrapper(  # type: ignore[no-untyped-call]
            "anthropic._utils._transform",
            "transform",
            _TransformWrapper(),
        )

        self._original_async_transform = _transform_module.async_transform
        wrap_function_wrapper(  # type: ignore[no-untyped-call]
            "anthropic._utils._transform",
            "async_transform",
            _AsyncTransformWrapper(),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        import anthropic._utils._transform as _transform_module
        from anthropic.resources.beta.messages import AsyncMessages as AsyncBetaMessages
        from anthropic.resources.beta.messages import Messages as BetaMessages
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

        if self._original_messages_stream is not None:
            Messages.stream = self._original_messages_stream  # type: ignore[method-assign]
        if self._original_async_messages_stream is not None:
            AsyncMessages.stream = self._original_async_messages_stream  # type: ignore[method-assign]

        if self._original_messages_parse is not None:
            Messages.parse = self._original_messages_parse  # type: ignore[method-assign]
        if self._original_async_messages_parse is not None:
            AsyncMessages.parse = self._original_async_messages_parse  # type: ignore[method-assign]

        if self._original_beta_messages_create is not None:
            BetaMessages.create = self._original_beta_messages_create  # type: ignore[method-assign]
        if self._original_async_beta_messages_create is not None:
            AsyncBetaMessages.create = self._original_async_beta_messages_create  # type: ignore[method-assign]

        if self._original_beta_messages_stream is not None:
            BetaMessages.stream = self._original_beta_messages_stream  # type: ignore[method-assign]
        if self._original_async_beta_messages_stream is not None:
            AsyncBetaMessages.stream = self._original_async_beta_messages_stream  # type: ignore[method-assign]

        if self._original_beta_messages_parse is not None:
            BetaMessages.parse = self._original_beta_messages_parse  # type: ignore[method-assign]
        if self._original_async_beta_messages_parse is not None:
            AsyncBetaMessages.parse = self._original_async_beta_messages_parse  # type: ignore[method-assign]

        if self._original_transform is not None:
            _transform_module.transform = self._original_transform
        if self._original_async_transform is not None:
            _transform_module.async_transform = self._original_async_transform
