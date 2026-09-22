import logging
from importlib import import_module
from types import ModuleType
from typing import Any, Collection, Optional, Tuple

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import (  # type: ignore[attr-defined]
    BaseInstrumentor,
)
from wrapt import wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.anthropic._wrappers import (
    _AsyncMessagesStreamWrapper,
    _AsyncMessageStreamManager,
    _AsyncMessagesWrapper,
    _AsyncTransformWrapper,
    _BetaAsyncMessageStreamManager,
    _BetaMessageStreamManager,
    _MessagesStreamWrapper,
    _MessageStreamManager,
    _MessagesWrapper,
    _TransformWrapper,
)
from openinference.instrumentation.anthropic.version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_instruments = ("anthropic >= 1.0.0",)

# The private function that prepares an Anthropic request body is patched to enrich the
# recorded invocation parameters. anthropic>=1.8.0 renamed it from
# anthropic._utils._transform.transform to anthropic._utils._prepare.prepare_request_data and
# calls it from anthropic._base_client, which binds it as a module global, so each version has
# to be patched where its own call site resolves the name. Newest layout first.
_TRANSFORM_TARGETS = (
    ("anthropic._base_client", "prepare_request_data", "async_prepare_request_data"),
    ("anthropic._utils._transform", "transform", "async_transform"),
)


def _resolve_transform_target() -> Optional[Tuple[ModuleType, str, str]]:
    """
    Returns the module to patch and the names of the sync and async request-body preparation
    functions it calls, or None if this version of anthropic has neither known layout.
    """
    for module_name, transform_name, async_transform_name in _TRANSFORM_TARGETS:
        try:
            module = import_module(module_name)
        except ImportError:
            continue
        if hasattr(module, transform_name) and hasattr(module, async_transform_name):
            return module, transform_name, async_transform_name
    return None


class AnthropicInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """An instrumentor for the Anthropic framework."""

    __slots__ = (
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
        "_transform_target",
        "_instruments",
        "_tracer",
    )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        from anthropic.resources.beta.messages import AsyncMessages as AsyncBetaMessages
        from anthropic.resources.beta.messages import Messages as BetaMessages
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

        self._original_messages_create = Messages.create
        wrap_function_wrapper(
            "anthropic.resources.messages",
            "Messages.create",
            _MessagesWrapper(
                tracer=self._tracer,  # type: ignore[arg-type]
                span_name="messages.create",
            ),
        )

        self._original_async_messages_create = AsyncMessages.create
        wrap_function_wrapper(
            "anthropic.resources.messages",
            "AsyncMessages.create",
            _AsyncMessagesWrapper(
                tracer=self._tracer,  # type: ignore[arg-type]
                span_name="messages.create",
            ),
        )

        self._original_messages_stream = Messages.stream
        wrap_function_wrapper(
            "anthropic.resources.messages",
            "Messages.stream",
            _MessagesStreamWrapper(
                tracer=self._tracer,  # type: ignore[arg-type]
                span_name="messages.stream",
                manager_class=_MessageStreamManager,
            ),
        )

        self._original_async_messages_stream = AsyncMessages.stream
        wrap_function_wrapper(
            "anthropic.resources.messages",
            "AsyncMessages.stream",
            _AsyncMessagesStreamWrapper(
                tracer=self._tracer,  # type: ignore[arg-type]
                span_name="messages.stream",
                manager_class=_AsyncMessageStreamManager,
            ),
        )

        self._original_messages_parse = Messages.parse
        wrap_function_wrapper(
            "anthropic.resources.messages",
            "Messages.parse",
            _MessagesWrapper(
                tracer=self._tracer,  # type: ignore[arg-type]
                span_name="messages.parse",
            ),
        )

        self._original_async_messages_parse = AsyncMessages.parse
        wrap_function_wrapper(
            "anthropic.resources.messages",
            "AsyncMessages.parse",
            _AsyncMessagesWrapper(
                tracer=self._tracer,  # type: ignore[arg-type]
                span_name="messages.parse",
            ),
        )

        self._original_beta_messages_create = BetaMessages.create
        wrap_function_wrapper(
            "anthropic.resources.beta.messages",
            "Messages.create",
            _MessagesWrapper(
                tracer=self._tracer,  # type: ignore[arg-type]
                span_name="beta.messages.create",
            ),
        )

        self._original_async_beta_messages_create = AsyncBetaMessages.create
        wrap_function_wrapper(
            "anthropic.resources.beta.messages",
            "AsyncMessages.create",
            _AsyncMessagesWrapper(
                tracer=self._tracer,  # type: ignore[arg-type]
                span_name="beta.messages.create",
            ),
        )

        self._original_beta_messages_stream = BetaMessages.stream
        wrap_function_wrapper(
            "anthropic.resources.beta.messages",
            "Messages.stream",
            _MessagesStreamWrapper(
                tracer=self._tracer,  # type: ignore[arg-type]
                span_name="beta.messages.stream",
                manager_class=_BetaMessageStreamManager,  # type: ignore[arg-type]
            ),
        )

        self._original_async_beta_messages_stream = AsyncBetaMessages.stream
        wrap_function_wrapper(
            "anthropic.resources.beta.messages",
            "AsyncMessages.stream",
            _AsyncMessagesStreamWrapper(
                tracer=self._tracer,  # type: ignore[arg-type]
                span_name="beta.messages.stream",
                manager_class=_BetaAsyncMessageStreamManager,  # type: ignore[arg-type]
            ),
        )

        self._original_beta_messages_parse = BetaMessages.parse
        wrap_function_wrapper(
            "anthropic.resources.beta.messages",
            "Messages.parse",
            _MessagesWrapper(
                tracer=self._tracer,  # type: ignore[arg-type]
                span_name="beta.messages.parse",
            ),
        )

        self._original_async_beta_messages_parse = AsyncBetaMessages.parse
        wrap_function_wrapper(
            "anthropic.resources.beta.messages",
            "AsyncMessages.parse",
            _AsyncMessagesWrapper(
                tracer=self._tracer,  # type: ignore[arg-type]
                span_name="beta.messages.parse",
            ),
        )

        self._transform_target = _resolve_transform_target()
        self._original_transform = None
        self._original_async_transform = None
        if self._transform_target is None:
            logger.warning(
                "Could not find the request body preparation functions of this anthropic "
                "version. Some invocation parameters may be missing from LLM spans."
            )
        else:
            transform_module, transform_name, async_transform_name = self._transform_target

            self._original_transform = getattr(transform_module, transform_name)
            wrap_function_wrapper(
                transform_module.__name__,
                transform_name,
                _TransformWrapper(),
            )

            self._original_async_transform = getattr(transform_module, async_transform_name)
            wrap_function_wrapper(
                transform_module.__name__,
                async_transform_name,
                _AsyncTransformWrapper(),
            )

    def _uninstrument(self, **kwargs: Any) -> None:
        from anthropic.resources.beta.messages import AsyncMessages as AsyncBetaMessages
        from anthropic.resources.beta.messages import Messages as BetaMessages
        from anthropic.resources.messages import AsyncMessages, Messages

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

        if self._transform_target is not None:
            transform_module, transform_name, async_transform_name = self._transform_target
            if self._original_transform is not None:
                setattr(transform_module, transform_name, self._original_transform)
            if self._original_async_transform is not None:
                setattr(transform_module, async_transform_name, self._original_async_transform)
            self._transform_target = None
