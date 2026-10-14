import logging
from importlib import import_module
from typing import Any, Collection, List, Tuple

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import (  # type: ignore[attr-defined]
    BaseInstrumentor,
)
from wrapt import wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.anthropic._wrappers import (
    _AsyncCountTokensWrapper,
    _AsyncMessagesStreamWrapper,
    _AsyncMessageStreamManager,
    _AsyncMessagesWrapper,
    _AsyncPrepareRequestDataWrapper,
    _BetaAsyncMessageStreamManager,
    _BetaMessageStreamManager,
    _CountTokensWrapper,
    _MessagesStreamWrapper,
    _MessageStreamManager,
    _MessagesWrapper,
    _PrepareRequestDataWrapper,
    _WithTracer,
)
from openinference.instrumentation.anthropic.version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_instruments = ("anthropic >= 1.8.0",)

# The modules of the beta messages resources of the Bedrock and Vertex AI clients. Their classes
# copy the first-party methods when they are defined, so wrapping the first-party classes misses
# them, and they are wrapped as well.
_PLATFORM_BETA_MESSAGES_MODULES = (
    "anthropic.lib.bedrock._beta_messages",
    "anthropic.lib.vertex._beta_messages",
)


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
        "_original_prepare_request_data",
        "_original_async_prepare_request_data",
        "_originals",
        "_config",
        "_instruments",
        "_tracer",
    )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        from anthropic.resources.beta.messages import AsyncMessages as AsyncBetaMessages
        from anthropic.resources.beta.messages import Messages as BetaMessages
        from anthropic.resources.messages import AsyncMessages, Messages

        # before the first-party methods are wrapped, so that the platform classes copy the
        # originals if their modules have not been imported yet
        platform_beta_messages_modules = _import_platform_beta_messages_modules()
        # the original attributes of the classes wrapped by _wrap, to restore when uninstrumenting
        self._originals: List[Tuple[Any, str, Any]] = []

        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        if not (config := kwargs.get("config")):
            config = TraceConfig()
        else:
            assert isinstance(config, TraceConfig)
        # for the attributes that the tracer's masking does not cover
        self._config = config
        self._tracer = OITracer(
            trace_api.get_tracer(__name__, __version__, tracer_provider),
            config=config,
        )

        self._original_messages_create = Messages.create
        wrap_function_wrapper(
            "anthropic.resources.messages",
            "Messages.create",
            _MessagesWrapper(
                config=self._config,
                tracer=self._tracer,  # type: ignore[arg-type]
                span_name="messages.create",
            ),
        )

        self._original_async_messages_create = AsyncMessages.create
        wrap_function_wrapper(
            "anthropic.resources.messages",
            "AsyncMessages.create",
            _AsyncMessagesWrapper(
                config=self._config,
                tracer=self._tracer,  # type: ignore[arg-type]
                span_name="messages.create",
            ),
        )

        self._original_messages_stream = Messages.stream
        wrap_function_wrapper(
            "anthropic.resources.messages",
            "Messages.stream",
            _MessagesStreamWrapper(
                config=self._config,
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
                config=self._config,
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
                config=self._config,
                tracer=self._tracer,  # type: ignore[arg-type]
                span_name="messages.parse",
            ),
        )

        self._original_async_messages_parse = AsyncMessages.parse
        wrap_function_wrapper(
            "anthropic.resources.messages",
            "AsyncMessages.parse",
            _AsyncMessagesWrapper(
                config=self._config,
                tracer=self._tracer,  # type: ignore[arg-type]
                span_name="messages.parse",
            ),
        )

        self._original_beta_messages_create = BetaMessages.create
        wrap_function_wrapper(
            "anthropic.resources.beta.messages",
            "Messages.create",
            _MessagesWrapper(
                config=self._config,
                tracer=self._tracer,  # type: ignore[arg-type]
                span_name="beta.messages.create",
            ),
        )

        self._original_async_beta_messages_create = AsyncBetaMessages.create
        wrap_function_wrapper(
            "anthropic.resources.beta.messages",
            "AsyncMessages.create",
            _AsyncMessagesWrapper(
                config=self._config,
                tracer=self._tracer,  # type: ignore[arg-type]
                span_name="beta.messages.create",
            ),
        )

        self._original_beta_messages_stream = BetaMessages.stream
        wrap_function_wrapper(
            "anthropic.resources.beta.messages",
            "Messages.stream",
            _MessagesStreamWrapper(
                config=self._config,
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
                config=self._config,
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
                config=self._config,
                tracer=self._tracer,  # type: ignore[arg-type]
                span_name="beta.messages.parse",
            ),
        )

        self._original_async_beta_messages_parse = AsyncBetaMessages.parse
        wrap_function_wrapper(
            "anthropic.resources.beta.messages",
            "AsyncMessages.parse",
            _AsyncMessagesWrapper(
                config=self._config,
                tracer=self._tracer,  # type: ignore[arg-type]
                span_name="beta.messages.parse",
            ),
        )

        for cls, async_cls, span_name in [
            (Messages, AsyncMessages, "messages.count_tokens"),
            (BetaMessages, AsyncBetaMessages, "beta.messages.count_tokens"),
        ]:
            self._wrap_count_tokens(cls, async_cls, span_name)

        for module in platform_beta_messages_modules:
            self._wrap_platform_beta_messages(module)

        self._wrap_request_preparation()

    def _wrap(self, cls: Any, name: str, wrapper: Any) -> None:
        # only a method of the class itself, e.g. not one a platform class lacks and inherits
        if (original := cls.__dict__.get(name)) is None:
            return
        wrap_function_wrapper(cls, name, wrapper)
        self._originals.append((cls, name, original))

    def _wrap_count_tokens(self, cls: Any, async_cls: Any, span_name: str) -> None:
        # an OITracer, which the wrappers' annotations do not name
        tracer: Any = self._tracer
        self._wrap(
            cls,
            "count_tokens",
            _CountTokensWrapper(config=self._config, tracer=tracer, span_name=span_name),
        )
        self._wrap(
            async_cls,
            "count_tokens",
            _AsyncCountTokensWrapper(config=self._config, tracer=tracer, span_name=span_name),
        )

    def _wrap_platform_beta_messages(self, module: Any) -> None:
        # an OITracer, which the wrappers' annotations do not name
        tracer: Any = self._tracer
        sync_cls, async_cls = module.Messages, module.AsyncMessages
        for method, span_name in [
            ("create", "beta.messages.create"),
            ("parse", "beta.messages.parse"),
        ]:
            self._wrap(
                sync_cls,
                method,
                _MessagesWrapper(config=self._config, tracer=tracer, span_name=span_name),
            )
            self._wrap(
                async_cls,
                method,
                _AsyncMessagesWrapper(config=self._config, tracer=tracer, span_name=span_name),
            )
        self._wrap(
            sync_cls,
            "stream",
            _MessagesStreamWrapper(
                config=self._config,
                tracer=tracer,
                span_name="beta.messages.stream",
                manager_class=_BetaMessageStreamManager,  # type: ignore[arg-type]
            ),
        )
        self._wrap(
            async_cls,
            "stream",
            _AsyncMessagesStreamWrapper(
                config=self._config,
                tracer=tracer,
                span_name="beta.messages.stream",
                manager_class=_BetaAsyncMessageStreamManager,  # type: ignore[arg-type]
            ),
        )
        self._wrap_count_tokens(sync_cls, async_cls, "beta.messages.count_tokens")

    def _wrap_request_preparation(self) -> None:
        from anthropic import _base_client

        self._original_prepare_request_data = None
        self._original_async_prepare_request_data = None
        try:
            original = getattr(_base_client, "prepare_request_data")
            async_original = getattr(_base_client, "async_prepare_request_data")
        except AttributeError:
            logger.warning(
                "Could not find Anthropic request preparation functions. Some invocation "
                "parameters may be missing from LLM spans."
            )
            return
        wrap_function_wrapper(_base_client, "prepare_request_data", _PrepareRequestDataWrapper())
        wrap_function_wrapper(
            _base_client, "async_prepare_request_data", _AsyncPrepareRequestDataWrapper()
        )
        self._original_prepare_request_data = original
        self._original_async_prepare_request_data = async_original

    def _uninstrument(self, **kwargs: Any) -> None:
        from anthropic import _base_client
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

        for cls, name, original in reversed(getattr(self, "_originals", ())):
            setattr(cls, name, original)
        self._originals = []

        # cached accessors (like with_raw_response) keep the wrapped method
        for wrapper in list(_WithTracer._instances):
            wrapper._active = False

        if self._original_prepare_request_data is not None:
            setattr(_base_client, "prepare_request_data", self._original_prepare_request_data)
        if self._original_async_prepare_request_data is not None:
            setattr(
                _base_client,
                "async_prepare_request_data",
                self._original_async_prepare_request_data,
            )


def _import_platform_beta_messages_modules() -> List[Any]:
    modules = []
    for name in _PLATFORM_BETA_MESSAGES_MODULES:
        try:
            modules.append(import_module(name))
        except Exception:
            logger.debug("Could not import %s, so it is not instrumented", name, exc_info=True)
    return modules
