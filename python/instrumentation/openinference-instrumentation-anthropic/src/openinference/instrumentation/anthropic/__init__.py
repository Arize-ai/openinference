import logging
import re
from importlib import import_module
from typing import Any, Callable, Collection, NamedTuple, Optional, Tuple

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
    _AsyncPrepareRequestDataWrapper,
    _AsyncTransformWrapper,
    _BetaAsyncMessageStreamManager,
    _BetaMessageStreamManager,
    _MessagesStreamWrapper,
    _MessageStreamManager,
    _MessagesWrapper,
    _PrepareRequestDataWrapper,
    _TransformWrapper,
)
from openinference.instrumentation.anthropic.version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_instruments = ("anthropic >= 1.0.0",)


class _RequestPreparation(NamedTuple):
    """
    The private anthropic functions that prepare a request body, patched to record the body as
    sent, e.g. with the ``stream`` flag that messages.stream() adds, as invocation parameters.
    """

    module: str  # the module the SDK looks the functions up in, so the one to patch
    sync_name: str
    async_name: str
    sync_wrapper: Callable[..., Any]
    async_wrapper: Callable[..., Any]


# anthropic<1.8.0: the resource methods prepare the body when they build the request, through
# anthropic._utils._transform.maybe_transform, which calls transform as a module global.
_TRANSFORM = _RequestPreparation(
    "anthropic._utils._transform",
    "transform",
    "async_transform",
    _TransformWrapper(),
    _AsyncTransformWrapper(),
)

# anthropic>=1.8.0: anthropic._utils._prepare.prepare_request_data, which prepares query
# parameters too, called when the request is sent from anthropic._base_client, which imports it
# as a module global.
_PREPARE_REQUEST_DATA = _RequestPreparation(
    "anthropic._base_client",
    "prepare_request_data",
    "async_prepare_request_data",
    _PrepareRequestDataWrapper(),
    _AsyncPrepareRequestDataWrapper(),
)
_PREPARE_REQUEST_DATA_VERSION = (1, 8, 0)


def _get_anthropic_version() -> Optional[Tuple[int, int, int]]:
    """
    The version of the anthropic code that is imported, or None if it cannot be parsed. Any
    pre-release suffix is ignored, so 1.8.0rc1 counts as 1.8.0.
    """
    from anthropic import __version__ as anthropic_version

    if (match := re.match(r"(\d+)\.(\d+)\.(\d+)", anthropic_version)) is None:
        return None
    return int(match[1]), int(match[2]), int(match[3])


def _get_request_preparation() -> Optional[_RequestPreparation]:
    if (anthropic_version := _get_anthropic_version()) is None:
        return None
    if anthropic_version >= _PREPARE_REQUEST_DATA_VERSION:
        return _PREPARE_REQUEST_DATA
    return _TRANSFORM


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
        "_request_preparation",
        "_original_request_preparation",
        "_original_async_request_preparation",
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

        self._wrap_request_preparation()

    def _wrap_request_preparation(self) -> None:
        """
        The patch only enriches the invocation parameters, so failing to apply it must not stop
        the instrumentation.
        """
        from anthropic import __version__ as anthropic_version

        self._request_preparation = None
        self._original_request_preparation = None
        self._original_async_request_preparation = None
        if (request_preparation := _get_request_preparation()) is None:
            logger.warning(
                "Could not parse the anthropic version %r. Some invocation parameters may be "
                "missing from LLM spans.",
                anthropic_version,
            )
            return
        try:
            module = import_module(request_preparation.module)
            original = getattr(module, request_preparation.sync_name)
            async_original = getattr(module, request_preparation.async_name)
        except (ImportError, AttributeError):
            logger.warning(
                "Could not find %s.%s in anthropic %s. Some invocation parameters may be "
                "missing from LLM spans.",
                request_preparation.module,
                request_preparation.sync_name,
                anthropic_version,
            )
            return
        wrap_function_wrapper(
            module, request_preparation.sync_name, request_preparation.sync_wrapper
        )
        wrap_function_wrapper(
            module, request_preparation.async_name, request_preparation.async_wrapper
        )
        self._request_preparation = request_preparation
        self._original_request_preparation = original
        self._original_async_request_preparation = async_original

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

        if self._request_preparation is not None:
            module = import_module(self._request_preparation.module)
            setattr(module, self._request_preparation.sync_name, self._original_request_preparation)
            setattr(
                module,
                self._request_preparation.async_name,
                self._original_async_request_preparation,
            )
            self._request_preparation = None
