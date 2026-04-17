import logging
from typing import (
    Any,
    Collection,
)

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from wrapt import wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.mistralai._chat_wrapper import (
    _AsyncChatWrapper,
    _AsyncStreamChatWrapper,
    _SyncChatWrapper,
)
from openinference.instrumentation.mistralai.package import _instruments
from openinference.instrumentation.mistralai.version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_MODULE = "mistralai"


class MistralAIInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumentor for mistralai
    """

    __slots__ = (
        "_tracer",
        "_original_sync_chat_method",
        "_original_sync_stream_chat_method",
        "_original_async_chat_method",
        "_original_async_stream_chat_method",
        "_original_sync_agent_method",
        "_original_sync_stream_agent_method",
        "_original_async_agent_method",
        "_original_async_stream_agent_method",
    )

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

        try:
            import mistralai.client as mistralai_client
            from mistralai.client.agents import Agents
            from mistralai.client.chat import Chat
        except ImportError as err:
            raise Exception(
                "Could not import mistralai. Please install with `pip install mistralai`."
            ) from err

        self._original_sync_chat_method = Chat.complete
        self._original_sync_stream_chat_method = Chat.stream
        self._original_async_chat_method = Chat.complete_async
        self._original_async_stream_chat_method = Chat.stream_async
        self._original_sync_agent_method = Agents.complete
        self._original_sync_stream_agent_method = Agents.stream
        self._original_async_agent_method = Agents.complete_async
        self._original_async_stream_agent_method = Agents.stream_async
        wrap_function_wrapper(
            "mistralai.client.chat",
            "Chat.complete",
            _SyncChatWrapper("MistralClient.chat", self._tracer, mistralai_client),
        )

        wrap_function_wrapper(
            "mistralai.client.chat",
            "Chat.stream",
            _SyncChatWrapper("MistralClient.chat", self._tracer, mistralai_client),
        )

        wrap_function_wrapper(
            "mistralai.client.chat",
            "Chat.complete_async",
            _AsyncChatWrapper("MistralAsyncClient.chat", self._tracer, mistralai_client),
        )

        wrap_function_wrapper(
            "mistralai.client.chat",
            "Chat.stream_async",
            _AsyncStreamChatWrapper("MistralAsyncClient.chat", self._tracer, mistralai_client),
        )

        wrap_function_wrapper(
            "mistralai.client.agents",
            "Agents.complete",
            _SyncChatWrapper("MistralClient.agents", self._tracer, mistralai_client),
        )

        wrap_function_wrapper(
            "mistralai.client.agents",
            "Agents.stream",
            _SyncChatWrapper("MistralClient.agents", self._tracer, mistralai_client),
        )

        wrap_function_wrapper(
            "mistralai.client.agents",
            "Agents.complete_async",
            _AsyncChatWrapper("MistralAsyncClient.agents", self._tracer, mistralai_client),
        )

        wrap_function_wrapper(
            "mistralai.client.agents",
            "Agents.stream_async",
            _AsyncStreamChatWrapper("MistralAsyncClient.agents", self._tracer, mistralai_client),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        from mistralai.client.agents import Agents
        from mistralai.client.chat import Chat

        Chat.complete = self._original_sync_chat_method  # type: ignore[method-assign]
        Chat.stream = self._original_sync_stream_chat_method  # type: ignore[method-assign]
        Chat.complete_async = self._original_async_chat_method  # type: ignore[method-assign]
        Chat.stream_async = self._original_async_stream_chat_method  # type: ignore[method-assign]
        Agents.complete = self._original_sync_agent_method  # type: ignore[method-assign]
        Agents.stream = self._original_sync_stream_agent_method  # type: ignore[method-assign]
        Agents.complete_async = self._original_async_agent_method  # type: ignore[method-assign]
        Agents.stream_async = self._original_async_stream_agent_method  # type: ignore[method-assign]
