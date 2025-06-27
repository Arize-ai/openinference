import logging
from typing import Any, Collection, List, Tuple

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import (  # type: ignore[attr-defined]
    BaseInstrumentor,
)
from wrapt import resolve_path, wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.autogen_agentchat.version import __version__

_instruments = ("autogen-agentchat >= 0.5.1",)

logger = logging.getLogger(__name__)


class AutogenAgentChatInstrumentor(BaseInstrumentor):  # type: ignore
    """An instrumentor for autogen-agentchat"""

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

        from autogen_ext.models.openai import BaseOpenAIChatCompletionClient

        from autogen_agentchat.agents import AssistantAgent, BaseChatAgent
        from autogen_agentchat.teams import BaseGroupChat
        from openinference.instrumentation.autogen_agentchat._wrappers import (
            _AssistantAgentOnMessagesStreamWrapper,
            _BaseChatAgentOnMessagesStreamWrapper,
            _BaseGroupChatRunStreamWrapper,
            _BaseOpenAIChatCompletionClientCreateStreamWrapper,
            _BaseOpenAIChatCompletionClientCreateWrapper,
        )

        self._originals: List[Tuple[Any, Any, Any]] = []

        method_wrappers: dict[Any, Any] = {
            AssistantAgent.on_messages_stream: _AssistantAgentOnMessagesStreamWrapper(self._tracer),
            BaseChatAgent.on_messages_stream: _BaseChatAgentOnMessagesStreamWrapper(self._tracer),
            BaseGroupChat.run_stream: _BaseGroupChatRunStreamWrapper(self._tracer),
            BaseOpenAIChatCompletionClient.create: _BaseOpenAIChatCompletionClientCreateWrapper(
                self._tracer
            ),
            BaseOpenAIChatCompletionClient.create_stream: (
                _BaseOpenAIChatCompletionClientCreateStreamWrapper(self._tracer)
            ),
        }

        for method, wrapper in method_wrappers.items():
            module, name = method.__module__, method.__qualname__
            self._originals.append(resolve_path(module, name))
            wrap_function_wrapper(module, name, wrapper)

    def _uninstrument(self, **kwargs: Any) -> None:
        for parent, attribute, original in getattr(self, "_originals", ()):
            setattr(parent, attribute, original)
