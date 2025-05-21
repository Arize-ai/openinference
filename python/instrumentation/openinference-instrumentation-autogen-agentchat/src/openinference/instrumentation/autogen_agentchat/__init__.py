import logging
from importlib import import_module
from typing import Any, Collection

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import (  # type: ignore[attr-defined]
    BaseInstrumentor,
)
from wrapt import wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.autogen_agentchat._wrappers import (
    _BaseAgentRunWrapper,
    _BaseGroupChatRunStreamWrapper,
    _PublishMessageWrapper,
    _SendMessageWrapper,
    _ToolsRunJSONWrapper,
)
from openinference.instrumentation.autogen_agentchat.version import __version__

_instruments = ("autogen-agentchat >= 0.5.1",)

logger = logging.getLogger(__name__)


class AutogenAgentChatInstrumentor(BaseInstrumentor):  # type: ignore
    """An instrumentor for autogen-agentchat"""

    __slots__ = (
        "_tracer",
        "_original_base_agent_run_method",
        "_original_tools_run_json",
        "_original_group_chat_run_stream_method",
        "_original_single_threaded_agent_runtime_send_message",
        "_original_single_threaded_agent_runtime_publish_message",
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

        self._original_base_agent_run_method = getattr(
            import_module("autogen_agentchat.agents").BaseChatAgent, "run", None
        )
        wrap_function_wrapper(
            module="autogen_agentchat.agents",
            name="BaseChatAgent.run",
            wrapper=_BaseAgentRunWrapper(tracer=self._tracer),
        )

        self._original_tools_run_json = getattr(
            import_module("autogen_core.tools").BaseTool, "run_json", None
        )
        wrap_function_wrapper(
            module="autogen_core.tools",
            name="BaseTool.run_json",
            wrapper=_ToolsRunJSONWrapper(tracer=self._tracer),
        )

        self._original_group_chat_run_stream_method = getattr(
            import_module("autogen_agentchat.teams._group_chat._base_group_chat").BaseGroupChat,
            "run_stream",
            None,
        )
        wrap_function_wrapper(
            module="autogen_agentchat.teams._group_chat._base_group_chat",
            name="BaseGroupChat.run_stream",
            wrapper=_BaseGroupChatRunStreamWrapper(tracer=self._tracer),
        )

        self._original_single_threaded_agent_runtime_send_message = getattr(
            import_module("autogen_core._single_threaded_agent_runtime").SingleThreadedAgentRuntime,
            "send_message",
            None,
        )
        wrap_function_wrapper(
            module="autogen_core._single_threaded_agent_runtime",
            name="SingleThreadedAgentRuntime.send_message",
            wrapper=_SendMessageWrapper(tracer=self._tracer),
        )

        self._original_single_threaded_agent_runtime_publish_message = getattr(
            import_module("autogen_core._single_threaded_agent_runtime").SingleThreadedAgentRuntime,
            "publish_message",
            None,
        )
        wrap_function_wrapper(
            module="autogen_core._single_threaded_agent_runtime",
            name="SingleThreadedAgentRuntime.publish_message",
            wrapper=_PublishMessageWrapper(tracer=self._tracer),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        if self._original_base_agent_run_method is not None:
            agent_module = import_module("autogen_agentchat.agents")
            agent_module.BaseChatAgent.run = self._original_base_agent_run_method
            self._original_base_agent_run_method = None

        if self._original_tools_run_json is not None:
            tools_module = import_module("autogen_core.tools")
            tools_module.BaseTool.run_json = self._original_tools_run_json
            self._original_tools_run_json = None

        if self._original_group_chat_run_stream_method is not None:
            group_chat_module = import_module(
                "autogen_agentchat.teams._group_chat._base_group_chat"
            )
            group_chat_module.BaseGroupChat.run_stream = self._original_group_chat_run_stream_method
            self._original_group_chat_run_stream_method = None

        if self._original_single_threaded_agent_runtime_send_message is not None:
            single_threaded_agent_runtime_module = import_module(
                "autogen_core._single_threaded_agent_runtime"
            )
            single_threaded_agent_runtime_module.SingleThreadedAgentRuntime.send_message = (
                self._original_single_threaded_agent_runtime_send_message
            )
            self._original_single_threaded_agent_runtime_send_message = None

        if self._original_single_threaded_agent_runtime_publish_message is not None:
            single_threaded_agent_runtime_module = import_module(
                "autogen_core._single_threaded_agent_runtime"
            )
            single_threaded_agent_runtime_module.SingleThreadedAgentRuntime.publish_message = (
                self._original_single_threaded_agent_runtime_publish_message
            )
            self._original_single_threaded_agent_runtime_publish_message = None
