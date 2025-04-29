import logging
from typing import Any, Collection, Tuple, Dict

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import (  # type: ignore[attr-defined]
    BaseInstrumentor,
)

from wrapt import wrap_function_wrapper
from openinference.instrumentation import (
    OITracer,
    TraceConfig,
)

from ._wrappers import _AssistantAgentWrapper
from .version import __version__

_instruments = ("autogen-agentchat >= 0.5.1",)

logger = logging.getLogger(__name__)


class AutogenInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """An instrumentor for the Autogen framework."""

    __slots__ = ("_tracer",)

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
        from autogen_agentchat.agents import AssistantAgent
        self._original_on_messages = AssistantAgent.on_messages
        self._original_call_llm = AssistantAgent._call_llm
        self._original_execute_tool = AssistantAgent._execute_tool_call

        # Create wrapper instance
        wrapper = _AssistantAgentWrapper(tracer=self._tracer)

        # Wrap AssistantAgent methods
        wrap_function_wrapper(
            module="autogen_agentchat.agents",
            name="AssistantAgent.on_messages",
            wrapper=wrapper.on_messages_wrapper,
        )
        wrap_function_wrapper(
            module="autogen_agentchat.agents",
            name="AssistantAgent._execute_tool_call",
            wrapper=wrapper.execute_tool_wrapper,
        )
        wrap_function_wrapper(
            module="autogen_agentchat.agents",
            name="AssistantAgent._call_llm",
            wrapper=wrapper.call_llm_wrapper,
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        from autogen_agentchat.agents import AssistantAgent
        if self._original_on_messages is not None:
            AssistantAgent.on_messages = self._original_on_messages
        if self._original_call_llm is not None:
            AssistantAgent.call_llm = self._original_call_llm
        if self._original_execute_tool is not None:
            AssistantAgent.execute_tool = self._original_execute_tool