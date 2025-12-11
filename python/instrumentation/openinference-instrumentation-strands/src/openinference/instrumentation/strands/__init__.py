import logging
from typing import Any, Collection

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import (  # type: ignore[attr-defined]
    BaseInstrumentor,
)
from wrapt import wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.strands._wrappers import (
    _AgentInvokeAsyncWrapper,
    _AgentStreamAsyncWrapper,
    _EventLoopCycleWrapper,
    _ToolExecutorExecuteWrapper,
    _ToolStreamWrapper,
)
from openinference.instrumentation.strands.package import _instruments
from openinference.instrumentation.strands.version import __version__

logger = logging.getLogger(__name__)


class StrandsInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """An instrumentor for the Strands Agents framework.

    This instrumentor provides automatic tracing for Strands Agents, capturing:
    - Agent invocations (streaming and non-streaming)
    - Event loop cycles
    - Tool executions
    - Multi-agent interactions

    Example:
        ```python
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from openinference.instrumentation.strands import StrandsInstrumentor

        # Set up tracer provider
        tracer_provider = TracerProvider()
        trace.set_tracer_provider(tracer_provider)

        # Instrument Strands
        StrandsInstrumentor().instrument()

        # Use Strands as normal
        from strands import Agent
        agent = Agent()
        result = agent("Hello, world!")
        ```
    """

    __slots__ = (
        "_original_agent_invoke_async",
        "_original_agent_stream_async",
        "_original_event_loop_cycle",
        "_original_tool_executor_execute",
        "_original_agent_tool_stream",
        "_tracer",
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

        # Instrument Agent.invoke_async
        try:
            from strands.agent.agent import Agent

            self._original_agent_invoke_async = Agent.invoke_async
            wrap_function_wrapper(
                module="strands.agent.agent",
                name="Agent.invoke_async",
                wrapper=_AgentInvokeAsyncWrapper(tracer=self._tracer),
            )
            logger.debug("instrumented Agent.invoke_async")
        except Exception as e:
            logger.warning("failed to instrument Agent.invoke_async: %s", e)

        # Instrument Agent.stream_async
        try:
            from strands.agent.agent import Agent

            self._original_agent_stream_async = Agent.stream_async
            wrap_function_wrapper(
                module="strands.agent.agent",
                name="Agent.stream_async",
                wrapper=_AgentStreamAsyncWrapper(tracer=self._tracer),
            )
            logger.debug("instrumented Agent.stream_async")
        except Exception as e:
            logger.warning("failed to instrument Agent.stream_async: %s", e)

        # Instrument event_loop_cycle
        try:
            from strands.event_loop import event_loop

            self._original_event_loop_cycle = event_loop.event_loop_cycle
            wrap_function_wrapper(
                module="strands.event_loop.event_loop",
                name="event_loop_cycle",
                wrapper=_EventLoopCycleWrapper(tracer=self._tracer),
            )
            logger.debug("instrumented event_loop_cycle")
        except Exception as e:
            logger.warning("failed to instrument event_loop_cycle: %s", e)

        # Instrument ToolExecutor._execute
        try:
            from strands.tools.executors._executor import ToolExecutor

            self._original_tool_executor_execute = ToolExecutor._execute
            wrap_function_wrapper(
                module="strands.tools.executors._executor",
                name="ToolExecutor._execute",
                wrapper=_ToolExecutorExecuteWrapper(tracer=self._tracer),
            )
            logger.debug("instrumented ToolExecutor._execute")
        except Exception as e:
            logger.warning("failed to instrument ToolExecutor._execute: %s", e)

        # Instrument AgentTool.stream for individual tool tracing
        try:
            from strands.tools.tools import AgentTool  # type: ignore[attr-defined]

            self._original_agent_tool_stream = AgentTool.stream
            wrap_function_wrapper(
                module="strands.tools.tools",
                name="AgentTool.stream",
                wrapper=_ToolStreamWrapper(tracer=self._tracer),
            )
            logger.debug("instrumented AgentTool.stream")
        except Exception as e:
            logger.warning("failed to instrument AgentTool.stream: %s", e)

    def _uninstrument(self, **kwargs: Any) -> None:
        try:
            from strands.agent.agent import Agent

            if self._original_agent_invoke_async is not None:
                Agent.invoke_async = self._original_agent_invoke_async  # type: ignore[method-assign]
            if self._original_agent_stream_async is not None:
                Agent.stream_async = self._original_agent_stream_async  # type: ignore[method-assign]
        except Exception as e:
            logger.warning("failed to uninstrument Agent methods: %s", e)

        try:
            from strands.event_loop import event_loop

            if self._original_event_loop_cycle is not None:
                event_loop.event_loop_cycle = self._original_event_loop_cycle
        except Exception as e:
            logger.warning("failed to uninstrument event_loop_cycle: %s", e)

        try:
            from strands.tools.executors._executor import ToolExecutor

            if self._original_tool_executor_execute is not None:
                ToolExecutor._execute = self._original_tool_executor_execute  # type: ignore[method-assign]
        except Exception as e:
            logger.warning("failed to uninstrument ToolExecutor._execute: %s", e)

        try:
            from strands.tools.tools import AgentTool  # type: ignore[attr-defined]

            if self._original_agent_tool_stream is not None:
                AgentTool.stream = self._original_agent_tool_stream  # type: ignore[method-assign]
        except Exception as e:
            logger.warning("failed to uninstrument AgentTool.stream: %s", e)


__all__ = ["StrandsInstrumentor"]
