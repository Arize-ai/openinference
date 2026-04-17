import logging
from importlib import import_module
from typing import Any, Collection

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from wrapt import wrap_function_wrapper

from openinference.instrumentation import (
    OITracer,
    TraceConfig,
)
from openinference.instrumentation.crewai._wrappers import (
    _AgentKickoffWrapper,
    _BaseToolRunWrapper,
    _CrewKickoffWrapper,
    _ExecuteCoreWrapper,
    _ExecuteWithoutTimeoutContextDescriptor,
    _FlowExecuteMethodWrapper,
    _FlowKickoffAsyncWrapper,
    _FlowKickoffWrapper,
    _LongTermMemorySaveWrapper,
    _LongTermMemorySearchWrapper,
    _ShortTermMemorySaveWrapper,
    _ShortTermMemorySearchWrapper,
)
from openinference.instrumentation.crewai.version import __version__

_instruments = ("crewai >= 1.10.1",)

logger = logging.getLogger(__name__)


class CrewAIInstrumentor(BaseInstrumentor):  # type: ignore
    __slots__ = (
        "_original_execute_core",
        "_original_crew_kickoff",
        "_original_flow_kickoff",
        "_original_flow_kickoff_async",
        "_original_flow_execute_method",
        "_original_execute_without_timeout",
        "_original_execute_single_native_tool_call",
        "_original_long_term_memory_save",
        "_original_long_term_memory_search",
        "_original_short_term_memory_save",
        "_original_short_term_memory_search",
        "_original_base_tool_run",
        "_original_agent_kickoff",
        "_tracer",
        "_event_listener",
    )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _patch_context_thread_propagation(self) -> None:
        agent_module = import_module("crewai.agent.core")
        Agent = agent_module.Agent
        self._original_execute_without_timeout = getattr(Agent, "_execute_without_timeout", None)
        if self._original_execute_without_timeout is not None:
            Agent._execute_without_timeout = _ExecuteWithoutTimeoutContextDescriptor(
                self._original_execute_without_timeout
            )

        try:
            executor_module = import_module("crewai.agents.crew_agent_executor")
        except ModuleNotFoundError:
            self._original_execute_single_native_tool_call = None
        else:
            CrewAgentExecutor = executor_module.CrewAgentExecutor
            self._original_execute_single_native_tool_call = getattr(
                CrewAgentExecutor, "_execute_single_native_tool_call", None
            )
            if self._original_execute_single_native_tool_call is not None:
                CrewAgentExecutor._execute_single_native_tool_call = (
                    _ExecuteWithoutTimeoutContextDescriptor(
                        self._original_execute_single_native_tool_call
                    )
                )

    def _restore_context_thread_propagation(self) -> None:
        if getattr(self, "_original_execute_without_timeout", None) is not None:
            agent_module = import_module("crewai.agent.core")
            agent_module.Agent._execute_without_timeout = self._original_execute_without_timeout
            self._original_execute_without_timeout = None

        if getattr(self, "_original_execute_single_native_tool_call", None) is not None:
            executor_module = import_module("crewai.agents.crew_agent_executor")
            executor_module.CrewAgentExecutor._execute_single_native_tool_call = (
                self._original_execute_single_native_tool_call
            )
            self._original_execute_single_native_tool_call = None

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        if not (config := kwargs.get("config")):
            config = TraceConfig()
        else:
            assert isinstance(config, TraceConfig)

        self._patch_context_thread_propagation()

        use_event_listener = kwargs.get("use_event_listener", False)
        if use_event_listener:
            from openinference.instrumentation.crewai._event_listener import (
                OpenInferenceEventListener,
            )

            create_llm_spans = kwargs.get("create_llm_spans", True)
            self._event_listener = OpenInferenceEventListener(
                tracer_provider=tracer_provider,
                config=config,
                create_llm_spans=create_llm_spans,
            )
            return

        self._event_listener = None  # type: ignore[assignment]
        self._tracer = OITracer(
            trace_api.get_tracer(__name__, __version__, tracer_provider),
            config=config,
        )

        execute_core_wrapper = _ExecuteCoreWrapper(tracer=self._tracer)
        self._original_execute_core = getattr(import_module("crewai").Task, "_execute_core", None)
        wrap_function_wrapper(
            "crewai",
            "Task._execute_core",
            execute_core_wrapper,
        )

        crew_kickoff_wrapper = _CrewKickoffWrapper(tracer=self._tracer)
        self._original_crew_kickoff = getattr(import_module("crewai").Crew, "kickoff", None)
        wrap_function_wrapper(
            "crewai",
            "Crew.kickoff",
            crew_kickoff_wrapper,
        )

        flow_kickoff_wrapper = _FlowKickoffWrapper(tracer=self._tracer)
        self._original_flow_kickoff = getattr(import_module("crewai").Flow, "kickoff", None)
        wrap_function_wrapper(
            "crewai",
            "Flow.kickoff",
            flow_kickoff_wrapper,
        )

        flow_kickoff_async_wrapper = _FlowKickoffAsyncWrapper(tracer=self._tracer)
        self._original_flow_kickoff_async = getattr(
            import_module("crewai").Flow, "kickoff_async", None
        )
        wrap_function_wrapper(
            "crewai",
            "Flow.kickoff_async",
            flow_kickoff_async_wrapper,
        )

        flow_execute_method_wrapper = _FlowExecuteMethodWrapper(tracer=self._tracer)
        self._original_flow_execute_method = getattr(
            import_module("crewai.flow.flow").Flow, "_execute_method", None
        )
        if self._original_flow_execute_method is not None:
            wrap_function_wrapper(
                "crewai.flow.flow",
                "Flow._execute_method",
                flow_execute_method_wrapper,
            )

        agent_kickoff_wrapper = _AgentKickoffWrapper(tracer=self._tracer)
        self._original_agent_kickoff = getattr(import_module("crewai").Agent, "kickoff", None)
        if self._original_agent_kickoff is not None:
            wrap_function_wrapper(
                "crewai",
                "Agent.kickoff",
                agent_kickoff_wrapper,
            )

        try:
            long_term_memory_module = import_module("crewai.memory.long_term.long_term_memory")
        except ModuleNotFoundError:
            # CrewAI 1.10+ removed long_term in favor of unified Memory
            self._original_long_term_memory_save = None
            self._original_long_term_memory_search = None
        else:
            long_term_memory_save_wrapper = _LongTermMemorySaveWrapper(tracer=self._tracer)
            long_term_memory_search_wrapper = _LongTermMemorySearchWrapper(tracer=self._tracer)
            self._original_long_term_memory_save = getattr(
                long_term_memory_module.LongTermMemory, "save", None
            )
            wrap_function_wrapper(
                "crewai.memory.long_term.long_term_memory",
                "LongTermMemory.save",
                long_term_memory_save_wrapper,
            )
            self._original_long_term_memory_search = getattr(
                long_term_memory_module.LongTermMemory, "search", None
            )
            wrap_function_wrapper(
                "crewai.memory.long_term.long_term_memory",
                "LongTermMemory.search",
                long_term_memory_search_wrapper,
            )

        try:
            short_term_memory_module = import_module("crewai.memory.short_term.short_term_memory")
        except ModuleNotFoundError:
            # CrewAI 1.10+ removed short_term in favor of unified Memory
            self._original_short_term_memory_save = None
            self._original_short_term_memory_search = None
        else:
            short_term_memory_save_wrapper = _ShortTermMemorySaveWrapper(tracer=self._tracer)
            short_term_memory_search_wrapper = _ShortTermMemorySearchWrapper(tracer=self._tracer)
            self._original_short_term_memory_save = getattr(
                short_term_memory_module.ShortTermMemory, "save", None
            )
            wrap_function_wrapper(
                "crewai.memory.short_term.short_term_memory",
                "ShortTermMemory.save",
                short_term_memory_save_wrapper,
            )
            self._original_short_term_memory_search = getattr(
                short_term_memory_module.ShortTermMemory, "search", None
            )
            wrap_function_wrapper(
                "crewai.memory.short_term.short_term_memory",
                "ShortTermMemory.search",
                short_term_memory_search_wrapper,
            )

        base_tool_run_wrapper = _BaseToolRunWrapper(tracer=self._tracer)
        self._original_base_tool_run = getattr(
            import_module("crewai.tools.base_tool").BaseTool, "run", None
        )
        wrap_function_wrapper(
            "crewai.tools.base_tool",
            "BaseTool.run",
            base_tool_run_wrapper,
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        if getattr(self, "_event_listener", None) is not None:
            self._event_listener.shutdown()
            self._event_listener = None  # type: ignore[assignment]

        if getattr(self, "_original_execute_core", None) is not None:
            task_module = import_module("crewai")
            task_module.Task._execute_core = self._original_execute_core
            self._original_execute_core = None

        if getattr(self, "_original_crew_kickoff", None) is not None:
            crew_module = import_module("crewai")
            crew_module.Crew.kickoff = self._original_crew_kickoff
            self._original_crew_kickoff = None

        if getattr(self, "_original_flow_kickoff", None) is not None:
            crew_module = import_module("crewai")
            crew_module.Flow.kickoff = self._original_flow_kickoff
            self._original_flow_kickoff = None

        if getattr(self, "_original_flow_kickoff_async", None) is not None:
            crew_module = import_module("crewai")
            crew_module.Flow.kickoff_async = self._original_flow_kickoff_async
            self._original_flow_kickoff_async = None

        if getattr(self, "_original_flow_execute_method", None) is not None:
            flow_module = import_module("crewai.flow.flow")
            flow_module.Flow._execute_method = self._original_flow_execute_method
            self._original_flow_execute_method = None

        if getattr(self, "_original_agent_kickoff", None) is not None:
            agent_module = import_module("crewai")
            agent_module.Agent.kickoff = self._original_agent_kickoff
            self._original_agent_kickoff = None

        if getattr(self, "_original_long_term_memory_save", None) is not None:
            long_term_memory_module = import_module("crewai.memory.long_term.long_term_memory")
            long_term_memory_module.LongTermMemory.save = self._original_long_term_memory_save
            self._original_long_term_memory_save = None

        if getattr(self, "_original_long_term_memory_search", None) is not None:
            long_term_memory_module = import_module("crewai.memory.long_term.long_term_memory")
            long_term_memory_module.LongTermMemory.search = self._original_long_term_memory_search
            self._original_long_term_memory_search = None

        if getattr(self, "_original_short_term_memory_save", None) is not None:
            short_term_memory_module = import_module("crewai.memory.short_term.short_term_memory")
            short_term_memory_module.ShortTermMemory.save = self._original_short_term_memory_save
            self._original_short_term_memory_save = None

        if getattr(self, "_original_short_term_memory_search", None) is not None:
            short_term_memory_module = import_module("crewai.memory.short_term.short_term_memory")
            short_term_memory_module.ShortTermMemory.search = (
                self._original_short_term_memory_search
            )
            self._original_short_term_memory_search = None

        if getattr(self, "_original_base_tool_run", None) is not None:
            base_tool_module = import_module("crewai.tools.base_tool")
            base_tool_module.BaseTool.run = self._original_base_tool_run
            self._original_base_tool_run = None

        self._restore_context_thread_propagation()
