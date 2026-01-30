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
    _BaseToolRunWrapper,
    _CrewKickoffWrapper,
    _ExecuteCoreWrapper,
    _FlowKickoffAsyncWrapper,
    _LongTermMemorySaveWrapper,
    _LongTermMemorySearchWrapper,
    _ShortTermMemorySaveWrapper,
    _ShortTermMemorySearchWrapper,
)
from openinference.instrumentation.crewai.version import __version__

_instruments = ("crewai >= 1.9.0",)

logger = logging.getLogger(__name__)


class CrewAIInstrumentor(BaseInstrumentor):  # type: ignore
    __slots__ = (
        "_original_execute_core",
        "_original_crew_kickoff",
        "_original_flow_kickoff_async",
        "_original_long_term_memory_save",
        "_original_long_term_memory_search",
        "_original_short_term_memory_save",
        "_original_short_term_memory_search",
        "_original_base_tool_run",
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

        execute_core_wrapper = _ExecuteCoreWrapper(tracer=self._tracer)
        self._original_execute_core = getattr(import_module("crewai").Task, "_execute_core", None)
        wrap_function_wrapper(
            module="crewai",
            name="Task._execute_core",
            wrapper=execute_core_wrapper,
        )

        crew_kickoff_wrapper = _CrewKickoffWrapper(tracer=self._tracer)
        self._original_crew_kickoff = getattr(import_module("crewai").Crew, "kickoff", None)
        wrap_function_wrapper(
            module="crewai",
            name="Crew.kickoff",
            wrapper=crew_kickoff_wrapper,
        )

        flow_kickoff_async_wrapper = _FlowKickoffAsyncWrapper(tracer=self._tracer)
        self._original_flow_kickoff_async = getattr(
            import_module("crewai").Flow, "kickoff_async", None
        )
        wrap_function_wrapper(
            module="crewai",
            name="Flow.kickoff_async",
            wrapper=flow_kickoff_async_wrapper,
        )

        long_term_memory_save_wrapper = _LongTermMemorySaveWrapper(tracer=self._tracer)
        self._original_long_term_memory_save = getattr(
            import_module("crewai.memory.long_term.long_term_memory").LongTermMemory, "save", None
        )
        wrap_function_wrapper(
            module="crewai.memory.long_term.long_term_memory",
            name="LongTermMemory.save",
            wrapper=long_term_memory_save_wrapper,
        )

        long_term_memory_search_wrapper = _LongTermMemorySearchWrapper(tracer=self._tracer)
        self._original_long_term_memory_search = getattr(
            import_module("crewai.memory.long_term.long_term_memory").LongTermMemory, "search", None
        )
        wrap_function_wrapper(
            module="crewai.memory.long_term.long_term_memory",
            name="LongTermMemory.search",
            wrapper=long_term_memory_search_wrapper,
        )

        short_term_memory_save_wrapper = _ShortTermMemorySaveWrapper(tracer=self._tracer)
        self._original_short_term_memory_save = getattr(
            import_module("crewai.memory.short_term.short_term_memory").ShortTermMemory,
            "save",
            None,
        )
        wrap_function_wrapper(
            module="crewai.memory.short_term.short_term_memory",
            name="ShortTermMemory.save",
            wrapper=short_term_memory_save_wrapper,
        )

        short_term_memory_search_wrapper = _ShortTermMemorySearchWrapper(tracer=self._tracer)
        self._original_short_term_memory_search = getattr(
            import_module("crewai.memory.short_term.short_term_memory").ShortTermMemory,
            "search",
            None,
        )
        wrap_function_wrapper(
            module="crewai.memory.short_term.short_term_memory",
            name="ShortTermMemory.search",
            wrapper=short_term_memory_search_wrapper,
        )

        base_tool_run_wrapper = _BaseToolRunWrapper(tracer=self._tracer)
        self._original_base_tool_run = getattr(
            import_module("crewai.tools.base_tool").BaseTool, "run", None
        )
        wrap_function_wrapper(
            module="crewai.tools.base_tool",
            name="BaseTool.run",
            wrapper=base_tool_run_wrapper,
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        if self._original_execute_core is not None:
            task_module = import_module("crewai")
            task_module.Task._execute_core = self._original_execute_core
            self._original_execute_core = None

        if self._original_crew_kickoff is not None:
            crew_module = import_module("crewai")
            crew_module.Crew.kickoff = self._original_crew_kickoff
            self._original_crew_kickoff = None

        if self._original_flow_kickoff_async is not None:
            crew_module = import_module("crewai")
            crew_module.Flow.kickoff_async = self._original_flow_kickoff_async
            self._original_flow_kickoff_async = None

        if self._original_long_term_memory_save is not None:
            long_term_memory_module = import_module("crewai.memory.long_term.long_term_memory")
            long_term_memory_module.LongTermMemory.save = self._original_long_term_memory_save
            self._original_long_term_memory_save = None

        if self._original_long_term_memory_search is not None:
            long_term_memory_module = import_module("crewai.memory.long_term.long_term_memory")
            long_term_memory_module.LongTermMemory.search = self._original_long_term_memory_search
            self._original_long_term_memory_search = None

        if self._original_short_term_memory_save is not None:
            short_term_memory_module = import_module("crewai.memory.short_term.short_term_memory")
            short_term_memory_module.ShortTermMemory.save = self._original_short_term_memory_save
            self._original_short_term_memory_save = None

        if self._original_short_term_memory_search is not None:
            short_term_memory_module = import_module("crewai.memory.short_term.short_term_memory")
            short_term_memory_module.ShortTermMemory.search = (
                self._original_short_term_memory_search
            )
            self._original_short_term_memory_search = None

        if self._original_base_tool_run is not None:
            base_tool_module = import_module("crewai.tools.base_tool")
            base_tool_module.BaseTool.run = self._original_base_tool_run
            self._original_base_tool_run = None
