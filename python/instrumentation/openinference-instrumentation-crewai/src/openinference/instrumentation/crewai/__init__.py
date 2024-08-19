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
    _ExecuteCoreWrapper,
    _KickoffWrapper,
    _ToolUseWrapper,
)
from openinference.instrumentation.crewai.version import __version__

_instruments = ("crewai >= 0.41.1",)

logger = logging.getLogger(__name__)


class CrewAIInstrumentor(BaseInstrumentor):  # type: ignore
    __slots__ = (
        "_original_execute_core",
        "_original_kickoff",
        "_original_tool_use",
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

        kickoff_wrapper = _KickoffWrapper(tracer=self._tracer)
        self._original_kickoff = getattr(import_module("crewai").Crew, "kickoff", None)
        wrap_function_wrapper(
            module="crewai",
            name="Crew.kickoff",
            wrapper=kickoff_wrapper,
        )

        use_wrapper = _ToolUseWrapper(tracer=self._tracer)
        self._original_tool_use = getattr(
            import_module("crewai.tools.tool_usage").ToolUsage, "_use", None
        )
        wrap_function_wrapper(
            module="crewai.tools.tool_usage",
            name="ToolUsage._use",
            wrapper=use_wrapper,
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        if self._original_execute_core is not None:
            task_module = import_module("crewai")
            task_module.Task._execute_core = self._original_execute_core
            self._original_execute_core = None

        if self._original_kickoff is not None:
            crew_module = import_module("crewai")
            crew_module.Crew.kickoff = self._original_kickoff
            self._original_kickoff = None

        if self._original_tool_use is not None:
            tool_usage_module = import_module("crewai.tools.tool_usage")
            tool_usage_module.ToolUsage._use = self._original_tool_use
            self._original_tool_use = None
