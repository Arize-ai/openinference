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
from openinference.instrumentation.smolagents._wrappers import (
    _RunWrapper,
    _StepWrapper,
    _ModelWrapper,
    _ToolCallWrapper,
)
from openinference.instrumentation.smolagents.version import __version__

_instruments = ("smolagents >= 1.1.0",)

logger = logging.getLogger(__name__)


class SmolagentsInstrumentor(BaseInstrumentor):  # type: ignore
    __slots__ = (
        "_original_run",
        "_original_step",
        "_original_tool_call",
        "_original_model_generate",
        "_original_model_get_tool_call",
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

        run_wrapper = _RunWrapper(tracer=self._tracer)
        self._original_run = getattr(import_module("smolagents.agents").MultiStepAgent, "run", None)
        wrap_function_wrapper(
            module="smolagents",
            name="MultiStepAgent.run",
            wrapper=run_wrapper,
        )

        step_wrapper_code = _StepWrapper(tracer=self._tracer)
        self._original_step = getattr(import_module("smolagents.agents").CodeAgent, "step", None)
        wrap_function_wrapper(
            module="smolagents",
            name="CodeAgent.step",
            wrapper=step_wrapper_code,
        )

        step_wrapper_tool_calling = _StepWrapper(tracer=self._tracer)
        self._original_step = getattr(import_module("smolagents.agents").ToolCallingAgent, "step", None)
        wrap_function_wrapper(
            module="smolagents",
            name="ToolCallingAgent.step",
            wrapper=step_wrapper_tool_calling,
        )

        model_generate_wrapper = _ModelWrapper(tracer=self._tracer)
        self._original_model_generate = getattr(import_module("smolagents").Model, "__call__", None)
        wrap_function_wrapper(
            module="smolagents",
            name="Model.__call__",
            wrapper=model_generate_wrapper,
        )

        model_tool_call_wrapper = _ModelWrapper(tracer=self._tracer)
        self._original_model_tool_call = getattr(import_module("smolagents").Model, "get_tool_call", None)
        wrap_function_wrapper(
            module="smolagents",
            name="Model.get_tool_call",
            wrapper=model_tool_call_wrapper,
        )

        tool_call_wrapper = _ToolCallWrapper(tracer=self._tracer)
        self._original_tool_call = getattr(import_module("smolagents.tools").Tool, "__call__", None)
        wrap_function_wrapper(
            module="smolagents",
            name="Tool.__call__",
            wrapper=tool_call_wrapper,
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        if self._original_run is not None:
            smolagents_module = import_module("smolagents.agents")
            smolagents_module.MultiStepAgent.run = self._original_run
            self._original_run = None

        if self._original_step is not None:
            smolagents_module = import_module("smolagents.agents")
            smolagents_module.MultiStepAgent.step = self._original_step
            self._original_step = None

        if self._original_model_generate is not None:
            smolagents_module = import_module("smolagents.models")
            smolagents_module.MultimodelAgent.model = self._original_model_generate
            self._original_step = None

        if self._original_model_get_tool_call is not None:
            smolagents_module = import_module("smolagents.models")
            smolagents_module.MultimodelAgent.model = self._original_model_get_tool_call
            self._original_step = None

        if self._original_tool_call is not None:
            tool_usage_module = import_module("smolagents.tools")
            tool_usage_module.Tool.__call__ = self._original_tool_call
            self._original_tool_call = None
