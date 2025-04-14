from typing import Any, Callable, Collection, Optional

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from wrapt import wrap_function_wrapper

from openinference.instrumentation import (
    OITracer,
    TraceConfig,
)
from openinference.instrumentation.smolagents._wrappers import (
    _ModelWrapper,
    _RunWrapper,
    _StepWrapper,
    _ToolCallWrapper,
)
from openinference.instrumentation.smolagents.version import __version__

_instruments = ("smolagents >= 1.2.2.dev0",)


class SmolagentsInstrumentor(BaseInstrumentor):  # type: ignore
    __slots__ = (
        "_original_run_method",
        "_original_step_methods",
        "_original_tool_call_method",
        "_original_model_call_methods",
        "_tracer",
    )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        import smolagents
        from smolagents import CodeAgent, MultiStepAgent, Tool, ToolCallingAgent, models

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
        self._original_run_method = getattr(MultiStepAgent, "run", None)
        wrap_function_wrapper(
            module="smolagents",
            name="MultiStepAgent.run",
            wrapper=run_wrapper,
        )

        self._original_step_methods: Optional[dict[type, Optional[Callable[..., Any]]]] = {}
        step_wrapper = _StepWrapper(tracer=self._tracer)
        for step_cls in [CodeAgent, ToolCallingAgent]:
            self._original_step_methods[step_cls] = getattr(step_cls, "step", None)
            wrap_function_wrapper(
                module="smolagents",
                name=f"{step_cls.__name__}.step",
                wrapper=step_wrapper,
            )

        self._original_model_call_methods: Optional[dict[type, Callable[..., Any]]] = {}

        exported_model_subclasses = [
            attr
            for _, attr in vars(smolagents).items()
            if isinstance(attr, type) and issubclass(attr, models.Model)
        ]

        for model_subclass in exported_model_subclasses:
            model_subclass_wrapper = _ModelWrapper(tracer=self._tracer)
            self._original_model_call_methods[model_subclass] = getattr(model_subclass, "__call__")
            wrap_function_wrapper(
                module="smolagents",
                name=model_subclass.__name__ + ".__call__",
                wrapper=model_subclass_wrapper,
            )

        tool_call_wrapper = _ToolCallWrapper(tracer=self._tracer)
        self._original_tool_call_method = getattr(Tool, "__call__", None)
        wrap_function_wrapper(
            module="smolagents",
            name="Tool.__call__",
            wrapper=tool_call_wrapper,
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        from smolagents import MultiStepAgent, Tool

        if self._original_run_method is not None:
            MultiStepAgent.run = self._original_run_method
            self._original_run_method = None

        if self._original_step_methods is not None:
            for step_cls, original_step_method in self._original_step_methods.items():
                setattr(step_cls, "step", original_step_method)
            self._original_step_methods = None

        if self._original_model_call_methods is not None:
            for (
                model_subclass,
                original_model_call_method,
            ) in self._original_model_call_methods.items():
                setattr(model_subclass, "__call__", original_model_call_method)
            self._original_model_call_methods = None

        if self._original_tool_call_method is not None:
            Tool.__call__ = self._original_tool_call_method
            self._original_tool_call_method = None
