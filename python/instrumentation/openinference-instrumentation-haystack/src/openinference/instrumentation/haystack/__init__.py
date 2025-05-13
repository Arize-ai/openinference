import logging
from typing import Any, Callable, Collection

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import (  # type: ignore[attr-defined]
    BaseInstrumentor,
)
from wrapt import wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.haystack._wrappers import (
    _ComponentRunWrapper,
    _PipelineRunComponentWrapper,
    _PipelineWrapper,
)
from openinference.instrumentation.haystack.version import __version__

logger = logging.getLogger(__name__)

_instruments = ("haystack-ai >= 2.13.0",)


class HaystackInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """An instrumentor for the Haystack framework."""

    __slots__ = (
        "_original_pipeline_run",
        "_original_pipeline_run_component",
        "_original_component_run_methods",
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
        import haystack

        self._original_pipeline_run = haystack.Pipeline.run
        wrap_function_wrapper(
            module="haystack.core.pipeline.pipeline",
            name="Pipeline.run",
            wrapper=_PipelineWrapper(tracer=self._tracer),
        )
        from haystack.core.pipeline.pipeline import Pipeline

        original = Pipeline.__dict__["_run_component"]
        self._original_pipeline_run_component = original.__func__
        self._original_component_run_methods: dict[type[Any], Callable[..., Any]] = {}

        def wrap_component_run_method(
            component_cls: type[Any], run_method: Callable[..., Any]
        ) -> None:
            if component_cls not in self._original_component_run_methods:
                self._original_component_run_methods[component_cls] = run_method
                wrap_function_wrapper(
                    module=component_cls.__module__,
                    name=f"{component_cls.__name__}.run",
                    wrapper=_ComponentRunWrapper(tracer=self._tracer),
                )

        wrap_function_wrapper(
            Pipeline,
            "_run_component",
            _PipelineRunComponentWrapper(
                tracer=self._tracer, wrap_component_run_method=wrap_component_run_method
            ),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        import haystack

        if self._original_pipeline_run is not None:
            haystack.Pipeline.run = self._original_pipeline_run

        if self._original_pipeline_run_component is not None:
            from haystack.core.pipeline.pipeline import Pipeline

            Pipeline._run_component = staticmethod(self._original_pipeline_run_component)

        for component_cls, original_run_method in self._original_component_run_methods.items():
            component_cls.run = original_run_method
