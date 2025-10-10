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
    _AsyncComponentRunWrapper,
    _PipelineRunComponentWrapper,
    _AsyncPipelineRunComponentWrapper,  # Async pipeline run component wrapper
    _PipelineWrapper,
    _AsyncPipelineWrapper,  # Async pipeline wrapper
)
from openinference.instrumentation.haystack.version import __version__

logger = logging.getLogger(__name__)

_instruments = ("haystack-ai >= 2.18.0",)


class HaystackInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """An instrumentor for the Haystack framework."""

    __slots__ = (
        "_original_pipeline_run",
        "_original_pipeline_run_async",
        "_original_pipeline_run_component",
        "_original_component_run_methods",
        "_original_pipeline_run_async_component",
        "_original_component_run_async_methods",
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
        # Instrument AsyncPipeline.run_async with _AsyncPipelineWrapper
        self._original_pipeline_run_async = haystack.AsyncPipeline.run_async
        wrap_function_wrapper(
            module="haystack.core.pipeline.async_pipeline",
            name="AsyncPipeline.run_async",
            wrapper=_AsyncPipelineWrapper(tracer=self._tracer),
        )

        from haystack.core.pipeline.pipeline import Pipeline
        from haystack.core.pipeline.async_pipeline import AsyncPipeline

        original = Pipeline.__dict__["_run_component"]
        self._original_pipeline_run_component = original.__func__
        self._original_component_run_methods: dict[type[Any], Callable[..., Any]] = {}
        self._original_component_run_async_methods: dict[type[Any], Callable[..., Any]] = {}

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

        def wrap_component_run_async_method(
            component_cls: type[Any], run_method: Callable[..., Any]
        ) -> None:
            if component_cls not in self._original_component_run_async_methods:
                self._original_component_run_async_methods[component_cls] = getattr(component_cls, "run_async")
                wrap_function_wrapper(
                    module=component_cls.__module__,
                    name=f"{component_cls.__name__}.run_async",
                    wrapper=_AsyncComponentRunWrapper(tracer=self._tracer),
                )

        wrap_function_wrapper(
            Pipeline,
            "_run_component",
            _PipelineRunComponentWrapper(
                tracer=self._tracer, wrap_component_run_method=wrap_component_run_method
            )
        )
        original_async = AsyncPipeline.__dict__["_run_component_async"]
        self._original_pipeline_run_component_async = original_async.__func__
        wrap_function_wrapper(
            AsyncPipeline,
            "_run_component_async",
            _AsyncPipelineRunComponentWrapper(
                tracer=self._tracer, wrap_component_run_method=wrap_component_run_async_method
            )
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        import haystack

        if self._original_pipeline_run is not None:
            setattr(haystack.Pipeline, "run", self._original_pipeline_run)

        if self._original_pipeline_run_component is not None:
            from haystack.core.pipeline.pipeline import Pipeline

            setattr(Pipeline, "_run_component", staticmethod(self._original_pipeline_run_component))

        for component_cls, original_run_method in self._original_component_run_methods.items():
            setattr(component_cls, "run", original_run_method)

        if self._original_pipeline_run_async is not None:
            setattr(haystack.AsyncPipeline, "run_async", self._original_pipeline_run)

        if self._original_pipeline_run_async_component is not None:
            from haystack.core.pipeline.async_pipeline import AsyncPipeline

            setattr(AsyncPipeline, "_run_async_component", staticmethod(self._original_pipeline_run_async_component))

        for component_cls, original_run_method in self._original_component_run_async_methods.items():
            setattr(component_cls, "run_async", original_run_method)
