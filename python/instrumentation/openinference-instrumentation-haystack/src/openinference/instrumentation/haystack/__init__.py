import logging
from typing import Any, Callable, Collection

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import (  # type: ignore[attr-defined]
    BaseInstrumentor,
)
from wrapt import wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.haystack._wrappers import (
    _AsyncComponentRunWrapper,
    _AsyncPipelineRunAsyncGeneratorWrapper,
    _AsyncPipelineRunComponentWrapper,
    _AsyncPipelineWrapper,
    _ComponentRunWrapper,
    _PipelineRunComponentWrapper,
    _PipelineWrapper,
)
from openinference.instrumentation.haystack.version import __version__

logger = logging.getLogger(__name__)

_instruments = ("haystack-ai >= 2.18.0",)


class HaystackInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """An instrumentor for the Haystack framework."""

    __slots__ = (
        "_original_pipeline_run",
        "_original_pipeline_run_component",
        "_original_component_run_methods",
        "_original_async_pipeline_run",
        "_original_async_pipeline_run_async",
        "_original_async_pipeline_run_async_generator",
        "_original_async_pipeline_run_component_async",
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
        self._original_async_pipeline_run = haystack.AsyncPipeline.run
        wrap_function_wrapper(
            module="haystack.core.pipeline.async_pipeline",
            name="AsyncPipeline.run",
            wrapper=_PipelineWrapper(tracer=self._tracer),
        )
        self._original_async_pipeline_run_async = haystack.AsyncPipeline.run_async
        wrap_function_wrapper(
            module="haystack.core.pipeline.async_pipeline",
            name="AsyncPipeline.run_async",
            wrapper=_AsyncPipelineWrapper(tracer=self._tracer),
        )
        self._original_async_pipeline_run_async_generator = (
            haystack.AsyncPipeline.run_async_generator
        )
        wrap_function_wrapper(
            module="haystack.core.pipeline.async_pipeline",
            name="AsyncPipeline.run_async_generator",
            wrapper=_AsyncPipelineRunAsyncGeneratorWrapper(tracer=self._tracer),
        )

        from haystack.core.pipeline.async_pipeline import AsyncPipeline
        from haystack.core.pipeline.pipeline import Pipeline

        original = Pipeline.__dict__["_run_component"]
        self._original_pipeline_run_component = original.__func__
        self._original_component_run_methods: dict[type[Any], Callable[..., Any]] = {}
        self._original_component_run_async_methods: dict[type[Any], Callable[..., Any]] = {}

        def wrap_component_run_method(
            component_cls: type[Any], run_method: Callable[..., Any]
        ) -> None:
            # To avoid double wrapping, we only wrap the class bound "run" method here.

            method_name = getattr(run_method, "__name__", None)
            if method_name is None and hasattr(run_method, "__func__"):
                method_name = getattr(run_method.__func__, "__name__", None)

            if method_name == "run" and component_cls not in self._original_component_run_methods:
                class_method = getattr(component_cls, "run")
                self._original_component_run_methods[component_cls] = class_method
                wrap_function_wrapper(
                    module=component_cls.__module__,
                    name=f"{component_cls.__name__}.run",
                    wrapper=_ComponentRunWrapper(tracer=self._tracer),
                )
            if (
                method_name == "run_async"
                and component_cls not in self._original_component_run_async_methods
            ):
                class_method = getattr(component_cls, "run_async")
                self._original_component_run_async_methods[component_cls] = class_method
                wrap_function_wrapper(
                    module=component_cls.__module__,
                    name=f"{component_cls.__name__}.{method_name}",
                    wrapper=_AsyncComponentRunWrapper(tracer=self._tracer),
                )

        wrap_function_wrapper(
            Pipeline,
            "_run_component",
            _PipelineRunComponentWrapper(
                tracer=self._tracer, wrap_component_run_method=wrap_component_run_method
            ),
        )

        async_original = AsyncPipeline.__dict__["_run_component_async"]
        self._original_async_pipeline_run_component_async = async_original.__func__

        wrap_function_wrapper(
            AsyncPipeline,
            "_run_component_async",
            _AsyncPipelineRunComponentWrapper(
                tracer=self._tracer, wrap_component_run_method=wrap_component_run_method
            ),
        )
        from haystack.core.component.component import component

        for class_path, cls in component.registry.items():
            # Ensure the class looks like a Component (has run or run_async)
            if hasattr(cls, "run"):
                wrap_component_run_method(cls, cls.run)
            if hasattr(cls, "run_async"):
                wrap_component_run_method(cls, cls.run_async)

    def _uninstrument(self, **kwargs: Any) -> None:
        import haystack

        if self._original_pipeline_run is not None:
            setattr(haystack.Pipeline, "run", self._original_pipeline_run)

        if self._original_async_pipeline_run is not None:
            setattr(haystack.AsyncPipeline, "run", self._original_async_pipeline_run)

        if self._original_async_pipeline_run_async is not None:
            setattr(haystack.AsyncPipeline, "run_async", self._original_async_pipeline_run_async)

        if self._original_async_pipeline_run_async_generator is not None:
            setattr(
                haystack.AsyncPipeline,
                "run_async_generator",
                self._original_async_pipeline_run_async_generator,
            )

        if self._original_pipeline_run_component is not None:
            from haystack.core.pipeline.pipeline import Pipeline

            setattr(Pipeline, "_run_component", staticmethod(self._original_pipeline_run_component))

        for component_cls, original_run_method in self._original_component_run_methods.items():
            setattr(component_cls, "run", original_run_method)

        if self._original_async_pipeline_run_component_async is not None:
            from haystack.core.pipeline.async_pipeline import AsyncPipeline

            setattr(
                AsyncPipeline,
                "_run_component_async",
                staticmethod(self._original_async_pipeline_run_component_async),
            )

        for component_cls, original_run_mt in self._original_component_run_async_methods.items():
            setattr(component_cls, "run_async", original_run_mt)
