import logging
from importlib import import_module
from typing import Any, Collection

from openinference.instrumentation.haystack._wrappers import _ComponentWrapper, _PipelineWrapper
from openinference.instrumentation.haystack.version import __version__
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)

_instruments = ("haystack-ai >= 2.0.0",)


class HaystackInstrumentor(BaseInstrumentor):
    """An instrumentor for the Haystack framework."""

    __slots__ = ("_original_pipeline_run", "_original_pipeline_run_component")

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        tracer = get_tracer(__name__, __version__, tracer_provider)
        haystack = import_module("haystack.core.pipeline.pipeline")
        self._original_pipeline_run = haystack.Pipeline.run
        self._original_pipeline_run_component = haystack.Pipeline._run_component

        # Creating a parent span for the Pipeline
        wrap_function_wrapper(
            module="haystack.core.pipeline.pipeline",
            name="Pipeline.run",
            wrapper=_PipelineWrapper(tracer=tracer),
        )
        # Creating child spans for every Component in the Pipeline
        wrap_function_wrapper(
            module="haystack.core.pipeline.pipeline",
            name="Pipeline._run_component",
            wrapper=_ComponentWrapper(tracer=tracer),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        haystack = import_module("haystack.core.pipeline.pipeline")
        haystack.Pipeline.run = self._original_pipeline_run
        haystack.Pipeline._run_component = self._original_pipeline_run_component
