import logging
from typing import Any, Collection

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import (  # type: ignore[attr-defined]
    BaseInstrumentor,
)
from wrapt import wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.haystack._wrappers import _ComponentWrapper, _PipelineWrapper
from openinference.instrumentation.haystack.version import __version__

logger = logging.getLogger(__name__)

_instruments = ("haystack-ai >= 2.0.0",)


class HaystackInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """An instrumentor for the Haystack framework."""

    __slots__ = ("_original_pipeline_run", "_original_pipeline_run_component", "_tracer")

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        import haystack

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

        self._original_pipeline_run = haystack.Pipeline.run
        wrap_function_wrapper(
            module="haystack.core.pipeline.pipeline",
            name="Pipeline.run",
            wrapper=_PipelineWrapper(tracer=self._tracer),
        )
        self._original_pipeline_run_component = haystack.Pipeline._run_component
        wrap_function_wrapper(
            module="haystack.core.pipeline.pipeline",
            name="Pipeline._run_component",
            wrapper=_ComponentWrapper(tracer=self._tracer),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        import haystack

        if self._original_pipeline_run is not None:
            haystack.Pipeline.run = self._original_pipeline_run

        if self._original_pipeline_run_component is not None:
            haystack.Pipeline._run_component = self._original_pipeline_run_component
