import logging
from importlib import import_module
from typing import Collection
from wrapt import wrap_function_wrapper

from opentelemetry.trace import get_tracer
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from openinference.instrumentation.haystack.version import __version__
from openinference.instrumentation.haystack._wrap_pipeline import _PipelineWrapper

logger = logging.getLogger(__name__)

_instruments = ("haystack-ai >= 2.0.0",)

_MODULE = "haystack.core.pipeline.pipeline"


class HaystackInstrumentor(BaseInstrumentor):
    """An instrumentor for the Haystack framework."""

    __slots__ = (
        "_original_pipeline_run",
    )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = get_tracer_provider()
        tracer = get_tracer(__name__, __version__, tracer_provider)
        haystack = import_module(_MODULE)
        self._original_pipeline_run = haystack.Pipeline.run
        wrap_function_wrapper(
            module=_MODULE,
            name="Pipeline.run",
            wrapper=_PipelineWrapper(tracer=tracer),
        )

    def _uninstrument(self, **kwargs):
        haystack = import_module(_MODULE)
        haystack.Pipeline.run = self._original_pipeline_run
