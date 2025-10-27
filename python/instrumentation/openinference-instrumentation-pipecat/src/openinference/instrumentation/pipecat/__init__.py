"""OpenInference instrumentation for Pipecat."""

import logging
from typing import Any, Collection

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from wrapt import wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.pipecat.package import _instruments
from openinference.instrumentation.pipecat.version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ["PipecatInstrumentor"]


class PipecatInstrumentor(BaseInstrumentor):
    """
    An instrumentor for Pipecat pipelines.

    Automatically instruments PipelineTask to observe frame flow and create
    OpenInference-compliant spans for LLM, TTS, and STT services.
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def create_observer(self):
        """
        Create an OpenInferenceObserver manually.

        Returns:
            OpenInferenceObserver instance

        Raises:
            RuntimeError: If instrumentor is not instrumented yet
        """
        if not self.is_instrumented_by_opentelemetry:
            raise RuntimeError(
                "Instrumentor must be instrumented before creating observers. "
                "Call .instrument() first."
            )

        from openinference.instrumentation.pipecat._observer import OpenInferenceObserver

        return OpenInferenceObserver(tracer=self._tracer, config=self._config)

    def _instrument(self, **kwargs: Any) -> None:
        """
        Instrument Pipecat by wrapping PipelineTask.__init__ to inject observer.
        """
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()

        if not (config := kwargs.get("config")):
            config = TraceConfig()
        else:
            assert isinstance(config, TraceConfig)

        # Create OITracer
        tracer = OITracer(
            trace_api.get_tracer(__name__, __version__, tracer_provider),
            config=config,
        )

        # Store for creating observers
        self._tracer = tracer
        self._config = config

        try:
            # Import Pipecat classes
            from pipecat.pipeline.task import PipelineTask

            # Store original __init__
            self._original_task_init = PipelineTask.__init__

            # Wrap PipelineTask.__init__ to inject our observer
            wrap_function_wrapper(
                module="pipecat.pipeline.task",
                name="PipelineTask.__init__",
                wrapper=_TaskInitWrapper(tracer=tracer, config=config),
            )

            logger.info("Pipecat instrumentation enabled")

        except ImportError as e:
            logger.warning(f"Failed to instrument Pipecat: {e}")

    def _uninstrument(self, **kwargs: Any) -> None:
        """
        Uninstrument Pipecat by restoring original PipelineTask.__init__.
        """
        try:
            from pipecat.pipeline.task import PipelineTask

            if hasattr(self, "_original_task_init"):
                PipelineTask.__init__ = self._original_task_init
                logger.info("Pipecat instrumentation disabled")
        except (ImportError, AttributeError):
            pass


class _TaskInitWrapper:
    """Wrapper for PipelineTask.__init__ to inject OpenInferenceObserver."""

    def __init__(self, tracer: OITracer, config: TraceConfig):
        self._tracer = tracer
        self._config = config

    def __call__(self, wrapped, instance, args, kwargs):
        """
        Call original __init__, then inject our observer.

        This creates a new observer instance for each task (thread-safe).
        """
        # Call original __init__
        wrapped(*args, **kwargs)

        # Create observer for this task
        from openinference.instrumentation.pipecat._observer import OpenInferenceObserver

        observer = OpenInferenceObserver(tracer=self._tracer, config=self._config)

        # Inject observer into task
        instance.add_observer(observer)

        logger.debug(f"Injected OpenInferenceObserver into PipelineTask {id(instance)}")
