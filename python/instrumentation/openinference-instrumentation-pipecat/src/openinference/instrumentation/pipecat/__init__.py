"""OpenInference instrumentation for Pipecat."""

import logging
from typing import Any, Callable, Collection, Dict, Optional, Tuple

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from wrapt import wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig

from .package import _instruments
from .version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.INFO)

__all__ = ["PipecatInstrumentor"]


class PipecatInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumentor for Pipecat pipelines.

    Automatically instruments PipelineWorker to observe frame flow and create
    OpenInference-compliant spans for LLM, TTS, and STT services. The
    deprecated ``PipelineTask`` alias (a thin subclass of ``PipelineWorker``)
    is covered transparently because its ``__init__`` calls ``super().__init__``.
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments if isinstance(_instruments, tuple) else ()

    def create_observer(self) -> Any:
        """
        Create an OpenInferenceObserver manually.

        Returns:
            OpenInferenceObserver instance

        Raises:
            RuntimeError: If instrumentor is not instrumented yet
        """
        from ._observer import OpenInferenceObserver

        if not self.is_instrumented_by_opentelemetry:
            raise RuntimeError(
                "Instrumentor must be instrumented before creating observers. "
                "Call .instrument() first."
            )

        return OpenInferenceObserver(tracer=self._tracer, config=self._config)  # type: ignore[arg-type,unused-ignore]

    def _instrument(self, **kwargs: Any) -> None:
        """
        Instrument Pipecat by wrapping PipelineWorker.__init__ to inject observer.

        Args:
            tracer_provider: OpenTelemetry TracerProvider
            config: OpenInference TraceConfig
            debug_log_filename: Optional debug log filename to use for all observers
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
        self._debug_log_filename = kwargs.get("debug_log_filename")

        try:
            from pipecat.pipeline.worker import PipelineWorker

            # Store original __init__
            self._original_worker_init = PipelineWorker.__init__

            # Wrap PipelineWorker.__init__ to inject our observer. PipelineTask
            # (the deprecated alias) is a thin subclass of PipelineWorker whose
            # __init__ delegates via super().__init__, so it is covered by the
            # same wrap — instantiating PipelineTask still fires the wrapper
            # exactly once.
            wrap_function_wrapper(
                "pipecat.pipeline.worker",
                "PipelineWorker.__init__",
                _PipelineInitWrapper(
                    tracer=tracer,
                    config=config,
                    default_debug_log_filename=self._debug_log_filename,
                ),
            )

            logger.info("Pipecat instrumentation enabled")

        except ImportError as e:
            logger.warning(f"Failed to instrument Pipecat: {e}")

    def _uninstrument(self, **kwargs: Any) -> None:
        """
        Uninstrument Pipecat by restoring original PipelineWorker.__init__.
        """
        try:
            from pipecat.pipeline.worker import PipelineWorker

            if hasattr(self, "_original_worker_init"):
                PipelineWorker.__init__ = self._original_worker_init  # type: ignore
                logger.info("Pipecat instrumentation disabled")
        except (ImportError, AttributeError):
            pass


class _PipelineInitWrapper:
    """Wrapper for PipelineWorker.__init__ to inject OpenInferenceObserver."""

    def __init__(
        self,
        tracer: OITracer,
        config: TraceConfig,
        default_debug_log_filename: Optional[str] = None,
    ):
        self._tracer = tracer
        self._config = config
        self._default_debug_log_filename = default_debug_log_filename

    def __call__(
        self,
        wrapped: Callable[[Any, Any], Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> None:
        """
        Call original __init__, then inject our observer.

        This creates a new observer instance for each task (thread-safe).
        """
        # Call original __init__
        wrapped(*args, **kwargs)

        # Extract conversation_id from PipelineWorker if available
        # PipelineWorker stores it as _conversation_id (private attribute)
        conversation_id = getattr(instance, "_conversation_id", None)
        additional_span_attributes = getattr(instance, "_additional_span_attributes", None)

        # Use task-specific debug log filename if set, otherwise use default from instrument()
        debug_log_filename = (
            getattr(instance, "_debug_log_filename", None) or self._default_debug_log_filename
        )

        from openinference.instrumentation.pipecat._observer import OpenInferenceObserver

        observer = OpenInferenceObserver(
            tracer=self._tracer,  # type: ignore[arg-type,unused-ignore]
            config=self._config,
            conversation_id=conversation_id,
            debug_log_filename=debug_log_filename,
            additional_span_attributes=additional_span_attributes,
        )

        # Inject observer into task
        instance.add_observer(observer)

        logger.info(
            f"Injected OpenInferenceObserver into {type(instance).__name__} {id(instance)} "
        )
        if additional_span_attributes:
            logger.info(f"Additional span attributes: {str(additional_span_attributes)}")
        if conversation_id:
            logger.info(f"Conversation ID: {conversation_id}")
