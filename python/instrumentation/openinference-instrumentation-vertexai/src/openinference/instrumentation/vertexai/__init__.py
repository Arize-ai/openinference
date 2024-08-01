import logging
from typing import Any, Collection

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.vertexai import _instrumentation_status
from openinference.instrumentation.vertexai.package import _instruments
from openinference.instrumentation.vertexai.version import __version__
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from opentelemetry.trace import get_tracer, get_tracer_provider
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class VertexAIInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumentor for `google-cloud-aiplatform` or the `vertexai` SDK
    """

    _status = _instrumentation_status

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = get_tracer_provider()
        if not (config := kwargs.get("config")):
            config = TraceConfig()
        else:
            assert isinstance(config, TraceConfig)
        tracer = OITracer(
            get_tracer(__name__, __version__, tracer_provider),
            config=config,
        )
        self._status._IS_INSTRUMENTED = True
        import google.api_core.gapic_v1 as gapic
        from openinference.instrumentation.vertexai._wrapper import _Wrapper

        for method in (gapic.method.wrap_method, gapic.method_async.wrap_method):
            wrap_function_wrapper(
                module=method.__module__,
                name=method.__name__,
                wrapper=lambda f, _, args, kwargs: _Wrapper(tracer)(f(*args, **kwargs)),
            )

    def _uninstrument(self, **kwargs: Any) -> None:
        self._status._IS_INSTRUMENTED = False
        import google.api_core.gapic_v1 as gapic

        for module in (gapic.method, gapic.method_async):
            if wrapped := getattr(module.wrap_method, "__wrapped__", None):
                setattr(module, "wrap_method", wrapped)
