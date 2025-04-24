import logging
from typing import Any, Collection

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import (  # type: ignore[attr-defined]
    BaseInstrumentor,
)

from wrapt import wrap_function_wrapper
from openinference.instrumentation import (
    OITracer,
    TraceConfig,
)

from openinference.instrumentation.autogen.version import __version__

_instruments = ("autogen-agentchat >= 0.5.1",)

logger = logging.getLogger(__name__)


class AutogenInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    __slots__ = ("_tracer",)

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


    def _uninstrument(self, **kwargs: Any) -> None:
        return
