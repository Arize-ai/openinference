import logging
from importlib import import_module
from typing import Any, Collection

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from wrapt import wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.portkey.version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_instruments = ("portkey_ai >= 0.1.0",)


class PortkeyInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """An instrumentor for the Portkey AI framework."""

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

        # TODO: Implement instrumentation for Portkey AI
        # This will involve wrapping the appropriate methods from the portkey_ai package
        # For example:
        # self._original_method = PortkeyClass.method
        # wrap_function_wrapper(
        #     module="portkey_ai.module",
        #     name="PortkeyClass.method",
        #     wrapper=_MethodWrapper(tracer=self._tracer),
        # )

    def _uninstrument(self, **kwargs: Any) -> None:
        # TODO: Implement uninstrumentation for Portkey AI
        # This will involve restoring the original methods
        # For example:
        # portkey_module = import_module("portkey_ai.module")
        # if self._original_method is not None:
        #     portkey_module.PortkeyClass.method = self._original_method
        pass 