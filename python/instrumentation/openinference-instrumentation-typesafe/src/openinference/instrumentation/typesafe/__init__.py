import logging
from importlib import import_module
from typing import Any, Collection

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from wrapt import wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.typesafe.version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_instruments = ("typesafe-sdk >= 0.6.0",)

# The clients are patched through the SDK's public re-export rather than their
# private defining modules; wrapt sets the attribute on the same class object either
# way, and `typesafe_sdk._core.*` carries no compatibility promise.
_MODULE = "typesafe_sdk"

__all__ = ("TypeSafeAIInstrumentor",)


class TypeSafeAIInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """Traces the TypeSafe AI Python SDK (``typesafe-sdk``) as OpenInference LLM spans.

    Wraps ``TypeSafeClient.system_one`` and ``AsyncTypeSafeClient.system_one``.
    ``instrument()`` accepts a ``tracer_provider`` and a ``TraceConfig``; ``uninstrument()``
    restores the original methods.
    """

    __slots__ = ("_original_system_one", "_original_async_system_one", "_tracer")

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        from openinference.instrumentation.typesafe._wrappers import (
            _AsyncSystemOneWrapper,
            _SystemOneWrapper,
        )

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

        module = import_module(_MODULE)
        self._original_system_one = module.TypeSafeClient.system_one
        self._original_async_system_one = module.AsyncTypeSafeClient.system_one
        wrap_function_wrapper(
            _MODULE,
            "TypeSafeClient.system_one",
            _SystemOneWrapper(tracer=self._tracer, span_name="TypeSafeClient"),
        )
        wrap_function_wrapper(
            _MODULE,
            "AsyncTypeSafeClient.system_one",
            _AsyncSystemOneWrapper(tracer=self._tracer, span_name="AsyncTypeSafeClient"),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        module = import_module(_MODULE)
        module.TypeSafeClient.system_one = self._original_system_one
        module.AsyncTypeSafeClient.system_one = self._original_async_system_one
