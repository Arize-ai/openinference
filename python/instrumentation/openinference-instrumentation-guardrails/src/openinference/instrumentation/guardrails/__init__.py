import logging
from importlib import import_module
from typing import Collection
from wrapt import wrap_function_wrapper

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from openinference.instrumentation.guardrails.version import __version__
from openinference.instrumentation.guardrails._wrap_guard_call import _GuardCallWrapper

logger = logging.getLogger(__name__)

_instruments = ("guardrails-ai >= 0.4.5",)

_MODULE = "guardrails.guard"


class GuardrailsInstrumentor(BaseInstrumentor):
    """An instrumentor for the Guardrails framework."""

    __slots__ = (
        "_original_guardrails_guard_call",
    )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_trace_provider()
        tracer = trace_api.get_tracer(__name__, __version__, tracer_provider)
        guardrails = import_module(_MODULE)
        self._original_pipeline_run = guardrails.Guard.__call__
        wrap_function_wrapper(
            module=_MODULE,
            name="Guard.__call__",
            wrapper=_GuardCallWrapper(tracer=tracer),
        )

    def _uninstrument(self, **kwargs):
        guardrails = import_module(_MODULE)
        guardrails.Guard.__call__ = self._original_guardrails_guard_call
