import logging
from importlib import import_module
from typing import Collection
from wrapt import wrap_function_wrapper

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from openinference.instrumentation.guardrails.version import __version__
from openinference.instrumentation.guardrails._wrap_guard_call import _GuardCallWrapper, _ParseCallableWrapper
import contextvars
from opentelemetry import context as otel_context

logger = logging.getLogger(__name__)

_instruments = ("guardrails-ai >= 0.4.5",)

_MODULE = "guardrails.guard"
_LLM_PROVIDERS_MODULE = 'guardrails.llm_providers'
_RUNNER_MODULE = 'guardrails.run'

from wrapt import ObjectProxy

original_context = contextvars.Context

def create_otel_preserving_context(wrapped):
    def _wrapped():
        # Create a new contextvars.Context
        new_context = wrapped()

        # Get the current OpenTelemetry context
        current_otel_context = otel_context.get_current()

        # Copy all items from the OpenTelemetry context to the new context
        for key, value in current_otel_context.items():
            new_context.run(contextvars.ContextVar(str(key)).set, value)
        return new_context

    return _wrapped

class OTelPreservingContext(ObjectProxy):
    def __init__(self, *args, **kwargs):
        context = original_context(*args, **kwargs)
        super().__init__(context)
        self._self_otel_context = otel_context.get_current()
        print("WRAPPER original CONTEXT {}".format(self._self_otel_context))

    def run(self, callable, *args, **kwargs):
        print("WRAPPER ABOUT TO ATTACH {}".format(self._self_otel_context))
        token = otel_context.attach(self._self_otel_context)
        try:
            return self.__wrapped__.run(callable, *args, **kwargs)
        finally:
            otel_context.detach(token)

def patch_contextvars():
    def wrapped_context(*args, **kwargs):
        return OTelPreservingContext(*args, **kwargs)
    contextvars.Context = wrapped_context

def unpatch_contextvars():
    contextvars.Context = original_context

class GuardrailsInstrumentor(BaseInstrumentor):
    """An instrumentor for the Guardrails framework."""

    __slots__ = (
        "_original_guardrails_guard_call",
        "_original_guardrails_llm_providers_call",
    )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_trace_provider()
        tracer = trace_api.get_tracer(__name__, __version__, tracer_provider)

        guardrails = import_module(_MODULE)
        self._original_pipeline_run = guardrails.Guard.__call__
        guard_wrapper = _GuardCallWrapper(tracer=tracer)
        wrap_function_wrapper(
            module=_MODULE,
            name="Guard.__call__",
            wrapper=guard_wrapper,
        )

        runner_wrapper = _ParseCallableWrapper(tracer=tracer)
        wrap_function_wrapper(
            module=_RUNNER_MODULE,
            name="Runner.step",
            wrapper=runner_wrapper,
        )

        patch_contextvars()


    def _uninstrument(self, **kwargs):
        guardrails = import_module(_MODULE)
        guardrails.Guard.__call__ = self._original_guardrails_guard_call

        llm_providers = import_module(_LLM_PROVIDERS_MODULE)
        llm_providers.PromptCallableBase.__call__ = self._original_guardrails_llm_providers_call

        unpatch_contextvars()