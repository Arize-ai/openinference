import logging
from importlib import import_module
from typing import Collection, Any
from wrapt import wrap_function_wrapper

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from openinference.instrumentation.guardrails.version import __version__
from openinference.instrumentation.guardrails._wrap_guard_call import _GuardCallWrapper, _ParseCallableWrapper, _PromptCallableWrapper, _PostValidationWrapper
import contextvars
from opentelemetry import context as otel_context
from wrapt import ObjectProxy, wrap_function_wrapper

logger = logging.getLogger(__name__)

_instruments = ("guardrails-ai >= 0.4.5",)

_VALIDATION_MODULE = "guardrails.validator_service"
_LLM_PROVIDERS_MODULE = 'guardrails.llm_providers'
_RUNNER_MODULE = 'guardrails.run'


class _Contextvars(ObjectProxy):  # type: ignore
    def __init__(self, cv: Any) -> None:
        super().__init__(cv)

    @staticmethod
    def Context() -> contextvars.Context:
        return contextvars.copy_context()


class GuardrailsInstrumentor(BaseInstrumentor):
    """An instrumentor for the Guardrails framework."""

    __slots__ = (
        "_original_guardrails_guard_call",
        "_original_guardrails_llm_providers_call",
        "_original_guardrails_runner_step",
        "_original_guardrails_validation_after_run",
    )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        tracer = trace_api.get_tracer(__name__, __version__, tracer_provider)

        import guardrails as gd

        gd.guard.contextvars = _Contextvars(gd.guard.contextvars)
        gd.async_guard.contextvars = _Contextvars(gd.async_guard.contextvars)
        for name in ("pydantic", "string", "rail_string", "rail"):
            wrap_function_wrapper(
                module="guardrails.guard",
                name=f"Guard.from_{name}",
                wrapper=lambda f, _, args, kwargs: f(*args, **{**kwargs, "tracer": tracer}),
            )        

        runner_wrapper = _ParseCallableWrapper(tracer=tracer)
        wrap_function_wrapper(
            module=_RUNNER_MODULE,
            name="Runner.step",
            wrapper=runner_wrapper,
        )

        prompt_callable_wrapper = _PromptCallableWrapper(tracer=tracer)
        wrap_function_wrapper(
            module=_LLM_PROVIDERS_MODULE,
            name="PromptCallableBase.__call__",
            wrapper=prompt_callable_wrapper,
        )

        post_validator_wrapper = _PostValidationWrapper(tracer=tracer)
        wrap_function_wrapper(
            module=_VALIDATION_MODULE,
            name="ValidatorServiceBase.after_run_validator",
            wrapper=post_validator_wrapper,
        )


    def _uninstrument(self, **kwargs):
        llm_providers = import_module(_LLM_PROVIDERS_MODULE)
        llm_providers.PromptCallableBase.__call__ = self._original_guardrails_llm_providers_call

        runner_module = import_module(_RUNNER_MODULE)
        runner_module.Runner.step = self._original_guardrails_runner_step

        validation_module = import_module(_VALIDATION_MODULE)
        validation_module.ValidatorServiceBase.after_run_validator = self._original_guardrails_validation_after_run

        import guardrails as gd
        gd.guard.contextvars = contextvars.Context
        gd.async_guard.contextvars = contextvars.Context