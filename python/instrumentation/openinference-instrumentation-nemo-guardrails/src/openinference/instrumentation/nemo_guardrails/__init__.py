import logging
from abc import ABC
from importlib import import_module
from typing import Any, Callable, Collection

from openinference.instrumentation.nemo_guardrails.version import __version__
from openinference.instrumentation.nemo_guardrails._wrappers import _ExecuteActionWrapper, _GenerateAsyncWrapper
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from wrapt import ObjectProxy, wrap_function_wrapper


logger = logging.getLogger(__name__)

_instruments = ("nemoguardrails >= 0.9.1",)


class NemoGuardrailsInstrumentor(BaseInstrumentor):  # type: ignore

    __slots__ = (
        "_original_execute_action",
    )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        tracer = trace_api.get_tracer(__name__, __version__, tracer_provider)
        
        module_name = "nemoguardrails.actions.action_dispatcher"
        function_name = "ActionDispatcher.execute_action"
        
        self._original_execute_action = getattr(import_module(module_name).ActionDispatcher, "execute_action", None)
        if self._original_execute_action is None:
            logger.error(f"Failed to get the original function: {module_name}.{function_name}")
            return
        
        execute_action_wrapper = _ExecuteActionWrapper(tracer=tracer)
        logger.info(f"Wrapping function: {module_name}.{function_name}")
        
        wrap_function_wrapper(
            module=module_name,
            name=function_name,
            wrapper=execute_action_wrapper,
        )
        module_name_async = "nemoguardrails"
        class_name_async = "LLMRails"
        function_name_async = "generate_async"
        
        self._original_generate_async = getattr(import_module(module_name_async).LLMRails, "generate_async", None)
        if self._original_generate_async is None:
            logger.error(f"Failed to get the original function: {module_name_async}.{class_name_async}.{function_name_async}")
            return
        
        generate_async_wrapper = _GenerateAsyncWrapper(tracer=tracer)
        logger.info(f"Wrapping function: {module_name_async}.{class_name_async}.{function_name_async}")
        
        wrap_function_wrapper(
            module=module_name_async,
            name=f"{class_name_async}.{function_name_async}",
            wrapper=generate_async_wrapper,
        )

        print(f"Successfully wrapped function: {module_name}.{function_name}")

    def _uninstrument(self, **kwargs: Any) -> None:
        return
