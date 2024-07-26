import logging
from importlib import import_module
from typing import Any, Collection

from openinference.instrumentation.nemo_guardrails.version import __version__
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from wrapt import ObjectProxy, wrap_function_wrapper


logger = logging.getLogger(__name__)

_instruments = ("nemoguardrails >= 0.9.1.1")


class _ExecuteActionWrapper(_WithTracer):
    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        if instance:
            span_name = f"{instance.__class__.__name__}.{wrapped.__name__}"
        else:
            span_name = wrapped.__name__
        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.GUARDRAIL,
                        INPUT_VALUE: _get_input_value(
                            wrapped,
                            *args,
                            **kwargs,
                        ),
                    }
                )
            ),
        ) as span:
            span.set_attributes(dict(get_attributes_from_context()))
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)
        return response


class NemoGuardrailsInstrumentor(BaseInstrumentor):  # type: ignore

    __slots__ = (
        "_original_guardrails_llm_providers_call",
    )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        tracer = trace_api.get_tracer(__name__, __version__, tracer_provider)
        
        execute_action_wrapper = _ExecuteActionWrapper(tracer=tracer)
        wrap_function_wrapper(
            module="colang.runtime",
            name="RuntimeV1_0.execute_action",
            wrapper=execute_action_wrapper,
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        return
