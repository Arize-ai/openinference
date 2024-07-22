import logging
from importlib import import_module
from typing import Any, Collection

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from wrapt import wrap_function_wrapper

_instruments = ("crewai >= 0.41.1",)

logger = logging.getLogger(__name__)

class _ExecuteCoreWrapper:
    def __init__(self, tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        if instance:
            span_name = f"{instance.__class__.__name__}.{wrapped.__name__}"
        else:
            span_name = wrapped.__name__
        with self._tracer.start_as_current_span(span_name) as span:
            try:
                instance = args[0]  # Assuming the first argument is the instance
                agent = instance.agent
                crew = agent.crew if agent else None
                task = instance

                if crew:
                    span.set_attribute("crew_key", crew.key)
                    span.set_attribute("crew_id", str(crew.id))
                span.set_attribute("task_key", task.key)
                span.set_attribute("task_id", str(task.id))

                if crew and crew.share_crew:
                    span.set_attribute("formatted_description", task.description)
                    span.set_attribute("formatted_expected_output", task.expected_output)

                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)
        return response

class CrewAIInstrumentor(BaseInstrumentor):  # type: ignore
    __slots__ = (
        "_original_function",
    )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        tracer = trace_api.get_tracer(__name__, __version__, tracer_provider)

        execute_core_wrapper = _ExecuteCoreWrapper(tracer=tracer)
        wrap_function_wrapper(
            module="crewai.task",
            name="Task._execute_core",
            wrapper=execute_core_wrapper,
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        return