import logging
import json
from importlib import import_module
from typing import Any, Callable, Collection, Mapping, Tuple

from opentelemetry import trace as trace_api
from openinference.instrumentation.crewai.version import __version__
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
                agent = args[0]  # Assuming the first argument is the instance
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


class _KickoffWrapper:
    def __init__(self, tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        span_name = f"{instance.__class__.__name__}.kickoff"
        with self._tracer.start_as_current_span(span_name) as span:
            try:
                crew = instance
                inputs = kwargs.get('inputs', None) or (args[0] if args else None)

                span.set_attribute("crew_key", crew.key)
                span.set_attribute("crew_id", str(crew.id))
                span.set_attribute("crew_inputs", json.dumps(inputs) if inputs else None)
                span.set_attribute("crew_agents", json.dumps([
                    {
                        "key": agent.key,
                        "id": str(agent.id),
                        "role": agent.role,
                        "goal": agent.goal,
                        "backstory": agent.backstory,
                        "verbose?": agent.verbose,
                        "max_iter": agent.max_iter,
                        "max_rpm": agent.max_rpm,
                        "i18n": agent.i18n.prompt_file,
                        "delegation_enabled": agent.allow_delegation,
                        "tools_names": [tool.name.casefold() for tool in agent.tools or []],
                    }
                    for agent in crew.agents
                ]))
                span.set_attribute("crew_tasks", json.dumps([
                    {
                        "id": str(task.id),
                        "description": task.description,
                        "expected_output": task.expected_output,
                        "async_execution?": task.async_execution,
                        "human_input?": task.human_input,
                        "agent_role": task.agent.role if task.agent else "None",
                        "agent_key": task.agent.key if task.agent else None,
                        "context": [task.description for task in task.context] if task.context else None,
                        "tools_names": [tool.name.casefold() for tool in task.tools or []],
                    }
                    for task in crew.tasks
                ]))

                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)
        return response


class _CrewExecution:
    def __init__(self, tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        if instance:
            span_name = f"{instance.__class__.__name__}.{wrapped.__name__}"
        else:
            span_name = wrapped.__name__
        with self._tracer.start_as_current_span(span_name) as span:
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)
        return response


class _ToolUse:
    def __init__(self, tracer):
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if instance:
            span_name = f"{instance.__class__.__name__}.{wrapped.__name__}"
        else:
            span_name = wrapped.__name__
        with self._tracer.start_as_current_span(span_name) as span:
            tool = kwargs.get("tool")
            tool_name = ""
            if tool:
                tool_name = tool.name
            span.set_attribute("function_calling_llm", instance.function_calling_llm)
            span.set_attribute("tool_name", tool_name)
            try:
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
            module="crewai",
            name="Task._execute_core",
            wrapper=execute_core_wrapper,
        )

        kickoff_wrapper = _KickoffWrapper(tracer=tracer)
        wrap_function_wrapper(
            module="crewai",
            name="Crew.kickoff",
            wrapper=kickoff_wrapper,
        )

        use_wrapper = _ToolUse(tracer=tracer)
        wrap_function_wrapper(
            module="crewai.tools.tool_usage",
            name="ToolUsage._use",
            wrapper=use_wrapper,
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        return
