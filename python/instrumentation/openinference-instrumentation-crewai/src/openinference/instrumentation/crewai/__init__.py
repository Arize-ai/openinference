import logging
import json
from enum import Enum
from importlib import import_module
from typing import Any, Callable, Collection, Iterator, Mapping, List, Optional, Tuple
from inspect import signature

from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

from opentelemetry import trace as trace_api
from openinference.instrumentation.crewai.version import __version__
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from openinference.instrumentation import get_attributes_from_context, safe_json_dumps
from wrapt import wrap_function_wrapper
from opentelemetry.util.types import AttributeValue

_instruments = ("crewai >= 0.41.1",)

logger = logging.getLogger(__name__)

class SafeJSONEncoder(json.JSONEncoder):
    """
    Safely encodes non-JSON-serializable objects.
    """

    def default(self, o: Any) -> Any:
        try:
            return super().default(o)
        except TypeError:
            if hasattr(o, "dict") and callable(o.dict):  # pydantic v1 models, e.g., from Cohere
                return o.dict()
            return repr(o)

def _flatten(mapping: Optional[Mapping[str, Any]]) -> Iterator[Tuple[str, AttributeValue]]:
    if not mapping:
        return
    for key, value in mapping.items():
        if value is None:
            continue
        if isinstance(value, Mapping):
            for sub_key, sub_value in _flatten(value):
                yield f"{key}.{sub_key}", sub_value
        elif isinstance(value, List) and any(isinstance(item, Mapping) for item in value):
            for index, sub_mapping in enumerate(value):
                for sub_key, sub_value in _flatten(sub_mapping):
                    yield f"{key}.{index}.{sub_key}", sub_value
        else:
            if isinstance(value, Enum):
                value = value.value
            yield key, value

def _get_input_value(method: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    """
    Parses a method call's inputs into a JSON string. Ensures a consistent
    output regardless of whether the those inputs are passed as positional or
    keyword arguments.
    """

    # For typical class methods, the corresponding instance of inspect.Signature
    # does not include the self parameter. However, the inspect.Signature
    # instance for __call__ does include the self parameter.
    method_signature = signature(method)
    first_parameter_name = next(iter(method_signature.parameters), None)
    signature_contains_self_parameter = first_parameter_name in ["self"]
    bound_arguments = method_signature.bind(
        *(
            [None]  # the value bound to the method's self argument is discarded below, so pass None
            if signature_contains_self_parameter
            else []  # no self parameter, so no need to pass a value
        ),
        *args,
        **kwargs,
    )
    return safe_json_dumps(
        {
            **{
                argument_name: argument_value
                for argument_name, argument_value in bound_arguments.arguments.items()
                if argument_name not in ["self", "kwargs"]
            },
            **bound_arguments.arguments.get("kwargs", {}),
        },
        cls=SafeJSONEncoder,
    )

class _ExecuteCoreWrapper:
    def __init__(self, tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        if instance:
            span_name = f"{instance.__class__.__name__}.{wrapped.__name__}"
        else:
            span_name = wrapped.__name__
        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT,
                        SpanAttributes.INPUT_VALUE: _get_input_value(
                            wrapped,
                            *args,
                            **kwargs,
                        ),
                    }
                )
            ),
        ) as span:
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
            span.set_attribute(OUTPUT_VALUE, response)
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

class _ToolUseWrapper:
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
        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL,
                        SpanAttributes.INPUT_VALUE: _get_input_value(
                            wrapped,
                            *args,
                            **kwargs,
                        ),
                    }
                )
            ),
        ) as span:
            tool = kwargs.get("tool")
            tool_name = ""
            if tool:
                tool_name = tool.name
            span.set_attribute("function_calling_llm", instance.function_calling_llm)
            span.set_attribute(SpanAttributes.TOOL_NAME, tool_name)
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)
            span.set_attribute(OUTPUT_VALUE, response)
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

        use_wrapper = _ToolUseWrapper(tracer=tracer)
        wrap_function_wrapper(
            module="crewai.tools.tool_usage",
            name="ToolUsage._use",
            wrapper=use_wrapper,
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        return

INPUT_VALUE = SpanAttributes.INPUT_VALUE
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
