import json
from enum import Enum
from inspect import signature
from typing import Any, Callable, Iterator, List, Mapping, Optional, Tuple, cast

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import (
    get_attributes_from_context,
    get_input_attributes,
    get_output_attributes,
    safe_json_dumps,
)
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes


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


def _get_crew_name(crew: Any) -> str:
    """Generate a meaningful crew name for span naming."""
    # First try to use crew.name attribute (user-friendly name)
    if hasattr(crew, "name") and crew.name and crew.name.strip():
        crew_name = str(crew.name).strip()
        # If it's not just the default "crew", use it
        if crew_name.lower() != "crew":
            return crew_name

    # Fallback to Crew with ID
    if hasattr(crew, "id") and crew.id is not None:
        return f"Crew_{str(crew.id)}"

    # Final fallback
    return "Crew"


def _get_flow_name(flow: Any) -> str:
    """Generate a meaningful flow name for span naming."""
    # First try to use flow.name attribute (user-friendly name)
    if hasattr(flow, "name") and flow.name and flow.name.strip():
        flow_name = str(flow.name).strip()
        # If it's not just the default "flow", use it
        if flow_name.lower() != "flow":
            return flow_name

    # Fallback to Flow with ID
    if hasattr(flow, "flow_id") and flow.flow_id is not None:
        return f"Flow_{str(flow.flow_id)}"

    # Final fallback
    return "Flow"


def _get_tool_span_name(
    instance: Any, wrapped: Callable[..., Any], kwargs: Mapping[str, Any]
) -> str:
    """Generate a meaningful tool span name including tool name."""
    base_method = wrapped.__name__

    # Try to get the tool name from kwargs
    tool = kwargs.get("tool")
    if tool and hasattr(tool, "name") and tool.name:
        tool_name = str(tool.name).strip()
        if tool_name:
            return f"{tool_name}.{base_method}"

    # Fallback to original naming if no tool name available
    if instance:
        return f"{instance.__class__.__name__}.{str(base_method)}"
    else:
        return str(base_method)


def _get_execute_core_span_name(instance: Any, wrapped: Callable[..., Any], agent: Any) -> str:
    """Generate a meaningful task span name using agent role."""
    base_method = wrapped.__name__

    if not instance:
        return str(base_method)

    # Get agent role for context - simplified to just use agent name
    if agent and hasattr(agent, "role") and agent.role:
        agent_role = str(agent.role).strip()
        if agent_role:
            return f"{agent_role}.{str(base_method)}"

    # Fallback to original naming if no agent role available
    if instance:
        return f"{instance.__class__.__name__}.{str(base_method)}"
    else:
        return str(base_method)


def _find_parent_agent(current_role: str, agents: List[Any]) -> Optional[str]:
    for i, a in enumerate(agents):
        if a.role == current_role and i != 0:
            parent_agent = agents[i - 1]
            if parent_agent.role:
                return cast(str, parent_agent.role)
    return None


class _ExecuteCoreWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        # task naming - include agent role and task context
        agent = args[0] if args else kwargs.get("agent")
        span_name = _get_execute_core_span_name(instance, wrapped, agent)
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
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            span.set_attribute("task_key", instance.key)
            span.set_attribute("task_id", str(instance.id))

            agent = args[0] if args else None
            # Conditionally set attributes for the agent, crew, and task
            if agent:
                span.set_attribute(SpanAttributes.GRAPH_NODE_ID, agent.role)
                crew = agent.crew
                if crew:
                    span.set_attribute("crew_key", crew.key)
                    span.set_attribute("crew_id", str(crew.id))

                    # Find the current agent, the previous agent is the parent node
                    parent_agent_role = _find_parent_agent(agent.role, crew.agents)
                    if parent_agent_role:
                        span.set_attribute(SpanAttributes.GRAPH_NODE_PARENT_ID, parent_agent_role)

                    if crew.share_crew:
                        span.set_attribute("formatted_description", instance.description)
                        span.set_attribute("formatted_expected_output", instance.expected_output)

            # Run the wrapped function, and capture errors before re-raising
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise

            span.set_status(trace_api.StatusCode.OK)
            span.set_attributes(dict(get_output_attributes(response)))
            span.set_attributes(dict(get_attributes_from_context()))
        return response


class _CrewKickoffWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        # Enhanced crew naming - use meaningful crew name instead of generic "Crew.kickoff"
        crew_name = _get_crew_name(instance)
        span_name = f"{crew_name}.kickoff"
        with self._tracer.start_as_current_span(
            span_name,
            record_exception=False,
            set_status_on_exception=False,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN,
                    }
                )
            ),
        ) as span:
            crew = instance
            inputs = kwargs.get("inputs", None) or (args[0] if args else None)

            if inputs is not None:
                span.set_attributes(dict(get_input_attributes(inputs)))

            span.set_attribute("crew_key", crew.key)
            span.set_attribute("crew_id", str(crew.id))
            span.set_attribute("crew_inputs", json.dumps(inputs) if inputs else "")
            span.set_attribute(
                "crew_agents",
                json.dumps(
                    [
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
                    ]
                ),
            )
            span.set_attribute(
                "crew_tasks",
                json.dumps(
                    [
                        {
                            "id": str(task.id),
                            "description": task.description,
                            "expected_output": task.expected_output,
                            "async_execution?": task.async_execution,
                            "human_input?": task.human_input,
                            "agent_role": task.agent.role if task.agent else "None",
                            "agent_key": task.agent.key if task.agent else None,
                            "context": [task.description for task in task.context]
                            if isinstance(task.context, list)
                            else None,
                            "tools_names": [tool.name.casefold() for tool in task.tools or []],
                        }
                        for task in crew.tasks
                    ]
                ),
            )
            try:
                crew_output = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)

            span.set_attributes(dict(get_output_attributes(crew_output)))
            span.set_attributes(dict(get_attributes_from_context()))
        return crew_output


class _FlowKickoffAsyncWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)
        # Enhanced flow naming - use meaningful flow name instead of generic "Flow.kickoff"
        flow_name = _get_flow_name(instance)
        span_name = f"{flow_name}.kickoff"
        with self._tracer.start_as_current_span(
            span_name,
            record_exception=False,
            set_status_on_exception=False,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN,
                    }
                )
            ),
        ) as span:
            flow = instance
            inputs = kwargs.get("inputs", None) or (args[0] if args else None)

            if inputs is not None:
                span.set_attributes(dict(get_input_attributes(inputs)))

            span.set_attribute("flow_id", str(flow.flow_id))
            span.set_attribute("flow_inputs", json.dumps(inputs) if inputs else "")

            try:
                flow_output = await wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)

            span.set_attributes(dict(get_output_attributes(flow_output)))
            span.set_attributes(dict(get_attributes_from_context()))
        return flow_output


class _ToolUseWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        # Enhanced tool naming - include actual tool name
        span_name = _get_tool_span_name(instance, wrapped, kwargs)
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
            record_exception=False,
            set_status_on_exception=False,
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
            span.set_attributes(dict(get_output_attributes(response)))
            span.set_attributes(dict(get_attributes_from_context()))
        return response


INPUT_VALUE = SpanAttributes.INPUT_VALUE
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
