import json
from enum import Enum
from inspect import signature
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import get_attributes_from_context, safe_json_dumps
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)

if TYPE_CHECKING:
    from smolagents.tools import Tool


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
    arguments = _bind_arguments(method, *args, **kwargs)
    arguments = _strip_method_args(arguments)
    return safe_json_dumps(arguments)


def _bind_arguments(method: Callable[..., Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
    method_signature = signature(method)
    bound_args = method_signature.bind(*args, **kwargs)
    bound_args.apply_defaults()
    return bound_args.arguments


def _strip_method_args(arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in arguments.items() if key not in ("self", "cls")}


class _RunWrapper:
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
        span_name = f"{instance.__class__.__name__}.run"
        with self._tracer.start_as_current_span(
            span_name,
            record_exception=False,
            set_status_on_exception=False,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: AGENT,
                        INPUT_VALUE: _get_input_value(
                            wrapped,
                            *args,
                            **kwargs,
                        ),
                    }
                )
            ),
        ) as span:
            agent = instance
            task = agent.task
            additional_args = kwargs.get("additional_args", None)

            # span.set_attribute("agent_key", agent.key)
            span.set_attribute(INPUT_VALUE, task)
            span.set_attribute("agent_name", str(agent.__class__.__name__))
            span.set_attribute("task", task)
            span.set_attribute(
                "additional_args", json.dumps(additional_args) if additional_args else ""
            )
            span.set_attribute(
                "additional_args", json.dumps(additional_args) if additional_args else ""
            )
            model_id = f" - {agent.model.model_id}" if hasattr(agent.model, "model_id") else ""
            model_type = f"{type(agent.model).__name__}"
            span.set_attribute("model", model_type + model_id)
            span.set_attribute("max_steps", agent.max_steps)
            span.set_attribute("tools_names", list(agent.tools.keys()))

            span.set_attribute(
                "managed_agents",
                json.dumps(
                    [
                        {
                            "name": managed_agent.name,
                            "description": managed_agent.description,
                            "additional_prompting": managed_agent.additional_prompting,
                            "model": f"{type(managed_agent.agent.model).__name__}"
                            + f" - {managed_agent.agent.model.model_id}"
                            if hasattr(managed_agent.agent.model, "model_id")
                            else "",
                            "max_steps": managed_agent.agent.max_steps,
                            "tools_names": list(agent.tools.keys()),
                        }
                        for managed_agent in agent.managed_agents.values()
                    ]
                ),
            )
            try:
                agent_output = wrapped(*args, **kwargs)
                span.set_attribute(LLM_TOKEN_COUNT_PROMPT, agent.monitor.total_input_token_count)
                span.set_attribute(
                    LLM_TOKEN_COUNT_COMPLETION, agent.monitor.total_output_token_count
                )
                span.set_attribute(
                    LLM_TOKEN_COUNT_TOTAL,
                    agent.monitor.total_input_token_count + agent.monitor.total_output_token_count,
                )

            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)
            span.set_attribute(OUTPUT_VALUE, str(agent_output))
            span.set_attributes(dict(get_attributes_from_context()))
        return agent_output


class _StepWrapper:
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
        agent = instance
        span_name = f"Step {agent.step_number}"
        with self._tracer.start_as_current_span(
            span_name,
            record_exception=False,
            set_status_on_exception=False,
            attributes={
                OPENINFERENCE_SPAN_KIND: CHAIN,
            },
        ) as span:
            try:
                result = wrapped(*args, **kwargs)
                if result is not None:
                    span.set_attribute("Final answer", result)
                step_log = args[0]  # ActionStep
                # if step_log.error is not None:
                #     span.set_status(
                #         trace_api.Status(trace_api.StatusCode.ERROR, str(step_log.error))
                #     )
                #     span.record_exception(str(step_log.error))

            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_attribute("Tool calls", step_log.tool_calls)
            if step_log.error is not None:
                span.set_attribute("Error", str(step_log.error))
            span.set_attribute("Observations", step_log.observations)
            span.set_status(trace_api.StatusCode.OK)
            span.set_attribute(OUTPUT_VALUE, step_log.observations)
            span.set_attributes(dict(get_attributes_from_context()))
        return result


def _llm_input_messages(arguments: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    if isinstance(prompt := arguments.get("prompt"), str):
        yield f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}", "user"
        yield f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}", prompt
    elif isinstance(messages := arguments.get("messages"), list):
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            if (role := message.get("role", None)) is not None:
                yield (
                    f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_ROLE}",
                    role,
                )
            if (content := message.get("content", None)) is not None:
                yield (
                    f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_CONTENT}",
                    content,
                )


def _llm_output_messages(output_message: Any) -> Iterator[Tuple[str, Any]]:
    if (role := getattr(output_message, "role", None)) is not None:
        yield (
            f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}",
            role,
        )
    if (content := getattr(output_message, "content", None)) is not None:
        yield (
            f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}",
            content,
        )
    if isinstance(tool_calls := getattr(output_message, "tool_calls", None), list):
        for tool_call_index, tool_call in enumerate(tool_calls):
            if (tool_call_id := getattr(tool_call, "id", None)) is not None:
                yield (
                    f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.{tool_call_index}.{TOOL_CALL_ID}",
                    tool_call_id,
                )
            if (function := getattr(tool_call, "function", None)) is not None:
                if (name := getattr(function, "name", None)) is not None:
                    yield (
                        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.{tool_call_index}.{TOOL_CALL_FUNCTION_NAME}",
                        name,
                    )
                if isinstance(arguments := getattr(function, "arguments", None), str):
                    yield (
                        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.{tool_call_index}.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                        arguments,
                    )


def _output_value_and_mime_type(output: Any) -> Iterator[Tuple[str, Any]]:
    yield OUTPUT_MIME_TYPE, JSON
    yield OUTPUT_VALUE, output.model_dump_json()


def _llm_invocation_parameters(
    model: Any, arguments: Mapping[str, Any]
) -> Iterator[Tuple[str, Any]]:
    model_kwargs = _ if isinstance(_ := getattr(model, "kwargs", {}), dict) else {}
    kwargs = _ if isinstance(_ := arguments.get("kwargs"), dict) else {}
    yield LLM_INVOCATION_PARAMETERS, safe_json_dumps(model_kwargs | kwargs)


def _llm_tools(tools_to_call_from: List[Any]) -> Iterator[Tuple[str, Any]]:
    from smolagents import Tool
    from smolagents.models import get_json_schema  # type:ignore[import-untyped]

    if not isinstance(tools_to_call_from, list):
        return
    for tool_index, tool in enumerate(tools_to_call_from):
        if isinstance(tool, Tool):
            yield (
                f"{LLM_TOOLS}.{tool_index}.{TOOL_JSON_SCHEMA}",
                safe_json_dumps(get_json_schema(tool)),
            )


def _tools(tool: "Tool") -> Iterator[Tuple[str, Any]]:
    if tool_name := getattr(tool, "name", None):
        yield TOOL_NAME, tool_name
    if tool_description := getattr(tool, "description", None):
        yield TOOL_DESCRIPTION, tool_description
    yield TOOL_PARAMETERS, safe_json_dumps(tool.inputs)


def _input_value_and_mime_type(arguments: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    yield INPUT_MIME_TYPE, JSON
    yield INPUT_VALUE, safe_json_dumps(arguments)


class _ModelWrapper:
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
        arguments = _bind_arguments(wrapped, *args, **kwargs)
        if instance:
            span_name = f"{instance.__class__.__name__}.__call__"
        else:
            span_name = wrapped.__name__
        model = instance
        with self._tracer.start_as_current_span(
            span_name,
            attributes={
                OPENINFERENCE_SPAN_KIND: LLM,
                **dict(_input_value_and_mime_type(arguments)),
                **dict(_llm_invocation_parameters(instance, arguments)),
                **dict(_llm_input_messages(arguments)),
                **dict(get_attributes_from_context()),
            },
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                output_message = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_attribute(LLM_TOKEN_COUNT_PROMPT, model.last_input_token_count)
            span.set_attribute(LLM_TOKEN_COUNT_COMPLETION, model.last_output_token_count)
            span.set_attribute(LLM_MODEL_NAME, model.model_id)
            span.set_attribute(
                LLM_TOKEN_COUNT_TOTAL, model.last_input_token_count + model.last_output_token_count
            )
            span.set_status(trace_api.StatusCode.OK)
            span.set_attribute(OUTPUT_VALUE, output_message)
            span.set_attributes(dict(_llm_output_messages(output_message)))
            span.set_attributes(dict(_llm_tools(arguments.get("tools_to_call_from", []))))
            span.set_attributes(dict(_output_value_and_mime_type(output_message)))
        return output_message


class _ToolCallWrapper:
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
        if instance:
            span_name = f"{instance.__class__.__name__}"
        else:
            span_name = wrapped.__name__
        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: TOOL,
                        INPUT_VALUE: _get_input_value(
                            wrapped,
                            *args,
                            **kwargs,
                        ),
                        **dict(_tools(instance)),
                    }
                )
            ),
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)
            span.set_attributes(dict(get_attributes_from_context()))
            span.set_attributes(
                dict(
                    _output_value_and_mime_type_for_tool_span(
                        response=response,
                        output_type=instance.output_type,
                    )
                )
            )
        return response


def _output_value_and_mime_type_for_tool_span(
    response: Any, output_type: str
) -> Iterator[Tuple[str, Any]]:
    if output_type in (
        "string",
        "boolean",
        "integer",
        "number",
    ):
        yield OUTPUT_VALUE, response
        yield OUTPUT_MIME_TYPE, TEXT
    elif output_type == "object":
        yield OUTPUT_VALUE, safe_json_dumps(response)
        yield OUTPUT_MIME_TYPE, JSON

    # TODO: handle other types


# span attributes
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
LLM_PROMPTS = SpanAttributes.LLM_PROMPTS
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
LLM_TOOLS = SpanAttributes.LLM_TOOLS
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
TOOL_DESCRIPTION = SpanAttributes.TOOL_DESCRIPTION
TOOL_NAME = SpanAttributes.TOOL_NAME
TOOL_PARAMETERS = SpanAttributes.TOOL_PARAMETERS

# message attributes
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON = MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON
MESSAGE_FUNCTION_CALL_NAME = MessageAttributes.MESSAGE_FUNCTION_CALL_NAME
MESSAGE_NAME = MessageAttributes.MESSAGE_NAME
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS

# mime types
JSON = OpenInferenceMimeTypeValues.JSON.value
TEXT = OpenInferenceMimeTypeValues.TEXT.value

# span kinds
AGENT = OpenInferenceSpanKindValues.AGENT.value
CHAIN = OpenInferenceSpanKindValues.CHAIN.value
LLM = OpenInferenceSpanKindValues.LLM.value
TOOL = OpenInferenceSpanKindValues.TOOL.value

# tool attributes
TOOL_JSON_SCHEMA = ToolAttributes.TOOL_JSON_SCHEMA

# tool call attributes
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
TOOL_CALL_ID = ToolCallAttributes.TOOL_CALL_ID
