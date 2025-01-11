import json
from enum import Enum
from inspect import signature
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import get_attributes_from_context, safe_json_dumps
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)


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
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN,
                    }
                )
            ),
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

            if step_log.error is not None:
                span.set_attribute("Error", str(step_log.error))
            span.set_attribute("Observations", step_log.observations)
            span.set_status(trace_api.StatusCode.OK)
            span.set_attribute(OUTPUT_VALUE, step_log.observations)
            span.set_attributes(dict(get_attributes_from_context()))
        return result


def _bind_arguments(method: Callable[..., Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
    method_signature = signature(method)
    bound_args = method_signature.bind(*args, **kwargs)
    bound_args.apply_defaults()
    return bound_args.arguments


def _llm_input_messages(arguments: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    if isinstance(prompt := arguments.get("prompt"), str):
        yield f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}", "user"
        yield f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}", prompt
    elif isinstance(messages := arguments.get("messages"), list):
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            if (role := message.get("role", None)) is not None:
                yield (
                    f"{SpanAttributes.LLM_INPUT_MESSAGES}.{i}.{MessageAttributes.MESSAGE_ROLE}",
                    role,
                )
            if (content := message.get("content", None)) is not None:
                yield (
                    f"{SpanAttributes.LLM_INPUT_MESSAGES}.{i}.{MessageAttributes.MESSAGE_CONTENT}",
                    content,
                )


def _llm_invocation_parameters(
    model: Any, arguments: Mapping[str, Any]
) -> Iterator[Tuple[str, Any]]:
    model_kwargs = _ if isinstance(_ := getattr(model, "kwargs", {}), dict) else {}
    kwargs = _ if isinstance(_ := arguments.get("kwargs"), dict) else {}
    yield SpanAttributes.LLM_INVOCATION_PARAMETERS, safe_json_dumps(model_kwargs | kwargs)


def _input_value_and_mime_type(arguments: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    yield SpanAttributes.INPUT_MIME_TYPE, OpenInferenceMimeTypeValues.JSON.value
    yield SpanAttributes.INPUT_VALUE, safe_json_dumps(arguments)


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
            span_name = f"{instance.__class__.__name__}"
        else:
            span_name = wrapped.__name__
        model = instance
        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM,
                        **dict(_input_value_and_mime_type(arguments)),
                        **dict(_llm_invocation_parameters(instance, arguments)),
                        **dict(_llm_input_messages(arguments)),
                        **dict(get_attributes_from_context()),
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
            span.set_attribute(LLM_TOKEN_COUNT_PROMPT, model.last_input_token_count)
            span.set_attribute(LLM_TOKEN_COUNT_COMPLETION, model.last_output_token_count)
            span.set_attribute(LLM_MODEL_NAME, model.model_id)
            span.set_attribute(
                LLM_TOKEN_COUNT_TOTAL, model.last_input_token_count + model.last_output_token_count
            )
            span.set_status(trace_api.StatusCode.OK)
            span.set_attribute(OUTPUT_VALUE, response)
            span.set_attributes(dict(get_attributes_from_context()))
        return response


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
            span.set_attribute(SpanAttributes.TOOL_NAME, f"{instance.__class__.name}")
            span.set_attribute(SpanAttributes.TOOL_PARAMETERS, json.dumps(kwargs))
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)
            span.set_attribute(OUTPUT_VALUE, response)
            span.set_attributes(dict(get_attributes_from_context()))
        return response


INPUT_VALUE = SpanAttributes.INPUT_VALUE
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
