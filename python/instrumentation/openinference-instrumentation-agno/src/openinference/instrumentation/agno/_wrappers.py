import json
from enum import Enum
from inspect import signature
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    Mapping,
    Optional,
    OrderedDict,
    Tuple,
    Union,
)

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue

from agno.agent import Agent
from agno.models.base import Model
from agno.team import Team
from agno.tools.function import Function, FunctionCall
from agno.tools.toolkit import Toolkit
from openinference.instrumentation import get_attributes_from_context, safe_json_dumps
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)


def _flatten(mapping: Optional[Mapping[str, Any]]) -> Iterator[Tuple[str, AttributeValue]]:
    if not mapping:
        return
    for key, value in mapping.items():
        if value is None:
            continue
        if isinstance(value, Mapping):
            for sub_key, sub_value in _flatten(value):
                yield f"{key}.{sub_key}", sub_value
        elif isinstance(value, list) and any(isinstance(item, Mapping) for item in value):
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
    arguments = bound_args.arguments
    arguments = OrderedDict(
        {key: value for key, value in arguments.items() if value is not None and value != {}}
    )
    return arguments


def _strip_method_args(arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in arguments.items() if key not in ("self", "cls")}


def _agent_run_attributes(
    agent: Union[Agent, Team], key_suffix: str = ""
) -> Iterator[Tuple[str, AttributeValue]]:
    if isinstance(agent, Team):
        yield f"agno{key_suffix}.team", agent.name or ""
        for member in agent.members:
            yield from _agent_run_attributes(member, f".{member.name}")
    elif isinstance(agent, Agent):
        if agent.name:
            yield f"agno{key_suffix}.agent", agent.name or ""

        if agent.session_id:
            yield SESSION_ID, agent.session_id

        if agent.knowledge:
            yield f"agno{key_suffix}.knowledge", agent.knowledge.__class__.__name__

        if agent.tools:
            tool_names = []
            for tool in agent.tools:
                if isinstance(tool, Function):
                    tool_names.append(tool.name)
                elif isinstance(tool, Toolkit):
                    tool_names.extend([f for f in tool.functions.keys()])
                elif callable(tool):
                    tool_names.append(tool.__name__)
                else:
                    tool_names.append(str(tool))
            yield "agno{key_suffix}.tools", tool_names


class _RunWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    def run(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        agent = instance
        if hasattr(agent, "name") and agent.name:
            agent_name = agent.name.replace(" ", "_").replace("-", "_")
        else:
            agent_name = "Agent"
        span_name = f"{agent_name}.run"

        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: AGENT,
                        INPUT_VALUE: _get_input_value(
                            wrapped,
                            *args,
                            **kwargs,
                        ),
                        **dict(_agent_run_attributes(agent)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        ) as span:
            try:
                run_response = wrapped(*args, **kwargs)
                span.set_status(trace_api.StatusCode.OK)
                span.set_attribute(OUTPUT_VALUE, run_response.to_json())
                span.set_attribute(OUTPUT_MIME_TYPE, JSON)
                return run_response

            except Exception as e:
                span.set_status(trace_api.StatusCode.ERROR, str(e))
                raise

    def run_stream(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        agent = instance
        if hasattr(agent, "name") and agent.name:
            agent_name = agent.name.replace(" ", "_").replace("-", "_")
        else:
            agent_name = "Agent"
        span_name = f"{agent_name}.run"

        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: AGENT,
                        INPUT_VALUE: _get_input_value(
                            wrapped,
                            *args,
                            **kwargs,
                        ),
                        **dict(_agent_run_attributes(agent)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        ) as span:
            try:
                yield from wrapped(*args, **kwargs)
                run_response = agent.run_response
                span.set_status(trace_api.StatusCode.OK)
                span.set_attribute(OUTPUT_VALUE, run_response.to_json())
                span.set_attribute(OUTPUT_MIME_TYPE, JSON)

            except Exception as e:
                span.set_status(trace_api.StatusCode.ERROR, str(e))
                raise

    async def arun(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            response = await wrapped(*args, **kwargs)
            return response

        agent = instance
        if hasattr(agent, "name"):
            agent_name = agent.name.replace(" ", "_").replace("-", "_")
        else:
            agent_name = "Agent"
        span_name = f"{agent_name}.run"

        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: AGENT,
                        INPUT_VALUE: _get_input_value(
                            wrapped,
                            *args,
                            **kwargs,
                        ),
                        **dict(_agent_run_attributes(agent)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        ) as span:
            try:
                run_response = await wrapped(*args, **kwargs)
                span.set_status(trace_api.StatusCode.OK)
                span.set_attribute(OUTPUT_VALUE, run_response.to_json())
                span.set_attribute(OUTPUT_MIME_TYPE, JSON)
                return run_response
            except Exception as e:
                span.set_status(trace_api.StatusCode.ERROR, str(e))
                raise

    async def arun_stream(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            async for response in await wrapped(*args, **kwargs):
                yield response

        agent = instance
        if hasattr(agent, "name") and agent.name:
            agent_name = agent.name.replace(" ", "_").replace("-", "_")
        else:
            agent_name = "Agent"
        span_name = f"{agent_name}.run"

        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: AGENT,
                        INPUT_VALUE: _get_input_value(
                            wrapped,
                            *args,
                            **kwargs,
                        ),
                        **dict(_agent_run_attributes(agent)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        ) as span:
            try:
                async for response in wrapped(*args, **kwargs):  # type: ignore[attr-defined]
                    yield response
                run_response = agent.run_response
                span.set_status(trace_api.StatusCode.OK)
                span.set_attribute(OUTPUT_VALUE, run_response.to_json())
                span.set_attribute(OUTPUT_MIME_TYPE, JSON)
            except Exception as e:
                span.set_status(trace_api.StatusCode.ERROR, str(e))
                raise


def _llm_input_messages(arguments: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    def process_message(idx: int, role: str, content: str) -> Iterator[Tuple[str, Any]]:
        yield f"{LLM_INPUT_MESSAGES}.{idx}.{MESSAGE_ROLE}", role
        yield f"{LLM_INPUT_MESSAGES}.{idx}.{MESSAGE_CONTENT}", content

    messages = arguments.get("messages", [])
    for i, message in enumerate(messages):
        role, content = message.role, message.get_content_string()
        if content:
            yield from process_message(i, role, content)

    tools = arguments.get("tools", [])
    for tool_index, tool in enumerate(tools):
        yield f"{LLM_TOOLS}.{tool_index}.{TOOL_JSON_SCHEMA}", safe_json_dumps(tool)


def _llm_invocation_parameters(
    model: Model, arguments: Optional[Mapping[str, Any]] = None
) -> Iterator[Tuple[str, Any]]:
    request_kwargs = {}
    # TODO (v2.0.0): with the cleanup of the agno.models.base.Model class we will
    # handle these attributes in a more consistent way.
    if getattr(model, "request_kwargs", None):
        request_kwargs = model.request_kwargs  # type: ignore[attr-defined]
    if getattr(model, "request_params", None):
        request_kwargs = model.request_params  # type: ignore[attr-defined]
    if getattr(model, "get_request_kwargs", None):
        request_kwargs = model.get_request_kwargs()  # type: ignore[attr-defined]
    if getattr(model, "get_request_params", None):
        # Special handling for OpenAIResponses model
        if model.__class__.__name__ == "OpenAIResponses" and arguments:
            messages = arguments.get("messages", [])
            request_kwargs = model.get_request_params(messages=messages)  # type: ignore[attr-defined]
        else:
            request_kwargs = model.get_request_params()  # type: ignore[attr-defined]

    if request_kwargs:
        filtered_kwargs = _filter_sensitive_params(request_kwargs)
        yield LLM_INVOCATION_PARAMETERS, safe_json_dumps(filtered_kwargs)


def _filter_sensitive_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Filter out sensitive parameters from model request parameters."""
    sensitive_keys = frozenset(
        [
            "api_key",
            "api_base",
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_access_key",
            "aws_secret_key",
            "azure_endpoint",
            "azure_deployment",
            "azure_ad_token",
            "azure_ad_token_provider",
        ]
    )

    return {
        key: "[REDACTED]"
        if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys)
        else value
        for key, value in params.items()
    }


def _input_value_and_mime_type(arguments: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    yield INPUT_MIME_TYPE, JSON
    yield INPUT_VALUE, safe_json_dumps(arguments)


def _output_value_and_mime_type(output: str) -> Iterator[Tuple[str, Any]]:
    yield OUTPUT_MIME_TYPE, JSON
    yield OUTPUT_VALUE, output


def _parse_model_output(output: Any) -> str:
    if hasattr(output, "model_dump_json"):
        return output.model_dump_json()  # type: ignore[no-any-return]
    elif isinstance(output, dict):
        return json.dumps(output)
    else:
        return str(output)


class _ModelWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    def run(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        arguments = _bind_arguments(wrapped, *args, **kwargs)

        model = instance
        model_name = model.name
        span_name = f"{model_name}.invoke"

        with self._tracer.start_as_current_span(
            span_name,
            attributes={
                OPENINFERENCE_SPAN_KIND: LLM,
                **dict(_input_value_and_mime_type(arguments)),
                **dict(_llm_invocation_parameters(model, arguments)),
                **dict(_llm_input_messages(arguments)),
                **dict(get_attributes_from_context()),
            },
        ) as span:
            span.set_status(trace_api.StatusCode.OK)
            span.set_attribute(LLM_MODEL_NAME, model.id)
            span.set_attribute(LLM_PROVIDER, model.provider)

            response = wrapped(*args, **kwargs)
            output_message = _parse_model_output(response)

            span.set_attributes(dict(_output_value_and_mime_type(output_message)))
            return response

    def run_stream(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        arguments = _bind_arguments(wrapped, *args, **kwargs)

        model = instance
        model_name = model.name
        span_name = f"{model_name}.invoke_stream"

        with self._tracer.start_as_current_span(
            span_name,
            attributes={
                OPENINFERENCE_SPAN_KIND: LLM,
                **dict(_input_value_and_mime_type(arguments)),
                **dict(_llm_invocation_parameters(model)),
                **dict(_llm_input_messages(arguments)),
                **dict(get_attributes_from_context()),
            },
        ) as span:
            span.set_status(trace_api.StatusCode.OK)
            span.set_attribute(LLM_MODEL_NAME, model.id)
            span.set_attribute(LLM_PROVIDER, model.provider)

            responses = []
            for chunk in wrapped(*args, **kwargs):
                responses.append(chunk)
                yield chunk
            output_message = json.dumps([_parse_model_output(response) for response in responses])
            span.set_attributes(dict(_output_value_and_mime_type(output_message)))

    async def arun(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        arguments = _bind_arguments(wrapped, *args, **kwargs)

        model = instance
        model_name = model.name
        span_name = f"{model_name}.ainvoke"

        with self._tracer.start_as_current_span(
            span_name,
            attributes={
                OPENINFERENCE_SPAN_KIND: LLM,
                **dict(_input_value_and_mime_type(arguments)),
                **dict(_llm_invocation_parameters(model)),
                **dict(_llm_input_messages(arguments)),
                **dict(get_attributes_from_context()),
            },
        ) as span:
            span.set_status(trace_api.StatusCode.OK)
            span.set_attribute(LLM_MODEL_NAME, model.id)
            span.set_attribute(LLM_PROVIDER, model.provider)

            response = await wrapped(*args, **kwargs)
            output_message = _parse_model_output(response)

            span.set_attributes(dict(_output_value_and_mime_type(output_message)))
            return response

    async def arun_stream(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            async for response in wrapped(*args, **kwargs):  # type: ignore[attr-defined]
                yield response
            return

        arguments = _bind_arguments(wrapped, *args, **kwargs)

        model = instance
        model_name = model.name
        span_name = f"{model_name}.ainvoke_stream"

        with self._tracer.start_as_current_span(
            span_name,
            attributes={
                OPENINFERENCE_SPAN_KIND: LLM,
                **dict(_input_value_and_mime_type(arguments)),
                **dict(_llm_invocation_parameters(model)),
                **dict(_llm_input_messages(arguments)),
                **dict(get_attributes_from_context()),
            },
        ) as span:
            span.set_status(trace_api.StatusCode.OK)
            span.set_attribute(LLM_MODEL_NAME, model.id)
            span.set_attribute(LLM_PROVIDER, model.provider)

            responses = []
            async for chunk in wrapped(*args, **kwargs):  # type: ignore[attr-defined]
                responses.append(chunk)
                yield chunk
            output_message = json.dumps([_parse_model_output(response) for response in responses])
            span.set_attributes(dict(_output_value_and_mime_type(output_message)))


def _function_call_attributes(function_call: FunctionCall) -> Iterator[Tuple[str, Any]]:
    function = function_call.function
    function_name = function.name
    function_arguments = function_call.arguments

    yield TOOL_NAME, function_name

    if function_description := getattr(function, "description", None):
        yield TOOL_DESCRIPTION, function_description
    yield TOOL_PARAMETERS, safe_json_dumps(function_arguments)


def _output_value_and_mime_type_for_tool_span(result: Any) -> Iterator[Tuple[str, Any]]:
    yield OUTPUT_VALUE, str(result)
    yield OUTPUT_MIME_TYPE, TEXT


class _FunctionCallWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    def run(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        function_call = instance
        function = function_call.function
        function_name = function.name
        function_arguments = function_call.arguments

        span_name = f"{function_name}"

        with self._tracer.start_as_current_span(
            span_name,
            attributes={
                OPENINFERENCE_SPAN_KIND: TOOL,
                INPUT_VALUE: safe_json_dumps(function_arguments),
                **dict(_function_call_attributes(function_call)),
                **dict(get_attributes_from_context()),
            },
        ) as span:
            response = wrapped(*args, **kwargs)
            function_result = function_call.result

            span.set_status(trace_api.StatusCode.OK)
            span.set_attributes(
                dict(
                    _output_value_and_mime_type_for_tool_span(
                        result=function_result,
                    )
                )
            )
        return response

    async def arun(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        function_call = instance
        function = function_call.function
        function_name = function.name

        span_name = f"{function_name}"

        with self._tracer.start_as_current_span(
            span_name,
            attributes={
                OPENINFERENCE_SPAN_KIND: TOOL,
                INPUT_VALUE: _get_input_value(
                    wrapped,
                    *args,
                    **kwargs,
                ),
                **dict(_function_call_attributes(function_call)),
                **dict(get_attributes_from_context()),
            },
        ) as span:
            response = await wrapped(*args, **kwargs)
            function_result = function_call.result

            span.set_status(trace_api.StatusCode.OK)
            span.set_attributes(
                dict(
                    _output_value_and_mime_type_for_tool_span(
                        result=function_result,
                    )
                )
            )
        return response


# span attributes
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
SESSION_ID = SpanAttributes.SESSION_ID
LLM_TOOLS = SpanAttributes.LLM_TOOLS
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_PROVIDER = SpanAttributes.LLM_PROVIDER
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
LLM_PROMPTS = SpanAttributes.LLM_PROMPTS
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
LLM_FUNCTION_CALL = SpanAttributes.LLM_FUNCTION_CALL
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
TEXT = OpenInferenceMimeTypeValues.TEXT.value
JSON = OpenInferenceMimeTypeValues.JSON.value

# span kinds
AGENT = OpenInferenceSpanKindValues.AGENT.value
LLM = OpenInferenceSpanKindValues.LLM.value
TOOL = OpenInferenceSpanKindValues.TOOL.value

# tool attributes
TOOL_JSON_SCHEMA = ToolAttributes.TOOL_JSON_SCHEMA

# tool call attributes
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
TOOL_CALL_ID = ToolCallAttributes.TOOL_CALL_ID
