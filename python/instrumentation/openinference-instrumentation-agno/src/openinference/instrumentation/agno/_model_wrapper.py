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
)

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue

from agno.models.base import Model
from openinference.instrumentation import get_attributes_from_context, safe_json_dumps
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)


def _get_attr(obj: Any, key: str, default: Any = None) -> Any:
    """Helper function to get attribute from either dict or object."""
    if obj is None:
        return default
    if isinstance(obj, dict):  # It's a dict
        return obj.get(key, default)
    else:  # It's an object with attributes
        return getattr(obj, key, default)


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


def _bind_arguments(method: Callable[..., Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
    method_signature = signature(method)
    bound_args = method_signature.bind(*args, **kwargs)
    bound_args.apply_defaults()
    arguments = bound_args.arguments
    arguments = OrderedDict(
        {key: value for key, value in arguments.items() if value is not None and value != {}}
    )
    return arguments


def _llm_input_messages(arguments: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    def process_message(idx: int, message: Any) -> Iterator[Tuple[str, Any]]:
        yield f"{LLM_INPUT_MESSAGES}.{idx}.{MESSAGE_ROLE}", message.role
        if message.content:
            yield f"{LLM_INPUT_MESSAGES}.{idx}.{MESSAGE_CONTENT}", message.get_content_string()
        if message.tool_calls:
            for tool_call_index, tool_call in enumerate(message.tool_calls):
                yield (
                    f"{LLM_INPUT_MESSAGES}.{idx}.{MESSAGE_TOOL_CALLS}.{tool_call_index}.{TOOL_CALL_ID}",
                    _get_attr(tool_call, "id"),
                )
                function_obj = _get_attr(tool_call, "function", {})
                yield (
                    f"{LLM_INPUT_MESSAGES}.{idx}.{MESSAGE_TOOL_CALLS}.{tool_call_index}.{TOOL_CALL_FUNCTION_NAME}",
                    _get_attr(function_obj, "name"),
                )
                yield (
                    f"{LLM_INPUT_MESSAGES}.{idx}.{MESSAGE_TOOL_CALLS}.{tool_call_index}.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                    safe_json_dumps(_get_attr(function_obj, "arguments", {})),
                )

    messages = arguments.get("messages", [])
    for i, message in enumerate(messages):
        if message.role in ["system", "user", "assistant", "tool"]:
            yield from process_message(i, message)
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

    cleaned_input = []
    for message in arguments.get("messages", []):
        message_dict = message.to_dict()
        message_dict = {k: v for k, v in message_dict.items() if v is not None}
        cleaned_input.append(message_dict)
    yield INPUT_VALUE, safe_json_dumps({"messages": cleaned_input})


def _output_value_and_mime_type(output: str) -> Iterator[Tuple[str, Any]]:
    yield OUTPUT_MIME_TYPE, JSON

    # Try to parse the output and extract LLM_OUTPUT_MESSAGES
    try:
        output_data = json.loads(output)
        if isinstance(output_data, dict):
            # Extract message information for LLM_OUTPUT_MESSAGES (only core message fields)
            messages = []
            message = {}

            if role := output_data.get("role"):
                message["role"] = role

            if content := output_data.get("content"):
                message["content"] = content

            # Only include tool_calls if they exist and are not empty
            if tool_calls := output_data.get("tool_calls"):
                if tool_calls:  # Only include if not empty list
                    message["tool_calls"] = tool_calls

            messages.append(message)
            for i, message in enumerate(messages):
                yield f"{LLM_OUTPUT_MESSAGES}.{i}", safe_json_dumps(message)

            yield OUTPUT_VALUE, safe_json_dumps(messages)

    except (json.JSONDecodeError, TypeError):
        # Fall back to the original output if parsing fails
        yield OUTPUT_VALUE, output


def _parse_model_output(output: Any) -> str:
    if hasattr(output, "role") or hasattr(output, "content") or hasattr(output, "tool_calls"):
        try:
            result_dict = {
                "created_at": getattr(output, "created_at", None),
            }

            if hasattr(output, "role"):
                result_dict["role"] = output.role
            if hasattr(output, "content"):
                result_dict["content"] = output.content
            if hasattr(output, "tool_calls"):
                result_dict["tool_calls"] = output.tool_calls

            # Add response_usage if available
            if hasattr(output, "response_usage") and output.response_usage:
                result_dict["response_usage"] = {
                    "input_tokens": getattr(output.response_usage, "input_tokens", None),
                    "output_tokens": getattr(output.response_usage, "output_tokens", None),
                    "total_tokens": getattr(output.response_usage, "total_tokens", None),
                }

            return json.dumps(result_dict)
        except Exception:
            pass

    return json.dumps(output) if isinstance(output, dict) else str(output)


def _parse_model_output_stream(output: Any) -> Dict[str, Any]:
    # Accumulate all content and tool calls across chunks
    accumulated_content = ""
    all_tool_calls: list[Dict[str, Any]] = []

    for chunk in output:
        # Accumulate content from this chunk
        if chunk.content:
            accumulated_content += chunk.content

        # Collect tool calls from this chunk
        if chunk.tool_calls:
            for tool_call in chunk.tool_calls:
                if _get_attr(tool_call, "id"):
                    tool_call_dict = {
                        "id": _get_attr(tool_call, "id"),
                        "type": _get_attr(tool_call, "type"),
                    }
                    function_obj = _get_attr(tool_call, "function")
                    if function_obj:
                        tool_call_dict["function"] = {
                            "name": _get_attr(function_obj, "name"),
                            "arguments": _get_attr(function_obj, "arguments"),
                        }
                    all_tool_calls.append(tool_call_dict)

    # Create single message with accumulated content and all tool calls
    messages: list[Dict[str, Any]] = []
    if accumulated_content or all_tool_calls:
        result_dict: Dict[str, Any] = {"role": "assistant"}

        if accumulated_content:
            result_dict["content"] = accumulated_content

        if all_tool_calls:
            result_dict["tool_calls"] = all_tool_calls

        messages.append(result_dict)

    return {"messages": messages}


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
                **dict(_llm_input_messages(arguments)),
                **dict(_llm_invocation_parameters(model, arguments)),
                **dict(get_attributes_from_context()),
            },
        ) as span:
            span.set_status(trace_api.StatusCode.OK)
            span.set_attribute(LLM_MODEL_NAME, model.id)
            span.set_attribute(LLM_PROVIDER, model.provider)

            response = wrapped(*args, **kwargs)
            output_message = _parse_model_output(response)
            span.set_attributes(dict(_output_value_and_mime_type(output_message)))

            # Extract and set token usage from the response
            if hasattr(response, "response_usage") and response.response_usage:
                metrics = response.response_usage

                # Set token usage attributes
                if hasattr(metrics, "input_tokens") and metrics.input_tokens:
                    span.set_attribute(LLM_TOKEN_COUNT_PROMPT, metrics.input_tokens)

                if hasattr(metrics, "output_tokens") and metrics.output_tokens:
                    span.set_attribute(LLM_TOKEN_COUNT_COMPLETION, metrics.output_tokens)

                # Set cache-related tokens if available
                if hasattr(metrics, "cache_read_tokens") and metrics.cache_read_tokens:
                    span.set_attribute(
                        LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ, metrics.cache_read_tokens
                    )

                if hasattr(metrics, "cache_write_tokens") and metrics.cache_write_tokens:
                    span.set_attribute(
                        LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE, metrics.cache_write_tokens
                    )

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
            # Token usage will be set after streaming completes based on final response

            responses = []
            for chunk in wrapped(*args, **kwargs):
                responses.append(chunk)
                yield chunk

            output_message_dict = _parse_model_output_stream(responses)
            output_message = json.dumps(output_message_dict)
            span.set_attribute(OUTPUT_MIME_TYPE, JSON)
            span.set_attribute(OUTPUT_VALUE, output_message)

            # Find the final response with complete metrics (last one with response_usage)
            final_response_with_metrics = None
            for response in reversed(responses):  # Check from last to first
                if hasattr(response, "response_usage") and response.response_usage:
                    final_response_with_metrics = response
                    break

            # Extract and set token usage from the final response
            if final_response_with_metrics and final_response_with_metrics.response_usage:
                metrics = final_response_with_metrics.response_usage

                # Set token usage attributes
                if hasattr(metrics, "input_tokens") and metrics.input_tokens:
                    span.set_attribute(LLM_TOKEN_COUNT_PROMPT, metrics.input_tokens)

                if hasattr(metrics, "output_tokens") and metrics.output_tokens:
                    span.set_attribute(LLM_TOKEN_COUNT_COMPLETION, metrics.output_tokens)

                # Set cache-related tokens if available
                if hasattr(metrics, "cache_read_tokens") and metrics.cache_read_tokens:
                    span.set_attribute(
                        LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ, metrics.cache_read_tokens
                    )

                if hasattr(metrics, "cache_write_tokens") and metrics.cache_write_tokens:
                    span.set_attribute(
                        LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE, metrics.cache_write_tokens
                    )

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

            # Extract and set token usage from the response
            if hasattr(response, "response_usage") and response.response_usage:
                metrics = response.response_usage

                # Set token usage attributes
                if hasattr(metrics, "input_tokens") and metrics.input_tokens:
                    span.set_attribute(LLM_TOKEN_COUNT_PROMPT, metrics.input_tokens)

                if hasattr(metrics, "output_tokens") and metrics.output_tokens:
                    span.set_attribute(LLM_TOKEN_COUNT_COMPLETION, metrics.output_tokens)

                if hasattr(metrics, "total_tokens") and metrics.total_tokens:
                    span.set_attribute(LLM_TOKEN_COUNT_TOTAL, metrics.total_tokens)

                # Set cache-related tokens if available
                if hasattr(metrics, "cache_read_tokens") and metrics.cache_read_tokens:
                    span.set_attribute(
                        LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ, metrics.cache_read_tokens
                    )

                if hasattr(metrics, "cache_write_tokens") and metrics.cache_write_tokens:
                    span.set_attribute(
                        LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE, metrics.cache_write_tokens
                    )

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
            # Token usage will be set after streaming completes based on final response

            responses = []
            async for chunk in wrapped(*args, **kwargs):  # type: ignore[attr-defined]
                responses.append(chunk)
                yield chunk

            output_message_dict = _parse_model_output_stream(responses)
            output_message = json.dumps(output_message_dict)
            span.set_attribute(OUTPUT_MIME_TYPE, JSON)
            span.set_attribute(OUTPUT_VALUE, output_message)

            # Find the final response with complete metrics (last one with response_usage)
            final_response_with_metrics = None
            for response in reversed(responses):  # Check from last to first
                if hasattr(response, "response_usage") and response.response_usage:
                    final_response_with_metrics = response
                    break

            # Extract and set token usage from the final response
            if final_response_with_metrics and final_response_with_metrics.response_usage:
                metrics = final_response_with_metrics.response_usage

                # Set token usage attributes
                if hasattr(metrics, "input_tokens") and metrics.input_tokens:
                    span.set_attribute(LLM_TOKEN_COUNT_PROMPT, metrics.input_tokens)

                if hasattr(metrics, "output_tokens") and metrics.output_tokens:
                    span.set_attribute(LLM_TOKEN_COUNT_COMPLETION, metrics.output_tokens)

                # Set cache-related tokens if available
                if hasattr(metrics, "cache_read_tokens") and metrics.cache_read_tokens:
                    span.set_attribute(
                        LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ, metrics.cache_read_tokens
                    )

                if hasattr(metrics, "cache_write_tokens") and metrics.cache_write_tokens:
                    span.set_attribute(
                        LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE, metrics.cache_write_tokens
                    )


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
USER_ID = SpanAttributes.USER_ID

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

# tool attributes
TOOL_JSON_SCHEMA = ToolAttributes.TOOL_JSON_SCHEMA
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
TOOL_CALL_ID = ToolCallAttributes.TOOL_CALL_ID

# span kinds
LLM = OpenInferenceSpanKindValues.LLM.value

LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ = SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ
LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE = (
    SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE
)
