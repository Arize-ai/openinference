"""
Attribute extraction utilities for Anthropic models in AWS Bedrock.

This module provides functions to extract and format attributes from Anthropic model
requests and responses for OpenInference tracing. It handles message parsing, token
counting, parameter validation, and attribute formatting specific to Anthropic's
Claude VV3+ models running on AWS Bedrock.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, List, Mapping, Tuple

from opentelemetry.trace import Span

from openinference.instrumentation import (
    Image,
    ImageMessageContent,
    Message,
    TextMessageContent,
    ToolCall,
    ToolCallFunction,
    get_input_attributes,
    get_llm_attributes,
    get_llm_model_name_attributes,
    get_llm_output_message_attributes,
    get_output_attributes,
    get_span_kind_attributes,
)
from openinference.semconv.trace import (
    OpenInferenceLLMProviderValues,
    OpenInferenceLLMSystemValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)


def _get_input_messages(content: Any) -> List[Message]:
    """
    Convert raw message content to OpenInference Message objects.

    Processes various message types including text, images, tool calls, and tool results
    from Anthropic message format into standardized Message objects for tracing.

    Args:
        content: Raw message content from the request, typically a list of message dictionaries

    Returns:
        List of Message objects with properly formatted content, roles, and tool calls
        Handles multiple content types:
        - Text messages
        - Image messages (base64 encoded)
        - Tool use calls
        - Tool result responses
    """
    messages = []
    for message in content:
        if isinstance(message.get("content"), str):
            messages.append(Message(content=message.get("content"), role=message.get("role")))
        elif isinstance(message.get("content"), Iterable):
            contents: List[Any] = []
            tool_calls: List[ToolCall] = []
            for part_content in message.get("content"):
                if part_content.get("type") == "image":
                    base64_img = part_content.get("source", {}).get("data")
                    media_type = part_content.get("source", {}).get("media_type") or "image/png"
                    image = Image(url=f"data:{media_type};base64,{base64_img}")
                    contents.append(ImageMessageContent(image=image, type="image"))
                elif part_content.get("type") == "tool_use":
                    tool_call = ToolCall(
                        id=part_content.get("id", ""),
                        function=ToolCallFunction(
                            name=part_content.get("name", ""), arguments=part_content.get("input")
                        ),
                    )
                    tool_calls.append(tool_call)
                elif part_content.get("type") == "tool_result":
                    tool_call_result = Message(
                        tool_call_id=part_content.get("tool_use_id", ""),
                        content=part_content.get("content", ""),
                        role=message.get("role"),
                    )
                    messages.append(tool_call_result)
                else:
                    contents.append(TextMessageContent(text=part_content.get("text"), type="text"))
            if contents or tool_calls:
                messages.append(
                    Message(contents=contents, tool_calls=tool_calls, role=message.get("role"))
                )
    return messages


def _get_llm_token_counts(usage: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Extract token count information from usage statistics.

    Processes token usage data from Anthropic responses, including prompt tokens,
    completion tokens, and cache-related token counts.

    Args:
        usage: Dictionary containing token usage statistics from the response

    Yields:
        Tuples of (attribute_name, token_count) for various token types:
        - Prompt tokens (including cache creation and read)
        - Completion tokens
        - Cache-specific token details
    """
    prompt_tokens = sum(
        usage.get(key, 0)
        for key in [
            "input_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ]
    )
    if prompt_tokens:
        yield LLM_TOKEN_COUNT_PROMPT, prompt_tokens

    if usage.get("output_tokens"):
        yield LLM_TOKEN_COUNT_COMPLETION, usage.get("output_tokens")
    if usage.get("cache_read_input_tokens"):
        yield LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ, usage.get("cache_read_input_tokens")
    if usage.get("cache_creation_input_tokens"):
        yield LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE, usage.get("cache_creation_input_tokens")


def _get_llm_model_name_from_input(model_id: str) -> str:
    """
    Extract the model name from a Bedrock model ID.

    Parses Bedrock model IDs which typically follow the format "vendor.model_name"
    and extracts just the model name portion.

    Args:
        model_id: The full Bedrock model identifier (e.g., "anthropic.claude-3-sonnet-20240229
        -v1:0")

    Returns:
        The extracted model name, or the original model_id if parsing fails
    """
    if model_id and model_id.count(".") == 1:
        vendor, model_name = model_id.split(".")
        return model_name
    return model_id


def _get_invocation_parameters(kwargs: Mapping[str, Any]) -> Any:
    """
    Extract invocation parameters from the request kwargs.

    Filters the request parameters to include only those that should be
    recorded as invocation parameters, excluding standard fields like
    messages and modelId.

    Args:
        kwargs: The complete request parameters mapping

    Returns:
        Dictionary containing only the invocation parameters that should
        be recorded for tracing purposes
    """
    return {key: value for key, value in kwargs.items() if _validate_invocation_parameter(key)}


def _get_output_messages(response: Any) -> List[Message]:
    """
    Extract output messages and tool calls from the Anthropic response.

    Processes the response content to extract both text messages and tool calls,
    converting them into standardized Message objects for tracing.

    Args:
        response: The response body from the Anthropic model

    Returns:
        List of Message objects containing the response content and any tool calls
    """
    output_messages = []
    for block in response.get("content", []):
        if block.get("type") == "tool_use":
            tool_call_function = ToolCallFunction(
                name=block.get("name", ""), arguments=block.get("input")
            )
            tool_calls = [ToolCall(id=block.get("id", ""), function=tool_call_function)]
            output_messages.append(
                Message(
                    tool_call_id=block.get("tool_call_id", ""),
                    role=response.get("role"),
                    tool_calls=tool_calls,
                )
            )

        if block.get("type") == "text":
            output_messages.append(
                Message(content=block.get("text"), role=response.get("role") or "assistant")
            )
    return output_messages


def _validate_invocation_parameter(parameter: Any) -> bool:
    """
    Validate whether a parameter should be included in invocation parameters.

    Determines if a given parameter key should be recorded as part of the
    invocation parameters for tracing purposes.

    Args:
        parameter: The parameter key to validate

    Returns:
        True if the parameter should be included in invocation parameters,
        False if it should be excluded
    """
    excluded_params = (
        "messages",
        "modelId",
    )
    return parameter not in excluded_params


def get_llm_input_attributes(request_body: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    """
    Extract and format input attributes for LLM tracing from Anthropic request.

    Processes the request body and model information to create a comprehensive
    set of attributes for OpenInference tracing, including model details,
    input messages, and invocation parameters.

    Args:
        request_body: The complete request body sent to the Anthropic model
        model_id: The Bedrock model identifier

    Returns:
        Dictionary containing formatted attributes for tracing:
        - LLM system and provider information
        - Model name and parameters
        - Input messages and content
        - Span kind and input attributes
    """
    invocation_parameters = _get_invocation_parameters(request_body)
    input_messages = _get_input_messages(request_body.get("messages"))
    return {
        **get_llm_attributes(
            model_name=_get_llm_model_name_from_input(model_id),
            system=OpenInferenceLLMSystemValues.ANTHROPIC.value,
            provider=OpenInferenceLLMProviderValues.AWS.value,
            invocation_parameters=invocation_parameters,
            input_messages=input_messages,
        ),
        **get_span_kind_attributes(OpenInferenceSpanKindValues.LLM),
        **get_input_attributes(request_body),
    }


def get_llm_output_attributes(response_body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and format output attributes for LLM tracing from Anthropic response.

    Processes the response body to create attributes for OpenInference tracing,
    including output messages, token usage, and model information.

    Args:
        response_body: The complete response body from the Anthropic model

    Returns:
        Dictionary containing formatted attributes for tracing:
        - Output messages and content
        - Token count statistics
        - Model name information
        - General output attributes
    """
    output_messages = _get_output_messages(response_body)
    return {
        **get_llm_output_message_attributes(output_messages),
        **get_output_attributes(response_body),
        **get_llm_model_name_attributes(response_body.get("model")),
        **dict(_get_llm_token_counts(response_body.get("usage") or {})),
    }


def set_input_attributes(span: Span, request_body: Dict[str, Any], model_id: str) -> None:
    """
    Set input attributes on the span for Anthropic model requests.

    Convenience function that extracts and sets all input-related attributes
    from the request body and model ID onto the provided span.

    Args:
        span: The OpenTelemetry span to set attributes on
        request_body: The complete request body sent to the Anthropic model
        model_id: The Bedrock model identifier
    """
    span.set_attributes(get_llm_input_attributes(request_body, model_id))


def set_response_attributes(span: Span, response_body: Dict[str, Any]) -> None:
    """
    Set response attributes on the span for Anthropic model responses.

    Convenience function that extracts and sets all response-related attributes
    from the response body onto the provided span.

    Args:
        span: The OpenTelemetry span to set attributes on
        response_body: The complete response body from the Anthropic model
    """
    span.set_attributes(get_llm_output_attributes(response_body))


# Token count attribute constants for convenience
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ = SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ
LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE = (
    SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE
)
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
