"""
Attribute extraction utilities for AWS Bedrock invoke_model operations.

This module provides functions to extract and set OpenInference tracing attributes
from AWS Bedrock invoke_model requests and responses. It handles model identification,
token counting, parameter extraction, and response processing for various model
providers supported by Bedrock.

The module supports multiple model providers including:
- AI21 Labs
- Anthropic (Claude V1, V2 models)
- Cohere
- Meta (Llama models)
- Amazon (Titan, Nova)
- And other Bedrock-supported models
"""

from typing import Any, Dict, List

from opentelemetry import trace as trace_api
from opentelemetry.trace import Span
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import (
    Message,
    TextMessageContent,
    Tool,
    ToolCall,
    ToolCallFunction,
    get_llm_attributes,
    get_llm_output_message_attributes,
    get_llm_tool_attributes,
    get_output_attributes,
    safe_json_dumps,
)
from openinference.semconv.trace import (
    OpenInferenceLLMProviderValues,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)


def _set_span_attribute(span: trace_api.Span, name: str, value: AttributeValue) -> None:
    """
    Set a span attribute if the value is not None or empty string.

    This helper function provides safe attribute setting by checking for None
    and empty string values before setting the attribute on the span.

    Args:
        span: The OpenTelemetry span to set the attribute on
        name: The attribute name/key
        value: The attribute value to set
    """
    if value is not None and value != "":
        span.set_attribute(name, value)


def _set_model_name_attributes(
    span: Span, response_body: Dict[str, Any], kwargs: Dict[str, Any]
) -> None:
    """
    Set model name and output value attributes based on the model vendor.

    Extracts the model ID from kwargs and sets the LLM_MODEL_NAME attribute.
    Also extracts the appropriate output content based on the model vendor's
    response format and sets the OUTPUT_VALUE attribute.

    Args:
        span: The OpenTelemetry span to set attributes on
        response_body: The response body from the Bedrock model
        kwargs: The original request kwargs containing modelId.
        Different model vendors use different response formats:
        - AI21: Uses "completions" field
        - Anthropic: Uses "completion" field
        - Cohere: Uses "generations" field
        - Meta: Uses "generation" field
        - Amazon Nova: Uses "output.message.content" field
    """
    content = ""
    model_id = kwargs.get("modelId")
    if model_id:
        _set_span_attribute(span, SpanAttributes.LLM_MODEL_NAME, model_id)
    if "amazon.nova" in str(model_id):
        message = response_body.get("output", {}).get("message")
        if message:
            span.set_attributes(get_output_attributes(message))
        output_messages = _build_nova_output_messages(response_body)
        if output_messages:
            span.set_attributes(get_llm_output_message_attributes(output_messages))
    else:
        vendor = ""
        if model_id and isinstance(model_id, str):
            (vendor, *_) = model_id.split(".")
        if vendor == "amazon":
            # Titan format: {results: [{outputText: "..."}]}
            results = response_body.get("results", [])
            content = results[0].get("outputText", "") if results else ""
        elif vendor == "ai21":
            content = str(response_body.get("completions"))
        elif vendor == "anthropic":
            content = str(response_body.get("completion"))
        elif vendor == "cohere":
            content = str(response_body.get("generations"))
        elif vendor == "meta":
            content = str(response_body.get("generation"))
    if content:
        _set_span_attribute(span, SpanAttributes.OUTPUT_VALUE, content)


def _set_token_count_attributes(span: Span, metadata: Dict[str, Any]) -> None:
    """
    Set token count attributes from Bedrock response metadata.

    Extracts token count information from the HTTP headers in the response
    metadata and sets the appropriate LLM token count attributes on the span.

    Args:
        span: The OpenTelemetry span to set attributes on
        metadata: The response metadata containing HTTPHeaders

        Bedrock provides token counts in HTTP headers:
        - x-amzn-bedrock-input-token-count: Prompt tokens used
        - x-amzn-bedrock-output-token-count: Completion tokens generated
        Total tokens are calculated as the sum of input and output tokens.
    """
    if headers := metadata.get("HTTPHeaders"):
        if input_token_count := headers.get("x-amzn-bedrock-input-token-count"):
            input_token_count = int(input_token_count)
            _set_span_attribute(span, SpanAttributes.LLM_TOKEN_COUNT_PROMPT, input_token_count)
        if response_token_count := headers.get("x-amzn-bedrock-output-token-count"):
            response_token_count = int(response_token_count)
            _set_span_attribute(
                span,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
                response_token_count,
            )
        if total_token_count := (
            input_token_count + response_token_count
            if input_token_count and response_token_count
            else None
        ):
            _set_span_attribute(span, SpanAttributes.LLM_TOKEN_COUNT_TOTAL, total_token_count)


def set_input_attributes(span: Span, request_body: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
    """
    Set input-related attributes on the span from the request body.

    Extracts the prompt from the request body and sets it as the INPUT_VALUE
    attribute. Also serializes the remaining request parameters as invocation
    parameters and sets the span kind to LLM.

    Args:
        span: The OpenTelemetry span to set attributes on
        request_body: The request body containing prompt and other parameters
        kwargs:  request input params
    """
    model_id = kwargs.get("modelId")
    if model_id:
        _set_span_attribute(span, SpanAttributes.LLM_MODEL_NAME, model_id)

    span.set_attribute(SpanAttributes.LLM_PROVIDER, OpenInferenceLLMProviderValues.AWS.value)

    if "amazon.nova" in str(model_id):
        input_value = safe_json_dumps(request_body.get("messages", []))
        span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, OpenInferenceMimeTypeValues.JSON.value)
        invocation_parameters = safe_json_dumps(request_body.get("inferenceConfig", {}))
        input_messages = _build_nova_input_messages(request_body)
        if input_messages:
            span.set_attributes(get_llm_attributes(input_messages=input_messages))
        tools = _build_nova_tools(request_body)
        if tools:
            span.set_attributes(get_llm_tool_attributes(tools))
    else:
        # All other models (anthropic completion style, cohere, meta, ai21):
        # input is the prompt field, remaining body fields are invocation params
        input_value = request_body.pop("prompt", None)
        invocation_parameters = safe_json_dumps(request_body)

    _set_span_attribute(span, SpanAttributes.INPUT_VALUE, input_value)
    _set_span_attribute(span, SpanAttributes.LLM_INVOCATION_PARAMETERS, invocation_parameters)
    span.set_attribute(
        SpanAttributes.OPENINFERENCE_SPAN_KIND,
        OpenInferenceSpanKindValues.LLM.value,
    )


def set_response_attributes(
    span: Span, kwargs: Dict[str, Any], response_body: Dict[str, Any], response: Dict[str, Any]
) -> None:
    """
    Set response-related attributes on the span from the model response.

    Orchestrates the setting of model name, output value, and token count
    attributes by calling the appropriate helper functions.

    Args:
        span: The OpenTelemetry span to set attributes on
        kwargs: The original request kwargs containing modelId
        response_body: The response body from the Bedrock model
        response: The complete response object including metadata
    """
    _set_model_name_attributes(span, response_body, kwargs)
    if metadata := response.get("ResponseMetadata"):
        _set_token_count_attributes(span, metadata)
    if "amazon.nova" in str(kwargs.get("modelId")):
        _set_nova_body_token_attributes(span, response_body)


def is_claude_message_api(model_id: str) -> bool:
    """
    Determine if the model ID corresponds to Claude's Messages API.

    Checks if the given model ID represents an Anthropic Claude model that
    uses the Messages API format (as opposed to the legacy Text Completions API).

    Args:
        model_id: The Bedrock model identifier to check

    Returns:
        True if the model uses Claude's Messages API, False otherwise

    Note:
        Claude v2 and Claude Instant v1 models use the legacy Text Completions API,
        while newer Claude models (v3+) use the Messages API format.

    Examples:
        >>> is_claude_message_api("anthropic.claude-3-sonnet-20240229-v1:0")
        True
        >>> is_claude_message_api("anthropic.claude-v2")
        False
        >>> is_claude_message_api("ai21.j2-ultra-v1")
        False
    """
    return (
        model_id is not None
        and "anthropic" in model_id
        and "claude-v2" not in str(model_id)
        and "claude-instant-v1" not in str(model_id)
    )


def _build_nova_tools(request_body: Dict[str, Any]) -> List[Tool]:
    """Extract Nova tool definitions from a request body's toolConfig."""
    tool_config = request_body.get("toolConfig")
    if not isinstance(tool_config, dict):
        return []
    raw_tools = tool_config.get("tools")
    if not isinstance(raw_tools, list):
        return []
    tools: List[Tool] = []
    for tool in raw_tools:
        # Record each tool element verbatim (keeping the Converse tagged-union
        # envelope) to match the OpenAI/Anthropic instrumentors and the Converse
        # extractor, and to preserve non-toolSpec members such as systemTool.
        if isinstance(tool, dict):
            tools.append(Tool(json_schema=dict(tool)))
    return tools


def _build_nova_input_messages(request_body: Dict[str, Any]) -> List[Message]:
    messages: List[Message] = []
    for system_block in request_body.get("system", []):
        if isinstance(system_block, dict) and (text := system_block.get("text")):
            messages.append(Message(role="system", content=text))
    for msg in request_body.get("messages", []):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        contents: List[Any] = []
        tool_calls: List[ToolCall] = []
        tool_results: List[Message] = []
        for block in msg.get("content", []):
            if not isinstance(block, dict):
                continue
            if text := block.get("text"):
                contents.append(TextMessageContent(text=text, type="text"))
            elif tool_use := block.get("toolUse"):
                tool_calls.append(
                    ToolCall(
                        id=tool_use.get("toolUseId", ""),
                        function=ToolCallFunction(
                            name=tool_use.get("name", ""),
                            arguments=tool_use.get("input"),
                        ),
                    )
                )
            elif tool_result := block.get("toolResult"):
                tool_use_id = tool_result.get("toolUseId", "")
                result_parts: List[str] = []
                for result_block in tool_result.get("content", []):
                    if not isinstance(result_block, dict):
                        continue
                    if result_text := result_block.get("text"):
                        result_parts.append(result_text)
                    elif result_json := result_block.get("json"):
                        result_parts.append(safe_json_dumps(result_json))
                result_content = "\n".join(result_parts) if result_parts else ""
                tr_msg = Message(role=role, tool_call_id=tool_use_id)
                if result_content:
                    tr_msg["content"] = result_content
                tool_results.append(tr_msg)
        messages.extend(tool_results)
        if contents or tool_calls:
            msg_obj = Message(role=role)
            if contents:
                msg_obj["contents"] = contents
            if tool_calls:
                msg_obj["tool_calls"] = tool_calls
            messages.append(msg_obj)
    return messages


def _build_nova_output_messages(response_body: Dict[str, Any]) -> List[Message]:
    message = response_body.get("output", {}).get("message", {})
    if not message:
        return []
    role = message.get("role", "assistant")
    contents: List[Any] = []
    tool_calls: List[ToolCall] = []
    for block in message.get("content", []):
        if not isinstance(block, dict):
            continue
        if text := block.get("text"):
            contents.append(TextMessageContent(text=text, type="text"))
        elif tool_use := block.get("toolUse"):
            tool_calls.append(
                ToolCall(
                    id=tool_use.get("toolUseId", ""),
                    function=ToolCallFunction(
                        name=tool_use.get("name", ""),
                        arguments=tool_use.get("input"),
                    ),
                )
            )
    msg_obj = Message(role=role)
    if contents:
        msg_obj["contents"] = contents
    if tool_calls:
        msg_obj["tool_calls"] = tool_calls
    return [msg_obj]


def _set_nova_body_token_attributes(span: Span, response_body: Dict[str, Any]) -> None:
    """Set token count attributes from the Nova invoke_model response body usage field.

    Nova responses include usage in the JSON body in addition to HTTP headers.
    The body is the authoritative source for usage token fields.
    """
    usage = response_body.get("usage", {})
    if not isinstance(usage, dict):
        return
    input_tokens = usage.get("inputTokens")
    output_tokens = usage.get("outputTokens")
    if input_tokens is not None:
        _set_span_attribute(span, SpanAttributes.LLM_TOKEN_COUNT_PROMPT, int(input_tokens))
    if output_tokens is not None:
        _set_span_attribute(span, SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, int(output_tokens))
    total = usage.get("totalTokens")
    if total is None and input_tokens is not None and output_tokens is not None:
        total = int(input_tokens) + int(output_tokens)
    if total is not None:
        _set_span_attribute(span, SpanAttributes.LLM_TOKEN_COUNT_TOTAL, int(total))
    if (v := usage.get("cacheReadInputTokenCount")) is not None:
        _set_span_attribute(span, SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ, int(v))
    if (v := usage.get("cacheWriteInputTokenCount")) is not None:
        _set_span_attribute(span, SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE, int(v))
