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
- And other Bedrock-supported models
"""

from typing import Any, Dict

from opentelemetry import trace as trace_api
from opentelemetry.trace import Span
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import safe_json_dumps
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes


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
    """
    if model_id := kwargs.get("modelId"):
        _set_span_attribute(span, SpanAttributes.LLM_MODEL_NAME, model_id)
        vendor = None
        if isinstance(model_id, str):
            (vendor, *_) = model_id.split(".")

        if vendor == "ai21":
            content = str(response_body.get("completions"))
        elif vendor == "anthropic":
            content = str(response_body.get("completion"))
        elif vendor == "cohere":
            content = str(response_body.get("generations"))
        elif vendor == "meta":
            content = str(response_body.get("generation"))
        else:
            content = ""

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


def set_input_attributes(span: Span, request_body: Dict[str, Any]) -> None:
    """
    Set input-related attributes on the span from the request body.

    Extracts the prompt from the request body and sets it as the INPUT_VALUE
    attribute. Also serializes the remaining request parameters as invocation
    parameters and sets the span kind to LLM.

    Args:
        span: The OpenTelemetry span to set attributes on
        request_body: The request body containing prompt and other parameters
    """
    prompt = request_body.pop("prompt", None)
    invocation_parameters = safe_json_dumps(request_body)
    _set_span_attribute(span, SpanAttributes.INPUT_VALUE, prompt)
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
