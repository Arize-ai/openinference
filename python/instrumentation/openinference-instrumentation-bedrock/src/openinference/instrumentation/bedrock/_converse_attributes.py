"""Attribute extractor module for extracting attributes from Bedrock trace data."""

from __future__ import annotations

import base64
import logging
from typing import Any, Dict, List

from openinference.instrumentation import (
    Image,
    ImageMessageContent,
    Message,
    TextMessageContent,
    TokenCount,
    Tool,
    ToolCall,
    ToolCallFunction,
    get_input_attributes,
    get_llm_attributes,
    get_llm_token_count_attributes,
    get_output_attributes,
    get_span_kind_attributes,
    safe_json_dumps,
)
from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
)

logger = logging.getLogger(__name__)


def get_message_objects(message_list: List[Any]) -> List[Message]:
    """
    Convert a list of message dictionaries into a list of Message objects with extracted content,
    tool calls, and tool results.

    Args:
        message_list (List[Any]): List of message dictionaries from the request or response.

    Returns:
        List[Message]: List of processed Message objects with structured content and tool call
        information.
    """
    messages: List[Message] = []
    for message in message_list or []:
        role = message.get("role", "")
        contents: list[Any] = []  # Accept both TextMessageContent and ImageMessageContent
        tool_calls: list[ToolCall] = []
        message_obj = Message(role=role)
        for message_content in message.get("content", []):
            if message_text := message_content.get("text"):
                contents.append(TextMessageContent(text=message_text, type="text"))
            if image_content := message_content.get("image"):
                if content_bytes := image_content.get("source", {}).get("bytes"):
                    base64_img = base64.b64encode(content_bytes).decode("utf-8")
                    image_url = f"data:{image_content.get('format')};base64,{base64_img}"
                    contents.append(ImageMessageContent(type="image", image=Image(url=image_url)))
            if tool_use_content := message_content.get("toolUse"):
                tool_call_function = ToolCallFunction(
                    name=tool_use_content.get("name", ""),
                    arguments=tool_use_content.get("input", {}),
                )
                tool_calls.append(
                    ToolCall(id=tool_use_content.get("toolUseId", ""), function=tool_call_function)
                )
            if tool_result := message_content.get("toolResult"):
                message_obj["tool_call_id"] = tool_result.get("toolUseId", "")
                for tool_result_content in tool_result.get("content", []):
                    if message_text := tool_result_content.get("text"):
                        message_obj["content"] = message_text
                    if json_text := tool_result_content.get("json"):
                        message_obj["content"] = safe_json_dumps(json_text)
                    if tool_result_content.get("image"):
                        pass  # TODO: handle image tool result
                    if tool_result_content.get("video"):
                        pass  # TODO: handle video tool result
                    if tool_result_content.get("document"):
                        pass  # TODO: handle document tool result
        if contents:
            message_obj["contents"] = contents
        if tool_calls:
            message_obj["tool_calls"] = tool_calls
        messages.append(message_obj)
    return messages


def get_input_messages(request_data: Dict[str, Any]) -> List[Message]:
    """
    Extracts and constructs input message objects from the request data, including system prompts
     and user messages.

    Args:
        request_data (Dict[str, Any]): The request data containing system and user messages.

    Returns:
        List[Message]: List of Message objects representing the input messages.
    """
    messages: List[Message] = []
    if system_prompts := request_data.get("system"):
        for system_prompt in system_prompts:
            messages.append(Message(content=system_prompt.get("text"), role="system"))
    messages.extend(get_message_objects(request_data.get("messages") or []))
    return messages


def get_attributes_from_request_data(request_data: dict[str, Any]) -> dict[str, Any]:
    """
    Extract attributes from model invocation input.

    This method processes the model invocation input to extract relevant attributes
    such as the model name, invocation parameters, and input messages. It combines
    these attributes with LLM-specific attributes and span kind attributes.

    Args:
        request_data (dict[str, Any]): The model invocation input dictionary.

    Returns:
        dict[str, Any]: A dictionary of extracted attributes.
    """
    llm_attributes = {}

    if model_name := request_data.get("modelId"):
        llm_attributes["model_name"] = model_name

    if invocation_parameters := request_data.get("inferenceConfig"):
        llm_attributes["invocation_parameters"] = invocation_parameters

    if tool_config := request_data.get("toolConfig"):
        llm_attributes["tools"] = [
            Tool(json_schema=tool.get("toolSpec")) for tool in tool_config.get("tools", [])
        ] or tool_config.get("tools")

    # Get input and output messages
    llm_attributes["input_messages"] = get_input_messages(request_data)

    # Set attributes
    return {
        **get_llm_attributes(**llm_attributes),
        **get_span_kind_attributes(OpenInferenceSpanKindValues.LLM),
        **get_input_attributes(request_data.get("messages") or []),
    }


def get_token_counts(output_params: dict[str, Any]) -> TokenCount | None:
    """
    Get token counts from output parameters.

    Args:
        output_params (dict[str, Any]): The output parameters.

    Returns:
        TokenCount | None: A TokenCount object if token counts are found, None otherwise.
    """
    if usage := output_params.get("usage"):
        completion, prompt, total = 0, 0, 0
        if input_tokens := usage.get("inputTokens"):
            prompt = input_tokens
        if output_tokens := usage.get("outputTokens"):
            completion = output_tokens
        if total_tokens := usage.get("totalTokens"):
            total = total_tokens
        return TokenCount(prompt=prompt, completion=completion, total=total)
    return None


def get_attributes_from_response_data(
    request_data: Dict[str, Any], response_data: Dict[str, Any]
) -> Any:
    """
    Extracts and compiles LLM request and response attributes into a single dictionary.

    Args:
        request_data (dict[str, Any]): The original request payload sent to the LLM API.
        response_data (dict[str, Any]): The response payload received from the LLM API.

    Returns:
        dict[str, Any]: A dictionary containing merged LLM attributes, token counts, and output
        message attributes.
    """
    llm_attributes: Dict[str, Any] = {}
    stop_reason = response_data.get("stopReason")
    if stop_reason:
        llm_attributes["invocation_parameters"] = request_data.get("inferenceConfig") or {}
        llm_attributes["invocation_parameters"]["stop_reason"] = stop_reason

    message = response_data.get("output", {}).get("message")
    if message:
        llm_attributes["output_messages"] = get_message_objects([message])

    request_attributes = {
        **get_llm_attributes(**llm_attributes),
        **get_llm_token_count_attributes(get_token_counts(response_data or {})),
        **get_output_attributes(response_data.get("output", {}).get("message")),
    }
    return request_attributes
