"""Attribute extractor module for extracting attributes from Bedrock trace data."""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Union

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

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime.type_defs import (
        ContentBlockOutputTypeDef,
        ConverseRequestTypeDef,
        ConverseResponseTypeDef,
        ConverseStreamRequestTypeDef,
        ImageBlockOutputTypeDef,
        MessageOutputTypeDef,
        MessageUnionTypeDef,
        ToolResultBlockOutputTypeDef,
        ToolResultContentBlockOutputTypeDef,
        ToolUseBlockOutputTypeDef,
    )

logger = logging.getLogger(__name__)


def get_message_objects(message_list: Sequence[MessageUnionTypeDef]) -> List[Message]:
    """
    Convert a list of message dictionaries into a list of Message objects with extracted content,
    tool calls, and tool results.

    Args:
        message_list: List of message dicts (request or response) conforming to
            ``MessageUnionTypeDef``.

    Returns:
        List[Message]: List of processed Message objects with structured content and tool call
        information.
    """
    messages: List[Message] = []
    for message in message_list or []:
        role = message["role"]
        contents: list[Any] = []  # Accept both TextMessageContent and ImageMessageContent
        tool_calls: list[ToolCall] = []
        message_obj = Message(role=role)
        for message_content in message["content"]:
            _content: ContentBlockOutputTypeDef = message_content  # type: ignore[assignment]
            if "text" in _content:
                contents.append(TextMessageContent(text=_content["text"], type="text"))
            if "image" in _content:
                _image: ImageBlockOutputTypeDef = _content["image"]
                image_source = _image["source"]
                if "bytes" in image_source:
                    base64_img = base64.b64encode(image_source["bytes"]).decode("utf-8")
                    image_url = f"data:{_image['format']};base64,{base64_img}"
                    contents.append(ImageMessageContent(type="image", image=Image(url=image_url)))
            if "toolUse" in _content:
                _tool_use: ToolUseBlockOutputTypeDef = _content["toolUse"]
                tool_calls.append(
                    ToolCall(
                        id=_tool_use["toolUseId"],
                        function=ToolCallFunction(
                            name=_tool_use["name"], arguments=_tool_use["input"]
                        ),
                    )
                )
            if "toolResult" in _content:
                _tool_result: ToolResultBlockOutputTypeDef = _content["toolResult"]
                message_obj["tool_call_id"] = _tool_result["toolUseId"]
                for tool_result_content in _tool_result["content"]:
                    _tr_content: ToolResultContentBlockOutputTypeDef = tool_result_content
                    if "text" in _tr_content:
                        message_obj["content"] = _tr_content["text"]
                    if "json" in _tr_content:
                        message_obj["content"] = safe_json_dumps(_tr_content["json"])
                    if "image" in _tr_content:
                        pass  # TODO: handle image tool result
                    if "video" in _tr_content:
                        pass  # TODO: handle video tool result
                    if "document" in _tr_content:
                        pass  # TODO: handle document tool result
        if contents:
            message_obj["contents"] = contents
        if tool_calls:
            message_obj["tool_calls"] = tool_calls
        messages.append(message_obj)
    return messages


def get_input_messages(
    request_data: Union[ConverseRequestTypeDef, ConverseStreamRequestTypeDef],
) -> List[Message]:
    """
    Extracts and constructs input message objects from the request data, including system prompts
     and user messages.

    Args:
        request_data: The typed Bedrock converse (or converse_stream) request dict.

    Returns:
        List[Message]: List of Message objects representing the input messages.
    """
    messages: List[Message] = []
    if "system" in request_data:
        for system_prompt in request_data["system"]:
            msg = Message(role="system")
            if "text" in system_prompt:
                msg["content"] = system_prompt["text"]
            messages.append(msg)
    if "messages" in request_data:
        messages.extend(get_message_objects(list(request_data["messages"])))
    return messages


def get_attributes_from_request_data(
    request_data: Union[ConverseRequestTypeDef, ConverseStreamRequestTypeDef],
) -> dict[str, Any]:
    """
    Extract attributes from model invocation input.

    This method processes the model invocation input to extract relevant attributes
    such as the model name, invocation parameters, and input messages. It combines
    these attributes with LLM-specific attributes and span kind attributes.

    Args:
        request_data: The typed Bedrock converse (or converse_stream) request dict.

    Returns:
        dict[str, Any]: A dictionary of extracted attributes.
    """
    llm_attributes: dict[str, Any] = {"model_name": request_data["modelId"]}

    if "inferenceConfig" in request_data:
        llm_attributes["invocation_parameters"] = request_data["inferenceConfig"]

    if "toolConfig" in request_data:
        tool_config = request_data["toolConfig"]
        llm_attributes["tools"] = [
            Tool(json_schema=dict(tool["toolSpec"]))
            for tool in tool_config["tools"]
            if "toolSpec" in tool
        ] or list(tool_config["tools"])

    llm_attributes["input_messages"] = get_input_messages(request_data)

    return {
        **get_llm_attributes(**llm_attributes),
        **get_span_kind_attributes(OpenInferenceSpanKindValues.LLM),
        **get_input_attributes(
            list(request_data["messages"]) if "messages" in request_data else []
        ),
    }


def get_token_counts(output_params: ConverseResponseTypeDef) -> TokenCount | None:
    """
    Get token counts from output parameters.

    Args:
        output_params: The typed Bedrock converse response dict.

    Returns:
        TokenCount | None: A TokenCount object if token counts are found, None otherwise.
    """
    # usage is required in ConverseResponseTypeDef but may be absent from
    # stream-constructed responses when no metadata event was received.
    if "usage" not in output_params:
        return None
    usage = output_params["usage"]
    return TokenCount(
        prompt=usage["inputTokens"],
        completion=usage["outputTokens"],
        total=usage["totalTokens"],
    )


def get_attributes_from_response_data(
    request_data: Union[ConverseRequestTypeDef, ConverseStreamRequestTypeDef],
    response_data: ConverseResponseTypeDef,
) -> dict[str, Any]:
    """
    Extracts and compiles LLM request and response attributes into a single dictionary.

    Args:
        request_data: The typed Bedrock converse (or converse_stream) request dict.
        response_data: The typed Bedrock converse response dict.

    Returns:
        dict[str, Any]: A dictionary containing merged LLM attributes, token counts, and output
        message attributes.
    """
    llm_attributes: Dict[str, Any] = {}
    # stopReason is required in ConverseResponseTypeDef but may be absent from
    # stream-constructed responses when no messageStop event was received.
    if "stopReason" in response_data:
        llm_attributes["invocation_parameters"] = (
            dict(request_data["inferenceConfig"]) if "inferenceConfig" in request_data else {}
        )
        llm_attributes["invocation_parameters"]["stop_reason"] = response_data["stopReason"]

    # output is required in ConverseResponseTypeDef and always set by _construct_final_message.
    output = response_data["output"]
    message: MessageOutputTypeDef | None = output["message"] if "message" in output else None
    if message:
        llm_attributes["output_messages"] = get_message_objects([message])

    return {
        **get_llm_attributes(**llm_attributes),
        **get_llm_token_count_attributes(get_token_counts(response_data)),
        **get_output_attributes(message),
    }
