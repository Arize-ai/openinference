from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, TypeVar

from openinference.instrumentation import (
    Image,
    ImageMessageContent,
    Message,
    TextMessageContent,
    Tool,
    ToolCall,
    ToolCallFunction,
    get_llm_model_name_attributes,
    get_llm_output_message_attributes,
    get_llm_tool_attributes,
    get_output_attributes,
    safe_json_dumps,
)

T = TypeVar("T", bound=type)


def is_iterable_of(lst: Iterable[object], tp: type) -> bool:
    """
    Check if all elements in an iterable are instances of a given type.

    Args:
        lst (Iterable[object]): The iterable to check.
        tp (type): The type to check against.

    Returns:
        bool: True if all elements are instances of tp, False otherwise.
    """
    return isinstance(lst, Iterable) and all(isinstance(x, tp) for x in lst)


def _get_attributes_from_message_param_content_list(obj: Any) -> Any:
    contents: Any = []
    for i, item in enumerate(obj):
        if "type" not in item:
            continue
        if item["type"] in ["input_text", "output_text"]:
            if text := item.get("text"):
                contents.append(TextMessageContent(text=text, type="text"))
        elif item["type"] == "input_image":
            if "image_url" in item and (image_url := item["image_url"]):
                contents.append(ImageMessageContent(type="image", image=Image(url=image_url)))
        elif item["type"] == "refusal" and (refusal := item.get("refusal")):
            contents.append(TextMessageContent(text=refusal, type="text"))
    return contents


def _get_attributes_from_message_param(message: Mapping[str, Any]) -> Optional[Message]:
    """
    Extracts attributes from a message mapping and returns a Message object.

    Args:
        message (Mapping[str, Any]): The message dictionary to extract attributes from.

    Returns:
        Optional[Message]: A Message object with extracted attributes, or None if input is invalid.
    """
    if not hasattr(message, "get"):
        return None
    message_obj = Message()
    if role := message.get("role"):
        message_obj["role"] = role.value if isinstance(role, Enum) else role
    if tool_call_id := message.get("tool_call_id"):
        message_obj["tool_call_id"] = tool_call_id
    if content := message.get("content"):
        if isinstance(content, str):
            message_obj["content"] = content
        elif isinstance(content, Iterable):
            if contents := _get_attributes_from_message_param_content_list(content):
                message_obj["contents"] = contents
    return message_obj


def _get_attributes_from_response_input_item_param(obj: Any) -> Optional[Message]:
    """
    Extracts attributes from a single input item and returns a Message object.

    Args:
        obj (Any): The input item to extract attributes from.

    Returns:
        Optional[Message]: A Message object representing the input item, or None if not applicable.
    """
    if not isinstance(obj, dict):
        obj = getattr(obj, "__dict__", obj)
    obj_type = obj.get("type", "message")
    if obj_type == "message":
        return _get_attributes_from_message_param(obj)
    elif obj_type == "function_call":
        message_obj = Message(role="assistant")
        message_obj["role"] = "assistant"
        if (call_id := obj.get("call_id")) is not None:
            message_obj["tool_call_id"] = call_id
        function = ToolCallFunction(name=obj.get("name", ""), arguments=obj.get("arguments"))
        message_obj["tool_calls"] = [ToolCall(id=obj.get("id"), function=function)]
        return message_obj
    elif obj_type in ["function_call_output", "custom_tool_call_output"]:
        message_obj = Message(role="tool")
        if (call_id := obj.get("call_id")) is not None:
            message_obj["tool_call_id"] = call_id
        if (output := obj.get("output")) is not None:
            message_obj["content"] = output
        return message_obj
    elif obj_type == "reasoning":
        summary = obj.get("summary")
        if isinstance(summary, Iterable):
            contents = []
            for item in summary:
                if "type" not in item:
                    continue
                if item["type"] == "summary_text":
                    contents.append(TextMessageContent(text=item.get("text", ""), type="text"))
                else:
                    contents.append(TextMessageContent(text=safe_json_dumps(item), type="text"))
            return Message(role="assistant", contents=contents)
    elif obj_type in ["file_search_call", "web_search_call"]:
        function = ToolCallFunction(name=obj_type)
        return Message(
            role="assistant",
            tool_call_id=obj.get("call_id"),
            tool_calls=[ToolCall(id=obj.get("id"), function=function)],
        )
    elif obj_type in ["computer_call", "computer_call_output"]:
        function = ToolCallFunction(name=obj_type)
        return Message(
            role="tool",
            tool_call_id=obj.get("call_id"),
            tool_calls=[ToolCall(id=obj.get("id"), function=function)],
        )
    elif obj_type == "custom_tool_call":
        message_obj = Message(role="assistant")
        if (call_id := obj.get("call_id")) is not None:
            message_obj["tool_call_id"] = call_id
        function = ToolCallFunction(
            name=obj.get("name"), arguments=safe_json_dumps(obj.get("input"))
        )
        message_obj["tool_calls"] = [ToolCall(id=obj.get("id"), function=function)]
        return message_obj
    else:
        return Message(role="assistant", content=safe_json_dumps(obj))
    return None


def _get_attributes_from_response_input(kwargs: Dict[str, Any]) -> List[Message]:
    """
    Extracts input attributes from a response kwargs dictionary and returns a list of
    Message objects.

    Args:
        kwargs (Dict[str, Any]): The response input dictionary containing instructions and
        input messages.

    Returns:
        List[Message]: A list of Message objects representing the input messages and instructions.
    """
    messages = []
    if (instructions := kwargs.get("instructions")) is not None:
        messages.append(Message(role="system", content=instructions))
    if (inputs := kwargs.get("input")) is not None:
        if isinstance(inputs, str):
            messages.append(Message(role="user", content=inputs))
        elif isinstance(inputs, list):
            for item in inputs:
                if message_obj := _get_attributes_from_response_input_item_param(item):
                    messages.append(message_obj)
    return messages


def _get_attributes_from_response_output_item(obj: Any) -> Message:
    """
    Extracts output attributes from a response object and returns a Message object.

    Args:
        obj (Any): The response output item to extract attributes from.

    Returns:
        Message: A Message object representing the output item, including role, content,
        and tool calls.
    """
    message_obj = Message()
    contents: List[Any] = []
    tool_calls: List[ToolCall] = []
    obj_type = getattr(obj, "type", None) or (obj.get("type") if isinstance(obj, dict) else None)
    if obj_type == "message":
        message_obj["role"] = getattr(obj, "role", "assistant")
        for item in getattr(obj, "content", []):
            if hasattr(item, "text"):
                contents.append(TextMessageContent(text=getattr(item, "text", ""), type="text"))
            if hasattr(item, "refusal"):
                contents.append(TextMessageContent(text=getattr(item, "refusal", ""), type="text"))
    elif obj_type == "function_call":
        message_obj["role"] = "assistant"
        if (call_id := getattr(obj, "call_id", None)) is not None:
            message_obj["tool_call_id"] = call_id
        function = ToolCallFunction(
            name=getattr(obj, "name", ""), arguments=getattr(obj, "arguments", "")
        )
        tool_calls.append(ToolCall(id=getattr(obj, "id", ""), function=function))
    elif obj_type == "file_search_call":
        message_obj["role"] = "assistant"
        function = ToolCallFunction(name=obj_type)
        tool_calls.append(ToolCall(id=getattr(obj, "id", ""), function=function))
    elif obj_type == "computer_call":
        message_obj["role"] = "tool"
        function = ToolCallFunction(name=obj_type)
        tool_calls.append(ToolCall(id=getattr(obj, "id", ""), function=function))
    elif obj_type == "reasoning":
        summary = getattr(obj, "summary", None)
        if summary and isinstance(summary, Iterable):
            message_obj["role"] = "assistant"
            for item in summary:
                contents.append(TextMessageContent(text=getattr(item, "text", ""), type="text"))
            message_obj["contents"] = contents
    elif obj_type == "web_search_call":
        message_obj["role"] = "assistant"
        function = ToolCallFunction(name=obj_type)
        tool_calls.append(ToolCall(id=getattr(obj, "id", ""), function=function))
    elif obj_type == "custom_tool_call":
        message_obj["role"] = "assistant"
        if (call_id := getattr(obj, "call_id", None)) is not None:
            message_obj["tool_call_id"] = call_id
        function = ToolCallFunction(
            name=getattr(obj, "name", ""), arguments=safe_json_dumps(getattr(obj, "input", ""))
        )
        tool_calls.append(ToolCall(id=getattr(obj, "id", ""), function=function))
    else:
        message_obj["role"] = "assistant"
        try:
            message_obj["content"] = safe_json_dumps(obj)
        except Exception:
            pass
    if tool_calls:
        message_obj["tool_calls"] = tool_calls
    if contents:
        message_obj["contents"] = contents
    return message_obj


def _get_output_message_objects(obj: Any) -> List[Message]:
    """
    Converts the output field of a response object into a list of Message objects.

    Args:
        obj (Any): The response object containing an output field.

    Returns:
        List[Message]: A list of Message objects extracted from the output field.
    """
    messages = []
    output = getattr(obj, "output", None)
    if isinstance(output, Iterable):
        for item in output:
            if message := _get_attributes_from_response_output_item(item):
                messages.append(message)
    return messages


def _get_attributes_from_response_output(obj: Any) -> Dict[str, Any]:
    """
    Extracts output attributes from a response dictionary and returns a dictionary of attributes.

    Args:
        obj (Any): The response output Object.

    Returns:
        Dict[str, Any]: A dictionary containing model name, output message, and tool attributes.
    """
    tool_attributes: Mapping[str, Any] = {}
    tools = obj.get("tools")
    if isinstance(tools, Iterable):
        tool_attributes = get_llm_tool_attributes([Tool(json_schema=tool) for tool in tools])
    input_messages = _get_output_message_objects(obj)
    model_name_attributes = (
        get_llm_model_name_attributes(obj.get("model")) if obj.get("model") else {}
    )
    return {
        **model_name_attributes,
        **get_llm_output_message_attributes(input_messages),
        **get_output_attributes(input_messages),
        **tool_attributes,
    }
