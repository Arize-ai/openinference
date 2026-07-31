import logging
from typing import Any, Iterable, Mapping, Optional, Sequence, cast

from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import (
    Image,
    ImageMessageContent,
    Message,
    MessageContent,
    PromptDetails,
    TextMessageContent,
    TokenCount,
    Tool,
    ToolCall,
    ToolCallFunction,
    get_input_attributes,
    get_llm_attributes,
    get_llm_model_name_attributes,
    get_llm_output_message_attributes,
    get_llm_token_count_attributes,
    get_metadata_attributes,
    get_output_attributes,
    get_span_kind_attributes,
    safe_json_dumps,
)
from openinference.instrumentation.google_genai._utils import (
    _stop_on_exception_for_dict,
    _stop_on_exception_for_iter,
    get_attribute,
)
from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceLLMProviderValues,
    SpanAttributes,
    ToolCallAttributes,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _get_message_from_contents(role: str, contents: Any) -> Optional[Message]:
    message_contents: list[MessageContent] = []
    tool_calls: list[ToolCall] = []
    for content in contents or []:
        _append_content(content, message_contents, tool_calls)
    if not message_contents and not tool_calls:
        return None
    message = Message(role=role)
    if message_contents:
        message["contents"] = message_contents
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _append_content(
    content: Any,
    message_contents: list[MessageContent],
    tool_calls: list[ToolCall],
) -> None:
    content_type = get_attribute(content, "type")
    if content_type == "model_output":
        for item in get_attribute(content, "content") or []:
            _append_content(item, message_contents, tool_calls)
    elif content_type == "text":
        message_contents.append(
            TextMessageContent(type="text", text=get_attribute(content, "text", "") or "")
        )
    elif content_type == "function_call":
        name = get_attribute(content, "name")
        arguments = get_attribute(content, "arguments")
        tool_call = ToolCall(function=ToolCallFunction(name=name, arguments=arguments))
        if call_id := get_attribute(content, "id"):
            tool_call["id"] = call_id
        if sig := get_attribute(content, "signature"):
            tool_call["reasoning_signature"] = sig
        tool_calls.append(tool_call)
    elif content_type == "thought":
        signature = get_attribute(content, "signature")
        text: Optional[str] = None
        if summary := get_attribute(content, "summary"):
            if isinstance(summary, list):
                parts = [
                    get_attribute(item, "text")
                    for item in summary
                    if get_attribute(item, "type") == "text"
                ]
                joined = "".join(p for p in parts if p)
                if joined:
                    text = joined
        if text is None:
            text = get_attribute(content, "text")
        if text or signature:
            rc: dict[str, Any] = {"type": "reasoning"}
            if text:
                rc["text"] = text
            if signature:
                rc["signature"] = signature
            message_contents.append(cast(MessageContent, rc))
    elif content_type == "image":
        if uri := get_attribute(content, "uri"):
            message_contents.append(ImageMessageContent(type="image", image=Image(url=uri)))
        elif data := get_attribute(content, "data"):
            mime_type = get_attribute(content, "mime_type") or "image/png"
            url = f"data:{mime_type};base64,{data}"
            message_contents.append(ImageMessageContent(type="image", image=Image(url=url)))


def _get_reasoning_signature_attributes(
    messages: list[Message],
) -> dict[str, AttributeValue]:
    attrs: dict[str, AttributeValue] = {}
    base = SpanAttributes.LLM_OUTPUT_MESSAGES
    for msg_idx, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        for content_idx, content in enumerate(message.get("contents") or []):
            if not isinstance(content, dict):
                continue
            content_any: Any = content
            if content_any.get("type") == "reasoning" and isinstance(
                sig := content_any.get("signature"), str
            ):
                attrs[
                    f"{base}.{msg_idx}.{MessageAttributes.MESSAGE_CONTENTS}"
                    f".{content_idx}.{MessageContentAttributes.MESSAGE_CONTENT_SIGNATURE}"
                ] = sig

        for tc_idx, tool_call in enumerate(message.get("tool_calls") or []):
            if not isinstance(tool_call, dict):
                continue
            tc_any: Any = tool_call
            if isinstance(sig := tc_any.get("reasoning_signature"), str):
                attrs[
                    f"{base}.{msg_idx}.{MessageAttributes.MESSAGE_TOOL_CALLS}"
                    f".{tc_idx}.{ToolCallAttributes.TOOL_CALL_REASONING_SIGNATURE}"
                ] = sig
    return attrs


def get_output_messages(steps: Any) -> list[Message]:
    messages: list[Message] = []
    contents: list[MessageContent] = []
    tool_calls: list[ToolCall] = []
    for step in steps or []:
        step_type = get_attribute(step, "type")
        if step_type == "function_result":
            messages.append(
                Message(
                    role="tool",
                    content=get_attribute(step, "result", "") or "",
                    tool_call_id=get_attribute(step, "call_id", "") or "",
                )
            )
        else:
            _append_content(step, contents, tool_calls)
    if contents or tool_calls or not messages:
        messages.append(Message(role="model", contents=contents, tool_calls=tool_calls))
    return messages


def get_message_objects(inputs: Any) -> list[Message]:
    messages: list[Message] = []
    if isinstance(inputs, str):
        # if the input is a simple string, treat it as a user message
        messages.append(Message(role="user", content=inputs))
    if isinstance(inputs, list):
        contents: list[MessageContent] = []
        for message in inputs:
            if isinstance(message, dict) or hasattr(message, "type"):
                message_type = get_attribute(message, "type")
                if message_type == "function_result":
                    messages.append(
                        Message(
                            role="tool",
                            content=get_attribute(message, "result", "") or "",
                            tool_call_id=get_attribute(message, "call_id") or "",
                        )
                    )
                elif message_type == "user_input":
                    if step_message := _get_message_from_contents(
                        "user", get_attribute(message, "content")
                    ):
                        messages.append(step_message)
                elif message_type == "model_output":
                    if step_message := _get_message_from_contents(
                        "model", get_attribute(message, "content")
                    ):
                        messages.append(step_message)
                elif message_type == "function_call":
                    messages.extend(get_output_messages([message]))
                elif message_type == "text":
                    contents.append(
                        TextMessageContent(text=get_attribute(message, "text") or "", type="text")
                    )
                elif message_type == "image":
                    _append_content(message, contents, [])
                elif isinstance(get_attribute(message, "content"), str):
                    messages.append(
                        Message(
                            role=get_attribute(message, "role", "user"),
                            content=get_attribute(message, "content", ""),
                        )
                    )
                elif isinstance(get_attribute(message, "content"), Sequence):
                    role = get_attribute(message, "role", "model")
                    if step_message := _get_message_from_contents(
                        role, get_attribute(message, "content")
                    ):
                        messages.append(step_message)
        if contents:
            messages.append(Message(role="user", contents=contents))
    return messages


def get_tools(request_params: Mapping[str, Any]) -> list[Tool]:
    tools = request_params.get("tools") or []
    return [Tool(json_schema=tool) for tool in tools]


def _as_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _sum_token_values(items: Any) -> int:
    total = 0
    for item in items or []:
        total += _as_int(get_attribute(item, "tokens") or get_attribute(item, "token_count"))
    return total


def get_token_object_from_response(response: Any) -> TokenCount:
    token_count = TokenCount()
    if usage := get_attribute(response, "usage"):
        cached_tokens = _as_int(get_attribute(usage, "total_cached_tokens"))
        prompt = (
            max(
                _as_int(get_attribute(usage, "total_input_tokens")),
                _sum_token_values(get_attribute(usage, "input_tokens_by_modality")),
            )
            + cached_tokens
        )
        completion = (
            _as_int(get_attribute(usage, "total_thought_tokens"))
            + _as_int(get_attribute(usage, "total_output_tokens"))
            + _as_int(get_attribute(usage, "total_tool_use_tokens"))
        )
        total = _as_int(get_attribute(usage, "total_tokens"))
        if total > prompt + completion and (prompt or completion):
            prompt = total - completion
        else:
            total = max(total, prompt + completion)
        token_count = TokenCount(
            total=total,
            prompt=prompt,
            completion=completion,
        )
        if cached_tokens:
            token_count["prompt_details"] = PromptDetails(cache_read=cached_tokens)
    return token_count


def is_agent_call(request_parameters: Mapping[str, Any]) -> bool:
    return isinstance(request_parameters.get("agent"), str) and not isinstance(
        request_parameters.get("model"), str
    )


def is_agent_response(response: Any) -> bool:
    return isinstance(get_attribute(response, "agent"), str) and not isinstance(
        get_attribute(response, "model"), str
    )


def get_attributes_from_request_object(
    request_parameters: Mapping[str, Any],
) -> dict[str, AttributeValue]:
    if is_agent_call(request_parameters):
        return {
            **get_span_kind_attributes("agent"),
            **get_input_attributes(request_parameters.get("input")),
        }

    input_messages = []
    if system_instruction := request_parameters.get("system_instruction"):
        input_messages.append(Message(role="system", content=system_instruction))
    input_messages.extend(get_message_objects(request_parameters.get("input")))
    invocation_parameters = dict(request_parameters.get("generation_config") or {})
    if response_format := request_parameters.get("response_format"):
        invocation_parameters["response_format"] = response_format
    metadata = {}
    if previous_interaction_id := request_parameters.get("previous_interaction_id"):
        metadata["previous_interaction_id"] = previous_interaction_id
    return {
        **get_llm_attributes(
            provider=OpenInferenceLLMProviderValues.GOOGLE.value,
            input_messages=input_messages,
            tools=get_tools(request_parameters),
            invocation_parameters=invocation_parameters,
        ),
        **get_metadata_attributes(metadata=metadata),
        **get_span_kind_attributes("llm"),
        **get_input_attributes(request_parameters.get("input")),
    }


def get_attributes_from_get_request_object(
    request_parameters: Mapping[str, Any],
) -> dict[str, AttributeValue]:
    metadata = {}
    if interaction_id := request_parameters.get("id"):
        metadata["interaction_id"] = interaction_id
    return get_metadata_attributes(metadata=metadata)


@_stop_on_exception_for_iter
def get_attributes_from_request(
    request_parameters: Mapping[str, Any],
) -> Iterable[tuple[str, AttributeValue]]:
    attributes = get_attributes_from_request_object(request_parameters)
    for key, value in attributes.items():
        yield key, value


@_stop_on_exception_for_iter
def get_attributes_from_get_request(
    request_parameters: Mapping[str, Any],
) -> Iterable[tuple[str, AttributeValue]]:
    attributes = get_attributes_from_get_request_object(request_parameters)
    for key, value in attributes.items():
        yield key, value


@_stop_on_exception_for_dict
def get_attributes_from_response(
    request_parameters: Mapping[str, Any],
    response: Any,
) -> dict[str, AttributeValue]:
    if not response:
        return {}

    if is_agent_call(request_parameters) or is_agent_response(response):
        return {
            **get_span_kind_attributes("agent"),
            **get_output_attributes(safe_json_dumps(response)),
        }

    steps = get_attribute(response, "steps")
    output_messages = get_output_messages(steps)
    return {
        **get_llm_attributes(provider=OpenInferenceLLMProviderValues.GOOGLE.value),
        **get_span_kind_attributes("llm"),
        **get_llm_model_name_attributes(get_attribute(response, "model")),
        **get_output_attributes(safe_json_dumps(steps)),
        **get_llm_output_message_attributes(output_messages),
        **_get_reasoning_signature_attributes(output_messages),
        **get_llm_token_count_attributes(get_token_object_from_response(response)),
    }
