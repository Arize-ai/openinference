from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, List, Mapping, Tuple

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
    messages = list()
    for message in content:
        if isinstance(message.get("content"), str):
            messages.append(Message(content=message.get("content"), role=message.get("role")))
        elif isinstance(message.get("content"), Iterable):
            contents: List[Any] = list()
            tool_calls: List[ToolCall] = list()
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
    if prompt_tokens := (
        usage.get("input_tokens")
        or 0
        + (usage.get("cache_creation_input_tokens") or 0)
        + (usage.get("cache_read_input_tokens") or 0)
    ):
        yield LLM_TOKEN_COUNT_PROMPT, prompt_tokens

    if usage.get("output_tokens"):
        yield LLM_TOKEN_COUNT_COMPLETION, usage.get("output_tokens")
    if usage.get("cache_read_input_tokens"):
        yield LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ, usage.get("cache_read_input_tokens")
    if usage.get("cache_creation_input_tokens"):
        yield LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE, usage.get("cache_creation_input_tokens")


def _get_llm_model_name_from_input(model_id: str) -> str:
    if model_id and model_id.count(".") == 1:
        vendor, model_name = model_id.split(".")
        return model_name
    return model_id


def _get_invocation_parameters(kwargs: Mapping[str, Any]) -> Any:
    """
    Extracts the invocation parameters from the call
    """
    invocation_parameters = {}
    for key, value in kwargs.items():
        if _validate_invocation_parameter(key):
            invocation_parameters[key] = value
    return invocation_parameters


def _get_output_messages(response: Any) -> Any:
    """
    Extracts the tool call information from the response
    """
    output_messages = list()
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
    Validates the invocation parameters.
    """
    valid_params = (
        "max_tokens",
        "max_tokens_to_sample",
        "model",
        "metadata",
        "stop_sequences",
        "stream",
        "system",
        "temperature",
        "tool_choice",
        "tools",
        "top_k",
        "top_p",
    )
    return parameter in valid_params


def get_llm_input_attributes(request_body: Dict[str, Any], model_id: str) -> Dict[str, Any]:
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
    output_messages = _get_output_messages(response_body)
    return {
        **get_llm_output_message_attributes(output_messages),
        **get_output_attributes(response_body),
        **get_llm_model_name_attributes(response_body.get("model")),
        **dict(_get_llm_token_counts(response_body.get("usage") or {})),
    }


LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ = SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ
LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE = (
    SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE
)
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
