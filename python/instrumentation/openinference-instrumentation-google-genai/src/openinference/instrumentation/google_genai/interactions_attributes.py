import logging
from typing import Any, Mapping

from openinference.instrumentation import (
    get_llm_attributes,
    get_span_kind_attributes,
    Message, get_input_attributes, get_output_attributes, safe_json_dumps, get_llm_output_message_attributes,
    TokenCount, get_llm_model_name_attributes, get_llm_token_count_attributes, PromptDetails, Tool
)
from openinference.semconv.trace import OpenInferenceLLMProviderValues

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_output_messages(outputs):
    from google.genai._interactions import types
    messages = []
    for output in outputs or []:
        if isinstance(output, types.TextContent):
            messages.append(Message(
                role="model",
                content=output.text
            ))
        elif isinstance(output, types.FunctionCallContent):
            function_call_content = {
                "name": output.name,
                "arguments": output.arguments
            }
            messages.append(Message(
                role="model",
                content=function_call_content
            ))
        elif isinstance(output, types.ImageContent):
            # TODO: Handle Image Content types
            pass
    return messages


def get_message_objects(inputs):
    messages = []
    if isinstance(inputs, str):
        # if the input is a simple string, treat it as a user message
        messages.append(Message(role="user", content=inputs))
    if isinstance(inputs, list):
        for message in inputs:
            if isinstance(message, dict):
                if isinstance(message.get("content"), str):
                    messages.append(Message(
                        role=message.get("role", "user"),
                        content=message.get("content", "")
                    ))
                elif isinstance(message.get("content"), list):
                    messages.extend(get_output_messages(message.get("content")))
    return messages


def get_tools(request_params):
    tools = request_params.get("tools") or []
    return [Tool(json_schema=tool) for tool in tools]


def get_token_object_from_response(response: Any):
    if hasattr(response, "usage"):
        usage = response.usage
        return TokenCount(
            total=usage.total_tokens,
            prompt=usage.total_input_tokens,
            completion=usage.total_thought_tokens + usage.total_output_tokens,
        )


def get_attributes_from_request(
        request_parameters: Mapping[str, Any],
):
    input_messages = []
    if request_parameters.get("system_instruction"):
        input_messages.append(Message(role="system", content=request_parameters.get("system_instruction")))
    input_messages.extend(get_message_objects(request_parameters.get("input")))
    invocation_parameters = request_parameters.get("generation_config") or {}
    return {
        **get_llm_attributes(
            provider=OpenInferenceLLMProviderValues.GOOGLE.value,
            input_messages=input_messages,
            tools=get_tools(request_parameters),
            invocation_parameters=invocation_parameters
        ),
        **get_span_kind_attributes("llm"),
        **get_input_attributes(request_parameters.get("input"))
    }


def get_attributes_from_response(
        response: Any,
):
    return {
        **get_llm_model_name_attributes(response.model),
        **get_output_attributes(safe_json_dumps(response.outputs)),
        **get_llm_output_message_attributes(get_output_messages(response.outputs)),
        **get_llm_token_count_attributes(get_token_object_from_response(response))
    }
