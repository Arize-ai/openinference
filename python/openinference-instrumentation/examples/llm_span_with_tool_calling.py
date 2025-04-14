import json
from typing import Any, Dict, List, Optional, Union

from openai import NOT_GIVEN, OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from phoenix.otel import register
from typing_extensions import assert_never

import openinference.instrumentation as oi
from openinference.instrumentation import (
    get_input_attributes,
    get_llm_attributes,
    get_output_attributes,
)

tracer_provider = register(protocol="http/protobuf")
tracer = tracer_provider.get_tracer(__name__)

openai_client = OpenAI()


def get_oi_attributes_from_openai_inputs(
    input_messages: List[ChatCompletionMessageParam],
    invocation_parameters: Dict[str, Any],
    tools: Optional[List[ChatCompletionToolParam]] = None,
) -> Dict[str, Any]:
    oi_input_messages = [
        convert_openai_message_param_to_oi_message(message) for message in input_messages
    ]
    oi_tools = [convert_openai_tool_param_to_oi_tool(tool) for tool in tools or []]
    return {
        **get_input_attributes(
            {
                "input_messages": input_messages,
                "invocation_parameters": invocation_parameters,
            }
        ),
        **get_llm_attributes(
            provider="openai",
            system="openai",
            model_name="gpt-4o",
            input_messages=oi_input_messages,
            invocation_parameters=invocation_parameters,
            tools=oi_tools,
        ),
    }


def convert_openai_message_param_to_oi_message(
    message_param: Union[ChatCompletionMessageParam, ChatCompletionMessage],
) -> oi.Message:
    if isinstance(message_param, ChatCompletionMessage):
        role: str = message_param.role
        oi_message = oi.Message(
            role=role,
        )
        if isinstance(content := message_param.content, str):
            oi_message["content"] = content
        if message_param.tool_calls is not None:
            oi_tool_calls: List[oi.ToolCall] = []
            for tool_call in message_param.tool_calls:
                function = tool_call.function
                oi_tool_calls.append(
                    oi.ToolCall(
                        id=tool_call.id,
                        function=oi.ToolCallFunction(
                            name=function.name,
                            arguments=function.arguments,
                        ),
                    )
                )
            oi_message["tool_calls"] = oi_tool_calls
        return oi_message
    if isinstance(message_param, dict):
        role = message_param["role"]
        assert isinstance(message_param["content"], str)
        content = message_param["content"]
    else:
        assert_never(message_param)
    return oi.Message(role=role, content=content)


def convert_openai_tool_param_to_oi_tool(
    tool_param: ChatCompletionToolParam,
) -> oi.Tool:
    assert tool_param["type"] == "function"
    return oi.Tool(
        json_schema=dict(tool_param),
    )


def get_oi_attributes_from_openai_outputs(
    response: ChatCompletion,
) -> Dict[str, Any]:
    choices = response.choices
    assert len(choices) == 1
    message = response.choices[0].message
    role = message.role
    oi_message = oi.Message(role=role)
    if isinstance(message.content, str):
        oi_message["content"] = message.content
    if isinstance(message.tool_calls, list):
        oi_tool_calls: List[oi.ToolCall] = []
        for tool_call in message.tool_calls:
            tool_call_id = tool_call.id
            function_name = tool_call.function.name
            function_arguments = tool_call.function.arguments
            oi_tool_calls.append(
                oi.ToolCall(
                    id=tool_call_id,
                    function=oi.ToolCallFunction(
                        name=function_name,
                        arguments=function_arguments,
                    ),
                )
            )
        oi_message["tool_calls"] = oi_tool_calls
    output_messages = [oi_message]
    token_usage = response.usage
    assert token_usage is not None
    prompt_tokens = token_usage.prompt_tokens
    completion_tokens = token_usage.completion_tokens
    return {
        **get_llm_attributes(
            output_messages=output_messages,
            token_count={
                "completion": completion_tokens,
                "prompt": prompt_tokens,
            },
        ),
        **get_output_attributes(response),
    }


@tracer.llm(
    process_input=get_oi_attributes_from_openai_inputs,
    process_output=get_oi_attributes_from_openai_outputs,
)
def invoke_llm(
    input_messages: List[ChatCompletionMessageParam],
    invocation_parameters: Dict[str, Any],
    tools: Optional[List[ChatCompletionToolParam]] = None,
) -> ChatCompletion:
    response = openai_client.chat.completions.create(
        model="gpt-4",
        tools=tools or NOT_GIVEN,
        messages=input_messages,
        **invocation_parameters,
    )
    assert isinstance(response, ChatCompletion)
    return response


@tracer.tool
def get_weather(city: str) -> str:
    """
    Imagine you were making an API call here.
    """
    return "sunny"


input_messages = [
    ChatCompletionUserMessageParam(
        role="user",
        content="What's the weather like in San Francisco?",
    )
]
invocation_parameters = {
    "temperature": 0.5,
}
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "finds the weather for a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to find the weather for, e.g. 'London'",
                    }
                },
                "required": ["city"],
            },
        },
    },
]
response = invoke_llm(input_messages, invocation_parameters, tools=tools)
output_message = response.choices[0].message
tool_call = output_message.tool_calls[0]
city = json.loads(tool_call.function.arguments)["city"]
weather = get_weather(city)
print(f"{city} is {weather}")
input_messages.append(output_message)
input_messages.append(
    ChatCompletionToolMessageParam(
        content=weather,
        role="tool",
        tool_call_id=tool_call.id,
    )
)
invoke_llm(input_messages, invocation_parameters)
