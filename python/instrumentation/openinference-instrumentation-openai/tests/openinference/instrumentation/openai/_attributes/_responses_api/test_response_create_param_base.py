import json

import pytest
from openai.types.responses import FunctionToolParam
from openai.types.responses.response_create_params import ResponseCreateParamsBase
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation.openai._attributes._responses_api import _ResponsesApiAttributes


class TestResponseCreateParamsBase:
    @pytest.mark.parametrize(
        "obj,expected",
        [
            pytest.param(
                ResponseCreateParamsBase(
                    model="gpt-4",
                    input="Hello, world!",
                    instructions="Be helpful",
                    tools=[
                        FunctionToolParam(
                            type="function",
                            name="get_weather",
                            description="Get the weather",
                            parameters={"type": "object", "properties": {}},
                            strict=True,
                        )
                    ],
                    temperature=0.7,
                    top_p=0.9,
                    max_output_tokens=100,
                ),
                {
                    "llm.model_name": "gpt-4",
                    "llm.input_messages.0.message.role": "system",
                    "llm.input_messages.0.message.content": "Be helpful",
                    "llm.input_messages.1.message.role": "user",
                    "llm.input_messages.1.message.content": "Hello, world!",
                    "llm.tools.0.tool.json_schema": json.dumps(
                        {
                            "type": "function",
                            "name": "get_weather",
                            "description": "Get the weather",
                            "parameters": {"type": "object", "properties": {}},
                            "strict": True,
                        }
                    ),
                    "llm.invocation_parameters": json.dumps(
                        {
                            "model": "gpt-4",
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "max_output_tokens": 100,
                        }
                    ),
                },
                id="basic_params_with_tools",
            ),
            pytest.param(
                ResponseCreateParamsBase(
                    model="gpt-4",
                    input="Hello, world!",
                ),
                {
                    "llm.model_name": "gpt-4",
                    "llm.input_messages.1.message.role": "user",
                    "llm.input_messages.1.message.content": "Hello, world!",
                    "llm.invocation_parameters": json.dumps({"model": "gpt-4"}),
                },
                id="minimal_params",
            ),
            pytest.param(
                ResponseCreateParamsBase(
                    model="gpt-4",
                    input="Hello",
                    instructions="Be helpful",
                ),
                {
                    "llm.model_name": "gpt-4",
                    "llm.input_messages.0.message.role": "system",
                    "llm.input_messages.0.message.content": "Be helpful",
                    "llm.input_messages.1.message.content": "Hello",
                    "llm.input_messages.1.message.role": "user",
                    "llm.invocation_parameters": json.dumps({"model": "gpt-4"}),
                },
                id="with_instructions_only",
            ),
            pytest.param(
                ResponseCreateParamsBase(
                    model="gpt-4",
                    input=[
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi there!"},
                    ],
                ),
                {
                    "llm.model_name": "gpt-4",
                    "llm.invocation_parameters": json.dumps({"model": "gpt-4"}),
                    "llm.input_messages.1.message.content": "Hello",
                    "llm.input_messages.1.message.role": "user",
                    "llm.input_messages.2.message.content": "Hi there!",
                    "llm.input_messages.2.message.role": "assistant",
                },
                id="with_input_list",
            ),
        ],
    )
    def test_ResponseCreateParamsBase(
        self,
        obj: ResponseCreateParamsBase,
        expected: dict[str, AttributeValue],
    ) -> None:
        actual = dict(_ResponsesApiAttributes._get_attributes_from_response_create_param_base(obj))
        assert actual == expected
