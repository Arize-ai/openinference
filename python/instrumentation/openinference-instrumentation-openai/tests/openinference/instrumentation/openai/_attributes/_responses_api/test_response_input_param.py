import pytest
from openai.types.responses import (
    ResponseFunctionToolCallParam,
    ResponseOutputMessageParam,
    ResponseReasoningItemParam,
)
from openai.types.responses.easy_input_message_param import EasyInputMessageParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam
from openai.types.responses.response_input_param import (
    FunctionCallOutput,
    Message,
)
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_reasoning_item_param import Summary
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation.openai._attributes._responses_api import (
    _ResponsesApiAttributes,
)


class TestResponseInputItemParam:
    @pytest.mark.parametrize(
        "obj,expected",
        [
            pytest.param(
                Message(
                    type="message",
                    content=[
                        ResponseInputTextParam(type="input_text", text="Hello, I'm John"),
                    ],
                    role="user",
                ),
                {
                    "message.role": "user",
                    "message.contents.0.message_content.type": "text",
                    "message.contents.0.message_content.text": "Hello, I'm John",
                },
                id="message_with_text",
            ),
            pytest.param(
                Message(
                    type="message",
                    content=[
                        ResponseInputTextParam(
                            type="input_text", text="I used the tool to get information."
                        )
                    ],
                    role="user",
                ),
                {
                    "message.role": "user",
                    "message.contents.0.message_content.type": "text",
                    "message.contents.0.message_content.text": "I used the tool to get information.",  # noqa: E501
                },
                id="message_with_tool_mention",
            ),
        ],
    )
    def test_Message(
        self,
        obj: Message,
        expected: dict[str, AttributeValue],
    ) -> None:
        actual = dict(_ResponsesApiAttributes._get_attributes_from_response_input_item_param(obj))
        assert actual == expected

    @pytest.mark.parametrize(
        "obj,expected",
        [
            pytest.param(
                EasyInputMessageParam(content="Hello, world!", role="user"),
                {
                    "message.role": "user",
                    "message.content": "Hello, world!",
                },
                id="basic_user_message",
            ),
            pytest.param(
                EasyInputMessageParam(content="Hi there!", role="assistant"),
                {
                    "message.role": "assistant",
                    "message.content": "Hi there!",
                },
                id="basic_assistant_message",
            ),
            pytest.param(
                EasyInputMessageParam(
                    content=[ResponseInputTextParam(type="input_text", text="Hello")],
                    role="user",
                ),
                {
                    "message.role": "user",
                    "message.contents.0.message_content.type": "text",
                    "message.contents.0.message_content.text": "Hello",
                },
                id="message_with_content_list",
            ),
            pytest.param(
                EasyInputMessageParam(
                    content=[
                        ResponseInputImageParam(
                            type="input_image",
                            image_url="https://example.com/image.jpg",
                            detail="low",
                        )
                    ],
                    role="user",
                ),
                {
                    "message.role": "user",
                    "message.contents.0.message_content.type": "image",
                    "message.contents.0.message_content.image.image.url": "https://example.com/image.jpg",  # noqa: E501
                },
                id="message_with_image_content",
            ),
            pytest.param(
                EasyInputMessageParam(
                    content=[
                        ResponseInputImageParam(
                            type="input_image",
                            image_url="https://example.com/image.jpg",
                            detail="low",
                        )
                    ],
                    role="user",
                ),
                {
                    "message.role": "user",
                    "message.contents.0.message_content.type": "image",
                    "message.contents.0.message_content.image.image.url": "https://example.com/image.jpg",  # noqa: E501
                },
                id="message_with_image_url_content",
            ),
            pytest.param(
                EasyInputMessageParam(
                    content=[
                        ResponseInputTextParam(type="input_text", text="Hello"),
                        ResponseInputTextParam(type="input_text", text="World"),
                    ],
                    role="user",
                ),
                {
                    "message.role": "user",
                    "message.contents.0.message_content.text": "Hello",
                    "message.contents.0.message_content.type": "text",
                    "message.contents.1.message_content.text": "World",
                    "message.contents.1.message_content.type": "text",
                },
                id="message_with_list_content",
            ),
        ],
    )
    def test_EasyInputMessageParam(
        self,
        obj: EasyInputMessageParam,
        expected: dict[str, AttributeValue],
    ) -> None:
        actual = dict(_ResponsesApiAttributes._get_attributes_from_response_input_item_param(obj))
        assert actual == expected

    @pytest.mark.parametrize(
        "obj,expected",
        [
            pytest.param(
                ResponseOutputMessageParam(
                    id="msg_123",
                    role="assistant",
                    content=[{"type": "output_text", "text": "Hello", "annotations": []}],
                    status="completed",
                    type="message",
                ),
                {
                    "message.role": "assistant",
                    "message.contents.0.message_content.type": "text",
                    "message.contents.0.message_content.text": "Hello",
                },
                id="basic_output_message",
            ),
            pytest.param(
                ResponseOutputMessageParam(
                    id="msg_123",
                    role="assistant",
                    content=[
                        {"type": "output_text", "text": "Hello", "annotations": []},
                        {"type": "output_text", "text": "World", "annotations": []},
                    ],
                    status="completed",
                    type="message",
                ),
                {
                    "message.role": "assistant",
                    "message.contents.0.message_content.type": "text",
                    "message.contents.0.message_content.text": "Hello",
                    "message.contents.1.message_content.type": "text",
                    "message.contents.1.message_content.text": "World",
                },
                id="output_message_with_multiple_contents",
            ),
        ],
    )
    def test_ResponseOutputMessageParam(
        self,
        obj: ResponseOutputMessageParam,
        expected: dict[str, AttributeValue],
    ) -> None:
        actual = dict(_ResponsesApiAttributes._get_attributes_from_response_input_item_param(obj))
        assert actual == expected

    @pytest.mark.parametrize(
        "obj,expected",
        [
            pytest.param(
                FunctionCallOutput(
                    type="function_call_output",
                    call_id="call_123",
                    output="The weather is sunny.",
                ),
                {
                    "message.role": "tool",
                    "message.tool_call_id": "call_123",
                    "message.content": "The weather is sunny.",
                },
                id="function_call_output_message",
            ),
            pytest.param(
                FunctionCallOutput(
                    type="function_call_output",
                    call_id="call_456",
                    output="",
                ),
                {
                    "message.role": "tool",
                    "message.tool_call_id": "call_456",
                    "message.content": "",
                },
                id="function_call_output_with_empty_content",
            ),
        ],
    )
    def test_FunctionCallOutput(
        self,
        obj: FunctionCallOutput,
        expected: dict[str, AttributeValue],
    ) -> None:
        actual = dict(_ResponsesApiAttributes._get_attributes_from_response_input_item_param(obj))
        assert actual == expected

    @pytest.mark.parametrize(
        "obj,expected",
        [
            pytest.param(
                ResponseFunctionToolCallParam(
                    type="function_call",
                    call_id="call_123",
                    name="get_weather",
                    arguments="{}",
                ),
                {
                    "message.role": "assistant",
                    "message.tool_calls.0.tool_call.id": "call_123",
                    "message.tool_calls.0.tool_call.function.name": "get_weather",
                },
                id="basic_function_call",
            ),
            pytest.param(
                ResponseFunctionToolCallParam(
                    type="function_call",
                    call_id="call_456",
                    name="get_stock_price",
                    arguments='{"symbol": "AAPL"}',
                ),
                {
                    "message.role": "assistant",
                    "message.tool_calls.0.tool_call.id": "call_456",
                    "message.tool_calls.0.tool_call.function.name": "get_stock_price",
                    "message.tool_calls.0.tool_call.function.arguments": '{"symbol": "AAPL"}',
                },
                id="function_call_with_arguments",
            ),
            pytest.param(
                ResponseFunctionToolCallParam(
                    type="function_call",
                    call_id="call_789",
                    name="search_database",
                    arguments='{"query": "find users", "limit": 10}',
                ),
                {
                    "message.role": "assistant",
                    "message.tool_calls.0.tool_call.id": "call_789",
                    "message.tool_calls.0.tool_call.function.name": "search_database",
                    "message.tool_calls.0.tool_call.function.arguments": '{"query": "find users", "limit": 10}',  # noqa: E501
                },
                id="function_call_with_complex_arguments",
            ),
        ],
    )
    def test_ResponseFunctionToolCallParam(
        self,
        obj: ResponseFunctionToolCallParam,
        expected: dict[str, AttributeValue],
    ) -> None:
        actual = dict(_ResponsesApiAttributes._get_attributes_from_response_input_item_param(obj))
        assert actual == expected

    @pytest.mark.parametrize(
        "obj,expected",
        [
            pytest.param(
                ResponseReasoningItemParam(
                    type="reasoning",
                    id="reason_123",
                    summary=[
                        Summary(
                            type="summary_text",
                            text="This is a reasoning step",
                        )
                    ],
                ),
                {
                    "message.role": "assistant",
                    "message.contents.0.message_content.type": "text",
                    "message.contents.0.message_content.text": "This is a reasoning step",
                },
                id="basic_reasoning_item",
            ),
            pytest.param(
                ResponseReasoningItemParam(
                    type="reasoning",
                    id="reason_456",
                    summary=[
                        Summary(
                            type="summary_text",
                            text="First reasoning step",
                        ),
                        Summary(
                            type="summary_text",
                            text="Second reasoning step",
                        ),
                    ],
                ),
                {
                    "message.role": "assistant",
                    "message.contents.0.message_content.type": "text",
                    "message.contents.0.message_content.text": "First reasoning step",
                    "message.contents.1.message_content.type": "text",
                    "message.contents.1.message_content.text": "Second reasoning step",
                },
                id="reasoning_item_with_multiple_steps",
            ),
        ],
    )
    def test_ResponseReasoningItemParam(
        self,
        obj: ResponseReasoningItemParam,
        expected: dict[str, AttributeValue],
    ) -> None:
        actual = dict(_ResponsesApiAttributes._get_attributes_from_response_input_item_param(obj))
        assert actual == expected
