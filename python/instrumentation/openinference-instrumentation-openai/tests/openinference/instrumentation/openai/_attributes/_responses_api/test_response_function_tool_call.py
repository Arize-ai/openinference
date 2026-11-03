import pytest
from openai.types.responses import ResponseFunctionToolCall
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation.openai._attributes._responses_api import _ResponsesApiAttributes


class TestResponseFunctionToolCall:
    @pytest.mark.parametrize(
        "obj,expected",
        [
            pytest.param(
                ResponseFunctionToolCall(
                    arguments='{"arg1": "value1", "arg2": 42}',
                    call_id="call_123",
                    name="test_function",
                    type="function_call",
                ),
                {
                    "tool_call.id": "call_123",
                    "tool_call.function.name": "test_function",
                    "tool_call.function.arguments": '{"arg1": "value1", "arg2": 42}',  # noqa: E501
                },
                id="with_arguments",
            ),
            pytest.param(
                ResponseFunctionToolCall(
                    arguments="{}",
                    call_id="call_456",
                    name="empty_function",
                    type="function_call",
                ),
                {
                    "tool_call.id": "call_456",
                    "tool_call.function.name": "empty_function",
                },
                id="with_empty_arguments",
            ),
            pytest.param(
                ResponseFunctionToolCall(
                    arguments='{"arg1": "value1"}',
                    call_id="call_789",
                    name="test_function",
                    type="function_call",
                    id="optional_id",
                    status="completed",
                ),
                {
                    "tool_call.id": "call_789",
                    "tool_call.function.name": "test_function",
                    "tool_call.function.arguments": '{"arg1": "value1"}',
                },
                id="with_optional_fields",
            ),
        ],
    )
    def test_ResponseFunctionToolCall(
        self,
        obj: ResponseFunctionToolCall,
        expected: dict[str, AttributeValue],
    ) -> None:
        actual = dict(_ResponsesApiAttributes._get_attributes_from_response_function_tool_call(obj))
        assert actual == expected
