import pytest
from openai.types.responses.response_reasoning_item_param import ResponseReasoningItemParam, Summary
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation.openai._attributes._responses_api import _ResponsesApiAttributes


class TestResponseReasoningItemParam:
    @pytest.mark.parametrize(
        "obj,expected",
        [
            pytest.param(
                ResponseReasoningItemParam(
                    type="reasoning",
                    id="reason_456",
                    summary=[
                        Summary(type="summary_text", text="First"),
                        Summary(type="summary_text", text="Second"),
                    ],
                ),
                {
                    "message.role": "assistant",
                    "message.contents.0.message_content.type": "reasoning",
                    "message.contents.0.message_content.text": "First\nSecond",
                },
                id="multiple_summary_entries_concatenated",
            ),
            pytest.param(
                ResponseReasoningItemParam(
                    type="reasoning",
                    id="reason_789",
                    summary=[
                        Summary(type="summary_text", text="Single summary"),
                    ],
                    encrypted_content="gAAAAA==",
                ),
                {
                    "message.role": "assistant",
                    "message.contents.0.message_content.type": "reasoning",
                    "message.contents.0.message_content.text": "Single summary",
                    "message.contents.0.message_content.encrypted_content": "gAAAAA==",
                },
                id="with_encrypted_content",
            ),
            pytest.param(
                ResponseReasoningItemParam(
                    type="reasoning",
                    id="reason_000",
                    summary=[],
                    encrypted_content="gAAAAA==",
                ),
                {
                    "message.role": "assistant",
                    "message.contents.0.message_content.type": "reasoning",
                    "message.contents.0.message_content.encrypted_content": "gAAAAA==",
                },
                id="empty_summary_with_encrypted_content",
            ),
        ],
    )
    def test_ResponseReasoningItemParam(
        self,
        obj: ResponseReasoningItemParam,
        expected: dict[str, AttributeValue],
    ) -> None:
        actual = dict(
            _ResponsesApiAttributes._get_attributes_from_response_reasoning_item_param(obj)
        )
        assert actual == expected
