import pytest
from openai.types.responses.response_reasoning_item import ResponseReasoningItem, Summary
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation.openai._attributes._responses_api import _ResponsesApiAttributes


class TestResponseReasoningItem:
    @pytest.mark.parametrize(
        "obj,expected",
        [
            pytest.param(
                ResponseReasoningItem(
                    type="reasoning",
                    id="reason_456",
                    summary=[
                        Summary(type="summary_text", text="First"),
                        Summary(type="summary_text", text="Second"),
                    ],
                ),
                {
                    "message.contents.0.message_content.type": "reasoning",
                    "message.contents.0.message_content.id": "reason_456",
                    "message.contents.0.message_content.text": "First\nSecond",
                },
                id="multiple_summary_entries_concatenated",
            ),
            pytest.param(
                ResponseReasoningItem(
                    type="reasoning",
                    id="reason_789",
                    summary=[
                        Summary(type="summary_text", text="Single summary"),
                    ],
                    encrypted_content="gAAAAA==",
                ),
                {
                    "message.contents.0.message_content.type": "reasoning",
                    "message.contents.0.message_content.id": "reason_789",
                    "message.contents.0.message_content.text": "Single summary",
                    "message.contents.0.message_content.encrypted_content": "gAAAAA==",
                },
                id="with_encrypted_content",
            ),
            pytest.param(
                ResponseReasoningItem(
                    type="reasoning",
                    id="reason_000",
                    summary=[],
                    encrypted_content="gAAAAA==",
                ),
                {
                    "message.contents.0.message_content.type": "reasoning",
                    "message.contents.0.message_content.id": "reason_000",
                    "message.contents.0.message_content.encrypted_content": "gAAAAA==",
                },
                id="empty_summary_with_encrypted_content",
            ),
        ],
    )
    def test_ResponseReasoningItem(
        self,
        obj: ResponseReasoningItem,
        expected: dict[str, AttributeValue],
    ) -> None:
        actual = dict(_ResponsesApiAttributes._get_attributes_from_response_reasoning_item(obj))
        assert actual == expected
