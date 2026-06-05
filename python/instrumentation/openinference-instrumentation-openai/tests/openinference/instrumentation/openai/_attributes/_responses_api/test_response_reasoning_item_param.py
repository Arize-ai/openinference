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
                    "message.contents.0.message_content.text": "First\n\nSecond",
                },
                id="multiple_summary_entries_concatenated",
            ),
            pytest.param(
                ResponseReasoningItemParam(
                    type="reasoning",
                    id="reason_single",
                    summary=[
                        Summary(type="summary_text", text="Only summary"),
                    ],
                ),
                {
                    "message.role": "assistant",
                    "message.contents.0.message_content.type": "reasoning",
                    "message.contents.0.message_content.text": "Only summary",
                },
                id="single_summary_entry",
            ),
            pytest.param(
                ResponseReasoningItemParam(
                    type="reasoning",
                    id="reason_encrypted",
                    summary=[],
                    encrypted_content="enc_abc123",
                ),
                {
                    "message.role": "assistant",
                    "message.contents.0.message_content.type": "reasoning",
                    "message.contents.0.message_content.encrypted_content": "enc_abc123",
                },
                id="encrypted_content_without_summary",
            ),
            pytest.param(
                ResponseReasoningItemParam(
                    type="reasoning",
                    id="reason_both",
                    summary=[
                        Summary(type="summary_text", text="Reasoning text"),
                    ],
                    encrypted_content="enc_xyz789",
                ),
                {
                    "message.role": "assistant",
                    "message.contents.0.message_content.type": "reasoning",
                    "message.contents.0.message_content.text": "Reasoning text",
                    "message.contents.0.message_content.encrypted_content": "enc_xyz789",
                },
                id="summary_and_encrypted_content",
            ),
            pytest.param(
                ResponseReasoningItemParam(
                    type="reasoning",
                    id="reason_no_content",
                    summary=[],
                ),
                {
                    "message.role": "assistant",
                },
                id="empty_summary_no_encrypted_content",
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

    def test_id_not_emitted(self) -> None:
        obj = ResponseReasoningItemParam(
            type="reasoning",
            id="should_not_appear",
            summary=[Summary(type="summary_text", text="text")],
        )
        actual = dict(
            _ResponsesApiAttributes._get_attributes_from_response_reasoning_item_param(obj)
        )
        assert not any("id" in key for key in actual)
