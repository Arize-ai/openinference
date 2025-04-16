import pytest
from openai.types.responses import ResponseOutputMessage, ResponseOutputText
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation.openai._attributes._responses_api import _ResponsesApiAttributes


class TestResponseOutputMessage:
    @pytest.mark.parametrize(
        "obj,expected",
        [
            pytest.param(
                ResponseOutputMessage(
                    id="msg_123",
                    role="assistant",
                    content=[
                        ResponseOutputText(
                            type="output_text",
                            text="Hello, world!",
                            annotations=[],
                        )
                    ],
                    status="completed",
                    type="message",
                ),
                {
                    "message.role": "assistant",
                    "message.contents.0.message_content.type": "text",
                    "message.contents.0.message_content.text": "Hello, world!",
                },
                id="basic_output_message",
            ),
            pytest.param(
                ResponseOutputMessage(
                    id="msg_456",
                    role="assistant",
                    content=[
                        ResponseOutputText(
                            type="output_text",
                            text="First part",
                            annotations=[],
                        ),
                        ResponseOutputText(
                            type="output_text",
                            text="Second part",
                            annotations=[],
                        ),
                    ],
                    status="completed",
                    type="message",
                ),
                {
                    "message.role": "assistant",
                    "message.contents.0.message_content.type": "text",
                    "message.contents.0.message_content.text": "First part",
                    "message.contents.1.message_content.type": "text",
                    "message.contents.1.message_content.text": "Second part",
                },
                id="message_with_multiple_contents",
            ),
        ],
    )
    def test_ResponseOutputMessage(
        self,
        obj: ResponseOutputMessage,
        expected: dict[str, AttributeValue],
    ) -> None:
        actual = dict(_ResponsesApiAttributes._get_attributes_from_response_output_message(obj))
        assert actual == expected
