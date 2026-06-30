import pytest
from openai.types.responses import ResponseOutputRefusal
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation.openai._attributes._responses_api import _ResponsesApiAttributes


class TestResponseOutputRefusal:
    @pytest.mark.parametrize(
        "obj,expected",
        [
            pytest.param(
                ResponseOutputRefusal(
                    type="refusal",
                    refusal="I cannot provide that information.",
                ),
                {
                    "message_content.type": "text",
                    "message_content.text": "I cannot provide that information.",
                },
                id="basic_refusal",
            ),
        ],
    )
    def test_ResponseOutputRefusal(
        self,
        obj: ResponseOutputRefusal,
        expected: dict[str, AttributeValue],
    ) -> None:
        actual = dict(_ResponsesApiAttributes._get_attributes_from_response_output_refusal(obj))
        assert actual == expected
