import pytest
from openai.types.responses import ResponseOutputTextParam
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation.openai._attributes._responses_api import _ResponsesApiAttributes


class TestResponseOutputTextParam:
    @pytest.mark.parametrize(
        "obj,expected",
        [
            pytest.param(
                ResponseOutputTextParam(
                    type="output_text",
                    text="This is a response text parameter",
                    annotations=[],
                ),
                {
                    "message_content.type": "text",
                    "message_content.text": "This is a response text parameter",
                },
                id="basic_output_text_param",
            ),
        ],
    )
    def test_ResponseOutputTextParam(
        self,
        obj: ResponseOutputTextParam,
        expected: dict[str, AttributeValue],
    ) -> None:
        actual = dict(_ResponsesApiAttributes._get_attributes_from_response_output_text_param(obj))
        assert actual == expected
