import pytest
from openai.types.responses import ResponseOutputRefusalParam
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation.openai._attributes._responses_api import _ResponsesApiAttributes


class TestResponseOutputRefusalParam:
    @pytest.mark.parametrize(
        "obj,expected",
        [
            pytest.param(
                ResponseOutputRefusalParam(
                    type="refusal",
                    refusal="I cannot provide that information.",
                ),
                {
                    "message_content.type": "text",
                    "message_content.text": "I cannot provide that information.",
                },
                id="basic_refusal_param",
            ),
        ],
    )
    def test_ResponseOutputRefusalParam(
        self,
        obj: ResponseOutputRefusalParam,
        expected: dict[str, AttributeValue],
    ) -> None:
        actual = dict(
            _ResponsesApiAttributes._get_attributes_from_response_output_refusal_param(obj)
        )
        assert actual == expected
