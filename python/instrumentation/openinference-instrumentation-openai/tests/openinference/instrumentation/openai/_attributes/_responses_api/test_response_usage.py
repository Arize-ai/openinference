import pytest
from openai.types.responses import ResponseUsage
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation.openai._attributes._responses_api import _ResponsesApiAttributes


class TestResponseUsage:
    @pytest.mark.parametrize(
        "obj,expected",
        [
            pytest.param(
                ResponseUsage(
                    input_tokens=100,
                    output_tokens=50,
                    total_tokens=150,
                    input_tokens_details=InputTokensDetails(
                        cached_tokens=20,
                    ),
                    output_tokens_details=OutputTokensDetails(
                        reasoning_tokens=10,
                    ),
                ),
                {
                    "llm.token_count.total": 150,
                    "llm.token_count.prompt": 100,
                    "llm.token_count.completion": 50,
                    "llm.token_count.prompt_details.cache_read": 20,
                    "llm.token_count.completion_details.reasoning": 10,
                },
                id="with_token_details",
            ),
        ],
    )
    def test_ResponseUsage(
        self,
        obj: ResponseUsage,
        expected: dict[str, AttributeValue],
    ) -> None:
        actual = dict(_ResponsesApiAttributes._get_attributes_from_response_usage(obj))
        assert actual == expected
