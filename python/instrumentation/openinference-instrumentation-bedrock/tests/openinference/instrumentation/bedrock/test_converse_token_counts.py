from typing import Any, Dict, cast

from openinference.instrumentation.bedrock._converse_attributes import (
    get_attributes_from_response_data,
)
from openinference.semconv.trace import SpanAttributes

REQUEST_DATA: Dict[str, Any] = {"modelId": "anthropic.claude-3", "messages": []}


def _response_data(usage: Dict[str, int]) -> Dict[str, Any]:
    return {
        "stopReason": "end_turn",
        "output": {"message": {"role": "assistant", "content": [{"text": "Answer."}]}},
        "usage": usage,
    }


def test_prompt_token_count_includes_cache_tokens() -> None:
    # Bedrock's inputTokens excludes cache reads and writes, while totalTokens includes them.
    usage = {
        "inputTokens": 4299,
        "outputTokens": 631,
        "totalTokens": 58291,
        "cacheReadInputTokens": 35574,
        "cacheWriteInputTokens": 17787,
    }
    attributes = get_attributes_from_response_data(
        cast(Any, REQUEST_DATA), cast(Any, _response_data(usage))
    )
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 4299 + 35574 + 17787
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == 631
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 58291
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ] == 35574
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE] == 17787
    assert (
        attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT]
        + attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION]
        == attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL]
    )


def test_prompt_token_count_without_cache_tokens() -> None:
    usage = {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}
    attributes = get_attributes_from_response_data(
        cast(Any, REQUEST_DATA), cast(Any, _response_data(usage))
    )
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 10
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == 5
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 15
    assert SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ not in attributes
