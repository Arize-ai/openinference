"""Regression test: completion text_tokens (a COUNT) must not be written to a COST attribute.

_set_token_counts_from_usage wrote completion_tokens_details.text_tokens (a token count) into
SpanAttributes.LLM_COST_COMPLETION_DETAILS_OUTPUT ("llm.cost.completion_details.output", a USD
cost field), so a cost dashboard reads e.g. "$500" for 500 text tokens. Every sibling writes to
llm.token_count.*. This routes it to the token-count attribute instead.

  with the fix -> PASS (count under token_count.*, NOT under llm.cost.*)
  without it   -> FAIL (count under the cost attribute)
"""
from types import SimpleNamespace

from openinference.instrumentation.litellm import _set_token_counts_from_usage
from openinference.semconv.trace import SpanAttributes


class FakeSpan:
    def __init__(self):
        self.attrs = {}

    def set_attribute(self, k, v):
        self.attrs[k] = v


def test_completion_text_tokens_go_to_token_count_not_cost():
    usage = SimpleNamespace(
        prompt_tokens=None, completion_tokens=None, total_tokens=None,
        completion_tokens_details=SimpleNamespace(text_tokens=500, reasoning_tokens=None, audio_tokens=None),
    )
    result = SimpleNamespace(usage=usage)
    span = FakeSpan()
    _set_token_counts_from_usage(span, result)
    assert SpanAttributes.LLM_COST_COMPLETION_DETAILS_OUTPUT not in span.attrs, span.attrs
    assert span.attrs.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_TEXT) == 500, span.attrs
