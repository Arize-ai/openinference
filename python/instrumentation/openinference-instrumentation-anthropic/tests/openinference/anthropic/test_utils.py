import logging
from types import SimpleNamespace
from typing import Iterator, Tuple

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation.anthropic._utils import _finish_tracing, _get_token_counts
from openinference.instrumentation.anthropic._with_span import _WithSpan


class _Attributes:
    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        yield "llm.system", "anthropic"


def _failing_params() -> Iterator[Tuple[str, AttributeValue]]:
    yield "llm.provider", "anthropic"
    raise RuntimeError("span finalization blew up")


def _make_usage(**kwargs: object) -> object:
    """Build a minimal Usage-like namespace for testing _get_token_counts."""
    defaults = {
        "input_tokens": None,
        "cache_creation_input_tokens": None,
        "cache_read_input_tokens": None,
        "output_tokens": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_get_token_counts_with_null_input_tokens_does_not_raise() -> None:
    """_get_token_counts must not raise TypeError when usage.input_tokens is None.

    Anthropic streaming snapshots can have input_tokens=None before the
    final message_delta event arrives.  Previously None + 0 raised TypeError,
    which escaped into _finish_tracing, causing the entire streaming LLM span
    to be exported with no attributes (model name, messages, token counts all
    dropped).
    """
    usage = _make_usage(input_tokens=None, output_tokens=10)
    attrs = dict(_get_token_counts(usage))  # must not raise
    # prompt_tokens = 0, so LLM_TOKEN_COUNT_PROMPT is absent
    assert "llm.token_count.prompt" not in attrs
    assert attrs["llm.token_count.completion"] == 10
    assert attrs["llm.token_count.total"] == 10


def test_get_token_counts_with_all_none_fields_does_not_raise() -> None:
    """When all Usage fields are None, _get_token_counts must yield nothing."""
    usage = _make_usage()
    attrs = dict(_get_token_counts(usage))
    assert attrs == {}


def test_get_token_counts_with_normal_usage() -> None:
    """Sanity-check: normal integer usage fields produce the expected attributes."""
    usage = _make_usage(input_tokens=100, output_tokens=50, cache_read_input_tokens=20)
    attrs = dict(_get_token_counts(usage))
    # prompt = 100 + 20 = 120
    assert attrs["llm.token_count.prompt"] == 120
    assert attrs["llm.token_count.completion"] == 50
    assert attrs["llm.token_count.total"] == 170
    assert attrs["llm.token_count.prompt_details.cache_read"] == 20


def test_finish_tracing_logs_and_recovers_when_finalization_fails(
    tracer_provider: TracerProvider,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An instrumentation failure while ending the span must not reach user code."""
    span = tracer_provider.get_tracer(__name__).start_span("Test")
    with_span = _WithSpan(span=span, params=_failing_params())

    with caplog.at_level(logging.ERROR):
        _finish_tracing(with_span=with_span, has_attributes=_Attributes())

    assert "Failed to finish tracing" in caplog.text
