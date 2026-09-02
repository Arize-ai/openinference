import logging
from types import SimpleNamespace
from typing import Iterator, Tuple

import pytest
from anthropic.types import Usage
from openinference.semconv.trace import SpanAttributes
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation.anthropic._utils import (
    _finish_tracing,
    _get_token_counts,
)
from openinference.instrumentation.anthropic._with_span import _WithSpan


class _Attributes:
    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        yield "llm.system", "anthropic"


def _failing_params() -> Iterator[Tuple[str, AttributeValue]]:
    yield "llm.provider", "anthropic"
    raise RuntimeError("span finalization blew up")


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


def test_get_token_counts_standard_usage() -> None:
    usage = Usage.construct(
        input_tokens=10,
        output_tokens=20,
        cache_creation_input_tokens=5,
        cache_read_input_tokens=2,
    )
    attributes = dict(_get_token_counts(usage))
    assert attributes == {
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 17,
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 20,
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ: 2,
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE: 5,
        SpanAttributes.LLM_TOKEN_COUNT_TOTAL: 37,
    }


def test_get_token_counts_none_input_tokens_guard() -> None:
    """Regression test for Issue #3499: NoneType input_tokens should not raise TypeError."""
    usage = Usage.construct(
        input_tokens=None,
        output_tokens=15,
        cache_creation_input_tokens=3,
        cache_read_input_tokens=7,
    )
    attributes = dict(_get_token_counts(usage))
    assert attributes == {
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 10,
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 15,
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ: 7,
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE: 3,
        SpanAttributes.LLM_TOKEN_COUNT_TOTAL: 25,
    }


def test_get_token_counts_all_none() -> None:
    usage = Usage.construct(
        input_tokens=None,
        output_tokens=None,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
    )
    attributes = dict(_get_token_counts(usage))
    assert attributes == {}


def test_get_token_counts_only_output_tokens() -> None:
    usage = Usage.construct(
        input_tokens=None,
        output_tokens=50,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
    )
    attributes = dict(_get_token_counts(usage))
    assert attributes == {
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 50,
        SpanAttributes.LLM_TOKEN_COUNT_TOTAL: 50,
    }


def test_get_token_counts_only_input_tokens() -> None:
    usage = Usage.construct(
        input_tokens=42,
        output_tokens=None,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
    )
    attributes = dict(_get_token_counts(usage))
    assert attributes == {
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 42,
        SpanAttributes.LLM_TOKEN_COUNT_TOTAL: 42,
    }


def test_get_token_counts_duck_typed_object_with_none() -> None:
    usage = SimpleNamespace(
        input_tokens=None,
        output_tokens=None,
        cache_creation_input_tokens=8,
        cache_read_input_tokens=None,
    )
    attributes = dict(_get_token_counts(usage))  # type: ignore[arg-type]
    assert attributes == {
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 8,
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE: 8,
        SpanAttributes.LLM_TOKEN_COUNT_TOTAL: 8,
    }
