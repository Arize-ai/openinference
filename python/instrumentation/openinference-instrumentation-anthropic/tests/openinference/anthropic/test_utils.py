import logging
from typing import Iterator, Tuple

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation.anthropic._utils import _finish_tracing
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
