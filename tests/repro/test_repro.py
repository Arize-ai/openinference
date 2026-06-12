"""Regression test for issue #58: StrandsAgentsToOpenInferenceProcessor must not
mutate spans that were not produced by the Strands Agents framework.

This test uses MockReadableSpan directly (no TracerProvider / editable install
from the repo root required) so it can run from any working directory as long as
the package under python/instrumentation/openinference-instrumentation-strands-agents
is installed (editable or otherwise).
"""

from typing import Any, Dict, List, Optional

from opentelemetry.trace import SpanKind

from openinference.instrumentation.strands_agents.processor import (
    StrandsAgentsToOpenInferenceProcessor,
)
from openinference.semconv.trace import SpanAttributes

SENTINEL = "REPRO_BUG_SENTINEL"


class MockReadableSpan:
    """Minimal stand-in for opentelemetry.sdk.trace.ReadableSpan."""

    def __init__(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        events: Optional[List[Any]] = None,
    ) -> None:
        self.name = name
        self._attributes = attributes or {}
        self._events = events or []
        self.kind = SpanKind.INTERNAL
        self.parent = None

    def get_span_context(self) -> Any:
        class MockSpanContext:
            span_id = 99999

        return MockSpanContext()


def test_repro_non_strands_span_not_mutated() -> None:
    """Bug #58: on_end() must skip spans that are not from the Strands framework.

    A span with only non-gen_ai.* attributes (e.g. rpc.system, http.method) and
    a generic name must exit on_end() with its original attributes intact — the
    processor must NOT replace them with an OpenInference-formatted dict.
    """
    processor = StrandsAgentsToOpenInferenceProcessor()

    span = MockReadableSpan(
        name="non-framework-span",
        attributes={
            "rpc.system": "grpc",
            "http.method": "GET",
            "custom.key": "preserved_value",
        },
    )

    processor.on_end(span)  # type: ignore[arg-type]

    attrs = span._attributes

    # Original attributes must still be present at the top level
    assert attrs.get("rpc.system") == "grpc", (
        f"REPRO: {SENTINEL} — rpc.system was destroyed. Got: {attrs}"
    )
    assert attrs.get("http.method") == "GET", (
        f"REPRO: {SENTINEL} — http.method was destroyed. Got: {attrs}"
    )
    assert attrs.get("custom.key") == "preserved_value", (
        f"REPRO: {SENTINEL} — custom.key was destroyed. Got: {attrs}"
    )

    # The processor must NOT have injected OpenInference span-kind
    assert SpanAttributes.OPENINFERENCE_SPAN_KIND not in attrs, (
        f"REPRO: {SENTINEL} — OPENINFERENCE_SPAN_KIND was injected into a non-Strands span. "
        f"Got: {attrs}"
    )

    # The processor must NOT have packed attributes into a metadata blob
    assert SpanAttributes.METADATA not in attrs, (
        f"REPRO: {SENTINEL} — non-Strands attributes were packed into metadata JSON blob. "
        f"Got: {attrs}"
    )
