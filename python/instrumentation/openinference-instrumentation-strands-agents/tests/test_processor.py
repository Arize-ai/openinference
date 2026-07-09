"""Tests for Strands to OpenInference processor."""

from typing import Any, Dict, List, Optional

import pytest
from opentelemetry.trace import SpanKind, Status, StatusCode

from openinference.instrumentation.strands_agents.processor import (
    StrandsAgentsToOpenInferenceProcessor,
)
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes


class MockEvent:
    """Mock event for testing."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.attributes: Dict[str, Any] = {}


class MockReadableSpan:
    """Mock ReadableSpan for testing."""

    def __init__(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        events: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.name = name
        self._attributes = attributes or {}
        self._events = events or []
        self._status = Status(status_code=StatusCode.OK)
        self.kind = SpanKind.INTERNAL
        self.parent = None

    @property
    def status(self) -> Status:
        return self._status

    def get_span_context(self) -> Any:
        """Mock get_span_context."""

        class MockSpanContext:
            def __init__(self) -> None:
                self.span_id = 12345

        return MockSpanContext()

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON dict."""
        return {
            "name": self.name,
            "attributes": self._attributes,
            "events": self._events,
        }


class TestStrandsAgentsToOpenInferenceProcessor:
    """Test cases for StrandsAgentsToOpenInferenceProcessor."""

    def test_processor_can_be_instantiated(self) -> None:
        """Test that the processor can be instantiated."""
        processor = StrandsAgentsToOpenInferenceProcessor()
        assert processor is not None

    def test_processor_transforms_llm_span(self) -> None:
        """Test that the processor transforms LLM spans correctly."""
        processor = StrandsAgentsToOpenInferenceProcessor()
        span = MockReadableSpan(
            name="chat",
            attributes={
                "gen_ai.request.model": "gpt-4",
                "gen_ai.usage.input_tokens": 100,
                "gen_ai.usage.output_tokens": 50,
                "gen_ai.system": "strands-agents",
            },
        )

        # Call on_end to process the span
        processor.on_end(span)  # type: ignore[arg-type]

        # Check that attributes were transformed
        assert span._attributes.get(SpanAttributes.LLM_MODEL_NAME) == "gpt-4"
        assert span._attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 100
        assert span._attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 50
        assert (
            span._attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
            == OpenInferenceSpanKindValues.LLM.value
        )

    def test_processor_maps_cache_token_counts(self) -> None:
        """Cache read/write tokens map to prompt_details and roll up into the prompt count."""
        processor = StrandsAgentsToOpenInferenceProcessor()
        span = MockReadableSpan(
            name="chat",
            attributes={
                "gen_ai.request.model": "us.anthropic.claude-sonnet-4-20250514-v1:0",
                "gen_ai.usage.input_tokens": 4299,
                "gen_ai.usage.output_tokens": 631,
                "gen_ai.usage.cache_read_input_tokens": 35574,
                "gen_ai.usage.cache_write_input_tokens": 17787,
                "gen_ai.usage.total_tokens": 58291,
                "gen_ai.system": "strands-agents",
            },
        )

        processor.on_end(span)  # type: ignore[arg-type]

        attributes = span._attributes
        assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ) == 35574
        assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE) == 17787
        # Prompt aggregate includes cached tokens so prompt + completion == total
        assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 4299 + 35574 + 17787
        assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 631
        assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 58291

    def test_processor_omits_cache_details_when_zero(self) -> None:
        """Zero cache counts (Strands emits 0 when caching is unused) add no attributes."""
        processor = StrandsAgentsToOpenInferenceProcessor()
        span = MockReadableSpan(
            name="chat",
            attributes={
                "gen_ai.request.model": "gpt-4",
                "gen_ai.usage.input_tokens": 100,
                "gen_ai.usage.output_tokens": 50,
                "gen_ai.usage.total_tokens": 150,
                "gen_ai.usage.cache_read_input_tokens": 0,
                "gen_ai.usage.cache_write_input_tokens": 0,
                "gen_ai.system": "strands-agents",
            },
        )

        processor.on_end(span)  # type: ignore[arg-type]

        attributes = span._attributes
        assert SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ not in attributes
        assert SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE not in attributes
        assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 100
        assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 150

    def test_processor_transforms_agent_span(self) -> None:
        """Test that the processor transforms agent spans correctly."""
        processor = StrandsAgentsToOpenInferenceProcessor()
        span = MockReadableSpan(
            name="invoke_agent test_agent",
            attributes={
                "agent.name": "test_agent",
                "gen_ai.provider.name": "strands-agents",
            },
        )

        processor.on_end(span)  # type: ignore[arg-type]

        assert (
            span._attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
            == OpenInferenceSpanKindValues.AGENT.value
        )

    def test_processor_transforms_tool_span(self) -> None:
        """Test that the processor transforms tool spans correctly."""
        processor = StrandsAgentsToOpenInferenceProcessor()
        span = MockReadableSpan(
            name="execute_tool calculator",
            attributes={
                "gen_ai.tool.name": "calculator",
                "gen_ai.provider.name": "strands-agents",
            },
        )

        processor.on_end(span)  # type: ignore[arg-type]

        assert span._attributes.get(SpanAttributes.TOOL_NAME) == "calculator"
        assert (
            span._attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
            == OpenInferenceSpanKindValues.TOOL.value
        )

    def test_processor_transforms_chain_span(self) -> None:
        """Test that the processor transforms chain spans correctly."""
        processor = StrandsAgentsToOpenInferenceProcessor()
        span = MockReadableSpan(
            name="execute_event_loop_cycle",
            attributes={
                "event_loop.cycle_id": "cycle-123",
                "gen_ai.provider.name": "strands-agents",
            },
        )

        processor.on_end(span)  # type: ignore[arg-type]

        assert (
            span._attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
            == OpenInferenceSpanKindValues.CHAIN.value
        )

    @pytest.mark.parametrize(
        ("span_name", "attributes"),
        [
            # HTTP span
            (
                "http.request",
                {
                    "http.method": "GET",
                    "http.url": "https://example.com/api",
                    "http.status_code": 200,
                },
            ),
            # RPC span
            (
                "rpc.call",
                {
                    "rpc.system": "grpc",
                    "rpc.service": "my.Service",
                    "rpc.method": "DoWork",
                },
            ),
            # AWS span
            (
                "aws.request",
                {
                    "rpc.system": "aws-api",
                    "rpc.service": "S3",
                    "rpc.method": "GetObject",
                },
            ),
        ],
    )
    def test_processor_leaves_non_strands_spans_unchanged(
        self,
        span_name: str,
        attributes: Dict[str, Any],
    ) -> None:
        """Test that non-Strands spans are not modified by the processor."""
        processor = StrandsAgentsToOpenInferenceProcessor()
        span = MockReadableSpan(name=span_name, attributes=dict(attributes))

        processor.on_end(span)  # type: ignore[arg-type]

        assert span._attributes == attributes

    def test_mixed_trace_only_transforms_strands_spans(self) -> None:
        """Test that only Strands spans are transformed in mixed traces."""
        processor = StrandsAgentsToOpenInferenceProcessor()
        http_span = MockReadableSpan(
            name="http.request",
            attributes={
                "http.method": "GET",
                "http.url": "https://example.com",
            },
        )
        strands_span = MockReadableSpan(
            name="chat",
            attributes={
                "gen_ai.request.model": "gpt-4",
                "gen_ai.system": "strands-agents",
            },
        )

        processor.on_end(http_span)  # type: ignore[arg-type]
        processor.on_end(strands_span)  # type: ignore[arg-type]

        assert http_span._attributes.get("http.method") == "GET"
        assert SpanAttributes.LLM_MODEL_NAME in strands_span._attributes

    def test_processor_ignores_non_strands_genai_span(self) -> None:
        """Test that spans from other GenAI SDKs remain unchanged."""
        processor = StrandsAgentsToOpenInferenceProcessor()

        event = MockEvent("gen_ai.request")

        span = MockReadableSpan(
            name="chat",
            attributes={
                "gen_ai.request.model": "gpt-4",
                "gen_ai.system": "openai",
            },
            events=[event],
        )

        original_attrs = dict(span._attributes)
        original_events = list(span._events)
        original_status = span._status

        processor.on_end(span)  # type: ignore[arg-type]

        assert span._attributes == original_attrs
        assert span._events == original_events
        assert span._status is original_status

    def test_processor_ignores_event_loop_span_without_cycle_id(self) -> None:
        """Test that event loop spans without a cycle ID are ignored."""
        processor = StrandsAgentsToOpenInferenceProcessor()

        span = MockReadableSpan(
            name="execute_event_loop_cycle",
            attributes={},
        )

        processor.on_end(span)  # type: ignore[arg-type]

        assert SpanAttributes.OPENINFERENCE_SPAN_KIND not in span._attributes

    def test_processor_does_not_overwrite_error_status(self) -> None:
        """Test that processor does not overwrite ERROR spans."""
        processor = StrandsAgentsToOpenInferenceProcessor()
        span = MockReadableSpan(
            name="chat",
            attributes={
                "gen_ai.request.model": "gpt-4",
                "gen_ai.system": "strands-agents",
            },
        )
        span._status = Status(StatusCode.ERROR)

        processor.on_end(span)  # type: ignore[arg-type]

        assert span._status.status_code == StatusCode.ERROR

    def test_processor_sets_ok_status_when_not_error(self) -> None:
        """Test that spans without errors are normalized to OK."""
        processor = StrandsAgentsToOpenInferenceProcessor()
        span = MockReadableSpan(
            name="chat",
            attributes={
                "gen_ai.request.model": "gpt-4",
                "gen_ai.system": "strands-agents",
            },
        )
        span._status = Status(StatusCode.UNSET)

        processor.on_end(span)  # type: ignore[arg-type]

        assert span._status.status_code == StatusCode.OK

    def test_processor_handles_empty_attributes(self) -> None:
        """Test that the processor handles spans with no attributes."""
        processor = StrandsAgentsToOpenInferenceProcessor()
        span = MockReadableSpan(name="test_span", attributes={})

        # Should not raise an exception
        processor.on_end(span)  # type: ignore[arg-type]

    def test_processor_debug_mode(self) -> None:
        """Test that debug mode works."""
        processor = StrandsAgentsToOpenInferenceProcessor(debug=True)
        span = MockReadableSpan(
            name="chat",
            attributes={
                "gen_ai.request.model": "gpt-4",
                "gen_ai.system": "strands-agents",
            },
        )

        # Should not raise an exception
        processor.on_end(span)  # type: ignore[arg-type]
