"""Tests for Strands to OpenInference processor."""

from typing import Any, Dict, List, Optional

from opentelemetry.trace import SpanKind

from openinference.instrumentation.strands_agents.processor import (
    StrandsAgentsToOpenInferenceProcessor,
)
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes


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
        self.kind = SpanKind.INTERNAL
        self.parent = None

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
            },
        )

        processor.on_end(span)  # type: ignore[arg-type]

        assert (
            span._attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
            == OpenInferenceSpanKindValues.CHAIN.value
        )

    def test_processor_handles_empty_attributes(self) -> None:
        """Test that the processor handles spans with no attributes."""
        processor = StrandsAgentsToOpenInferenceProcessor()
        span = MockReadableSpan(name="test_span", attributes={})

        # Should not raise an exception
        processor.on_end(span)  # type: ignore[arg-type]

    def test_processor_leaves_non_strands_spans_unchanged(self) -> None:
        """Non-Strands spans must not have their attributes modified."""
        processor = StrandsAgentsToOpenInferenceProcessor()
        original_attrs = {
            "http.method": "GET",
            "http.url": "https://example.com/api",
            "http.status_code": 200,
        }
        span = MockReadableSpan(name="http.request", attributes=dict(original_attrs))

        processor.on_end(span)  # type: ignore[arg-type]

        assert span._attributes == original_attrs

    def test_processor_leaves_non_strands_span_with_openai_attrs_unchanged(self) -> None:
        """A span from another GenAI SDK (e.g. OpenAI) must not be touched."""
        processor = StrandsAgentsToOpenInferenceProcessor()
        original_attrs = {
            "gen_ai.system": "openai",
            "gen_ai.request.model": "gpt-4o",
            "gen_ai.usage.input_tokens": 42,
        }
        span = MockReadableSpan(name="openai.chat", attributes=dict(original_attrs))

        processor.on_end(span)  # type: ignore[arg-type]

        assert span._attributes == original_attrs

    def test_processor_debug_mode(self) -> None:
        """Test that debug mode works."""
        processor = StrandsAgentsToOpenInferenceProcessor(debug=True)
        span = MockReadableSpan(
            name="chat",
            attributes={
                "gen_ai.request.model": "gpt-4",
            },
        )

        # Should not raise an exception
        processor.on_end(span)  # type: ignore[arg-type]
