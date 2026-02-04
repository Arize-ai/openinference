"""Span state management for streaming operations."""

from typing import Dict, Optional

from opentelemetry import trace as trace_api
from opentelemetry.trace import Span

from openinference.instrumentation import get_attributes_from_context
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes


OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
SESSION_ID = SpanAttributes.SESSION_ID
LLM_SYSTEM = SpanAttributes.LLM_SYSTEM


class SpanManager:
    """Manages span lifecycle during streaming operations."""

    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer
        self._active_spans: Dict[str, Span] = {}

    def start_agent_span(
        self,
        name: str,
        session_id: str,
        parent_span: Optional[Span] = None,
    ) -> Span:
        """Start an AGENT span."""
        context = None
        if parent_span is not None:
            context = trace_api.set_span_in_context(parent_span)

        attributes = {
            OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
            LLM_SYSTEM: "claude_code",
            SESSION_ID: session_id,
        }
        attributes.update(dict(get_attributes_from_context()))

        span = self._tracer.start_span(
            name=name,
            context=context,
            attributes=attributes,
        )

        return span

    def start_llm_span(
        self,
        name: str,
        parent_span: Optional[Span] = None,
    ) -> Span:
        """Start an LLM span."""
        context = None
        if parent_span is not None:
            context = trace_api.set_span_in_context(parent_span)

        attributes = {
            OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
            LLM_SYSTEM: "claude_code",
        }
        attributes.update(dict(get_attributes_from_context()))

        span = self._tracer.start_span(
            name=name,
            context=context,
            attributes=attributes,
        )

        return span

    def start_tool_span(
        self,
        tool_name: str,
        parent_span: Optional[Span] = None,
    ) -> Span:
        """Start a TOOL span."""
        context = None
        if parent_span is not None:
            context = trace_api.set_span_in_context(parent_span)

        attributes = {
            OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL.value,
            SpanAttributes.TOOL_NAME: tool_name,
        }
        attributes.update(dict(get_attributes_from_context()))

        span = self._tracer.start_span(
            name=f"Tool: {tool_name}",
            context=context,
            attributes=attributes,
        )

        return span

    def end_span(self, span: Span) -> None:
        """End a span."""
        if span is not None:
            span.end()

    def end_all_spans(self) -> None:
        """End all tracked spans (cleanup on error)."""
        for span in self._active_spans.values():
            self.end_span(span)
        self._active_spans.clear()
