from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterator, Mapping

from agents.tracing import Span, Trace, TracingProcessor
from agents.tracing.span_data import (
    AgentSpanData,
    CustomSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GuardrailSpanData,
    HandoffSpanData,
    ResponseSpanData,
    SpanData,
)
from opentelemetry.context import attach, detach, set_value
from opentelemetry.trace import Span as OtelSpan
from opentelemetry.trace import (
    Status,
    StatusCode,
    Tracer,
    TracerProvider,
    set_span_in_context,
)
from opentelemetry.trace.propagation import _SPAN_KEY
from opentelemetry.util.types import AttributeValue

from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

logger = logging.getLogger(__name__)


class OpenInferenceTracingProcessor(TracingProcessor):  # type: ignore[misc]
    def __init__(
        self,
        tracer_provider: TracerProvider,
        **kwargs: Any,
    ) -> None:
        self._root_spans: dict[str, OtelSpan] = {}
        self._otel_spans: dict[str, OtelSpan] = {}
        self._tokens: dict[str, object] = {}
        self._tracer: Tracer = tracer_provider.get_tracer(__name__)

    def on_trace_start(self, trace: Trace) -> None:
        root_span = self._tracer.start_span(
            name=trace.name,
            attributes={
                OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
            },
        )
        self._root_spans[trace.trace_id] = root_span

    def on_trace_end(self, trace: Trace) -> None:
        root_span = self._root_spans.pop(trace.trace_id, None)
        if root_span:
            root_span.set_status(Status(StatusCode.OK))
            root_span.end()

    def on_span_start(self, span: Span[Any]) -> None:
        if not span.started_at:
            return
        start_time = datetime.fromisoformat(span.started_at)
        parent_span = (
            self._otel_spans.get(span.parent_id)
            if span.parent_id
            else self._root_spans.get(span.trace_id)
        )
        context = set_span_in_context(parent_span) if parent_span else None
        span_name = _get_span_name(span)
        otel_span = self._tracer.start_span(
            name=span_name,
            context=context,
            start_time=_as_utc_nano(start_time),
            attributes={OPENINFERENCE_SPAN_KIND: _get_span_kind(span.span_data)},
        )
        self._otel_spans[span.span_id] = otel_span
        token = attach(set_value(_SPAN_KEY, span))
        self._tokens[span.span_id] = token

    def on_span_end(self, span: Span[Any]) -> None:
        if token := self._tokens.pop(span.span_id, None):
            detach(token)  # type: ignore[arg-type]
        if not (otel_span := self._otel_spans.pop(span.span_id, None)):
            return
        if not span.ended_at:
            return
        end_time = datetime.fromisoformat(span.ended_at)
        flatten_attributes: dict[str, AttributeValue] = dict(_flatten(span.export()))
        otel_span.set_attributes(flatten_attributes)
        otel_span.end(_as_utc_nano(end_time))

    def force_flush(self) -> None:
        """Forces an immediate flush of all queued spans/traces."""
        # This implementation is deferred
        pass

    def shutdown(self) -> None:
        """Called when the application stops."""
        # This implementation is deferred
        pass


def _as_utc_nano(dt: datetime) -> int:
    return int(dt.astimezone(timezone.utc).timestamp() * 1_000_000_000)


def _get_span_name(obj: Span) -> str:
    if hasattr(data := obj.span_data, "name") and isinstance(name := data.name, str):
        return name
    return obj.span_data.type  # type: ignore[no-any-return]


def _get_span_kind(obj: SpanData) -> str:
    if isinstance(obj, AgentSpanData):
        return OpenInferenceSpanKindValues.AGENT.value
    if isinstance(obj, FunctionSpanData):
        return OpenInferenceSpanKindValues.TOOL.value
    if isinstance(obj, GenerationSpanData):
        return OpenInferenceSpanKindValues.LLM.value
    if isinstance(obj, ResponseSpanData):
        return OpenInferenceSpanKindValues.LLM.value
    if isinstance(obj, HandoffSpanData):
        return OpenInferenceSpanKindValues.CHAIN.value
    if isinstance(obj, CustomSpanData):
        return OpenInferenceSpanKindValues.CHAIN.value
    if isinstance(obj, GuardrailSpanData):
        return OpenInferenceSpanKindValues.CHAIN.value
    return OpenInferenceSpanKindValues.CHAIN.value


def _flatten(
    obj: Mapping[str, Any],
    prefix: str = "",
) -> Iterator[tuple[str, AttributeValue]]:
    for key, value in obj.items():
        if isinstance(value, dict):
            yield from _flatten(value, f"{prefix}{key}.")
        elif isinstance(value, (str, int, float, bool, str)):
            yield key, value
        else:
            yield key, str(value)


METADATA = SpanAttributes.METADATA
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
