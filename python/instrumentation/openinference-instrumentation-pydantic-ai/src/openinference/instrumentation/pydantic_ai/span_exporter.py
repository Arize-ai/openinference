from typing import Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from openinference.instrumentation.pydantic_ai.semantic_conventions import (
    get_attributes,
)
from openinference.instrumentation.pydantic_ai.utils import SpanFilter, should_export_span


class OpenInferenceSpanExporter(SpanExporter):
    def __init__(
        self, base_exporter: SpanExporter, span_filter: Optional[SpanFilter] = None
    ) -> None:
        self._base_exporter = base_exporter
        self._span_filter = span_filter

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        openinference_spans = []

        for span in spans:
            openinference_attributes = {}

            if span.attributes is not None:
                openinference_attributes_iter = get_attributes(span.attributes)
                openinference_attributes = dict(openinference_attributes_iter)

            openinference_span = ReadableSpan(
                name=span.name,
                context=span.context,
                parent=span.parent,
                resource=span.resource,
                attributes=openinference_attributes,
                events=span.events,
                links=span.links,
                kind=span.kind,
                status=span.status,
                start_time=span.start_time,
                end_time=span.end_time,
                instrumentation_scope=span.instrumentation_scope,
            )

            if not should_export_span(openinference_span, self._span_filter):
                continue

            openinference_spans.append(openinference_span)

        # Export the openinference spans
        result = self._base_exporter.export(openinference_spans)
        return result

    def shutdown(self) -> None:
        self._base_exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._base_exporter.force_flush(timeout_millis)
