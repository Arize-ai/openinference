from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace import ReadableSpan
from typing import Sequence
from openinference.instrumentation.pydantic.semantic_conventions import (
    OpenInferenceAttributesExtractor,
)
import json


class OpenInferenceSpanExporter(SpanExporter):
    def __init__(self, base_exporter):
        self._base_exporter = base_exporter
        self._extractor = OpenInferenceAttributesExtractor()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        openinference_spans = []

        for span in spans:

            openinference_attributes = {}

            if span.attributes is not None:

                openinference_attributes_iter = self._extractor.get_attributes(span.attributes)
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

            openinference_spans.append(openinference_span)

        # Export the openinference spans
        return self._base_exporter.export(openinference_spans)

    def shutdown(self):
        return self._base_exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._base_exporter.force_flush(timeout_millis)
