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
        enhanced_spans = []

        for span in spans:
            # Create a new span with the added attributes
            merged_attributes = {}

            if span.attributes is not None:
                attributes = {}

                # Add in the original attributes
                for key, value in span.attributes.items():
                    attributes[key] = value

                # Add in the OpenInference attributes
                openinference_attributes_iter = self._extractor.get_attributes(attributes)
                openinference_attributes = dict(openinference_attributes_iter)
                merged_attributes.update(openinference_attributes)

            # Create a new ReadableSpan with enhanced attributes
            enhanced_span = ReadableSpan(
                name=span.name,
                context=span.context,
                parent=span.parent,
                resource=span.resource,
                attributes=merged_attributes,
                events=span.events,
                links=span.links,
                kind=span.kind,
                status=span.status,
                start_time=span.start_time,
                end_time=span.end_time,
                instrumentation_scope=span.instrumentation_scope,
            )

            enhanced_spans.append(enhanced_span)

        # Export the enhanced spans
        return self._base_exporter.export(enhanced_spans)

    def shutdown(self):
        return self._base_exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._base_exporter.force_flush(timeout_millis)
