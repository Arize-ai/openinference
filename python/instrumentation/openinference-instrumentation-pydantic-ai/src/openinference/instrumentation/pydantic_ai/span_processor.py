import logging
from typing import Optional

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.trace import Span

from openinference.instrumentation.pydantic_ai.semantic_conventions import get_attributes
from openinference.instrumentation.pydantic_ai.utils import SpanFilter, should_export_span

logger = logging.getLogger(__name__)


class OpenInferenceSpanProcessor(SpanProcessor):
    """
    A standalone span processor that enhances spans with OpenInference attributes.

    This processor processes spans by enhancing them with OpenInference attributes
    and can optionally filter which spans are processed based on a filter function.
    Unlike the Simple and Batch processors, this processor only modifies span attributes
    in-place without exporting them - it's designed to work alongside other processors.

    Args:
        span_filter: Optional filter function to determine if a span should be processed

    Example:
        ```python
        from openinference.instrumentation.pydantic_ai import OpenInferenceSpanProcessor
        from openinference.instrumentation.pydantic_ai.utils import is_openinference_span
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        # Add the OpenInference processor to enhance spans
        openinference_processor = OpenInferenceSpanProcessor(
            span_filter=is_openinference_span
        )
        tracer_provider.add_span_processor(openinference_processor)

        # Add a separate exporter processor
        exporter = OTLPSpanExporter()
        export_processor = SimpleSpanProcessor(exporter)
        tracer_provider.add_span_processor(export_processor)
        ```
    """

    def __init__(self, span_filter: Optional[SpanFilter] = None):
        self._span_filter = span_filter

    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        """Called when a span starts. No processing needed at start."""
        pass

    def on_end(self, span: ReadableSpan) -> None:
        """Process a span when it ends by enhancing it with OpenInference attributes."""
        try:
            if not span.attributes:
                return

            # Get the openinference attributes from the span
            openinference_attributes_iter = get_attributes(span.attributes)
            openinference_attributes = dict(openinference_attributes_iter)

            # Combine the attributes with the openinference attributes
            span._attributes = {**span.attributes, **openinference_attributes}

            # Determine if the span should be exported
            if should_export_span(span, self._span_filter):
                super().on_end(span)

        except Exception as e:
            logger.warning(f"Error processing span in OpenInferenceSpanProcessor: {e}")

    def shutdown(self) -> None:
        """Shutdown the processor. No cleanup needed."""
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush any pending operations. Always returns True as there's nothing to flush."""
        return True
