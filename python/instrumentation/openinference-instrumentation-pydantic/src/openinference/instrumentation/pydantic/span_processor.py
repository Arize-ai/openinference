import logging
from typing import Optional, Dict, Any

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter, BatchSpanProcessor, SimpleSpanProcessor

from openinference.instrumentation.pydantic.utils import (
    # add_openinference_attributes_to_span,
    should_export_span,
    SpanFilter,
)

logger = logging.getLogger(__name__)


class OpenInferenceSimpleSpanProcessor(SimpleSpanProcessor):
    """
    Extends SimpleSpanProcessor to support OpenInference attributes.

    This processor enhances spans with OpenInference attributes before exporting them.
    It can be configured to selectively export only specific spans based on a filter.

    Args:
        exporter: The exporter to pass spans to
        span_filter: Optional filter function to determine if a span should be exported

    Example:
        ```python
        from openinference.instrumentation.pydantic import OpenInferenceSimpleSpanProcessor
        from openinference.instrumentation.pydantic.utils import is_openinference_span
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter()
        processor = OpenInferenceSimpleSpanProcessor(
            exporter=exporter,
            span_filter=is_openinference_span
        )
        tracer_provider.add_span_processor(processor)
        ```
    """

    def __init__(self, exporter: SpanExporter, span_filter: Optional[SpanFilter] = None):
        super().__init__(exporter)
        self._span_filter = span_filter

    def on_end(self, span: ReadableSpan) -> None:
        """Process a span when it ends."""
        try:
            # Add OpenInference attributes to the span
            # add_openinference_attributes_to_span(span)

            # skip the span if it does not contain the final_result attribute
            if span.attributes is None or "final_result" in span.attributes:
                return

            # Determine if the span should be exported
            if should_export_span(span, self._span_filter):
                # Call the parent implementation to export the span
                super().on_end(span)
        except Exception as e:
            logger.warning(f"Error processing span in OpenInferenceSimpleSpanProcessor: {e}")


class OpenInferenceBatchSpanProcessor(BatchSpanProcessor):
    """
    Extends BatchSpanProcessor to support OpenInference attributes.

    This processor enhances spans with OpenInference attributes before exporting them.
    It can be configured to selectively export only specific spans based on a filter.

    Args:
        exporter: The exporter to pass spans to
        span_filter: Optional filter function to determine if a span should be exported
        max_export_batch_size: The maximum batch size for export (default: 512)
        max_queue_size: The maximum queue size (default: 2048)
        schedule_delay_millis: The scheduling interval in milliseconds (default: 5000)
        export_timeout_millis: The export timeout in milliseconds (default: 30000)

    Example:
        ```python
        from openinference.instrumentation.pydantic import OpenInferenceBatchSpanProcessor
        from openinference.instrumentation.pydantic.utils import is_openinference_span
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter()
        processor = OpenInferenceBatchSpanProcessor(
            exporter=exporter,
            span_filter=is_openinference_span,
            max_queue_size=4096,
            schedule_delay_millis=10000
        )
        tracer_provider.add_span_processor(processor)
        ```
    """

    def __init__(
        self,
        exporter: SpanExporter,
        span_filter: Optional[SpanFilter] = None,
        max_export_batch_size: int = 512,
        max_queue_size: int = 2048,
        schedule_delay_millis: int = 5000,
        export_timeout_millis: int = 30000,
        # max_export_attempts: int = 3
    ):
        super().__init__(
            exporter,
            max_export_batch_size=max_export_batch_size,
            max_queue_size=max_queue_size,
            schedule_delay_millis=schedule_delay_millis,
            export_timeout_millis=export_timeout_millis,
            # max_export_attempts=max_export_attempts
        )
        self._span_filter = span_filter

    def on_end(self, span: ReadableSpan) -> None:
        """Process a span when it ends."""
        try:
            # Add OpenInference attributes to the span
            # add_openinference_attributes_to_span(span)

            # Determine if the span should be exported
            if should_export_span(span, self._span_filter):
                # Call the parent implementation to export the span
                super().on_end(span)
        except Exception as e:
            logger.warning(f"Error processing span in OpenInferenceBatchSpanProcessor: {e}")
