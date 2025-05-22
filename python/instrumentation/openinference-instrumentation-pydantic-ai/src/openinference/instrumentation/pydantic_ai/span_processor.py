import logging
from typing import Optional

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter

from openinference.instrumentation.pydantic_ai.semantic_conventions import get_attributes
from openinference.instrumentation.pydantic_ai.utils import SpanFilter, should_export_span

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
            if not span.attributes:
                super().on_end(span)
                return

            # Get the openinference attributes from the span
            openinference_attributes_iter = get_attributes(span.attributes)
            openinference_attributes = dict(openinference_attributes_iter)

            # combine the attributes with the openinference attributes
            span._attributes = {**span.attributes, **openinference_attributes}

            # Determine if the span should be exported
            if should_export_span(span, self._span_filter):
                super().on_end(span)

        except Exception as e:
            logger.warning(f"Error processing span in OpenInferenceSimpleSpanProcessor: {e}")


class OpenInferenceBatchSpanProcessor(BatchSpanProcessor):
    """
    Extends BatchSpanProcessor to support OpenInference attributes.

    This processor enhances spans with OpenInference attributes before batching and exporting them.
    It provides better performance for high-throughput applications by batching spans before export.
    It can be configured to selectively export only specific spans based on a filter.

    Args:
        exporter: The exporter to pass spans to
        max_queue_size: Maximum size of the queue for pending spans
        schedule_delay_millis: Time interval between consecutive exports
        max_export_batch_size: Maximum number of spans to export in a single batch
        export_timeout_millis: Maximum time to wait for export completion
        span_filter: Optional filter function to determine if a span should be exported

    Example:
        ```python
        from openinference.instrumentation.pydantic import OpenInferenceBatchSpanProcessor
        from openinference.instrumentation.pydantic.utils import is_openinference_span
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter()
        processor = OpenInferenceBatchSpanProcessor(
            exporter=exporter,
            max_queue_size=2048,
            schedule_delay_millis=5000,
            max_export_batch_size=512,
            span_filter=is_openinference_span
        )
        tracer_provider.add_span_processor(processor)
        ```
    """

    def __init__(
        self,
        exporter: SpanExporter,
        max_queue_size: int = 2048,
        schedule_delay_millis: int = 5000,
        max_export_batch_size: int = 512,
        export_timeout_millis: int = 30000,
        span_filter: Optional[SpanFilter] = None,
    ):
        super().__init__(
            span_exporter=exporter,
            max_queue_size=max_queue_size,
            schedule_delay_millis=schedule_delay_millis,
            max_export_batch_size=max_export_batch_size,
            export_timeout_millis=export_timeout_millis,
        )
        self._span_filter = span_filter

    def on_end(self, span: ReadableSpan) -> None:
        """Process a span when it ends."""
        try:
            if not span.attributes:
                super().on_end(span)
                return

            # Get the openinference attributes from the span
            openinference_attributes_iter = get_attributes(span.attributes)
            openinference_attributes = dict(openinference_attributes_iter)

            # Replace the attributes with the openinference attributes
            span._attributes = {**span.attributes, **openinference_attributes}

            # Determine if the span should be exported
            if should_export_span(span, self._span_filter):
                super().on_end(span)

        except Exception as e:
            logger.warning(f"Error processing span in OpenInferenceBatchSpanProcessor: {e}")
