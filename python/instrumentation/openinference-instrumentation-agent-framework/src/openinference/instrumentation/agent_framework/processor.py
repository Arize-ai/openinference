"""Microsoft Agent Framework to OpenInference Span Processor.

This module provides a span processor that converts Microsoft Agent Framework's native
OpenTelemetry spans (using GenAI semantic conventions) to OpenInference format for
compatibility with OpenInference-compliant backends like Arize Phoenix.

The processor transforms:
- GenAI attributes (gen_ai.*) to OpenInference attributes (llm.*, tool.*, etc.)
- Span names to OpenInference span kinds (AGENT, CHAIN, TOOL, LLM)
- Message structures to OpenInference flattened format
- Token usage attributes to OpenInference format
"""

import logging
from typing import Any, Dict, Optional

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.trace import Status, StatusCode

from openinference.instrumentation.agent_framework import __version__
from openinference.instrumentation.agent_framework.semantic_conventions import get_attributes
from openinference.instrumentation.agent_framework.utils import SpanFilter, should_export_span

logger = logging.getLogger(__name__)


class AgentFrameworkToOpenInferenceProcessor(SpanProcessor):
    """
    SpanProcessor that converts Microsoft Agent Framework telemetry attributes
    to OpenInference format for compatibility with OpenInference-compliant backends.

    This processor intercepts spans on completion and transforms their attributes
    from the GenAI semantic conventions used by Microsoft Agent Framework to the
    OpenInference semantic conventions.

    Usage:
        ```python
        from opentelemetry.sdk.trace import TracerProvider
        from openinference.instrumentation.agent_framework import (
            AgentFrameworkToOpenInferenceProcessor
        )

        provider = TracerProvider()
        provider.add_span_processor(
            AgentFrameworkToOpenInferenceProcessor(debug=False)
        )
        ```
    """

    def __init__(self, debug: bool = False, span_filter: Optional[SpanFilter] = None) -> None:
        """
        Initialize the processor.

        Args:
            debug: Whether to log debug information about transformations
            span_filter: Optional filter function to determine if a span should be processed
        """
        super().__init__()
        self.debug = debug
        self._span_filter = span_filter

    def on_start(self, span: ReadableSpan, parent_context: Any = None) -> None:
        """Called when a span is started. No-op for this processor."""
        pass

    def on_end(self, span: ReadableSpan) -> None:
        """
        Called when a span ends. Transform the span attributes from Microsoft Agent
        Framework GenAI format to OpenInference format.
        """
        if not hasattr(span, "_attributes") or not span._attributes:
            return

        try:
            # Get span context information
            span_id = span.get_span_context().span_id  # type: ignore[no-untyped-call]

            # Get OpenInference attributes from the transformation function
            openinference_attributes_iter = get_attributes(
                dict(span._attributes), span.name, span_id
            )
            openinference_attributes = dict(openinference_attributes_iter)

            # Merge with original attributes to preserve GenAI attributes that weren't transformed
            span._attributes = {**span.attributes, **openinference_attributes}  # type: ignore[dict-item]

            # MS Agent Framework only sets ERROR status, not OK - set OK for successful spans
            if not span.status.status_code == StatusCode.ERROR:
                span._status = Status(status_code=StatusCode.OK)

            # Determine if the span should be exported
            if should_export_span(span, self._span_filter):
                super().on_end(span)

            if self.debug:
                logger.info(
                    "span_name=<%s>, trans_attrs=<%d> | transformed span",
                    span.name,
                    len(openinference_attributes),
                )

        except Exception as e:
            span._status = Status(status_code=StatusCode.ERROR, description=str(e))
            logger.exception(e)
            logger.warning(f"Error processing span in AgentFrameworkToOpenInferenceProcessor: {e}")

    def shutdown(self) -> None:
        """Shutdown the processor."""
        pass

    def force_flush(self, timeout_millis: Optional[int] = None) -> bool:
        """Force flush any pending data."""
        return True

    def get_processor_info(self) -> Dict[str, Any]:
        """Get information about this processor's capabilities."""
        return {
            "processor_name": "AgentFrameworkToOpenInferenceProcessor",
            "version": __version__,
            "debug_enabled": self.debug,
            "supported_span_kinds": ["LLM", "AGENT", "CHAIN", "TOOL"],
            "supported_operations": [
                "chat",
                "execute_tool",
                "invoke_agent",
                "create_agent",
                "workflow.run",
                "executor.process",
            ],
            "features": [
                "Message extraction and transformation",
                "Token usage mapping",
                "Tool call processing",
                "Graph node hierarchy mapping",
                "Workflow/executor span support",
                "Invocation parameters mapping",
            ],
        }
