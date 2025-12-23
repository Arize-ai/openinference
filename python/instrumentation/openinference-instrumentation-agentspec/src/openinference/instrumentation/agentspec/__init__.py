import logging
import warnings
from contextlib import contextmanager
from typing import Any, Collection, Generator

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from opentelemetry.sdk.trace.export import SpanProcessor as OtelSdkSpanProcessor  # type: ignore
from pyagentspec.tracing.spanprocessor import SpanProcessor as AgentSpecSpanProcessor

from openinference.instrumentation.agentspec._openinference_spanprocessor import (
    OpenInferenceSpanProcessor,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class AgentSpecInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumentor for Agent Spec Tracing
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return ["pyagentspec >= 26.1.0"]

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider", None)):
            tracer_provider = trace_api.get_tracer_provider()
        if not (resource := kwargs.get("resource", None)):
            if hasattr(tracer_provider, "resource"):
                resource = tracer_provider.resource
        mask_sensitive_information = kwargs.get("mask_sensitive_information", False)

        from pyagentspec.tracing.trace import Trace, get_trace

        if get_trace() is not None:
            raise ValueError(
                "Agent Spec Trace already active, instrumentation is not allowed. "
                "Close any existing Agent Spec Trace before instrumenting your code."
            )

        # Look for the span processors attached to the tracer provider
        # and wrap them in an OTel Agent Spec Span Processor
        tp_span_processors: list[OtelSdkSpanProcessor] = []
        if hasattr(tracer_provider, "_active_span_processor"):
            tp_span_processors = getattr(
                tracer_provider._active_span_processor, "_span_processors", []
            )
        agentspec_spanprocessors: list[AgentSpecSpanProcessor] = [
            OpenInferenceSpanProcessor(
                otel_span_processor=otel_span_processor,
                resource=resource,
                mask_sensitive_information=mask_sensitive_information,
            )
            for otel_span_processor in tp_span_processors
        ]
        if len(agentspec_spanprocessors) == 0:
            warnings.warn(
                "Instrumenting a TracerProvider that has no SpanProcessors attached", UserWarning
            )

        trace = Trace(span_processors=agentspec_spanprocessors, shutdown_on_exit=False)
        trace._start()

    def _uninstrument(self, **kwargs: Any) -> None:
        from pyagentspec.tracing.trace import get_trace

        trace = get_trace()
        if trace is not None:
            try:
                trace._end()
            except Exception as e:
                # Whatever happens we do not crash during the final shutdown, but we warn the user
                logger.warning(f"Exception raised during Trace `end`: {e}")

    @contextmanager
    def instrument_context(self, **kwargs: Any) -> Generator[None, Any, None]:
        """Context manager to instrument and uninstrument the library"""
        self.instrument(**kwargs)
        yield
        if self.is_instrumented_by_opentelemetry:
            self.uninstrument(**kwargs)


__all__ = ["AgentSpecInstrumentor", "OpenInferenceSpanProcessor"]
