from typing import Optional
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor

from openinference.instrumentation.openllmetry import OpenLLToOIProcessor

class OpenLLMetryInstrumentor(BaseInstrumentor):
    """Instrumentor for OpenLLMetry to OpenInference conversion."""

    def instrumentation_dependencies(self) -> list[str]:
        return []

    def _instrument(self, **kwargs):
        """Add the OpenLLToOIProcessor to the tracer provider."""
        tracer_provider: Optional[TracerProvider] = kwargs.get("tracer_provider")
        if tracer_provider is not None:
            tracer_provider.add_span_processor(OpenLLToOIProcessor())

    def _uninstrument(self, **kwargs):
        """No cleanup needed."""
        pass 