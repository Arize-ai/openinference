"""OpenInference instrumentation for Microsoft Agent Framework.

This package provides a SpanProcessor that transforms Microsoft Agent Framework's
native OpenTelemetry spans (using GenAI semantic conventions) to OpenInference format
for compatibility with OpenInference-compliant backends like Arize Phoenix.

Usage:
    ```python
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from agent_framework.observability import configure_otel_providers
    from openinference.instrumentation.agent_framework import (
        AgentFrameworkToOpenInferenceProcessor
    )

    # Configure MS Agent Framework's native telemetry
    # Set OTEL_EXPORTER_OTLP_TRACES_ENDPOINT env var for Phoenix endpoint
    configure_otel_providers(enable_sensitive_data=True)

    # Add OpenInference processor to transform spans
    tracer_provider = trace.get_tracer_provider()
    if isinstance(tracer_provider, TracerProvider):
        tracer_provider.add_span_processor(
            AgentFrameworkToOpenInferenceProcessor()
        )

    # Use framework normally - spans will be transformed automatically
    from agent_framework.openai import OpenAIChatClient

    client = OpenAIChatClient(model_id="gpt-4o-mini")
    agent = client.create_agent(name="Assistant")
    response = await agent.run("Hello!")
    ```
"""

__version__ = "0.1.0"

from openinference.instrumentation.agent_framework.processor import (
    AgentFrameworkToOpenInferenceProcessor,
)
from openinference.instrumentation.agent_framework.utils import is_openinference_span

__all__ = [
    "AgentFrameworkToOpenInferenceProcessor",
    "is_openinference_span",
    "__version__",
]
