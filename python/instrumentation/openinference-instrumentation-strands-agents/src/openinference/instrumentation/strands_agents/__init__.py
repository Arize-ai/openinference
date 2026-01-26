"""OpenInference Instrumentation for Strands Agents.

This package provides a span processor that transforms Strands' native OpenTelemetry
spans (using GenAI semantic conventions) into OpenInference format for compatibility
with OpenInference-supported OTEL backends (Phoenix, Arize, etc.).

Usage:
    from strands.telemetry import StrandsTelemetry
    from openinference.instrumentation.strands_agents import StrandsAgentsToOpenInferenceProcessor

    # Setup Strands native telemetry
    telemetry = StrandsTelemetry()
    telemetry.setup_otlp_exporter(endpoint="http://localhost:6006/v1/traces")

    # Add OpenInference processor to transform spans
    telemetry.tracer_provider.add_span_processor(
        StrandsAgentsToOpenInferenceProcessor(debug=False)
    )

    # Now use your Strands agents normally - spans will be automatically transformed
    from strands import Agent
    from strands.models.openai import OpenAIModel

    model = OpenAIModel(model_id="gpt-4o-mini")
    agent = Agent(name="MyAgent", model=model)
    result = agent("Hello!")

The processor handles:
- Span kind mapping (invoke_agent → AGENT, execute_event_loop_cycle → CHAIN, etc.)
- Attribute transformation (gen_ai.* → llm.*, tool.*, agent.*)
- Event extraction (gen_ai.user.message → llm.input_messages)
- Token usage mapping (gen_ai.usage.* → llm.token_count.*)
- Tool parameter extraction
- Message structure flattening

For more information, see: https://github.com/Arize-ai/openinference
"""

from openinference.instrumentation.strands_agents.processor import (
    StrandsAgentsToOpenInferenceProcessor,
)

__all__ = ["StrandsAgentsToOpenInferenceProcessor"]

__version__ = "0.1.0"
