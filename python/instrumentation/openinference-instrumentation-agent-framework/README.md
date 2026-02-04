# OpenInference Microsoft Agent Framework Instrumentation

OpenInference span processor for Microsoft Agent Framework that transforms native OpenTelemetry spans to OpenInference format for compatibility with OpenInference-compliant backends like [Arize Phoenix](https://github.com/Arize-ai/phoenix).

## Installation

```bash
pip install openinference-instrumentation-agent-framework
```

## Overview

Microsoft Agent Framework emits telemetry using GenAI semantic conventions (`gen_ai.*` attributes). This package provides a `SpanProcessor` that transforms these spans to OpenInference format, enabling compatibility with observability tools that support the OpenInference standard.

## Usage

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
agent = client.create_agent(name="Assistant", instructions="You are helpful.")
response = await agent.run("Hello!")
```

## Transformation Details

### Span Kind Mapping

| MS Agent Framework Operation | OpenInference Span Kind |
|------------------------------|-------------------------|
| `chat` | LLM |
| `execute_tool` | TOOL |
| `invoke_agent` | AGENT |
| `create_agent` | AGENT |
| `workflow.run` | CHAIN |
| `executor.process` | CHAIN |

### Attribute Mapping

| Source (GenAI) | Target (OpenInference) |
|----------------|------------------------|
| `gen_ai.request.model` | `llm.model_name` |
| `gen_ai.provider.name` | `llm.provider` |
| `gen_ai.usage.input_tokens` | `llm.token_count.prompt` |
| `gen_ai.usage.output_tokens` | `llm.token_count.completion` |
| `gen_ai.input.messages` | `llm.input_messages.*` (flattened) |
| `gen_ai.output.messages` | `llm.output_messages.*` (flattened) |
| `gen_ai.tool.name` | `tool.name` |
| `gen_ai.tool.call.id` | `tool.call_id` |
| `gen_ai.tool.call.arguments` | `tool.parameters` |
| `gen_ai.conversation.id` | `session.id` |

### Message Format Transformation

MS Agent Framework messages:
```json
{
  "role": "user",
  "parts": [
    {"type": "text", "content": "Hello"}
  ]
}
```

Transformed to OpenInference flattened format:
```
llm.input_messages.0.message.role = "user"
llm.input_messages.0.message.content = "Hello"
```

## Debug Mode

Enable debug mode to log transformation details:

```python
processor = AgentFrameworkToOpenInferenceProcessor(debug=True)
```

## Requirements

- Python >= 3.10
- opentelemetry-api >= 1.39.0
- opentelemetry-sdk >= 1.39.0
- openinference-semantic-conventions >= 0.1.17

## License

Apache-2.0
