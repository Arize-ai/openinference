# OpenInference Microsoft Agent Framework Instrumentation

OpenInference span processor for Microsoft Agent Framework that transforms native OpenTelemetry spans to OpenInference format for compatibility with OpenInference-compliant backends like [Arize Phoenix](https://github.com/Arize-ai/phoenix).

**Tested with agent-framework `1.0.0b260130` (January 30, 2026)**

## Installation

```bash
pip install openinference-instrumentation-agent-framework
```

Install with agent-framework:
```bash
pip install openinference-instrumentation-agent-framework[instruments]
```

## Overview

Microsoft Agent Framework emits telemetry using GenAI semantic conventions (`gen_ai.*` attributes). This package provides a `SpanProcessor` that transforms these spans to OpenInference format, enabling compatibility with observability tools that support the OpenInference standard.

**Note:** Agent Framework is in beta and its API may change between versions. This instrumentation tracks the latest stable release.

## Usage

### Basic Setup

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from agent_framework.observability import enable_instrumentation
from openinference.instrumentation.agent_framework import (
    AgentFrameworkToOpenInferenceProcessor
)

# Create tracer provider
tracer_provider = TracerProvider()

# Add OpenInference processor to transform spans
tracer_provider.add_span_processor(
    AgentFrameworkToOpenInferenceProcessor()
)

# Add exporter (Console for demo, use OTLP for Phoenix)
tracer_provider.add_span_processor(
    SimpleSpanProcessor(ConsoleSpanExporter())
)

# Set as global tracer provider
trace.set_tracer_provider(tracer_provider)

# Enable agent-framework instrumentation
enable_instrumentation(enable_sensitive_data=True)

# Use framework normally - spans will be transformed automatically
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient(model_id="gpt-4o-mini", api_key="your-key")
agent = client.as_agent(name="Assistant", instructions="You are helpful.")
response = await agent.run("Hello!")
```

### With Phoenix (Arize)

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from agent_framework.observability import enable_instrumentation
from openinference.instrumentation.agent_framework import (
    AgentFrameworkToOpenInferenceProcessor
)

# Configure Phoenix endpoint
endpoint = "http://localhost:6006/v1/traces"

# Setup tracer with OpenInference processor
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(AgentFrameworkToOpenInferenceProcessor())
tracer_provider.add_span_processor(
    SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
)
trace.set_tracer_provider(tracer_provider)

# Enable instrumentation
enable_instrumentation(enable_sensitive_data=True)

# Your agent code here
```

## Transformation Details

### Span Kind Mapping

| MS Agent Framework Operation | OpenInference Span Kind |
|------------------------------|-------------------------|
| `chat` | LLM |
| `execute_tool` | TOOL |
| `invoke_agent` | AGENT |
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

- Python >= 3.10, < 3.15
- opentelemetry-api >= 1.39.0
- opentelemetry-sdk >= 1.39.0
- openinference-semantic-conventions >= 0.1.25
- agent-framework >= 1.0.0b260130 (optional, install with `[instruments]` extra)

## Important Notes

### Agent Framework API Stability

Microsoft Agent Framework is in active beta development. API changes between versions are possible:
- This instrumentation is tested against `agent-framework==1.0.0b260130`
- The `-latest` test variant tracks breaking changes in new releases
- If you encounter API compatibility issues, pin to the tested version:
  ```bash
  pip install agent-framework==1.0.0b260130
  ```

### Sensitive Data

Set `enable_sensitive_data=True` when calling `enable_instrumentation()` to capture message content in traces. This is required for full observability but may include PII.

## Development

### Running Tests

Tests use VCR cassettes to replay recorded API interactions:

```bash
# Run with recorded cassettes (no API key needed)
pytest tests/test_processor.py -v

# Re-record cassettes (requires OPENAI_API_KEY)
export OPENAI_API_KEY=your_key
rm -rf tests/cassettes/
pytest tests/test_processor.py -v --record-mode=rewrite
```

## License

Apache-2.0
