# OpenInference Strands Instrumentation

Python auto-instrumentation library for the [Strands Agents](https://strandsagents.com/) framework.

This package provides automatic tracing for Strands Agents applications, capturing detailed telemetry about agent invocations, event loops, tool executions, and multi-agent interactions.

## Installation

```bash
pip install openinference-instrumentation-strands
```

## Quickstart

Install packages needed for this quickstart.

```bash
pip install openinference-instrumentation-strands strands opentelemetry-sdk opentelemetry-exporter-otlp openai
```

Set up `StrandsInstrumentor` to trace your Strands Agents application and sends them to an OpenTelemetry collector or endpoint.

```python
import os
from openinference.instrumentation.strands import StrandsInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

# Configure OpenTelemetry to export to a collector
endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
trace_api.set_tracer_provider(tracer_provider)

# Instrument Strands
StrandsInstrumentor().instrument()

# Use Strands as normal
from strands import Agent, tool
from strands.models.openai import OpenAIModel

@tool
def get_weather(city: str) -> dict:
    """Get the weather for a city.
    
    Args:
        city: The city to get weather for
    """
    return {
        "status": "success",
        "content": [{"text": f"The weather in {city} is sunny!"}]
    }

# Create agent with OpenAI model (requires OPENAI_API_KEY environment variable)
model = OpenAIModel(model_id="gpt-4o-mini")
agent = Agent(model=model, tools=[get_weather])
result = agent("What's the weather in San Francisco?")
print(result.message)
```

## What Gets Traced

The Strands instrumentation automatically captures:

### Agent Invocations
- Agent name and ID
- Input prompts (text, content blocks, or messages)
- Output messages and stop reasons
- Token usage and metrics
- Available tools

### Event Loop Cycles
- Cycle IDs and execution flow
- Stop reasons (end_turn, tool_use, max_tokens, etc.)
- Messages exchanged during cycles

### Tool Executions
- Tool names and IDs
- Input parameters
- Output results
- Execution status (success/error)

### Multi-Agent Interactions
- Agent-to-agent communication
- Swarm orchestration
- Graph-based workflows

## Span Kinds

The instrumentation uses the following OpenInference span kinds:

- **AGENT**: Agent invocations (`agent.invoke()`, `agent.stream()`)
- **CHAIN**: Event loop cycles and multi-agent orchestration
- **TOOL**: Individual tool executions
- **LLM**: Model calls (when model instrumentation is also enabled)

## Configuration

### TraceConfig

You can configure the instrumentation behavior using `TraceConfig`:

```python
from openinference.instrumentation import TraceConfig

config = TraceConfig(
    # Configuration options
)

StrandsInstrumentor().instrument(config=config)
```

### Custom Tracer Provider

You can provide a custom tracer provider:

```python
from opentelemetry.sdk.trace import TracerProvider

tracer_provider = TracerProvider()
# ... configure tracer provider ...

StrandsInstrumentor().instrument(tracer_provider=tracer_provider)
```

## Compatibility

This instrumentation is compatible with:
- Strands >= 0.1.0
- Python >= 3.10

## Examples

See the [examples](./examples) directory for complete examples of using the Strands instrumentation.

## More Info

- [Strands Agents Documentation](https://strandsagents.com/)
- [OpenInference](https://github.com/Arize-ai/openinference)
- [Phoenix - Open-source LLM Observability](https://docs.arize.com/phoenix)

