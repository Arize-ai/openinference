# OpenInference Portkey AI Instrumentation

This package provides instrumentation for the [Portkey AI](https://portkey.ai) library using OpenInference.

## Installation

```bash
pip install openinference-instrumentation-portkey
```

## Usage

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.portkey import PortkeyInstrumentor

# Set up the tracer provider
trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    SimpleSpanProcessor(ConsoleSpanExporter())
)

# Instrument Portkey AI
PortkeyInstrumentor().instrument()

# Use Portkey AI as usual
# ...

# Uninstrument when done
PortkeyInstrumentor().uninstrument()
```

## Features

- Automatic instrumentation of Portkey AI API calls
- Capture of input messages, output messages, and model information
- Integration with OpenTelemetry for distributed tracing

## Requirements

- Python 3.9+
- Portkey AI 0.1.0+

## License

Apache License 2.0 