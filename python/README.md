# OpenInference Python

This is the Python version of OpenInference instrumentation, a framework for collecting traces from LLM applications.


# Getting Started
Instrumentation is the act of adding observability code to an app yourself.
If you’re instrumenting an app, you need to use the OpenTelemetry SDK for your language. You’ll then use the SDK to initialize OpenTelemetry and the API to instrument your code. This will emit telemetry from your app, and any library you installed that also comes with instrumentation.

# OpenAI Example
Install openinference instrumentation for OpenAI:

```shell
pip install openinference-instrumentation-openai
```

This assumes that you already have OpenAI installed. If not, you can install it using:

```shell
pip install "openai>=1.0.0"
```
Currently only openai>=1.0.0 is supported.

To export traces from the instrumentor to a collector, we use the OpenTelemetry SDK and the HTTP exporter. Install these using:

```shell
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
```

Below shows a simple application calling chat completions from OpenAI.

Note that the endpoint is set to collector running on `localhost:6006`, but can be changed if you are running a collector on a different location.

```python
import openai
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

# Set up the OpenTelemetry SDK tracer provider with an HTTP exporter.
# Change the endpoint if the collector is running at a different location.
endpoint = "http://localhost:6006/v1/traces"
resource = Resource(attributes={})
tracer_provider = trace_sdk.TracerProvider(resource=resource)
span_exporter = OTLPSpanExporter(endpoint=endpoint)
span_processor = SimpleSpanProcessor(span_exporter=span_exporter)
tracer_provider.add_span_processor(span_processor=span_processor)
trace_api.set_tracer_provider(tracer_provider=tracer_provider)

# Call the instrumentor to instrument OpenAI
OpenAIInstrumentor().instrument()

# Run the OpenAI application.
# Make you have your API key set in the environment variable OPENAI_API_KEY.
if __name__ == "__main__":
    response = openai.OpenAI().chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Write a haiku."}],
        max_tokens=20,
    )
    print(response.choices[0].message.content)
```

## Phoenix Collector

If you don't have collector running, you can use the Phoenix collector. Install it using:

```shell
pip install arize-phoenix
```

Then before running the example abov, start the collector using:

```python
import phoenix as px

px.launch_app()
```

By default, the Phoenix collector is running on `http://localhost:6006/v1/traces`.
