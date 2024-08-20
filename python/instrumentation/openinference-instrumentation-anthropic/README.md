# OpenInference Anthropic Instrumentation

Python autoinstrumentation library for the [Anthropic](https://www.anthropic.com/api) package

This package implements the following Anthropic clients:
- `Messages`
- `Completions`
- `AsyncMessages`
- `AsyncCompletions`

These traces are fully OpenTelemetry compatible and can be sent to an OpenTelemetry collector for viewing, such as [Arize `phoenix`](https://github.com/Arize-ai/phoenix).


## Installation

```shell
pip install openinference-instrumentation-anthropic
```

## Quickstart

Through your *terminal*, install required packages.

```shell
pip install openinference-instrumentation-anthropic anthropic arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp
```

You can start Phoenix with the following terminal command:
```shell
python -m phoenix.server.main serve
````
By default, Phoenix listens on `http://localhost:6006`. You can visit the app via a browser at the same address. (Phoenix does not send data over the internet. It only operates locally on your machine.)


Try the following code in a *Python file*.

1. Set up `AnthropicInstrumentor` to trace your application and sends the traces to Phoenix. 
2. Then, set your Anthropic API key as an environment variable. 
3. Lastly, create a Anthropic client, make a request, then go see your results in Phoenix at `http://localhost:6006`!

```python
import os
from anthropic import Anthropic
from openinference.instrumentation.anthropic import AnthropicInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

# Configure AnthropicInstrumentor with Phoenix endpoint
endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)

os.environ["ANTHROPIC_API_KEY"] = "YOUR_KEY_HERE"

client = Anthropic()

response = client.messages.create(
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Tell me about the history of Iceland!",
        }
    ],
    model="claude-3-opus-20240229",
)
print(response)
```

Now, on the Phoenix UI on your browser, you should see the traces from your Anthropic application. Click on a trace, then the "Attributes" tab will provide you with in-depth information regarding execution!
