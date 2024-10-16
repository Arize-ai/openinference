# OpenInference Groq Instrumentation

Python autoinstrumentation library for the [Groq](https://wow.groq.com/why-groq/) package

This package implements OpenInference tracing for both Groq and AsyncGroq clients.

These traces are fully OpenTelemetry compatible and can be sent to an OpenTelemetry collector for viewing, such as [Arize `phoenix`](https://github.com/Arize-ai/phoenix).


## Installation

```shell
pip install openinference-instrumentation-groq
```

## Quickstart

Through your *terminal*, install required packages.

```shell
pip install openinference-instrumentation-groq groq arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp
```

You can start Phoenix with the following terminal command:
```shell
python -m phoenix.server.main serve
````
By default, Phoenix listens on `http://localhost:6006`. You can visit the app via a browser at the same address. (Phoenix does not send data over the internet. It only operates locally on your machine.)


Try the following code in a *Python file*.

1. Set up `GroqInstrumentor` to trace your application and sends the traces to Phoenix. 
2. Then, set your Groq API key as an environment variable. 
3. Lastly, create a Groq client, make a request, then go see your results in Phoenix at `http://localhost:6006`!

```python
import os
from groq import Groq
from openinference.instrumentation.groq import GroqInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

# Configure GroqInstrumentor with Phoenix endpoint
endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

GroqInstrumentor().instrument(tracer_provider=tracer_provider)

os.environ["GROQ_API_KEY"] = "YOUR_KEY_HERE"

client = Groq()

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of low latency LLMs",
        }
    ],
    model="llama3-8b-8192",
)

if __name__ == "__main__":
    print(chat_completion.choices[0].message.content)
```

Now, on the Phoenix UI on your browser, you should see the traces from your Groq application. Click on a trace, then the "Attributes" tab will provide you with in-depth information regarding execution!

## More Info

* [More info on OpenInference and Phoenix](https://docs.arize.com/phoenix)
* [How to customize spans to track sessions, metadata, etc.](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)
* [How to account for private information and span payload customization](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration)
