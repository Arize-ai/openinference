# OpenInference Haystack Instrumentation

Python auto-instrumentation library for LLM applications implemented with [Haystack](https://haystack.deepset.ai/).

Haystack [Pipelines](https://docs.haystack.deepset.ai/docs/pipelines) and [Components](https://docs.haystack.deepset.ai/docs/components) (ex. PromptBuilder, OpenAIGenerator, etc.) are fully OpenTelemetry-compatible and can be sent to an OpenTelemetry collector for monitoring, such as [`arize-phoenix`](https://github.com/Arize-ai/phoenix).

## Installation

```shell
pip install openinference-instrumentation-haystack
```

## Quickstart

This quickstart shows you how to instrument your Haystack-orchestrated LLM application

Through your *terminal*, install required packages.

```shell
pip install openinference-instrumentation-haystack haystack-ai arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp
```

You can install Phoenix and start it with the following terminal commands:
```shell
pip install arize-phoenix
python -m phoenix.server.main serve
````
Start Phoenix in the background as a collector. By default, it listens on `http://localhost:6006`. You can visit the app via a browser at the same address. (Phoenix does not send data over the internet. It only operates locally on your machine.)


Try the following in a *Python file*.

Set up `HaystackInstrumentor` to trace your application and sends the traces to Phoenix at the endpoint defined below.

```python
from openinference.instrumentation.haystack import HaystackInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
import os

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "YOUR_KEY_HERE"

# Set up the tracer, using Arize Phoenix as the endpoint
endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
trace_api.set_tracer_provider(tracer_provider)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

# Instrument the Haystack application
HaystackInstrumentor().instrument()
```

Set up a simple Pipeline with a template using `OpenAIGenerator`.
```python
from haystack import Pipeline
from haystack.components.generators import OpenAIGenerator

# Initialize the pipeline
pipeline = Pipeline()

# Initialize the OpenAI generator component
llm = OpenAIGenerator(model="gpt-3.5-turbo")

# Add the generator component to the pipeline
pipeline.add_component("llm", llm)

# Define the question
question = "What is the location of the Hanging Gardens of Babylon?"

# Run the pipeline with the question
response = pipeline.run({"llm": {"prompt": question}})

print(response)
```
Now, on the Phoenix UI on your browser, you should see the traces from your Haystack application. Specifically, you can see attributes from the execution of the OpenAIGenerator.


## More Info

* [More info on OpenInference and Phoenix](https://docs.arize.com/phoenix)
* [How to customize spans to track sessions, metadata, etc.](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)
* [How to account for private information and span payload customization](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration)