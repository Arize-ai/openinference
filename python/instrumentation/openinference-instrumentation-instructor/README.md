# OpenInference Instructor Instrumentation

[![pypi](https://badge.fury.io/py/openinference-instrumentation-instructor.svg)](https://pypi.org/project/openinference-instrumentation-instructor/)

Python auto-instrumentation library for the (Instructor)[https://github.com/jxnl/instructor] library

## Installation

```shell
pip install openinference-instrumentation-instructor
```

## Quickstart

This quickstart shows you how to instrument Instructor

Install required packages.

```shell
pip install instructor arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp
```

Start Phoenix in the background as a collector. By default, it listens on `http://localhost:6006`. You can visit the app via a browser at the same address. (Phoenix does not send data over the internet. It only operates locally on your machine.)

```shell
python -m phoenix.server.main serve
```

Set up `InstructorInstrumentor` to trace your application and send the traces to Phoenix at the endpoint defined below. 
```python
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
from openinference.instrumentation.instructor import InstructorInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor

InstructorInstrumentor().instrument(tracer_provider=tracer_provider)

# Optionally instrument the OpenAI SDK to get additional observability
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
```

Simple Instructor example
```python
import instructor
from pydantic import BaseModel
from openai import OpenAI


# Define your desired output structure
class UserInfo(BaseModel):
    name: str
    age: int


# Patch the OpenAI client
client = instructor.from_openai(OpenAI())

# Extract structured data from natural language
user_info = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=UserInfo,
    messages=[{"role": "user", "content": "John Doe is 30 years old."}],
)
```

## More Info

* [More info on OpenInference and Phoenix](https://docs.arize.com/phoenix)
* [How to customize spans to track sessions, metadata, etc.](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)
* [How to account for private information and span payload customization](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration)