# OpenInference LangChain Instrumentation

Python auto-instrumentation library for LangChain.

These traces are fully OpenTelemetry compatible and can be sent to an OpenTelemetry collector for viewing, such as [`arize-phoenix`](https://github.com/Arize-ai/phoenix).

[![pypi](https://badge.fury.io/py/openinference-instrumentation-langchain.svg)](https://pypi.org/project/openinference-instrumentation-langchain/)

## Installation

```shell
pip install openinference-instrumentation-langchain
```

## Quickstart

Install packages needed for this demonstration.

```shell
pip install openinference-instrumentation-langchain langchain arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp
```

Start the Phoenix app in the background as a collector. By default, it listens on `http://localhost:6006`. You can visit the app via a browser at the same address.

The Phoenix app does not send data over the internet. It only operates locally on your machine.

```shell
python -m phoenix.server.main serve
```

The following Python code sets up the `LangChainInstrumentor` to trace `langchain` and send the traces to Phoenix at the endpoint shown below.

```python
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
trace_api.set_tracer_provider(tracer_provider)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

LangChainInstrumentor().instrument()
```

To demonstrate `langchain` tracing, we'll make a simple chain to tell a joke. First, configure your OpenAI credentials.

```python
import os

os.environ["OPENAI_API_KEY"] = "<your openai key>"
```

Now we can create a chain and run it.

```python
prompt_template = "Tell me a {adjective} joke"
prompt = PromptTemplate(input_variables=["adjective"], template=prompt_template)
llm = LLMChain(llm=OpenAI(), prompt=prompt, metadata={"category": "jokes"})
completion = llm.predict(adjective="funny", metadata={"variant": "funny"})
print(completion)
```

Visit the Phoenix app at `http://localhost:6006` to see the traces.

## More Info

* [More info on OpenInference and Phoenix](https://docs.arize.com/phoenix)
* [How to customize spans to track sessions, metadata, etc.](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)
* [How to account for private information and span payload customization](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration)