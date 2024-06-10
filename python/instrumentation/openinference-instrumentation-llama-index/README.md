# OpenInference LlamaIndex Instrumentation
Python auto-instrumentation library for LlamaIndex.

These traces are fully OpenTelemetry compatible and can be sent to an OpenTelemetry collector for viewing, such as [`arize-phoenix`](https://github.com/Arize-ai/phoenix).

[![pypi](https://badge.fury.io/py/openinference-instrumentation-llama-index.svg)](https://pypi.org/project/openinference-instrumentation-llama-index/)

## Installation

```shell
pip install openinference-instrumentation-llama-index
```

## Compatibility

| llama-index version | openinference-instrumentation-llama-index version |
|---------------------|---------------------------------------------------|
| \>=0.10.43          | \>=2.0.0                                          |
| \>=0.10.0, <0.10.43 | \>=1.0.0, <0.2                                    |
| \>=0.9.14, <0.10.0  | 0.1.3                                             |

## Quickstart

Install packages needed for this demonstration.

```shell
python -m pip install --upgrade \
    openinference-instrumentation-llama-index \
    opentelemetry-sdk \
    opentelemetry-exporter-otlp \
    "opentelemetry-proto>=1.12.0" \
    arize-phoenix
```

Start the Phoenix app in the background as a collector. By default, it listens on `http://localhost:6006`. You can visit the app via a browser at the same address.

The Phoenix app does not send data over the internet. It only operates locally on your machine.

```shell
python -m phoenix.server.main serve
```

The following Python code sets up the `LlamaIndexInstrumentor` to trace `llama-index` and send the traces to Phoenix at the endpoint shown below.

```python
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
```

To demonstrate tracing, we'll use LlamaIndex below to query a document. 

First, download a text file.

```python
import tempfile
from urllib.request import urlretrieve
from llama_index.core import SimpleDirectoryReader

url = "https://raw.githubusercontent.com/Arize-ai/phoenix-assets/main/data/paul_graham/paul_graham_essay.txt"
with tempfile.NamedTemporaryFile() as tf:
    urlretrieve(url, tf.name)
    documents = SimpleDirectoryReader(input_files=[tf.name]).load_data()
```

Next, we'll query using OpenAI. To do that you need to set up your OpenAI API key in an environment variable.

```python
import os

os.environ["OPENAI_API_KEY"] = "<your openai key>"
```

Now we can query the indexed documents.

```python
from llama_index.core import VectorStoreIndex

query_engine = VectorStoreIndex.from_documents(documents).as_query_engine()
print(query_engine.query("What did the author do growing up?"))
```

Visit the Phoenix app at `http://localhost:6006` to see the traces.

## More Info

More details about tracing with OpenInference and Phoenix can be found in the [Phoenix documentation](https://docs.arize.com/phoenix).

For AI/ML observability solutions in production, including a cloud-based trace collector, visit [Arize](https://docs.arize.com/arize).
