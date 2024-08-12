# llama-index OpenInference instrumentation example

This is a [LlamaIndex](https://www.llamaindex.ai/) project bootstrapped with [`create-llama`](https://github.com/run-llama/LlamaIndexTS/tree/main/packages/create-llama) and instrumented using OpenInference.

This example integrates three components:

- A NextJS frontend that provides an interface to a basic RAG chat application
- A Python FastAPI backend that serves a simple LlamaIndex RAG application. The LlamaIndex framework is instrumented using OpenInference to produce traces.
- A [Phoenix](https://github.com/Arize-ai/phoenix) server that acts as both a collector for OpenInference traces and as a trace UI for observability.

## Setup

This application is instrumented using OpenInference with one [instrumentation call](./backend/instrument.py):

```python
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


def instrument():
    resource = Resource(attributes={})
    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    span_exporter = OTLPSpanExporter(endpoint="http://phoenix:6006/v1/traces")
    span_processor = SimpleSpanProcessor(span_exporter=span_exporter)
    tracer_provider.add_span_processor(span_processor=span_processor)
    trace_api.set_tracer_provider(tracer_provider=tracer_provider)
    LlamaIndexInstrumentor().instrument()
```

By calling `instrument()` prior to starting the FastAPI server, traces for LlamaIndex will be sent to the phoenix server, providing deep observability into the underlying behavior of the application. This includes information about all of the retrieved context for a given query, and other relevant information such as reranking and synthesis steps that might occur prior to returning the final LLM response to the user.

To learn more about LlamaIndex, take a look at the following resources:

- [LlamaIndex Documentation](https://docs.llamaindex.ai) - learn about LlamaIndex (Python features).
- [LlamaIndexTS Documentation](https://ts.llamaindex.ai) - learn about LlamaIndex (Typescript features).

You can check out [the LlamaIndexTS GitHub repository](https://github.com/run-llama/LlamaIndexTS) - your feedback and contributions are welcome!

## Getting Started with Local Development

First, startup the backend as described in the [backend README](./backend/README.md).

- If you'd like, include your own data to build an index in [the data directory](./backend/data/)
- Build a simple index using LlamaIndex
- Ensure that your OpenAI API key is available to the application, either via the `OPENAI_API_KEY` environment variable or a `.env` file
- Start the backend server

Second, run the development server of the frontend as described in the [frontend README](./frontend/README.md).

Open [http://localhost:3000](http://localhost:3000) with your browser to use the chat interface to your RAG application.

Traces can be viewed using the [Phoenix UI](http://localhost:6006).

## Getting Started with Docker-Compose

1. If you'd like, add your own PDFs to `./backend/data` to build indexes over.
2. Follow the instructions in `backend/README.md` to install LlamaIndex using poetry and generate an index.
3. Ensure that your OpenAI API key is available to the application, either via the `OPENAI_API_KEY` environment variable or a `.env` file alongside `compose.yml`.
4. Ensure that Docker is installed and running.
5. Run the command `docker compose up --build` to spin up services for the frontend, backend, and Phoenix.
6. Once those services are running, open [http://localhost:3000](http://localhost:3000) to use the chat interface.
7. Traces can be viewed using the [Phoenix UI](http://localhost:6006).
8. When you're finished, run `docker compose down` to spin down the services.
