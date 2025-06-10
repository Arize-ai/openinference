# OpenInference smolagents Examples

This directory contains numerous examples that show how to use OpenInference to instrument smolagents applications.
Specifically, this uses the [openinference-instrumentation-smolagents](..) package from source in the parent directory.

## Installation

```shell
pip install -r requirements.txt
```

Start Phoenix in the background as a collector, which listens on `http://localhost:6006` and default gRPC port 4317.
Note that Phoenix does not send data over the internet. It only operates locally on your machine.

```shell
python -m phoenix.server.main serve
```

## Running

Copy [env.example](env.example) to `.env` and update variables your example uses, such as `OPENAI_API_KEY`.

Then, run an example like this:

```shell
dotenv run -- opentelemetry-instrument python managed_agent.py
```

Finally, browse for your trace in Phoenix at `http://localhost:6006`!

## Manual instrumentation

`opentelemetry-instrument` is the [Zero-code instrumentation](https://opentelemetry.io/docs/zero-code/python) approach
for Python. It avoids explicitly importing and configuring OpenTelemetry code in your main source. Alternatively, you
can copy-paste the following into your main source and run it without `opentelemetry-instrument`.

```python
from opentelemetry.sdk.trace import TracerProvider

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)
trace_provider = TracerProvider()
trace_provider.add_span_processor(SimpleSpanProcessor(otlp_exporter))

SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)
```
