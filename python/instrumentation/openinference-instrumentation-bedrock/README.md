# OpenInference AWS Bedrock Instrumentation

Python autoinstrumentation library for AWS Bedrock made using `boto3`.

This package implements OpenInference tracing for `invoke_model` calls made using a `boto3` `bedrock-runtime` client. These traces are fully OpenTelemetry compatible and can be sent to an OpenTelemetry collector for viewing, such as [Arize `phoenix`](https://github.com/Arize-ai/phoenix).

[![pypi](https://badge.fury.io/py/openinference-instrumentation-bedrock.svg)](https://pypi.org/project/openinference-instrumentation-bedrock/)

## Installation

```shell
pip install openinference-instrumentation-bedrock
```

## Quickstart

Install `openinference-instrumentation-bedrock`, `arize-phoenix` and `boto3`.

```shell
pip install openinference-instrumentation-bedrock arize-phoenix boto3
```

Ensure that `boto3` is [configured with AWS credentials](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html).

First, import dependencies required to autoinstrument AWS Bedrock and set up `phoenix` as an collector for OpenInference traces.

```python
from urllib.parse import urljoin

import boto3
import phoenix as px

from openinference.instrumentation.bedrock import BedrockInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
```

Next, we'll start a `phoenix` server and set it as a collector.

```python
px.launch_app()
session_url = px.active_session().url
phoenix_otlp_endpoint = urljoin(session_url, "v1/traces")
phoenix_exporter = OTLPSpanExporter(endpoint=phoenix_otlp_endpoint)
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter=phoenix_exporter))
trace_api.set_tracer_provider(tracer_provider=tracer_provider)
```

Instrumenting `boto3` is simple:

```python
BedrockInstrumentor().instrument()
```

Now, all calls to `invoke_model` are instrumented and can be viewed in the `phoenix` UI.

```python
session = boto3.session.Session()
client = session.client("bedrock-runtime")
prompt = b'{"prompt": "Human: Hello there, how are you? Assistant:", "max_tokens_to_sample": 1024}'
response = client.invoke_model(modelId="anthropic.claude-v2", body=prompt)
response_body = json.loads(response.get("body").read())
print(response_body["completion"])
```

## More Info

More documentation on tracing with OpenInference and `phoenix` can be found in the [`phoenix` documentation](https://docs.arize.com/phoenix).
