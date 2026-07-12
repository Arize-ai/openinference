# OpenInference AG2

[![pypi](https://badge.fury.io/py/openinference-instrumentation-ag2.svg)](https://pypi.org/project/openinference-instrumentation-ag2/)

OpenTelemetry-compatible tracing for AG2 agents. The instrumentor captures chat runs,
individual agent replies, and synchronous or asynchronous tool execution using the
OpenInference semantic conventions.

## Installation

```shell
pip install openinference-instrumentation-ag2 ag2
```

This release supports the `autogen` API provided by AG2 0.14. AG2 1.0 uses a new
middleware API and is not yet covered by this instrumentor.

## Quickstart

```python
from autogen import ConversableAgent
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.ag2 import AG2Instrumentor

tracer_provider = TracerProvider()
tracer_provider.add_span_processor(
    SimpleSpanProcessor(OTLPSpanExporter("http://localhost:6006/v1/traces"))
)
trace.set_tracer_provider(tracer_provider)
AG2Instrumentor().instrument(tracer_provider=tracer_provider)

assistant = ConversableAgent(
    "assistant",
    llm_config=False,
    human_input_mode="NEVER",
    default_auto_reply="Hello from AG2",
)
assistant.generate_reply(messages=[{"role": "user", "content": "Hello"}])
```

`AG2Instrumentor().uninstrument()` restores every patched AG2 method. The instrumentor
also respects OpenTelemetry tracing suppression, OpenInference context attributes, and
`TraceConfig` masking options.

## More Info

- [OpenInference](https://github.com/Arize-ai/openinference)
- [AG2](https://github.com/ag2ai/ag2)
- [Customizing spans](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)
- [Tracing configuration](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration)
