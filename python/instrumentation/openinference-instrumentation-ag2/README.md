# OpenInference AG2 Instrumentation

[![pypi](https://badge.fury.io/py/openinference-instrumentation-ag2.svg)](https://pypi.org/project/openinference-instrumentation-ag2/)

Python auto-instrumentation library for [AG2](https://github.com/ag2ai/ag2) agents, capturing chats,
agent replies, and synchronous or asynchronous tool execution.

The following instrumentation is fully OpenTelemetry-compatible and can be sent to an OpenTelemetry collector for monitoring, such as [Arize Phoenix](https://github.com/Arize-ai/phoenix) or [Arize AX](https://arize.com/docs/ax).

## Installation

```shell
pip install openinference-instrumentation-ag2
```

This release supports the `autogen` API provided by AG2 0.14. AG2 1.0 uses a new middleware API
and is not yet covered by this instrumentor.

## Quickstart

This quickstart shows you how to instrument your AG2 application.

You've already installed openinference-instrumentation-ag2. Next is to install packages for AG2,
Phoenix, and the exporter that sends traces to it.

```shell
pip install "ag2[openai]" arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp
```

Start the Phoenix app in the background as a collector:

```shell
phoenix serve
```

By default, it listens on `http://localhost:6006`. You can visit the app via a browser at the same address.

The Phoenix app does not send data over the internet. It only operates locally on your machine.

Create a simple AG2 agent:

```python example.py
import os

from autogen import ConversableAgent, LLMConfig
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.ag2 import AG2Instrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
# Optionally, you can also print the spans to the console.
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

trace_api.set_tracer_provider(tracer_provider=tracer_provider)

# Start instrumenting AG2
AG2Instrumentor().instrument()

llm_config = LLMConfig(
    {"api_type": "openai", "model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}
)

agent = ConversableAgent(
    name="helpful_agent",
    system_message="You are a helpful assistant.",
    llm_config=llm_config,
)

response = agent.run(message="What is the capital of France?", max_turns=1, user_input=False)
response.process()
```

Finally, run the example:

```shell
python example.py
```

Finally, browse for your trace in Phoenix at `http://localhost:6006`!

## Span kinds

| AG2 method | Span name | OpenInference span kind |
| --- | --- | --- |
| `initiate_chat` / `a_initiate_chat` (also used by `run` and `initiate_chats`) | `<agent>.initiate_chat` | `AGENT` |
| `generate_reply` / `a_generate_reply` | `<agent>.generate_reply` | `AGENT` |
| `execute_function` / `a_execute_function` | `<tool>` | `TOOL` |

`AG2Instrumentor().uninstrument()` restores every patched AG2 method. The instrumentor also respects
OpenTelemetry tracing suppression, OpenInference context attributes, and `TraceConfig` masking options.

## Examples

More examples covering tool calling, group chats, sequential chats, structured outputs, and the
async paths live in [`examples/`](examples). Two of them need no LLM API key, so they are the
quickest way to confirm traces are reaching Phoenix.

## More Info

* [More info on OpenInference and Phoenix](https://docs.arize.com/phoenix)
* [More info on OpenInference and Arize AX](https://arize.com/docs/ax)
* [How to customize spans to track sessions, metadata, etc.](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)
* [How to account for private information and span payload customization](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration)
