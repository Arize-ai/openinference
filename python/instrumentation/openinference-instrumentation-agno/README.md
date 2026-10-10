# OpenInference Agno Instrumentation

[![pypi](https://badge.fury.io/py/openinference-instrumentation-agno.svg)](https://pypi.org/project/openinference-instrumentation-agno/)

Python auto-instrumentation library for Agno Agents

The following instrumentation is fully OpenTelemetry-compatible and can be sent to an OpenTelemetry collector for monitoring, such as [Arize Phoenix](https://github.com/Arize-ai/phoenix), [Arize AX](https://arize.com/products/ax?utm_source=docs&utm_medium=web&utm_content=openinference), or [Langfuse](https://langfuse.com).

## Installation

```shell
pip install openinference-instrumentation-agno
```

## Quickstart

This quickstart shows you how to instrument your Agno Agent application.

You've already installed openinference-instrumentation-agno. Next is to install packages for agno,
Phoenix and `opentelemetry-instrument`, which exports traces to it.

```shell
pip install agno arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc opentelemetry-distro
```

Start the Phoenix app in the background as a collector:

```shell
phoenix serve
```

By default, it listens on `http://localhost:6006`. You can visit the app via a browser at the same address.

The Phoenix app does not send data over the internet. It only operates locally on your machine.

Create a simple Agno agent:

```python example.py
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools

from openinference.instrumentation.agno import AgnoInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
# Optionally, you can also print the spans to the console.
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

trace_api.set_tracer_provider(tracer_provider=tracer_provider)

# Start instrumenting agno
AgnoInstrumentor().instrument()


agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"), 
    tools=[DuckDuckGoTools()],
    markdown=True, 
    debug_mode=True,
)

agent.print_response("What is currently trending on Twitter?")
```

Finally, run the example:

```shell
python example.py
```

Finally, browse for your trace in Phoenix at `http://localhost:6006`!

## External agents: Claude, Codex, and other frameworks

On Agno versions that provide `BaseExternalAgent`, `AgnoInstrumentor` also
instruments its shared execution methods. This includes `ClaudeAgent`,
`CodexAgent`, `LangGraphAgent`, `DSPyAgent`, and other adapters using those
methods. Older Agno versions continue to use native-agent instrumentation.

External runs produce an `AGENT` span with input/output, Agno run/session/agent
identity, framework, final status, and available token/cost metrics. Sync,
async, streaming, and background execution are supported. Parent tracing
context is preserved across the base class's sync worker threads.

Streaming tool start/completion events produce child `TOOL` spans. Their
durations measure the interval between the observed events, which can differ
from actual execution time when a harness buffers events. Non-streamed runs
retain completed tool calls/results in the run span's output messages;
they do not create tool spans with inferred execution durations. Interrupted
tools are closed and marked with `agno.tool.status`.

When Agno supplies `ToolExecution.parent_tool_call_id`, nested tool spans attach
to the corresponding tool span, including parents whose completion arrived first.
Claude subagent calls therefore retain their SDK-reported hierarchy. Tools without
an observed parent stay under the run; a reported parent ID is retained as
`agno.tool.parent_call_id`. Instrumentation does not infer relationships from names
or timing. This requires an Agno version that preserves tool lineage; older
versions retain the flat tree. No tracing database migration is required.

This traces the Agno adapter boundary. It does not reconstruct individual model
requests or subagent activity hidden inside a Claude or Codex subprocess.
Additional framework instrumentation can supply deeper spans. If another
instrumenter already captures tools, avoid duplicate event-derived tool spans:

```python
AgnoInstrumentor().instrument(
    tracer_provider=tracer_provider,
    capture_external_tool_spans=False,
)
```

### Store traces in Agno's database

No additional collector or Arize backend is required. In a fresh process:

```python
from agno.agents.claude import ClaudeAgent
from agno.db.sqlite import SqliteDb
from agno.tracing import setup_tracing

db = SqliteDb(db_file="external-agent-traces.db")
setup_tracing(db=db)
agent = ClaudeAgent(name="Claude", db=db)
agent.print_response("Describe this project.", stream=True)
```

`setup_tracing()` registers this instrumenter and Agno's existing
`DatabaseSpanExporter`. If your application already has a tracer provider,
attach `DatabaseSpanExporter` to that provider and pass the same provider to
`AgnoInstrumentor().instrument()` instead. Use PostgreSQL for production.

For a local test of either harness, see [examples/external_agents.py](examples/external_agents.py).
Install the selected harness SDK (`claude-agent-sdk` or `openai-codex`) and
authenticate it before running. These examples require an Agno version that
includes the corresponding external adapter.

## More Info

* [More info on OpenInference and Phoenix](https://docs.arize.com/phoenix)
* [More info on OpenInference and Arize AX](https://arize.com/products/ax?utm_source=docs&utm_medium=web&utm_content=openinference)
* [How to customize spans to track sessions, metadata, etc.](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)
* [How to account for private information and span payload customization](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration)
