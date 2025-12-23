# OpenInference Agent Spec Instrumentation

Python auto-instrumentation library for Agent Spec.

Open Agent Spec (Agent Spec) is a portable, platform-agnostic configuration language that allows Agents and Agentic Systems to be described with high fidelity.
It defines the conceptual building blocks—called components—that make up agents in typical agent-based systems.
This includes the properties that configure each component and the semantics that govern their behavior.

Agent Spec Tracing is an extension of Agent Spec that standardizes how agent and flow executions emit traces.
It enables:
- Runtime adapters to emit consistent traces across different frameworks.
- Consumers (observability backends, UIs, developer tooling) to ingest one standardized format regardless of the producer.

You can find more information about Agent Spec and Agent Spec Tracing at:
- Agent Spec: https://oracle.github.io/agent-spec/development/agentspec/index.html
- Tracing: https://oracle.github.io/agent-spec/development/agentspec/tracing.html

The traces emitted by this instrumentation are fully OpenTelemetry compatible and can be sent to an OpenTelemetry 
collector for viewing, such as [`arize-phoenix`](https://github.com/Arize-ai/phoenix)

## Installation

```shell
pip install openinference-instrumentation-agentspec
```

## Quickstart

In this example we will instrument a small program that uses Agent Spec Tracing and observe 
the traces via [`arize-phoenix`](https://github.com/Arize-ai/phoenix).

Install packages.

```shell
pip install openinference-instrumentation-agentspec arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp
```

Start the phoenix server so that it is ready to collect traces.
The Phoenix server runs entirely on your machine and does not send data over the internet.

```shell
phoenix serve
```

In a python file (e.g., `agentspec_agent.py`) , set up the `AgentSpecInstrumentor` and configure 
the tracer to send traces to Phoenix.

```python
from pyagentspec.adapters.langgraph import AgentSpecLoader
from pyagentspec.agent import Agent
from pyagentspec.llms import OpenAiConfig

agent = Agent(
    name="assistant",
    description="An general purpose agent without tools",
    llm_config=OpenAiConfig(name="openai-gpt-5-mini", model_id="gpt-5-mini"),
    system_prompt="You are a helpful assistant. Help the user answering politely.",
)

langgraph_agent = AgentSpecLoader().load_component(agent)

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.agentspec import AgentSpecInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

AgentSpecInstrumentor().instrument(tracer_provider=tracer_provider)

while True:
    user_input = input("USER  >>> ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = langgraph_agent.invoke(
        input={"messages": [{"role": "user", "content": user_input}]},
        config={"configurable": {"thread_id": "1"}},
    )
    print("AGENT >>>", response['messages'][-1].content.strip())
```

Now simply run the python file and observe the traces in Phoenix.

```shell
python agentspec_agent.py
```
