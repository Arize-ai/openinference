# openinference-instrumentation-autogen

[![PyPI Version](https://img.shields.io/pypi/v/openinference-instrumentation-autogen.svg)](https://pypi.python.org/pypi/openinference-instrumentation-autogen)

OpenInference instrumentation for [AG2](https://github.com/ag2ai/ag2) (formerly AutoGen) v0.11+.

## Installation

```bash
pip install openinference-instrumentation-autogen ag2
```

## Usage

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from openinference.instrumentation.autogen import AutogenInstrumentor

tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(your_exporter))

AutogenInstrumentor().instrument(tracer_provider=tracer_provider)
```

## What is traced

| Entry point | Span kind | Key attributes |
|---|---|---|
| `ConversableAgent.initiate_chat` | CHAIN | initiator, recipient, max_turns |
| `ConversableAgent.generate_reply` | AGENT | agent name, system message, input/output |
| `ConversableAgent.execute_function` | TOOL | function name, arguments, output |
| `GroupChatManager.run_chat` | CHAIN | agents, max_round, selector, graph topology |
| `run_swarm` | CHAIN | agents, initial agent, max_rounds |
| `ReasoningAgent.generate_reply` | AGENT | method, beam_size, max_depth |
| `ConversableAgent.initiate_chats` | CHAIN | chat count |

Compatible with [Arize Phoenix](https://phoenix.arize.com) — GroupChat topology renders in Phoenix's graph view via `graph.node.id` attributes.

## Compatibility

- AG2 >= 0.11.0
- Python >= 3.10
