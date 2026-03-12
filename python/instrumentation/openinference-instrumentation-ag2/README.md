# openinference-instrumentation-ag2

[![PyPI Version](https://img.shields.io/pypi/v/openinference-instrumentation-ag2.svg)](https://pypi.python.org/pypi/openinference-instrumentation-ag2)

OpenInference instrumentation for [AG2](https://github.com/ag2ai/ag2) (formerly AutoGen).

## Installation

```bash
pip install openinference-instrumentation-ag2 ag2
```

## Usage

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from openinference.instrumentation.ag2 import AG2Instrumentor

tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(your_exporter))

AG2Instrumentor().instrument(tracer_provider=tracer_provider)
```

## What is traced

| Entry point | Span kind | Key attributes |
|---|---|---|
| `ConversableAgent.initiate_chat` | CHAIN | initiator, recipient, max_turns |
| `ConversableAgent.generate_reply` | AGENT | agent name, system message, input/output |
| `ConversableAgent.execute_function` | TOOL | function name, arguments, output |
| `GroupChatManager.run_chat` | CHAIN | agents, max_round, selector, graph topology |
| `initiate_swarm_chat` | CHAIN | agents, initial agent, handoff events |
| `ReasoningAgent.generate_response` | AGENT | method, beam_size, max_depth |
| `ConversableAgent.initiate_chats` | CHAIN | chat count, carryover mode |

Compatible with [Arize Phoenix](https://phoenix.arize.com) — GroupChat topology renders in Phoenix's graph view.

## Compatibility

- AG2 >= 0.11.0
- Python >= 3.9
