# OpenInference OpenAI Agents Instrumentation

[![pypi](https://badge.fury.io/py/openinference-instrumentation-openai-agents.svg)](https://pypi.org/project/openinference-instrumentation-openai-agents/)

Python auto-instrumentation library for OpenAI Agents python SDK.

The traces emitted by this instrumentation are fully OpenTelemetry compatible and can be sent to an OpenTelemetry collector for viewing, such as [Arize Phoenix](https://github.com/Arize-ai/phoenix) or [Arize AX](https://arize.com/).

## Installation

```shell
pip install openinference-instrumentation-openai-agents
```

## Quickstart

In this example we will instrument a small program that uses OpenAI and observe the traces via [`arize-phoenix`](https://github.com/Arize-ai/phoenix).

Install packages.

```shell
pip install openinference-instrumentation-openai-agents arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp
```

Start the phoenix server so that it is ready to collect traces.
The Phoenix server runs entirely on your machine and does not send data over the internet.

```shell
phoenix serve
```

In a python file, set up the `OpenAIAgentsInstrumentor` and configure the tracer to send traces to Phoenix.

```python
from agents import Agent, Runner
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
# Optionally, you can also print the spans to the console.
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)


agent = Agent(name="Assistant", instructions="You are a helpful assistant")
result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)
```

Since we are using OpenAI, we must set the `OPENAI_API_KEY` environment variable to authenticate with the OpenAI API.

```shell
export OPENAI_API_KEY=your-api-key
```

Now simply run the python file and observe the traces in Phoenix.

```shell
python your_file.py
```

## Realtime audio

`OpenAIAgentsInstrumentor().instrument(...)` also traces `agents.realtime.RealtimeSession` (the OpenAI Agents SDK's voice/audio runtime) when the realtime extras are installed. No additional setup is required — `instrument(...)` applies the realtime patches whenever `agents.realtime` is importable.

For each turn the instrumentor produces this span tree:

```
AUDIO   "conversation.turn"     ← parent; aggregated input/output transcripts, llm.model_name, llm.invocation_parameters
├─ USER  "user"                 ← input.audio.url (WAV data URI), input.audio.transcript, or input.value for text input
├─ LLM   "assistant"            ← output.audio.url, output.audio.transcript, token counts, time_to_first_token_ms
│  └─ TOOL "<tool_name>"        ← one per function call within the turn
└─ ...                          ← additional USER / LLM siblings for split input or tool round-trips
```

A runnable mic/speaker example with two function tools lives at [`examples/realtime_with_tools.py`](./examples/realtime_with_tools.py).

### Audio redaction

The realtime instrumentor recognizes three environment variables for redacting captured audio:

- `OPENINFERENCE_HIDE_INPUT_AUDIO` — when truthy (`1` / `true` / `yes` / `on`), drops `input.audio.url`, `input.audio.mime_type`, and `input.audio.transcript` from `USER` spans. Default: `false`.
- `OPENINFERENCE_HIDE_OUTPUT_AUDIO` — same shape, drops the `output.audio.*` attributes from `LLM` spans. Default: `false`.
- `OPENINFERENCE_BASE64_AUDIO_MAX_LENGTH` — caps the base64 payload length of audio `data:` URIs. The `data:audio/wav;base64,` prefix is always preserved. Default: `32000`.

`TraceConfig(hide_inputs=True)` and `TraceConfig(hide_outputs=True)` also cascade to the corresponding audio attributes.

## More Info

* [More info on OpenInference and Phoenix](https://docs.arize.com/phoenix)
* [More info on OpenInference and Arize AX](https://arize.com/)
* [How to customize spans to track sessions, metadata, etc.](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)
* [How to account for private information and span payload customization](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration)
