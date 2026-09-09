# OpenInference OpenAI Agents Instrumentation

[![pypi](https://badge.fury.io/py/openinference-instrumentation-openai-agents.svg)](https://pypi.org/project/openinference-instrumentation-openai-agents/)

Python auto-instrumentation library for OpenAI Agents python SDK.

The traces emitted by this instrumentation are fully OpenTelemetry compatible and can be sent to an OpenTelemetry collector for viewing, such as [Arize Phoenix](https://github.com/Arize-ai/phoenix) or [Arize AX](https://arize.com/products/ax?utm_source=docs&utm_medium=web&utm_content=openinference).

## Compatibility

| `openinference-instrumentation-openai-agents` | `openai-agents` | Python           |
| --------------------------------------------- | --------------- | ---------------- |
| `>=2.0`                                        | `>=0.11.0`      | `>=3.10, <3.15`  |
| `>=1.4.1, <2.0`                                | `>=0.2.6`       | `>=3.10, <3.15`  |
| `>=1.2.0, <1.4.1`                              | `>=0.2.6`       | `>=3.9, <3.14`   |
| `>=1.0.0, <1.2.0`                              | `>=0.1.0`       | `>=3.9, <3.14`   |
| `<1.0.0`                                       | `>=0.0.3`       | `>=3.9, <3.14`   |

Instrumentor `>=2.0` requires `openai-agents>=0.11.0`. Three things changed below that floor
which the instrumentor no longer accommodates: the run internals moved out of
`agents._run_impl` into `agents.run_internal` in 0.8.0, `openai-agents` moved from
`openai<2` to `openai>=2.9` in the same release, and tool namespaces
(`agents.tool_namespace`) arrived in 0.11.0. Tools grouped by a namespace report
`tool.description` and `tool.parameters` only on instrumentor `>=2.0`.

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

## Hosted search tools

The instrumentor records the Agents SDK's hosted `FileSearchTool` and `WebSearchTool`
calls on the LLM span, so a turn that searched is distinguishable from one that did not:

| What                                   | Where it appears                                                                                                                              |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| A file search the model requested      | LLM span output message, `tool_call.function.name = "file_search_call"` with the `queries` as `tool_call.function.arguments`, correlated by `tool_call.id` |
| A web search the model requested       | LLM span output message, `tool_call.function.name = "web_search_call"` with the `action` (search, open_page, find) as `tool_call.function.arguments`     |
| Retrieved file chunks                  | A following `tool` role message whose `message.content` is the `results` JSON (only when `FileSearchTool(include_search_results=True)`)        |
| Either call, replayed on the next turn | Next LLM span input message, same `tool_call.*` attributes (and the same `tool` message for results)                                            |

Hosted tools run inside the Responses API, so there is no separate `TOOL` span for them.
The call `status` is not recorded as an attribute; it remains in the raw `output.value`.

### Example

[`examples/hosted_search_tools.py`](./examples/hosted_search_tools.py) creates a throwaway
vector store from a small FAQ, answers one question with file search, then replays that
turn and answers a follow-up with web search. It needs an `OPENAI_API_KEY` and Phoenix at
`http://localhost:6006`; the vector store is deleted on exit.

```shell
pip install -r examples/requirements.txt
python examples/hosted_search_tools.py
```

Set `SEARCH_MODEL` to use a different model and `PHOENIX_PROJECT` to separate runs. In
Phoenix, open the first `response` LLM span to see the `file_search_call` in its output
messages, then the second to see it replayed as input alongside the new `web_search_call`.

## Computer use

The instrumentor records the OpenAI Agents SDK's built-in computer tool (`ComputerTool`)
so each turn of a computer-use loop is visible in Phoenix:

| What                                   | Where it appears                                                                                                 |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| The action the model requested         | LLM span output message, `tool_call.function.name = "computer_call"` with the `action` (or batched `actions`) as `tool_call.function.arguments` |
| The action, replayed on the next turn  | Next LLM span input message, same `tool_call.*` attributes, correlated by `tool_call.id`                        |
| The screenshot returned to the model   | Next LLM span input message, as structured image content (`message.contents.0.message_content.image.image.url`) |
| The computer tool span                 | A `TOOL` span named `computer`; its output is a `{"type": "computer_screenshot"}` placeholder                     |

Screenshots live only in the structured image attribute, so the standard image controls
apply to them. `OPENINFERENCE_HIDE_INPUT_IMAGES=true` removes them, and
`OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH` (default `32000` characters) redacts any
screenshot whose data URL is longer than the limit. Real screenshots are usually larger
than the default, so raise the limit (or configure a blob uploader) to keep them.
The raw `input.value` JSON and the tool span's `output.value` omit the screenshot data
URL so it cannot leak through an attribute the image settings do not cover. One
consequence: if a run stops right after a computer action (for example `max_turns` is
reached), that final screenshot is not sent back to the model and is therefore not in
the trace.

### Example

[`examples/computer_use.py`](./examples/computer_use.py) runs a live model against an
in-memory display. The model clicks a red button, receives a new screenshot, and reports
that the button turned green. It needs OpenAI Agents SDK 0.11.0 or later, Pillow, an
`OPENAI_API_KEY` with access to `gpt-5.4`, and Phoenix at `http://localhost:6006`.

```shell
pip install -r examples/requirements.txt
python examples/computer_use.py

# Verify screenshot masking, then size-based redaction
OPENINFERENCE_HIDE_INPUT_IMAGES=true python examples/computer_use.py
OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH=100 python examples/computer_use.py
```

Set `COMPUTER_MODEL` to use a different model and `PHOENIX_PROJECT` to separate runs.
In Phoenix, open the `computer` tool span to see the requested action, then the
following LLM span to see the screenshot the model received.

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
* [More info on OpenInference and Arize AX](https://arize.com/products/ax?utm_source=docs&utm_medium=web&utm_content=openinference)
* [How to customize spans to track sessions, metadata, etc.](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)
* [How to account for private information and span payload customization](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration)
