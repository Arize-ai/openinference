# OpenInference Qwen-Agent Instrumentation

[![pypi](https://badge.fury.io/py/openinference-instrumentation-qwen-agent.svg)](https://pypi.org/project/openinference-instrumentation-qwen-agent/)

Python auto-instrumentation library for [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent).

Agent runs, model calls and tool calls are traced and exported to an OpenInference-compatible collector such as [Arize Phoenix](https://github.com/Arize-ai/phoenix) or [Arize AX](https://arize.com).

## Installation

```shell
pip install openinference-instrumentation-qwen-agent
```

## Quickstart

Install the instrumentor, Qwen-Agent and an OTLP exporter:

```shell
pip install openinference-instrumentation-qwen-agent qwen-agent arize-phoenix-otel
```

Start a Phoenix instance to receive traces:

```shell
pip install arize-phoenix
phoenix serve
```

Then instrument and run an agent:

```python
from phoenix.otel import register
from qwen_agent.agents import Assistant

from openinference.instrumentation.qwen_agent import QwenAgentInstrumentor

tracer_provider = register(project_name="qwen-agent-demo")
QwenAgentInstrumentor().instrument(tracer_provider=tracer_provider)

bot = Assistant(
    llm={"model": "qwen-max", "model_type": "qwen_dashscope"},
    name="weather-assistant",
    description="Answers questions about the weather.",
    system_message="You are a helpful assistant.",
    function_list=["amap_weather"],
)

for _ in bot.run([{"role": "user", "content": "What is the weather in Beijing?"}]):
    pass
```

## What gets traced

| Wrapped method | Span kind | Notes |
|---|---|---|
| `Agent.run` | `AGENT` | One span per agent invocation, named `<agent name>.run`. `Agent.run_nonstream` calls `run` internally and is deliberately not wrapped, so it produces exactly one span. |
| `Agent.run` on a `Memory` | `CHAIN` | `Memory` is an `Agent` subclass used for file management and RAG rather than reasoning. `Assistant` runs it on every turn to gather knowledge. |
| `BaseChatModel.chat` | `LLM` | One span per model call. Every backend routes through this method, so DashScope, the OpenAI-compatible backends, Azure, `transformers` and OpenVINO are all covered. |
| `Agent._call_tool` | `TOOL` | One span per tool call, named `<tool name>.call`, carrying the tool's JSON schema, arguments, result and — recovered from the message history — its `tool_call.id`. |
| `Agent._call_tool` on the `retrieval` tool | `RETRIEVER` | qwen-agent's document retrieval becomes a retriever span, with each retrieved chunk recorded as a document. See below. |

Multi-agent setups need no extra configuration: `Router` and `GroupChat` invoke
`run` on their member agents, so nested agents appear as child spans.

## Token counts

Token counts are recorded **only when Qwen-Agent itself exposes them**, which
depends on the backend:

| `model_type` | SDK underneath | Usage visible to Qwen-Agent | Where token counts land |
|---|---|---|---|
| `qwen_dashscope`, `qwenvl_dashscope` | `dashscope` | Yes — on `Message.extra["model_service_info"]` | This instrumentor's `LLM` span |
| `oai`, `azure`, `qwenvl_oai`, `qwenomni_oai` | `openai` | No — Qwen-Agent discards the response | The nested OpenAI-SDK span, but only for `stream=False` (see below) |
| `transformers`, `openvino` | local | No | Not available |

On an OpenAI-compatible backend, install
[`openinference-instrumentation-openai`](https://pypi.org/project/openinference-instrumentation-openai/)
alongside this package to capture token counts:

```python
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.instrumentation.qwen_agent import QwenAgentInstrumentor

QwenAgentInstrumentor().instrument(tracer_provider=tracer_provider)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
```

The OpenAI span nests inside the Qwen-Agent `LLM` span, and only one of the two
carries the token counts, so trace-level totals stay correct. Token counts are
never aggregated onto `AGENT` spans for the same reason — trace totals are summed
across every span in a trace, so a duplicated count would inflate them.

**Streaming caveat.** Qwen-Agent never sends `stream_options={"include_usage":
true}`, and an OpenAI-compatible server returns no usage block in a streamed
response unless asked. Agents always call the model with `stream=True`
(`Agent._call_llm` hard-codes it), so on an OpenAI-compatible backend an agent
run has no token counts available from either span — there is nothing to record
rather than something being dropped. Direct `chat(stream=False)` calls do get
counts, on the OpenAI span. DashScope is unaffected, because it reports usage
through `Message.extra` on both paths.

**Do not enable `OpenAIInstrumentor` just for token counts during agent runs.**
Because of the streaming caveat there are none to gain, and some platforms —
Arize AX among them — estimate `llm.token_count.*` server-side for LLM spans
that arrive without it. A nested OpenAI-SDK LLM span inside each Qwen-Agent
`LLM` span then gets its own estimate, and the trace total counts the same model
call twice. Enable it when you need the wire-level request/response detail, or
when real usage is available (a non-streaming call), and prefer the DashScope
backend when accurate token counts matter.

## Document retrieval

Calls to qwen-agent's built-in `retrieval` tool produce a `RETRIEVER` span rather
than a `TOOL` span, so the retrieved text shows up as documents:

```
AGENT     docs-assistant.run
  CHAIN     Memory.run
  LLM       TextChatAtOAI.chat
  RETRIEVER retrieval.call
  LLM       TextChatAtOAI.chat
```

`Retrieval.call` returns one entry per source document, shaped
`{"url": ..., "text": [chunk, ...]}`. Each retrieved chunk becomes one document:
`document.content` is the chunk, `document.id` is the source url, and
`document.metadata` carries that url plus the chunk's index within its document.

`Memory` — the `Agent` subclass that owns the `retrieval` and `doc_parser` tools —
stays a `CHAIN`, because it also parses files and generates search keywords. The
retrieval it performs appears as the `RETRIEVER` span beneath it.

Detection is by class, so a `Retrieval` subclass is recognised too, including
one registered under a different name. A differently implemented tool that merely
uses the name `retrieval` is *not* matched — it would not return the document
shape a retriever span needs. If the payload has an unrecognised shape the span
records no documents rather than guessing.

## Known limitations

- `FnCallAgent._call_tool` returns early when the model calls a tool that is not
  registered, without reaching `Agent._call_tool`, so that case produces no
  `TOOL` span.
- `Agent._call_tool` catches tool exceptions and returns the error text as the
  tool result, so a failing tool yields a successful `TOOL` span whose output is
  the error message. Only `ToolServiceError` and `DocParserError` propagate, and
  those are recorded on the span.
- Qwen-Agent does not request usage for streamed responses, so agent runs on an
  OpenAI-compatible backend have no token counts at all. See the streaming caveat
  above.
- Retrieved documents carry no `document.score`. qwen-agent computes relevance
  scores in `BaseSearch.sort_by_scores` but discards them in `get_topk`, so the
  scores never reach the tool's return value.
- `llm.provider` is set only for `model_type="azure"`. Qwen-Agent does not retain
  the base URL on the model instance, so an OpenAI-compatible endpoint cannot be
  attributed to a provider, and DashScope has no OpenInference provider value
  yet. The raw value is always available as `qwen_agent.llm.model_type`.

## More info

- [OpenInference tracing configuration](https://arize.com/docs/phoenix/tracing/how-to-tracing/customize-traces)
- [OpenInference specification](https://github.com/Arize-ai/openinference/tree/main/spec)
- [Examples](./examples)
