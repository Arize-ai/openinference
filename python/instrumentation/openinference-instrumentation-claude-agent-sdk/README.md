# OpenInference Claude Agent SDK Instrumentation

Python auto-instrumentation for the [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) (Python). Traces **`query()`** and **`ClaudeSDKClient`** as OpenInference AGENT spans with prompt input, result output, session/model metadata, token counts, and tool child spans via hook injection.

- **`query()`** – One span per call (one-off sessions).
- **`ClaudeSDKClient`** – One span per response turn: each time you iterate **`receive_response()`**, a span is created for that turn. Use for [continuous conversations](https://platform.claude.com/docs/en/agent-sdk/python#claudesdkclient).
- **Tools** – Tool calls are captured as child **TOOL** spans via Claude Agent SDK hooks (PreToolUse/PostToolUse/PostToolUseFailure).
- **Subagents** – Work delegated through a subagent tool such as `Task` is grouped under a nested **AGENT** span, with the subagent's own tool calls as its children.

For detailed LLM and tool spans inside agent runs, use [openinference-instrumentation-anthropic](https://github.com/Arize-ai/openinference/tree/main/python/instrumentation/openinference-instrumentation-anthropic) together with this package; the Agent SDK uses the Anthropic API under the hood.

Traces are OpenTelemetry-compatible and can be sent to any OTLP collector, [Arize Phoenix](https://github.com/Arize-ai/phoenix) (local), [Phoenix Cloud](https://arize.com/docs/phoenix/get-started/get-started-tracing), or [Arize AX](https://arize.com/products/ax?utm_source=docs&utm_medium=web&utm_content=openinference).

## Installation

```shell
pip install openinference-instrumentation-claude-agent-sdk
```

## Quickstart

```shell
pip install openinference-instrumentation-claude-agent-sdk claude-agent-sdk arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp
```

**Option A – Phoenix Cloud:** Create a free [Phoenix Cloud](https://arize.com/docs/phoenix/get-started/get-started-tracing) account, create a space, and set `PHOENIX_COLLECTOR_ENDPOINT` (e.g. `https://<host>/v1/traces`) and `PHOENIX_API_KEY`. The snippet below reads both.

**Option B – Local Phoenix:** Start Phoenix, then run your script:

```shell
python -m phoenix.server.main serve
```

Then in Python:

```python
import asyncio
import os
from claude_agent_sdk import query, ClaudeAgentOptions
from openinference.instrumentation.claude_agent_sdk import ClaudeAgentSDKInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

# Phoenix Cloud: set PHOENIX_COLLECTOR_ENDPOINT and PHOENIX_API_KEY. Otherwise defaults to local Phoenix.
endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://127.0.0.1:6006/v1/traces")
api_key = os.environ.get("PHOENIX_API_KEY")
headers = {"authorization": f"Bearer {api_key}"} if api_key else None
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint, headers=headers)))
ClaudeAgentSDKInstrumentor().instrument(tracer_provider=tracer_provider)

async def main():
    async for message in query(
        prompt="What files are in this directory?",
        options=ClaudeAgentOptions(allowed_tools=["Bash", "Glob"]),
    ):
        if hasattr(message, "result"):
            print(message.result)

asyncio.run(main())
```

View traces in [Phoenix Cloud](https://arize.com/docs/phoenix/get-started/get-started-tracing), at `http://localhost:6006` when running Phoenix locally, or in [Arize AX](https://arize.com/products/ax?utm_source=docs&utm_medium=web&utm_content=openinference).

## Examples

Run the [example](examples/) in this repo from the package directory:

```bash
pip install -r examples/requirements.txt
export ANTHROPIC_API_KEY=your-key
python examples/example.py
```

The example always exports spans over OTLP, defaulting to a local Phoenix at `http://127.0.0.1:6006` (start it first, or set `PHOENIX_COLLECTOR_ENDPOINT` and, for Phoenix Cloud, `PHOENIX_API_KEY`). See [examples/README.md](examples/README.md) for what the example does.

## What is instrumented

- **`query()`** – Each call is wrapped in a single AGENT span named `ClaudeAgentSDK.query` with:
  - **Input**: prompt text or JSON (for async message iterables)
  - **Output**: result text/JSON from the SDK result message, plus `llm.output_messages` including any tool calls
  - **Metadata**: `session.id`, `llm.model_name`, `llm.finish_reason`, `llm.provider`/`llm.system` (`anthropic`), token counts (prompt, completion, total, cache read/write), and `llm.cost.total` when available
  - **Tools**: TOOL child spans created via SDK hooks, with `tool.name`, input parameters, and output
  - **Subagents**: a nested AGENT span named `ClaudeAgentSDK.<tool>` (e.g. `ClaudeAgentSDK.Task`) with `agent.name` set, parenting the subagent's TOOL spans

- **`ClaudeSDKClient`** – For multi-turn conversations:
  - **`connect(prompt=...)`** and **`query(prompt)`** record the prompt for the next response.
  - Each **`receive_response()`** iteration is wrapped in an AGENT span named `ClaudeAgentSDK.ClaudeSDKClient.receive_response` with the same input/output/metadata/tool/subagent spans as above.
  - **`receive_messages()`** is not wrapped; use `receive_response()` to get a span per turn.

LLM spans for the SDK's internal Anthropic API calls are not created by this package; add `openinference-instrumentation-anthropic` and instrument Anthropic for that.

## More Info

* [More info on OpenInference and Phoenix](https://docs.arize.com/phoenix)
* [More info on OpenInference and Arize AX](https://arize.com/products/ax?utm_source=docs&utm_medium=web&utm_content=openinference)
