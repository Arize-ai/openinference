# OpenInference Claude Agent SDK Instrumentation

Python auto-instrumentation for the [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) (Python). Traces **`query()`** and **`ClaudeSDKClient`** as OpenInference AGENT spans with prompt input, result output, session/model metadata, token counts, and tool child spans via hook injection.

- **`query()`** – One span per call (one-off sessions).
- **`ClaudeSDKClient`** – One span per response turn: each time you iterate **`receive_response()`** (or **`receive_messages()`**), a span is created for that turn. Use for [continuous conversations](https://platform.claude.com/docs/en/agent-sdk/python#claudesdkclient).
- **Tools** – Tool calls are captured as child **TOOL** spans via Claude Agent SDK hooks (PreToolUse/PostToolUse/PostToolUseFailure).

For detailed LLM and tool spans inside agent runs, use [openinference-instrumentation-anthropic](https://github.com/Arize-ai/openinference/tree/main/python/instrumentation/openinference-instrumentation-anthropic) together with this package; the Agent SDK uses the Anthropic API under the hood.

Traces are OpenTelemetry-compatible and can be sent to any OTLP collector, [Arize Phoenix](https://github.com/Arize-ai/phoenix) (local), or [Phoenix Cloud](https://arize.com/docs/phoenix/get-started/get-started-tracing).

## Installation

```shell
pip install openinference-instrumentation-claude-agent-sdk
```

## Quickstart

```shell
pip install openinference-instrumentation-claude-agent-sdk claude-agent-sdk arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp
```

**Option A – Phoenix Cloud:** Create a free [Phoenix Cloud](https://arize.com/docs/phoenix/get-started/get-started-tracing) account, create a space, and set `PHOENIX_COLLECTOR_ENDPOINT` and `PHOENIX_API_KEY`. Use your collector endpoint (e.g. `https://<host>/v1/traces`) as `endpoint` below.

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

# Phoenix Cloud: set PHOENIX_COLLECTOR_ENDPOINT (and PHOENIX_API_KEY for auth). Else local.
endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://127.0.0.1:6006/v1/traces")
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
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

View traces in [Phoenix Cloud](https://arize.com/docs/phoenix/get-started/get-started-tracing) or at `http://localhost:6006` when running Phoenix locally.

## Examples

Run the [examples](examples/) in this repo:

```bash
pip install -r examples/requirements.txt
export ANTHROPIC_API_KEY=your-key
python examples/basic_query.py
```

| Example | Description |
|--------|-------------|
| [basic_query.py](examples/basic_query.py) | Simple `query()` with OTLP export to Phoenix |
| [query_with_tools.py](examples/query_with_tools.py) | `query()` with `ClaudeAgentOptions` and tools (Bash, Glob) |
| [client_basic.py](examples/client_basic.py) | `ClaudeSDKClient`: one turn (query + receive_response) |
| [client_multi_turn.py](examples/client_multi_turn.py) | `ClaudeSDKClient`: multi-turn conversation |
| [query_with_phoenix.py](examples/query_with_phoenix.py) | In-process Phoenix via `phoenix.otel.register()` (works with [Phoenix Cloud](https://arize.com/docs/phoenix/get-started/get-started-tracing) or local; requires `arize-phoenix`) |

See [examples/README.md](examples/README.md) for details.

## What is instrumented

- **`query()`** – Each call is wrapped in a single AGENT span named `ClaudeAgentSDK.query` with:
  - **Input**: prompt text or JSON (for async message iterables)
  - **Output**: result text/JSON from the SDK result message
  - **Metadata**: `session.id`, `llm.model_name`, token counts, and `llm.cost.total` when available
  - **Tools**: TOOL child spans created via SDK hooks

- **`ClaudeSDKClient`** – For multi-turn conversations:
  - **`connect(prompt=...)`** and **`query(prompt)`** record the prompt for the next response.
  - Each **`receive_response()`** iteration is wrapped in an AGENT span named `ClaudeAgentSDK.ClaudeSDKClient.receive_response` with the same input/output/metadata/tool spans as above.

Child LLM/tool spans (from the SDK’s internal Anthropic usage) are not created by this package; add `openinference-instrumentation-anthropic` and instrument Anthropic for that.
