# Claude Agent SDK instrumentation examples

Install dependencies (from this directory or the repo root):

```bash
pip install -r examples/requirements.txt
```

Set `ANTHROPIC_API_KEY` for any example that calls the agent.

You can view traces in **[Phoenix Cloud](https://arize.com/docs/phoenix/get-started/get-started-tracing)** (no local server) or run Phoenix locally (`python -m phoenix.server.main serve`) and view at http://127.0.0.1:6006.

| Example | Description |
|--------|-------------|
| **basic_query.py** | Simple `query()` call with OTLP export to Phoenix. Use [Phoenix Cloud](https://arize.com/docs/phoenix/get-started/get-started-tracing) or start local Phoenix with `python -m phoenix.server.main serve` first. |
| **query_with_tools.py** | `query()` with `ClaudeAgentOptions` (e.g. `allowed_tools=["Bash", "Glob"]`). |
| **client_basic.py** | `ClaudeSDKClient`: one turn (query + receive_response). One AGENT span per response. |
| **client_multi_turn.py** | `ClaudeSDKClient`: multi-turn conversation; one AGENT span per turn. |
| **query_with_phoenix.py** | Uses `phoenix.otel.register()` for in-process Phoenix export; works with Phoenix Cloud (set `PHOENIX_COLLECTOR_ENDPOINT`, `PHOENIX_API_KEY`) or local. Requires `pip install arize-phoenix`. |

Run any example:

```bash
python examples/basic_query.py
```

View traces in [Phoenix Cloud](https://arize.com/docs/phoenix/get-started/get-started-tracing) or at http://127.0.0.1:6006 when using local Phoenix.
