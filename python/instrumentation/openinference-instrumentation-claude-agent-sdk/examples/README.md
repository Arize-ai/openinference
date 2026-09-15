# Claude Agent SDK instrumentation example

Install dependencies from the package directory (`python/instrumentation/openinference-instrumentation-claude-agent-sdk`):

```bash
pip install -r examples/requirements.txt
```

Set `ANTHROPIC_API_KEY` for the example. Spans are always exported over OTLP: by default to a local Phoenix at http://127.0.0.1:6006 (start it first), or set `PHOENIX_COLLECTOR_ENDPOINT` to point at another Phoenix. If that Phoenix has auth enabled, also set `PHOENIX_API_KEY`; it is sent as a bearer token.

You can view traces in **[Phoenix Cloud](https://arize.com/docs/phoenix/get-started/get-started-tracing)** (no local server), run Phoenix locally (`python -m phoenix.server.main serve`) and view at http://127.0.0.1:6006, or send them to **[Arize AX](https://arize.com/products/ax?utm_source=docs&utm_medium=web&utm_content=openinference)**.

| Example | Description |
|--------|-------------|
| **example.py** | Runs one query that triggers Task -> Bash, then prints all captured span attributes: the top-level AGENT span, TOOL spans, and a nested AGENT span for the Task subagent. |

Run the example:

```bash
python examples/example.py
```

View traces in [Phoenix Cloud](https://arize.com/docs/phoenix/get-started/get-started-tracing), at http://127.0.0.1:6006 when using local Phoenix, or in [Arize AX](https://arize.com/products/ax?utm_source=docs&utm_medium=web&utm_content=openinference).
