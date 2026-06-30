# OpenInference GenAI Conformance MVP

A simplified harness that validates the OpenInference instrumentors for **Anthropic**, **OpenAI**, and **Google GenAI** against the [OpenTelemetry Generative AI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) using the official OTel [Weaver `registry live-check`](https://github.com/open-telemetry/weaver) tool.

```
anthropic_conformance.py    ┐
openai_conformance.py       ├─▶ <Provider>Instrumentor ──OTLP──▶ weaver registry live-check ──▶ console summary
google_genai_conformance.py ┘
        │
        └────HTTP────▶ mock_server.py  (Anthropic + OpenAI + Google GenAI endpoints)
```

Each instrumentor is configured with `TraceConfig(enable_genai_semconv=True)`, so every span carries both the native OpenInference attributes and the `gen_ai.*` attributes derived by the dual-write logic in `_genai_conversion.py`.

## Prerequisites

Only [`uv`](https://docs.astral.sh/uv/getting-started/installation/) and `git` are required. All Python dependencies are declared as PEP 723 inline metadata at the top of each script and resolved into ephemeral environments managed by uv. The local OpenInference packages (`openinference-instrumentation`, the per-provider instrumentor packages, and `openinference-semantic-conventions`) are installed editable from this repo via `[tool.uv.sources]` rather than from PyPI, so in-progress changes are picked up on the next run.

[run.py](./run.py) downloads Weaver and the semantic-conventions registry on first run and caches them under `~/.cache/oi-conformance/`. No real provider API keys or network access to provider servers is required.

## Running

```bash
uv run python/openinference-instrumentation/scripts/conformance/run.py
```

Optional flag:

- `--registry PATH` — use a local checkout of the semconv `model/` directory instead of the cached one.

## What it does

1. Allocates three free loopback ports atomically (mock server, weaver OTLP, weaver admin).
2. Starts [mock_server.py](./mock_server.py), which serves `POST /v1/messages` (Anthropic), `POST /v1/chat/completions` and `POST /v1/embeddings` (OpenAI), and `POST /v1beta/models/<model>:generateContent` (Google GenAI) with deterministic responses (`tool_use` / `tool_calls` / `functionCall` when the request has tool definitions, plain text otherwise).
3. Pre-resolves each provider script's uv environment so weaver's inactivity timeout doesn't fire while uv is installing dependencies on first run.
4. Starts `weaver registry live-check` and waits for its admin health endpoint.
5. Runs [anthropic_conformance.py](./anthropic_conformance.py), [openai_conformance.py](./openai_conformance.py), and [google_genai_conformance.py](./google_genai_conformance.py) in sequence; each exercises chat, tool calling, and (where applicable) embeddings against the instrumented client. The Anthropic script also creates manual TOOL, EMBEDDING, and RETRIEVER spans via `OITracer` to exercise dual-write paths that the SDK-level instrumentation alone doesn't reach.
6. Stops weaver via its `/stop` admin endpoint, then parses the JSON output in `./results/` and prints a console summary that lists registry attributes seen, non-registry attributes seen, missing `gen_ai.*` registry attributes, and advice-level counts (violations + improvements).

## Interpreting the summary

The summary has four sections:

- **Registry attributes seen** — `gen_ai.*` attributes recognized by the OTel semconv registry. This is what the dual-write should be producing.
- **Non-registry attributes seen** — OpenInference's native vocabulary (`llm.*`, `openinference.*`, `input.*`, `output.*`). These are expected and not violations on their own.
- **Missing registry attributes (`gen_ai.*`)** — registry attributes the instrumentation did not emit during this run. Some absences reflect scenarios the test app doesn't exercise; others reflect attributes that aren't auto-emittable (application-level concepts like `gen_ai.agent.*`, `gen_ai.evaluation.*`) or that this provider's API doesn't support.
- **Advice levels** — counts of `violation` / `improvement` advice from Weaver's per-attribute checks. The "violations" reported are predominantly `missing_attribute` advice for the OI native vocabulary that doesn't (and shouldn't) live in the OTel registry; the dual-write itself is well-formed. "Improvements" are mostly `not_stable` notes on `gen_ai.*` attributes that are still development-stage in the registry.

## Caveats

- This is an MVP. Coverage is the chat / tool / embeddings / retriever scenarios listed above; expand by adding scenarios to the provider scripts and corresponding response shapes to `mock_server.py`.
- The pinned versions are `WEAVER_VERSION=v0.23.0` and `SEMCONV_VERSION=v1.41.1` (see top of [run.py](./run.py)).
- The runner deletes and recreates `./results/` on each invocation.
