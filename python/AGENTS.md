# Python Workspace Guide

## Essential Commands

### Setup

```bash
# requires Python 3.10+
cd python
pip install tox-uv==1.11.2
pip install -r dev-requirements.txt
tox run -e add_symlinks  # required before any imports work
pip install -e openinference-instrumentation
```

### Testing and Quality

Uses **pytest-recording** (vcrpy) — cassettes are committed next to the test module
(`tests/openinference/instrumentation/<name>/cassettes/`) so CI needs no API key.

```bash
pytest tests/ -k test_name --record-mode=once  # record against live API
pytest tests/                                  # replay cassettes
```

Mark tests with `@pytest.mark.vcr`. Strip request/response headers in the `vcr_config`
fixture so credentials never land in cassettes, and default the API-key env var to a
placeholder in `conftest.py` so replay works without credentials. Use
`InMemorySpanExporter` to assert on spans.

Required test categories: suppress tracing, context attribute propagation, trace config masking.

### tox Commands

`ruff-mypy-test-openai` = hyphen-joined factors; not defined literally in `tox.ini`.

```bash
tox run -e test-openai            # run tests
tox run -e ruff-mypy-test-openai  # all checks
tox run-parallel                  # all CI checks
```

Token = strip `openinference-instrumentation-` prefix, replace remaining hyphens with underscores
(e.g. `openai`, `llama_index`). Full list in `python/tox.ini` `changedir` section.

## Architecture Overview

- **`openinference-instrumentation`**: Core framework — `OITracer`, `TraceConfig`, context managers (`using_session()`, `using_user()`, etc.), and span attribute builders
- **`openinference-semantic-conventions`**: Centralized span attribute definitions (`SpanAttributes`, `OpenInferenceSpanKindValues`, etc.)
- **Instrumentors**: `python/instrumentation/openinference-instrumentation-<name>/` — one package per AI library (openai, langchain, llama-index, crewai, …)

## Key Patterns

### Attribute Helpers

Always encouraged to use helpers from `openinference.instrumentation` before rolling custom solutions.
See full list: `openinference-instrumentation/src/openinference/instrumentation/_attributes.py`.
Never hand-spell semantic-convention attribute keys — build them from `SpanAttributes`,
`MessageAttributes`, `ToolAttributes`, etc.

### Required Features

1. **Suppress tracing** — check `context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY)` at the top of each wrapper; skip span creation if true.
2. **Context propagation** — `OITracer.start_span()` injects `get_attributes_from_context()` (session, user, metadata, tags) at span start automatically; do not also fetch and set them manually.
3. **OITracer** — wrap the raw OTel tracer with `OITracer(tracer_provider, config=TraceConfig())` for PII masking support.

### Logging and Diagnostics

- Use stdlib logging: `logger = logging.getLogger(__name__)` plus `logger.addHandler(logging.NullHandler())` at module top.
- There is no `diag` API in OpenTelemetry Python — `diag` is the JS convention (`@opentelemetry/api`). Stdlib logging is OTel Python's sanctioned self-diagnostics channel.
- Instrumentation must never raise into user code: wrap attribute extraction and span finalization in `try/except` with `logger.exception(...)`, and make sure the span still ends on failure.

### Instrumentor Conventions

- **Layout**: source in `src/openinference/instrumentation/<name>/`, tests mirror the namespace at `tests/openinference/instrumentation/<name>/`.
- **Span names**: name spans after the wrapped resource class (e.g. `Completions` / `AsyncCompletions`), consistent with sibling instrumentors — never ad-hoc lowercase names, which break cross-provider span-name queries.
- **Provider**: set `llm.provider` to its well-known value from `spec/semantic_conventions.md`; set `llm.system` only if a well-known value applies.
- **Request parameters**: filter out the SDK's "not given" sentinels (e.g. `Omit` / `NotGiven` in Stainless-generated SDKs) so unset defaults don't pollute `input.value` and `llm.invocation_parameters`.
- **Streaming**: when `create(stream=True)` returns a stream object, do not end the span — wrap the stream in a `wrapt.ObjectProxy` that accumulates chunks and finishes the span when iteration completes (see `openai/_stream.py`, `together/_stream.py`).

### New Instrumentor Checklist

- `pyproject.toml` entry points: both `opentelemetry_instrumentor` and `openinference_instrumentor`
- `python/tox.ini`: `envlist`, `changedir`, and `commands_pre` entries
- `.release-please-manifest.json` (`"0.1.0"`) and `release-please-config.json`
- Root `README.md`: libraries table and examples table
- Do **not** hand-edit `.github/dependabot.yml` — it is generated from package manifests by `scripts/generate_dependabot.py` via CI.
