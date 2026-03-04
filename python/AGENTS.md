# Python Workspace Guide

## Essential Commands

### Setup

```bash
# requires Python 3.9+
cd python
pip install tox-uv==1.11.2
pip install -r dev-requirements.txt
tox run -e add_symlinks  # required before any imports work
pip install -e openinference-instrumentation
```

### Testing and Quality

Uses **pytest-recording** (vcrpy) — cassettes in `tests/cassettes/` are committed so CI needs no API key.

```bash
pytest tests/ -k test_name --vcr-record=once  # record against live API
pytest tests/                                  # replay cassettes
```

Mark tests with `@pytest.mark.vcr`. Use `InMemorySpanExporter` to assert on spans.

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
See full list: `openinference-instrumentation/src/openinference/instrumentation/_attributes.py`

### Required Features

1. **Suppress tracing** — check `context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY)` at the top of each wrapper; skip span creation if true.
2. **Context propagation** — use `OITracer.start_span()` (handles automatically) or call `get_attributes_from_context()` manually.
3. **OITracer** — wrap the raw OTel tracer with `OITracer(tracer_provider, config=TraceConfig())` for PII masking support.
