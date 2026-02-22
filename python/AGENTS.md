# Python Workspace Guide

> **Start here — two things matter most:**
> 1. **VCR/cassette tests** are the workflow for both feature development and debugging —
>    read [Testing Patterns](#testing-patterns) before writing any code or running tests.
> 2. **Helpers from `openinference.instrumentation`** — check these before writing any
>    attribute-setting or span-creation logic; rolling custom solutions is the #1 review
>    blocker. See [Helper Functions](#helper-functions) for the full reference.

## Setup

```bash
cd python
pip install tox-uv==1.11.2
pip install -r dev-requirements.txt
tox run -e add_symlinks              # compose namespace package (required before any imports work)
pip install -e openinference-instrumentation  # editable install for development
```

---

## Testing Patterns

Every instrumentor uses **pytest-recording** (built on vcrpy) to capture real API
interactions as YAML cassettes, then replays them in CI. Combined with
`InMemorySpanExporter`, you can assert on span attributes without a live API key after the
initial recording.

### conftest.py setup

```python
import pytest
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from openinference.instrumentation.<name> import <Name>Instrumentor

@pytest.fixture(scope="session")
def vcr_config():
    return {
        "filter_headers": ["authorization", "api-key", "x-api-key"],
        "decode_compressed_response": True,
        "record_mode": "once",       # record only if cassette is absent, then replay
        "match_on": ["method", "scheme", "host", "port", "path", "query"],
    }

@pytest.fixture(scope="module")
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()

@pytest.fixture(scope="module")
def tracer_provider(in_memory_span_exporter):
    provider = trace_sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return provider

@pytest.fixture(autouse=True)
def instrument(tracer_provider, in_memory_span_exporter):
    in_memory_span_exporter.clear()          # clean slate before each test
    <Name>Instrumentor().instrument(tracer_provider=tracer_provider)
    yield
    <Name>Instrumentor().uninstrument()
    in_memory_span_exporter.clear()          # don't leak spans into next test
```

### Recording cassettes

```bash
# Record against the real API (requires API key) — creates cassettes/
pytest tests/ -k test_my_test --vcr-record=once

# Subsequent runs replay cassettes without an API key
pytest tests/
```

Cassettes are stored in `tests/cassettes/` and committed to git so CI needs no API key.

### Writing a test

```python
@pytest.mark.vcr
def test_basic_call(in_memory_span_exporter):
    my_framework_call(...)

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(spans[0].attributes or {})

    # Pop pattern: test fails loudly if attribute is absent
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(LLM_MODEL_NAME) == "gpt-4o"
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert not attributes  # strict: no unexpected attributes remain
```

### Required test categories

Test location: `tests/openinference/instrumentation/<name>/`

All three categories must be present:

- **Suppress tracing** — call inside `suppress_tracing()` context; assert zero spans produced.
- **Context attribute propagation** — call inside `using_session("id")` context; assert `session.id` is set on the span.
- **Trace config masking** — instrument with `TraceConfig(hide_inputs=True)`; assert `input.value` is absent from span attributes.

Pytest configuration in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

### Debugging with cassettes

To investigate unexpected span output or API response shape:

```bash
# Re-record a single test against the live API (overwrites existing cassette)
pytest tests/ -k test_my_test --vcr-record=all

# Record only cassettes that don't exist yet (safe default)
pytest tests/ -k test_my_test --vcr-record=once

# Run without cassettes (live API, no recording — useful for one-off inspection)
pytest tests/ -k test_my_test --vcr-record=none
```

Cassette files are plain YAML in `tests/cassettes/`. Open one to inspect the exact
request/response body the instrumentor saw — this is the fastest way to understand
why an attribute is missing or has an unexpected value.

---

## Helper Functions

**Check this before writing any attribute-setting or span-creation logic.**
`openinference.instrumentation` exports typed builders for every common span kind and
attribute group. Using them is required — not optional — because they handle edge cases
(mime type inference, safe JSON serialization, correct attribute key names) and their
absence is the #1 review blocker for new instrumentors.

Key helpers (all importable from `openinference.instrumentation`):

| Helper | Use case |
|--------|----------|
| `get_input_attributes(value)` | `input.value` + mime type, inferred or explicit |
| `get_output_attributes(value)` | `output.value` + mime type |
| `get_llm_attributes(...)` | model name, system, provider, messages, tools, token counts |
| `get_embedding_attributes(...)` | model name, embeddings list |
| `get_retriever_attributes(...)` | flattened `retrieval.documents` array |
| `get_reranker_attributes(...)` | input/output documents, query, model, top_k |
| `get_tool_attributes(...)` | name, description, parameters |
| `get_span_kind_attributes(kind)` | `openinference.span.kind` |
| `safe_json_dumps(obj)` | safe serializer for params, metadata, schemas |

For the full list see
`openinference-instrumentation/src/openinference/instrumentation/_attributes.py`.

---

## The Required Features

Every Python instrumentor must implement these features.

All spans must set `openinference.span.kind` (use `get_span_kind_attributes(...)` or the appropriate helper/builder).

### Feature 1 — Suppress Tracing

`BaseInstrumentor` manages the instrumentation lifecycle (whether to call `_instrument()` /
`_uninstrument()`), but it does not inject a suppression check into your wrapper functions.
Each wrapper is invoked once per API call and must check the suppression key before creating
a span for that specific request — the base class never sees those per-request calls.

```python
from opentelemetry import context as context_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY

def patched_function(*args, **kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return original_function(*args, **kwargs)  # skip this request; no span created
    # ... tracing logic ...
```

Implement `_uninstrument()` to reverse all monkey-patching and restore the originals.

### Feature 2 — Context Attribute Propagation

**`OITracer` handles this automatically** when you use `OITracer.start_span()` or
`OITracer.start_as_current_span()` — no extra code needed. You may still pass
`get_attributes_from_context()` explicitly when you defer attribute application (e.g., custom
span wrappers) or need to control attribute timing. Only call it manually when you bypass
`OITracer`'s `start_span`, and avoid double-setting unless you intentionally want overrides.

Available context managers (imported from `openinference.instrumentation`):

| Context Manager                                       | Purpose                                          |
| ----------------------------------------------------- | ------------------------------------------------ |
| `using_session(id)`                                   | Attach a session ID to all spans in the block    |
| `using_user(id)`                                      | Attach a user ID to all spans in the block       |
| `using_metadata({})`                                  | Attach custom metadata to all spans in the block |
| `using_tag([])`                                       | Attach tags to all spans in the block            |
| `using_prompt_template(template, version, variables)` | Attach prompt template info                      |
| `using_attributes(...)`                               | Set multiple context attributes at once          |

### Feature 3 — OITracer (TraceConfig masking)

Wrap the raw OTel tracer with `OITracer` so every span respects the user's `TraceConfig` settings (e.g., masking PII). Pass `tracer_provider` and `config` from `**kwargs` (defaulting to `TraceConfig()`) and use the resulting `OITracer` for all span creation — never the raw OTel tracer.

---

## Creating a New Python Instrumentor

The instrumentor must subclass `BaseInstrumentor` and implement `_instrument()` and `_uninstrument()`.

1. **Copy the canonical package** as a starting point:

   ```bash
   cp -r python/instrumentation/openinference-instrumentation-openai/ \
         python/instrumentation/openinference-instrumentation-<name>/
   ```

2. **Required files**:
   - `pyproject.toml` — package metadata, dependencies, tool config
   - `src/openinference/instrumentation/<name>/__init__.py` — instrumentor class
   - `src/openinference/instrumentation/<name>/version.py` — `__version__ = "0.1.0"` (hatch reads via `pyproject.toml`)
   - `src/openinference/instrumentation/<name>/package.py` — `_instruments` tuple and `_supports_metrics = False`
   - **`pyproject.toml` entry points** (required for auto-discovery by OTel and OpenInference tooling):
     ```toml
     [project.entry-points.opentelemetry_instrumentor]
     <name> = "openinference.instrumentation.<name>:<Name>Instrumentor"

     [project.entry-points.openinference_instrumentor]
     <name> = "openinference.instrumentation.<name>:<Name>Instrumentor"
     ```

3. **Register in `python/tox.ini`** (three sections must be updated):
   - Add to **`envlist`**: `py3{9,14}-ci-{<token>,<token>-latest}`
   - Add to **`changedir`**: `<token>: instrumentation/openinference-instrumentation-<name>/`
   - Add to **`commands_pre`** (standard 4-line pattern — uninstall, reinstall package, smoke-import, install test deps; add a `-latest` line to upgrade the upstream):
   ```ini
   <token>: uv pip install --reinstall-package openinference-instrumentation-<name> .
   # ... (see existing entries in tox.ini for the full pattern)
   ```

4. **Add to `release-please-config.json`** (repo root) so the new package is published to PyPI
   via release-please automation. See root `AGENTS.md` for details.

---

## tox Command Reference

### How tox factors work (critical, non-obvious)

An environment string like `ruff-mypy-test-openai` is a **hyphen-delimited conjunction** of 4 factors: `ruff`, `mypy`, `test`, and `openai`. tox creates one virtual environment named after the concatenation and runs all matching factor actions inside it. You will **not** find `ruff-mypy-test-openai` literally defined in `tox.ini` — it is assembled at runtime from its component factors.

The `-` is a conjunction, not a separator between distinct commands.

### Common commands

```bash
tox run-parallel                     # all CI checks in parallel (all envlist entries)
tox run -e ruff-openai               # format and lint the openai package
tox run -e mypy-openai               # type-check the openai package
tox run -e test-openai               # run tests for the openai package
tox run -e ruff-mypy-test-openai     # all three checks at once for openai
tox run -e ruff-openai,ruff-semconv  # multiple packages in one invocation (comma-separated)
```

Replace `openai` with any package token from the list below.

### Package tokens (use in tox commands)

Token = strip the `openinference-instrumentation-` prefix, then replace **all** remaining hyphens with underscores. Two top-level packages use short tokens:

| Token | Package |
|-------|---------|
| `semconv` | `openinference-semantic-conventions/` |
| `instrumentation` | `openinference-instrumentation/` |

All instrumentors under `instrumentation/` follow the derivation rule:

```
instrumentation/openinference-instrumentation-openai/          → openai
instrumentation/openinference-instrumentation-openai-agents/   → openai_agents
instrumentation/openinference-instrumentation-llama-index/     → llama_index
```

For the full list, see the `changedir` section of `python/tox.ini`.

---

## Additional Patterns

### Pattern 2 — Explicit PII masking at source

Don't rely solely on `TraceConfig` to strip secrets. Use one of three approaches in increasing safety order: **blacklist pop** (remove known-sensitive keys after copying the dict), **redact-list filter** (exclude at collection time with a deny-set), or **whitelist** (only collect explicitly known-safe keys — safest, won't leak new fields added by the upstream library).

### Pattern 3 — Prevent duplicate spans when frameworks nest instrumented clients

When instrumenting an agent framework that itself calls instrumented LLM clients, check
`trace_api.get_current_span()` for an active span whose `OPENINFERENCE_SPAN_KIND` attribute
equals `"LLM"` before creating a new LLM span.

---

## Common Pitfalls

### Pitfall 1 — Truthiness checks on numeric attributes

`if token_count:` silently drops legitimate zero values. Always use `if token_count is not None:`
for any numeric span attribute (token counts, scores, etc.). This is especially common with
cache/detail token fields where zero legitimately means "no cache activity".

### Pitfall 2 — `json.dumps()` instead of `safe_json_dumps()`

`json.dumps()` raises on non-serializable objects. Always use:

```python
from openinference.instrumentation import safe_json_dumps
span.set_attribute(SpanAttributes.LLM_INVOCATION_PARAMETERS, safe_json_dumps(params))
```

### Pitfall 3 — Importing optional dependencies at module level

Never import the instrumented library at module level in any module imported unconditionally (e.g., the package `__init__.py`) — it will crash the instrumentor if the library isn't installed. All target-library imports must be inside `_instrument()` or wrapper functions. The only exception is `TYPE_CHECKING`-guarded imports, which never execute at runtime.

---

## Publishing

**Conda-Forge**: after initial PyPI publication, create a feedstock once using `grayskull pypi <package-name>` to generate `meta.yaml`, then open a PR to `conda-forge/staged-recipes`. Subsequent releases are handled automatically by the conda-forge bot.

---

## See Also

- Root `AGENTS.md` — repository-wide guide, universal instrumentor requirements, release management
- `spec/semantic_conventions.md` — full attribute reference (language-agnostic)
