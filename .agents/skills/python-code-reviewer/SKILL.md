---
name: python-code-reviewer
description: >
  Review Python OpenInference instrumentation code for correctness and completeness.
  Use this skill when reviewing a Python instrumentor package — whether it's a new
  instrumentor, a PR that modifies one, or when the user asks to audit/review/check
  an existing instrumentor's code quality. Trigger on phrases like "review the
  instrumentor", "check the code", "audit the package", "is this instrumentor correct",
  or any request to validate an OpenInference Python instrumentation package against
  project standards.
invocable: true
---

# Python Code Reviewer for OpenInference Instrumentors

Review a Python OpenInference instrumentation package against the project's established
patterns and conventions. This is a checklist-driven review — go through each section,
report findings with file paths and line numbers, and surface issues organized by severity.

## Workflow

**Step 1: Identify the package to review**
- Ask the user which instrumentor to review if not already clear from context
- The package lives under `python/instrumentation/openinference-instrumentation-<name>/`
- Read the key files: `__init__.py`, `_wrappers.py` (or equivalent), `pyproject.toml`,
  and the full `tests/` directory

**Step 1.5: Pull the instrumented library source and use it as ground truth**

OpenInference instrumentors work by monkey-patching functions in the library they
instrument. All correctness judgments — whether wrappers target the right methods, handle
the right signatures, process the right data structures, and cover the right edge cases —
must be verified against the actual library source code. Do NOT make assumptions about
how the instrumented library works.

1. **Set up the tox environment** to install the pinned library version:
   ```bash
   cd python && uvx --with tox-uv tox run -e py314-ci-<pkg> -- --co -q
   ```
   (`-- --co -q` tells pytest to collect without running, which triggers the install.)
   If the `.tox` env already exists, skip this step.

2. **Locate the installed library source** at:
   ```
   python/.tox/py314-ci-<pkg>/lib/python3.14/site-packages/<library>/
   ```

3. **Reference the library source throughout the review.** Before flagging any finding,
   verify it against the actual code:
   - Are the monkey-patched methods/classes correct? Check they exist and have the
     expected signatures.
   - Are parameter types handled correctly? Read the real type annotations and defaults.
   - Are edge cases real? Check whether a supposed edge case can actually occur given
     the library's actual types, validation, and control flow.
   - Are attribute extractions correct? Verify field names, nesting, and optional vs.
     required fields against the library's actual data classes.

4. **Calibrate severity based on what the library actually does:**
   - A bug affecting types/paths the library actually uses → **High** or **Critical**
   - An edge case for a type that can't actually appear at runtime → **Low**
   - A missing handler for a type in the library's Union that is common → higher severity
   - A missing handler for a rare/internal type → lower severity

**Step 2: Run all four review sections below**

**Step 3: Present findings** organized by severity:
- **Critical**: Will cause incorrect behavior or CI failure
- **High**: Missing required convention or test coverage gap
- **Medium**: Deviates from established patterns but functional
- **Low**: Style or minor improvement suggestions

---

## Section 1: Test Setup and CI Config

The tox.ini install pattern matters because a broken pattern silently installs the wrong
version of the library, making the "pinned version" test target useless.

### 1.1 tox.ini install pattern

Read `python/tox.ini` and find the `commands_pre` entries for this package.

**Correct pattern** (google_adk style — 4 steps):
```
pkg: uv pip uninstall -r test-requirements.txt
pkg: uv pip install --reinstall-package openinference-instrumentation-pkg .
pkg: python -c 'import openinference.instrumentation.pkg'
pkg: uv pip install -r test-requirements.txt
```

**Broken pattern** (causes under-resolution — the pinned version test may silently test
the wrong version):
```
pkg: uv pip install --reinstall {toxinidir}/instrumentation/openinference-instrumentation-pkg[test]
```

Flag the broken pattern as **Critical** — it defeats the purpose of version-pinned testing.

### 1.2 test-requirements.txt

Check that `test-requirements.txt` exists in the package root and contains:
- A **pinned version** of the library being instrumented (e.g., `openai==2.8.0`)
- `opentelemetry-sdk`
- `pytest-recording` (if using VCR cassettes)
- Any other test utilities needed (pytest-asyncio, respx, responses, etc.)

If `test-requirements.txt` is missing entirely, flag as **Critical** (the correct tox
pattern depends on it).

### 1.3 Latest test target

Verify that the tox envlist has both pinned and `-latest` variants:
```
py3{10,14}-ci-{pkg,pkg-latest}
```

And that the `-latest` variant upgrades the library:
```
pkg-latest: uv pip install -U <library-name>
```

---

## Section 2: Testing Patterns

### 2.1 conftest.py fixtures

Read `tests/conftest.py` and verify these fixtures exist:

**Required fixtures:**
- `in_memory_span_exporter` — returns `InMemorySpanExporter()`
- `tracer_provider` — creates `TracerProvider` with `SimpleSpanProcessor` wired to the exporter
- `instrument` (autouse) — calls `Instrumentor().instrument(tracer_provider=...)`,
  clears exporter, yields, then calls `.uninstrument()` and clears again

**Scope considerations:**
- Session-scoped instrumentor fixtures are fine when the instrumentor is stateless
- Function-scoped exporter+provider is safer for test isolation but session-scoped works
  if tests clear the exporter properly

**VCR config fixture** (if using cassettes):
```python
@pytest.fixture(scope="session")
def vcr_config() -> dict[str, Any]:
    return {
        "before_record_request": _strip_request_headers,
        "before_record_response": _strip_response_headers,
        "decode_compressed_response": True,
        "record_mode": "once",
    }
```
With helper functions that strip sensitive headers from recorded cassettes.

### 2.2 pytest-recording / VCR cassettes

If the instrumentor calls external APIs (LLM providers, embedding services, etc.):
- Tests should use `@pytest.mark.vcr` decorator
- Cassettes should live in `tests/cassettes/` (pytest-recording default)
- Cassette YAML files should have headers stripped (no API keys recorded)
- `pytest-recording` should be in `test-requirements.txt`

If tests use mocking instead of VCR, that's acceptable but note it as a pattern difference.

### 2.3 Exhaustive attribute assertions (pop-style)

This is the most important testing pattern. Tests should verify ALL span attributes, not
just spot-check a few. The pattern prevents regressions where unexpected attributes appear
or expected ones disappear silently.

**Correct pattern:**
```python
attributes = dict(span.attributes or {})
assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.CHAIN.value
assert attributes.pop(INPUT_VALUE)
assert attributes.pop(INPUT_MIME_TYPE) == JSON
assert attributes.pop(OUTPUT_VALUE)
assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
# ... pop all remaining attributes ...
assert not attributes  # Nothing unexpected left
```

**What to flag:**
- Tests that only check a few attributes without the final `assert not attributes` — **High**
- Tests that use `span.attributes[KEY]` or `span.attributes.get(KEY)` instead of pop — **Medium**
  (functional but doesn't catch unexpected extras)
- Missing `assert not attributes` at the end — **High**

### 2.4 Context attribute propagation tests

There should be at least one test that uses `using_attributes()` context manager and
verifies that context attributes appear on spans:
```python
with using_attributes(
    session_id="test-session",
    user_id="test-user",
    metadata={"key": "value"},
    tags=["tag-1", "tag-2"],
    prompt_template="template {var}",
    prompt_template_version="v1.0",
    prompt_template_variables={"var": "value"},
):
    # run instrumented code
```

Then verify these attributes appear on the spans via pop assertions.

---

## Section 3: OpenInference Semantic Conventions

Check which conventions apply based on the type of library being instrumented. Not every
instrumentor needs every attribute — match the conventions to what the library actually does.

### 3.1 Always required

Every span must have:
- `OPENINFERENCE_SPAN_KIND` — set to the appropriate kind enum value
- `INPUT_VALUE` + `INPUT_MIME_TYPE` — what went into the operation
- `OUTPUT_VALUE` + `OUTPUT_MIME_TYPE` — what came out

MIME types should be `application/json` for structured data (dicts, Pydantic models) and
`text/plain` for strings. Flag if MIME type is missing when value is set — **High**.

When setting input/output attributes, the instrumentor should use:
```python
from openinference.instrumentation import get_input_attributes, get_output_attributes
span.set_attributes(dict(get_input_attributes(val, mime_type=OpenInferenceMimeTypeValues.JSON)))
```

### 3.2 LLM libraries (OpenAI, Anthropic, Mistral, LiteLLM, etc.)

These should set:
- `LLM_MODEL_NAME` — the model identifier
- `LLM_PROVIDER` — the provider name (e.g., "openai", "anthropic")
- `LLM_INVOCATION_PARAMETERS` — JSON of parameters like temperature, max_tokens
- `LLM_INPUT_MESSAGES` — array of message objects with role and content
- `LLM_OUTPUT_MESSAGES` — array of response message objects
- `LLM_TOKEN_COUNT_PROMPT` / `LLM_TOKEN_COUNT_COMPLETION` / `LLM_TOKEN_COUNT_TOTAL` — token usage

Span kind should be `LLM`.

### 3.3 Embedding libraries

- `EMBEDDING_MODEL_NAME`
- `EMBEDDING_EMBEDDINGS` — the embedding vectors (unless masked by TraceConfig)
- `EMBEDDING_TEXT` — input text

Span kind should be `EMBEDDING`.

### 3.4 Tool/function calling

When the library supports tool use or function calling:
- `TOOL_NAME` — name of the tool/function
- `TOOL_DESCRIPTION` — description (if available)
- `TOOL_PARAMETERS` — JSON schema of parameters

Span kind should be `TOOL`.

### 3.5 Agent/orchestration frameworks (CrewAI, LangChain, DSPy, etc.)

These typically produce multiple span kinds in a hierarchy:
- `CHAIN` for orchestration/workflow spans
- `AGENT` for agent execution spans
- `TOOL` for tool invocations
- `LLM` for underlying model calls

### 3.6 Retrieval libraries

- `RETRIEVAL_DOCUMENTS` — array of retrieved documents
- Document attributes: `DOCUMENT_ID`, `DOCUMENT_CONTENT`, `DOCUMENT_SCORE`, `DOCUMENT_METADATA`

Span kind should be `RETRIEVER`.

---

## Section 4: Span Hierarchy

### 4.1 Parent-child relationships

For instrumentors that create multiple spans, verify:
- Spans nest correctly (child span's `parent.span_id` matches parent span's `context.span_id`)
- All spans from a single operation share the same `trace_id`
- No orphaned root spans that should be children

Tests should verify hierarchy explicitly:
```python
trace_ids = {span.context.trace_id for span in spans}
assert len(trace_ids) == 1  # All in one trace

assert child_span.parent.span_id == parent_span.context.span_id
```

Flag missing hierarchy tests as **High** for multi-span instrumentors.

### 4.2 Correct span kinds in hierarchy

Common correct hierarchies:
- `CHAIN -> LLM` (simple chain with LLM call)
- `CHAIN -> AGENT -> TOOL` (agent framework)
- `CHAIN -> AGENT -> LLM` (agent making LLM calls)
- `CHAIN -> RETRIEVER -> EMBEDDING` (RAG pipeline)
- `CHAIN -> CHAIN -> LLM` (nested chains)

### 4.3 Thread/async context propagation

If the instrumented library uses threads or async:
- Verify that OTel context is properly propagated across thread boundaries
  (using `contextvars.copy_context()` if needed)
- For async code, ensure spans created in async functions are properly parented
- Flag if the library is known to use `ThreadPoolExecutor` or similar and the
  instrumentor doesn't handle context propagation — **Critical**

### 4.4 Suppress tracing support

Every wrapper should check suppression at the start:
```python
if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
    return wrapped(*args, **kwargs)
```

Missing suppression check — **High**.

### 4.5 TraceConfig masking support

The instrumentor should accept and respect `TraceConfig`:
- Pass it to `OITracer` or use it to mask attributes before setting them
- At minimum, `hide_inputs` and `hide_outputs` should work

Missing TraceConfig support — **Medium** (functional but incomplete).

---

## Presenting Results

Organize findings into a table:

| Severity | Section | Finding | Location |
|----------|---------|---------|----------|
| Critical | 1.1 | Uses broken tox install pattern | `python/tox.ini:142` |
| High | 2.3 | Tests don't use exhaustive pop assertions | `tests/test_instrumentor.py:85` |
| ... | ... | ... | ... |

Then list what's working well — positive findings help the user understand what doesn't
need to change.

Finally, ask the user what they'd like to do:
- "Fix the issues" — generate patches
- "Run the tests" — execute `tox run -e test-<pkg>`
- "Just reviewing" — done
