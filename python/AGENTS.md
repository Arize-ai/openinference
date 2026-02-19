# Python Workspace Guide

## Setup

```bash
cd python
pip install tox-uv==1.11.2
pip install -r dev-requirements.txt
tox run -e add_symlinks              # compose namespace package (required before any imports work)
pip install -e openinference-instrumentation  # editable install for development
```

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

| Token               | Package directory                                                  |
| ------------------- | ------------------------------------------------------------------ |
| `semconv`           | `openinference-semantic-conventions/`                              |
| `instrumentation`   | `openinference-instrumentation/`                                   |
| `openai`            | `instrumentation/openinference-instrumentation-openai/`            |
| `openai_agents`     | `instrumentation/openinference-instrumentation-openai-agents/`     |
| `anthropic`         | `instrumentation/openinference-instrumentation-anthropic/`         |
| `bedrock`           | `instrumentation/openinference-instrumentation-bedrock/`           |
| `mistralai`         | `instrumentation/openinference-instrumentation-mistralai/`         |
| `groq`              | `instrumentation/openinference-instrumentation-groq/`              |
| `litellm`           | `instrumentation/openinference-instrumentation-litellm/`           |
| `langchain`         | `instrumentation/openinference-instrumentation-langchain/`         |
| `llama_index`       | `instrumentation/openinference-instrumentation-llama-index/`       |
| `dspy`              | `instrumentation/openinference-instrumentation-dspy/`              |
| `instructor`        | `instrumentation/openinference-instrumentation-instructor/`        |
| `crewai`            | `instrumentation/openinference-instrumentation-crewai/`            |
| `haystack`          | `instrumentation/openinference-instrumentation-haystack/`          |
| `vertexai`          | `instrumentation/openinference-instrumentation-vertexai/`          |
| `smolagents`        | `instrumentation/openinference-instrumentation-smolagents/`        |
| `autogen`           | `instrumentation/openinference-instrumentation-autogen/`           |
| `autogen_agentchat` | `instrumentation/openinference-instrumentation-autogen-agentchat/` |
| `beeai`             | `instrumentation/openinference-instrumentation-beeai/`             |
| `portkey`           | `instrumentation/openinference-instrumentation-portkey/`           |
| `mcp`               | `instrumentation/openinference-instrumentation-mcp/`               |
| `google_genai`      | `instrumentation/openinference-instrumentation-google-genai/`      |
| `google_adk`        | `instrumentation/openinference-instrumentation-google-adk/`        |
| `pydantic_ai`       | `instrumentation/openinference-instrumentation-pydantic-ai/`       |
| `openllmetry`       | `instrumentation/openinference-instrumentation-openllmetry/`       |
| `openlit`           | `instrumentation/openinference-instrumentation-openlit/`           |
| `strands_agents`    | `instrumentation/openinference-instrumentation-strands-agents/`    |
| `pipecat`           | `instrumentation/openinference-instrumentation-pipecat/`           |
| `agent_framework`   | `instrumentation/openinference-instrumentation-agent-framework/`   |
| `agno`              | `instrumentation/openinference-instrumentation-agno/`              |
| `guardrails`        | `instrumentation/openinference-instrumentation-guardrails/`        |
| `agentspec`         | `instrumentation/openinference-instrumentation-agentspec/`         |

---

## The Three Required Features

Every Python instrumentor must implement these three features.

### Feature 1 — Suppress Tracing

Check the OTel context key before creating any span. When suppressed, call through to the original function without tracing.

```python
from opentelemetry import context as context_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY

def patched_function(*args, **kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return original_function(*args, **kwargs)  # skip tracing
    # ... tracing logic ...
```

To permanently disable tracing, implement `_uninstrument()` to reverse all monkey-patching.

### Feature 2 — Context Attribute Propagation

Read session ID, user ID, metadata, and tags from OTel context and attach them to spans.

```python
from opentelemetry.trace import Tracer
from openinference.instrumentation import get_attributes_from_context

# Pass at span creation time:
span = tracer.start_span(
    name="my-span",
    attributes=dict(get_attributes_from_context()),
)

# Or set after span creation:
with tracer.start_as_current_span(name="my-span") as span:
    span.set_attributes(dict(get_attributes_from_context()))
```

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

Wrap the raw OTel tracer with `OITracer` so every span respects the user's `TraceConfig` settings (e.g., masking PII).

```python
from openinference.instrumentation import OITracer, TraceConfig
import opentelemetry.trace as trace_api

def _instrument(self, **kwargs: Any) -> None:
    tracer_provider = kwargs.get("tracer_provider") or trace_api.get_tracer_provider()
    config = kwargs.get("config") or TraceConfig()
    assert isinstance(config, TraceConfig)
    tracer = OITracer(
        trace_api.get_tracer(__name__, __version__, tracer_provider),
        config=config,
    )
    # Use `tracer` (not the raw OTel tracer) for all span creation
```

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
   - `src/openinference/instrumentation/<name>/version.py` — `__version__` string
   - `src/openinference/instrumentation/<name>/package.py` — package name constant

3. **Register in `python/tox.ini`**:
   - Add `<token>: instrumentation/openinference-instrumentation-<name>/` under `changedir`
   - Add `<token>: uv pip install ...` lines under `commands_pre`

---

## Testing Patterns

Test location: `tests/openinference/instrumentation/<name>/`

Required test categories (all three must be present):

```python
# 1. Suppress tracing test
def test_suppress_tracing(instrumentor, tracer_provider):
    with suppress_tracing():
        result = my_framework_call()
    assert len(get_spans()) == 0  # no spans created

# 2. Context attribute propagation test
def test_context_attributes(instrumentor, tracer_provider):
    with using_session("test-session-id"):
        result = my_framework_call()
    spans = get_spans()
    assert spans[0].attributes["session.id"] == "test-session-id"

# 3. Trace configuration masking test
def test_trace_config_masking(tracer_provider):
    config = TraceConfig(hide_inputs=True)
    instrumentor = MyFrameworkInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider, config=config)
    result = my_framework_call(input="sensitive data")
    spans = get_spans()
    assert "input.value" not in spans[0].attributes
```

Pytest configuration in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

---

## Publishing

**Conda-Forge**: after initial PyPI publication, create a feedstock once using `grayskull pypi <package-name>` to generate `meta.yaml`, then open a PR to `conda-forge/staged-recipes`. Subsequent releases are handled automatically by the conda-forge bot.

---

## Preferred Patterns

### Pattern 1 — TypedDict + TypeGuard for structured extraction

When reading token counts or usage metadata from untyped dicts, define a `TypedDict` for the expected shape and a `TypeGuard` function for validation. This separates validation from extraction and enables mypy strictness.

```python
from typing import TypedDict
from typing_extensions import TypeGuard

class UsageDict(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

def is_usage_dict(obj: object) -> TypeGuard[UsageDict]:
    return (
        isinstance(obj, dict)
        and isinstance(obj.get("prompt_tokens"), int)
        and isinstance(obj.get("completion_tokens"), int)
        and isinstance(obj.get("total_tokens"), int)
    )

if is_usage_dict(usage):
    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, usage["prompt_tokens"])
```

### Pattern 2 — Graceful serialization fallback chain

For Pydantic/framework objects, never let serialization crash a span. Use a fallback chain:

```python
def _serialize(obj: Any) -> str:
    if hasattr(obj, "model_dump_json"):
        try:
            return obj.model_dump_json(exclude_unset=True)
        except Exception:
            pass
    if hasattr(obj, "model_dump"):
        return safe_json_dumps(obj.model_dump())
    if hasattr(obj, "dict"):
        return safe_json_dumps(obj.dict())
    return safe_json_dumps(obj)
```

### Pattern 3 — Explicit PII masking at source

Don't rely solely on `TraceConfig` to strip secrets. Three approaches, in increasing safety order:

```python
# Blacklist pop — remove known-sensitive keys (DSPy style)
params = request_params.copy()
params.pop("api_key", None)

# Redact-list filter — exclude at collection time (LiteLLM style)
params = {k: v for k, v in kwargs.items() if k not in {"api_key", "messages"}}

# Whitelist — only collect known-safe keys; won't leak new fields (Anthropic style, safest)
SAFE_PARAMS = {"max_tokens", "model", "temperature", "stream", "top_k", "top_p"}
params = {k: v for k, v in kwargs.items() if k in SAFE_PARAMS}

span.set_attribute(SpanAttributes.LLM_INVOCATION_PARAMETERS, safe_json_dumps(params))
```

### Pattern 4 — Prevent duplicate spans when frameworks nest instrumented clients

When instrumenting an agent framework that itself calls instrumented LLM clients, check for an active LLM parent span before creating a new one:

```python
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan
from openinference.semconv.trace import SpanAttributes

OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND

def _has_active_llm_parent_span() -> bool:
    span = trace_api.get_current_span()
    return (
        span.get_span_context().is_valid
        and span.is_recording()
        and isinstance(span, ReadableSpan)
        and (span.attributes or {}).get(OPENINFERENCE_SPAN_KIND) == "LLM"
    )
```

### Pattern 5 — Parametrized tests covering multiple input shapes

Use `@pytest.mark.parametrize` to cover dict inputs, Pydantic model inputs, and framework-specific message types in the same test. Reviewers consistently flag missing input-variant coverage.

```python
@pytest.mark.parametrize("messages", [
    [{"role": "user", "content": "hello"}],          # plain dict
    [UserMessage(content="hello")],                   # framework type
    [ChatCompletionUserMessageParam(role="user", content="hello")],  # typed param
])
def test_input_messages(instrumentor, tracer_provider, messages):
    my_framework_call(messages=messages)
    spans = get_spans()
    assert spans[0].attributes["llm.input_messages.0.message.role"] == "user"
```

### Pattern 6 — `assert_never` for exhaustive union matching

At the end of a discriminated-union dispatch, add `assert_never` so mypy flags unhandled members when a library adds new types. Used throughout the OpenAI, OpenAI Agents, LlamaIndex, Haystack, and Bedrock instrumentors.

**Variant A — `TYPE_CHECKING`-only** (preferred for external SDK types; zero runtime cost, mypy still enforces exhaustion):

```python
from typing import TYPE_CHECKING
from typing_extensions import assert_never

for item in content:
    if item["type"] == "input_text":
        yield from _get_text_attributes(item, prefix)
    elif item["type"] == "input_image":
        yield from _get_image_attributes(item, prefix)
    elif TYPE_CHECKING:
        assert_never(item["type"])  # never runs; mypy catches new union members
```

**Variant B — runtime assertion** (for internal enums you control; also raises `AssertionError` at runtime):

```python
else:
    assert_never(component_type)
```

---

## Common Pitfalls

### Pitfall 1 — Truthiness checks on numeric attributes

`if token_count:` silently drops legitimate zero values. Always use an explicit `None` check for any numeric span attribute (token counts, scores, etc.). This is especially common with cache/detail token fields where zero legitimately means "no cache activity".

```python
# Wrong — drops zero values (seen in LangChain cache token handling)
if cache_creation_input_tokens:
    yield LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE, cache_creation_input_tokens

# Correct
if cache_creation_input_tokens is not None:
    yield LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE, cache_creation_input_tokens
```

### Pitfall 2 — `json.dumps()` instead of `safe_json_dumps()`

`json.dumps()` raises on non-serializable objects (custom types, Pydantic models with non-JSON fields). Always use `safe_json_dumps` from `openinference.instrumentation` for all attribute serialization.

```python
# Wrong — raises TypeError on Pydantic models or custom objects
span.set_attribute(SpanAttributes.LLM_INVOCATION_PARAMETERS, json.dumps(params))

# Correct
from openinference.instrumentation import safe_json_dumps
span.set_attribute(SpanAttributes.LLM_INVOCATION_PARAMETERS, safe_json_dumps(params))
```

### Pitfall 3 — Importing optional dependencies at module level

A top-level `import my_framework` crashes the instrumentor module on load if the library isn't installed. Always defer to inside `_instrument()`:

```python
# Wrong — top-level import crashes on load
import my_framework

# Correct — deferred inside _instrument
def _instrument(self, **kwargs: Any) -> None:
    try:
        import my_framework
    except ImportError as e:
        raise RuntimeError("my-framework must be installed") from e
```

`TYPE_CHECKING`-guarded imports (for type annotations only) are fine at module level.

---

## See Also

- Root `AGENTS.md` — repository-wide guide, universal instrumentor requirements, release management
- `spec/AGENTS.md` — full semantic conventions attribute reference (language-agnostic)
