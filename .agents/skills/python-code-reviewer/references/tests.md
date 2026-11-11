# Tests

The script flags packages with no exhaustive assertion, no context-attribute test, no
error-path test, or a key-shaped secret in a cassette. Those are grep-level checks, so
read the tests to confirm them. Everything else here needs judgment.

## Fixtures

`tests/conftest.py` provides:
- `in_memory_span_exporter` returning `InMemorySpanExporter()`
- `tracer_provider` with a `SimpleSpanProcessor` wired to that exporter
- an autouse `instrument` fixture that calls `Instrumentor().instrument(tracer_provider=...)`,
  clears the exporter, yields, then calls `.uninstrument()` and clears again

A session-scoped instrumentor is fine when it is stateless. A session-scoped exporter is
fine only when every test clears it.

## Cassettes

Tests that hit a provider API use `@pytest.mark.vcr` with cassettes in `tests/cassettes/`.
The `vcr_config` fixture strips sensitive request and response headers and sets
`decode_compressed_response: True` and `record_mode: "once"`. A placeholder such as
`Bearer OPENAI_API_KEY` is fine. A real key in a cassette is Critical, and it needs a key
rotation as well as a commit. Mocks instead of VCR are acceptable; note the difference.

## Exhaustive assertions

Every span test checks every attribute, so an attribute that appears or disappears
unexpectedly fails the test:

```python
attributes = dict(span.attributes or {})
assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value
assert attributes.pop(INPUT_MIME_TYPE) == JSON
assert isinstance(attributes.pop(INPUT_VALUE), str)
# ... pop the rest ...
assert attributes == {}   # or: assert not attributes
```

A test with no final emptiness assertion is High. Reading `span.attributes[KEY]` without
popping is Medium.

## Context attributes

At least one test runs instrumented code inside `using_attributes(session_id=...,
user_id=..., metadata=..., tags=..., prompt_template=..., prompt_template_version=...,
prompt_template_variables=...)` and pops each of those attributes from the span (High if
missing).

## Error path

At least one test makes the wrapped call raise and asserts the span has
`StatusCode.ERROR`, an exception event, and that the original exception reached the caller
(High if missing).

## Streaming and uninstrument

When the library streams, tests cover sync and async streams, a stream the caller abandons
early, and the token counts accumulated from chunks. A test proves that calls after
`uninstrument()` produce no spans. Each gap is High.

## Hierarchy

Multi-span instrumentors assert one `trace_id` per operation and each child's
`parent.span_id` against its parent's `context.span_id` (High if missing).
