# Runtime behavior

The script catches suppression misuse and a package that never mentions suppression,
`TraceConfig`, or context attributes. Whether every wrapper honors them is judgment, so
read each wrapper.

## Suppression

Spans started through `OITracer` are suppressed for you: `OITracer.start_span` returns an
invalid span while suppression is on. A wrapper that starts spans another way, or does
other work before the span, checks suppression first and calls straight through:

```python
if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):   # common in this repo
    return wrapped(*args, **kwargs)

from opentelemetry.instrumentation.utils import is_instrumentation_enabled   # public API
if not is_instrumentation_enabled():
    return wrapped(*args, **kwargs)
```

`opentelemetry.instrumentation.utils.suppress_instrumentation()` is a context manager that
turns suppression on. `if suppress_instrumentation():` is always true and skips tracing on
every call (Critical). A wrapper with no check is High. Prove suppression with
`phoenix-verify`: one traced call and one suppressed call in a run must export one span.

## TraceConfig

The instrumentor accepts `config: TraceConfig` in `_instrument`, passes it to `OITracer`,
and so honors `hide_inputs`, `hide_outputs`, and the other flags in
`spec/configuration.md`. Missing support is High, since `CLAUDE.md` requires it of every
instrumentor. Attributes set outside `OITracer` must be masked by hand.

## Context attributes

Values from `using_attributes()` (session, user, metadata, tags, prompt template) reach every
span. `OITracer` adds them to spans it starts. Code that starts spans another way must add
`get_attributes_from_context()` itself. Missing propagation is High, also a `CLAUDE.md`
requirement.

## Errors

When the wrapped call raises, the span records the exception, sets status `ERROR`, ends,
and the original exception reaches the caller unchanged. A missing status or exception
event is High. Attribute extraction must never raise into user code: it catches and logs
its own errors, and the wrapped result still returns. Extraction that can raise into the
caller is Critical.

## Streaming

When the library returns a sync or async stream:
- the span stays open until the stream is exhausted, raises, or is closed, and ends once
- output messages and token counts accumulate from all chunks
- the returned proxy still behaves like the library's stream (context manager, `close()`,
  attribute access); compare with `openinference-instrumentation-openai`'s `_stream.py`

A span that never ends, or ends early, is Critical. Check what happens when the caller
breaks out of the loop.

## Uninstrument

`_uninstrument()` restores every patched attribute and unregisters every processor,
callback, or hook it registered with the library, restoring whatever an exclusive
registration replaced. Calls after `uninstrument()` produce no spans. A missing restore
or unregister is High.

## In-flight work across uninstrument

A run that is in flight when `uninstrument()` is called either finishes with complete
spans or stops producing them; it never exports half-built spans or orphan roots. An
`uninstrument()` followed by `instrument()` mid-run exports every span once, in one trace.
Orphan roots or duplicate spans are High. Prove the claim with `phoenix-verify` in
before/after mode.

## Threads and async

When the library hands work to a `ThreadPoolExecutor` or similar, the OTel context must
cross the boundary (`contextvars.copy_context()`), or child spans become orphan roots
(Critical). Async spans must parent to the caller's active span. Prove a parenting claim
with `phoenix-verify` and its span tree view.
