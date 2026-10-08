# Instrumentor requirements

Every instrumentor in this repo must meet five requirements (`CLAUDE.md` →
Requirements). Stage 4 uses this file to recognise an issue that touches one,
to name the code that honors it for a stage 6 pointer, and to add the checklist
an issue is missing.

The requirements are defined for each language in:

- Python — `python/DEVELOPMENT.md` → *Minimal Feature Set*, and the
  `python-code-reviewer` skill (§2.4 context propagation tests, §4.4 suppress
  tracing, §4.5 TraceConfig)
- JS — `js/CLAUDE.md` → *Every instrumentor must*
- Java — the `java-code-reviewer` skill (§2 required test coverage, §3
  TraceConfig hide flags, §4 `OITracer`)
- Masking flags — `spec/configuration.md`

Triage reads those documents; it does not run their procedures.

## The five requirements

### 1. Suppress tracing — `c/suppress-tracing`

Tracing can be paused (`suppress_tracing()` context) or stopped
(`uninstrument()`), and a wrapper must emit nothing while suppressed. This is
how Phoenix evals avoid tracing themselves and how nested instrumentors avoid
double spans.

| Language | Honoring it looks like |
| --- | --- |
| Python | `context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY)` checked at the top of every wrapper, returning `wrapped(*args, **kwargs)`; `_uninstrument()` restores every patched attribute |
| JS | `isTracingSuppressed(context.active())` from `@opentelemetry/core` checked before starting a span; `disable()` unpatches |
| Java | `SuppressTracing.isSuppressed()` checked before creating a span (applications wrap calls in `try (Scope ignored = SuppressTracing.begin())`); `OITracer` also honors `TraceConfig.suppressTracing` |

Cues: "spans emitted during evals", "suppress_tracing has no effect",
"uninstrument leaves the patch", "duplicate spans with sentry / logfire /
another instrumentor", "spans after `.uninstrument()`".

### 2. Context attributes — `c/context-attributes`

Attributes attached to the OTel context by the application must land on every
span the instrumentor creates: `session.id`, `user.id`, `metadata`, `tag.tags`,
and the `llm.prompt_template.*` trio.

| Language | Application API | Instrumentor side |
| --- | --- | --- |
| Python | `using_session`, `using_user`, `using_metadata`, `using_tags`, `using_prompt_template`, `using_attributes` (also usable as decorators) | `get_attributes_from_context()` merged into the span's attributes, or `OITracer` doing it |
| JS | `setSession`, `setUser`, `setMetadata`, `setTags`, `setPromptTemplate`, `setAttributes` on a `Context` | `OITracer` reads them; middleware-style packages must call `getAttributesFromContext` |
| Java | `ContextAttributes` helpers | `OITracer` |

Cues: "session id not on spans", "metadata missing from LLM span but present
on chain", "user.id only on the root span", "using_attributes ignored in the
async / streaming path", "decorator detaches before the coroutine runs".

### 3. TraceConfig masking — `c/trace-config`

`TraceConfig` (and the `OPENINFERENCE_HIDE_*` environment variables in
`spec/configuration.md`) lets a deployment keep prompts, completions, images,
embeddings, tool definitions and invocation parameters out of spans, replacing
them with `__REDACTED__`, and caps base64 images at
`OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH` (optionally handing them to a
`BlobUploader`). This is the repo's PII control; an instrumentor that bypasses
it leaks user data.

| Language | Honoring it looks like |
| --- | --- |
| Python | `_instrument()` accepts `config` and builds `OITracer(tracer, config=TraceConfig())`; every span is created through that tracer, never a raw OTel tracer |
| JS | The constructor takes `traceConfig` and wraps in `new OITracer({ tracer, traceConfig })` |
| Java | `OITracer` built with a `TraceConfig`; hide flags such as `hideInputMessages`, `hideOutputText` are respected |

Cues: any `hide_*` / `OPENINFERENCE_HIDE_*` name, "PII", "sensitive",
"redact", "`__REDACTED__`", "prompt still visible with hide_inputs", "image
data URI exceeds…", "BlobUploader", "masking not applied to tool call
arguments / streaming chunks / embeddings".

A bug where content reaches a span **despite** a hide flag is also `security`.

### 4. Semantic conventions — `c/semcov`

Attribute names, span kinds and MIME types come from the
`openinference-semantic-conventions` package for the language, never from
string literals, and match `spec/`. Every span carries
`openinference.span.kind`, and `input.value`/`output.value` carry a MIME type.
LLM spans carry model name, provider, invocation parameters, input and output
messages and token counts where the library exposes them.

Cues: "wrong span kind", "attribute named X should be Y", "tool call arguments
under the wrong key", "`gen_ai.*` not mapped to `llm.*`" (also `c/genai`),
"UNKNOWN span kind", "mime type missing", any proposal for a new attribute.

### 5. Tests — no label

Each instrumentor has a test suite with the shared fixtures (in-memory exporter,
tracer provider, autouse instrument/uninstrument), exhaustive pop-style
attribute assertions, and at least one test each for suppression, context
attributes and `TraceConfig`. Provider calls are recorded — `pytest-recording`
cassettes under `tests/cassettes/` in Python; recorded responses or mocks in JS
and Java.

Triage does not label this dimension. It matters in two places: the stage 6
comment should say whether a fix needs a **new recording against a live
provider** (an agent cannot make one; a newcomer may not be able to), and the
checklist below reminds the implementer.

## Where the code lives

For stage 6 pointers and the already-shipped `Grep`.

| | Python | JS | Java |
| --- | --- | --- | --- |
| Package root | `python/instrumentation/openinference-instrumentation-<name>/` | `js/packages/openinference-instrumentation-<name>/` or `js/packages/openinference-<name>/` | `java/instrumentation/openinference-instrumentation-<name>/` |
| Instrumentor entry | `src/openinference/instrumentation/<name>/__init__.py` (`_instrument`, `_uninstrument`) | `src/instrumentation.ts` (`patch`, `unpatch`) | `src/main/java/com/arize/instrumentation/<name>/` |
| Wrappers / extractors | `_wrappers.py`, `_wrapper.py`, `_handler.py`, `_request_attributes_extractor.py`, `_response_attributes_extractor.py`, `_stream.py`, `_utils.py` (varies by package) | `utils.ts`, `*Attributes.ts`, `responsesAttributes.ts` | listener / interceptor classes |
| Tests | `tests/`, `tests/cassettes/`, `tests/conftest.py` | `test/` | `src/test/` |
| CI entry | `python/tox.ini` (`commands_pre` for the package) | `package.json` scripts, pnpm workspace | `java/settings.gradle`, `build.gradle` |
| Changelog | `CHANGELOG.md` at the package root (release-please) | same | same |
| Core | `python/openinference-instrumentation/src/openinference/instrumentation/` (`config.py` = TraceConfig, `context_attributes.py`, `_tracers.py` = OITracer) | `js/packages/openinference-core/src/trace/` (`contextAttributes.ts`, `trace-config/`) | `java/openinference-instrumentation/src/main/java/com/arize/instrumentation/` (`TraceConfig.java`, `ContextAttributes.java`, `OITracer.java`) |
| Conventions | `python/openinference-semantic-conventions/` | `js/packages/openinference-semantic-conventions/` | `java/openinference-semantic-conventions/` |
| Spec | `spec/semantic_conventions.md`, `llm_spans.md`, `tool_calling.md`, `embedding_spans.md`, `multimodal_attributes.md`, `configuration.md`, `annotations.md` | | |

## The Requirements checklist

Stage 4(b) adds this to the triage block of an `enhancement` or
`new instrumentation` issue that adds spans, attributes or an instrumentor
without saying how it will meet the requirements. Keep the reporter's text
untouched; this goes inside the `<!-- triage:begin -->` block only.

Include only the rows that apply. Drop the language columns the issue does not
cover (stage 3 decided). For a **new instrumentor** include every row; for a
**new attribute on an existing instrumentor** the last three rows usually
suffice, plus the masking row if the attribute could carry user content.

```markdown
**Requirements** — every instrumentor must (see `CLAUDE.md` → Requirements):

- [ ] **Suppress tracing** — no spans while `suppress_tracing()` is active; `uninstrument()` removes the patch
- [ ] **Context attributes** — `session.id`, `user.id`, `metadata`, `tag.tags`, prompt template from `using_attributes` (Python) / `setSession` etc. (JS) appear on the new spans
- [ ] **TraceConfig masking** — new content attributes go through `OITracer` so `hide_inputs` / `hide_outputs` / `OPENINFERENCE_HIDE_*` redact them (`spec/configuration.md`)
- [ ] **Semantic conventions** — attribute names and span kind from the `openinference-semantic-conventions` package, MIME types set on `input.value` / `output.value`
- [ ] **Tests** — exhaustive attribute assertions, plus a case each for suppression, context attributes and masking; recorded fixtures under `tests/cassettes/` (Python)
```

Add a one-line pointer to the closest existing instrumentor when you know it
("Pattern: `openinference-instrumentation-groq` does the same for reasoning
tokens"). Do not add design guidance beyond that — the checklist is a reminder,
not a spec.
