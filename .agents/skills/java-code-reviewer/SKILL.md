---
name: java-code-reviewer
description: >
  Review Java OpenInference instrumentation code for correctness and completeness.
  Use this skill when reviewing a Java instrumentor package — whether it's a new
  instrumentor, a PR that modifies one, or when the user asks to audit/review/check
  an existing instrumentor's code quality. Trigger on phrases like "review the
  instrumentor", "check the Java code", "audit the package", "is this instrumentor correct",
  or any request to validate an OpenInference Java instrumentation package against
  project standards.
invocable: true
---

# Java Code Reviewer for OpenInference Instrumentors

Review a Java OpenInference instrumentation package against the project's established
patterns and conventions. Report findings with file paths and line numbers, organized
by severity (Critical / High / Medium / Low).

## Workflow

**Step 1: Identify the package to review**
- Ask the user which instrumentor to review if not already clear from context
- The package lives under `java/instrumentation/openinference-instrumentation-<name>/`
- Read the instrumentor source, `build.gradle`, and `src/test/` directory

**Step 2: Use the instrumented library source as ground truth**

Before flagging any finding, verify it against the actual library code. Do NOT assume
how the instrumented library works — read it. Do NOT present findings without having
read the library source first.

- Find the library version from `java/build.gradle` `ext` block
- Check `~/.gradle/caches/modules-2/files-2.1/` for cached sources
- If not cached, download the sources jar from Maven Central (`repo1.maven.org`).
  Some libraries split across multiple artifacts — check `build.gradle` dependency
  declarations and fetch all relevant ones.
- If you cannot obtain the source through any means, explicitly tell the user you
  were unable to verify against the library source before presenting findings.
- Calibrate severity by what the library actually does: a bug on a common code path is
  High/Critical; an edge case for a type that can't appear at runtime is Low

**Step 3: Run all review sections below**

**Step 4: Present findings** in a severity table, list what's working well, then ask
the user: fix issues, run tests (`./gradlew :instrumentation:...:test`), or done.

---

## Section 1: Gradle Setup

Read the instrumentor's `build.gradle` and the root `java/build.gradle`.

- Instrumented library must be `compileOnly` (not `implementation`) — **High**
- `openinference-instrumentation` must be `api`
- Version constants should be in root `ext` block, not hardcoded — **Medium**
- Module must be in `java/settings.gradle` — **Critical** if missing
- Run `cd java && ./gradlew spotlessCheck` (Palantir Java Format)

---

## Section 2: Testing Patterns

### Exhaustive attribute assertions

This is the most important testing pattern. Tests must verify ALL span attributes, not
spot-check a few. The remove-and-verify pattern catches both unexpected additions and
silent removals:

```java
Map<String, Object> attributes = new HashMap<>();
span.getAttributes().forEach((key, value) -> attributes.put(key.getKey(), value));

assertThat(attributes.remove("openinference.span.kind")).isEqualTo("LLM");
assertThat(attributes.remove("llm.model_name")).isEqualTo("gpt-4");
// ... remove and assert all remaining attributes ...
assertThat(attributes).isEmpty();  // Nothing unexpected left
```

Missing emptiness check — **High**.

### Other required test coverage

- Error handling: exception -> span has `StatusCode.ERROR` + recorded exception — **High**
- Context attribute propagation: session_id, user_id, metadata, tags
- TraceConfig masking: verify `hideInputMessages` etc. actually suppress attributes
- Missing test files entirely — **Critical**

---

## Section 3: OpenInference Semantic Conventions

Read `SemanticConventions.java` for the full attribute catalog:
`java/openinference-semantic-conventions/src/main/java/com/arize/semconv/trace/SemanticConventions.java`

Also read the spec files under `spec/` (`semantic_conventions.md`, `traces.md`,
`llm_spans.md`, `embedding_spans.md`, `tool_calling.md`) for expected behavior.

For the library type being reviewed, verify the instrumentor sets all applicable
attributes. Key checks:

- Every span needs `OPENINFERENCE_SPAN_KIND`, `INPUT_VALUE` + `INPUT_MIME_TYPE`,
  `OUTPUT_VALUE` + `OUTPUT_MIME_TYPE`. Missing MIME type when value is set — **High**
- All attributes should use constants from `SemanticConventions`, not hardcoded strings — **Medium**
- Read `TraceConfig.java` for the full list of hide flags; verify each is respected
  where applicable — **Medium** if missing

---

## Section 4: Span Lifecycle and Hierarchy

- `span.end()` must ALWAYS be called — **Critical** if missing
- `Scope` from `context.makeCurrent()` must be closed (try-with-resources) — **High**
- For multi-span instrumentors: verify parent-child nesting and shared `traceId`
- The instrumentor should use `OITracer` (not raw `Tracer`) to get TraceConfig support

---

## Presenting Results

Organize findings into a table:

| Severity | Section | Finding | Location |
|----------|---------|---------|----------|
| Critical | 4 | span.end() not called in error path | `SomeListener.java:142` |
| High | 2 | Tests don't verify all span attributes | `SomeTest.java:85` |
| ... | ... | ... | ... |

Then list what's working well — positive findings help the user understand what doesn't
need to change.
