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
patterns and conventions. This is a checklist-driven review — go through each section,
report findings with file paths and line numbers, and surface issues organized by severity.

## Workflow

**Step 1: Identify the package to review**
- Ask the user which instrumentor to review if not already clear from context
- The package lives under `java/instrumentation/openinference-instrumentation-<name>/`
- Read the key files: `*Instrumentor.java`, `*Listener.java` (or equivalent),
  `build.gradle`, and the full `src/test/` directory

**Step 2: Pull the instrumented library source and use it as ground truth**

OpenInference instrumentors work by hooking into the instrumented library's listener or
observation interfaces. All correctness judgments — whether listeners handle the right
event types, process the right data structures, and cover the right edge cases — must be
verified against the actual library source code. Do NOT make assumptions about how the
instrumented library works.

1. **Check the build.gradle dependencies** to find the exact library version. The root
   `java/build.gradle` defines version constants in the `ext` block (e.g.,
   `langchain4jVersion = '1.0.0'`).

2. **Locate the library source** from the Gradle cache:
   ```
   ~/.gradle/caches/modules-2/files-2.1/<group>/<artifact>/<version>/
   ```
   Or use the library's GitHub source at the pinned version tag.

3. **Reference the library source throughout the review.** Before flagging any finding,
   verify it against the actual code:
   - Are the listener/handler interfaces implemented correctly? Check method signatures.
   - Are parameter types handled correctly? Read the real type annotations.
   - Are edge cases real? Check whether a supposed edge case can actually occur given
     the library's actual types, validation, and control flow.
   - Are attribute extractions correct? Verify field names, nesting, and optional vs.
     required fields against the library's actual data classes.

4. **Calibrate severity based on what the library actually does:**
   - A bug affecting types/paths the library actually uses -> **High** or **Critical**
   - An edge case for a type that can't actually appear at runtime -> **Low**
   - A missing handler for a common type in a sealed class hierarchy -> higher severity
   - A missing handler for a rare/internal type -> lower severity

**Step 3: Run all review sections below**

**Step 4: Present findings** organized by severity:
- **Critical**: Will cause incorrect behavior or build failure
- **High**: Missing required convention or test coverage gap
- **Medium**: Deviates from established patterns but functional
- **Low**: Style or minor improvement suggestions

---

## Section 1: Build and CI Config

### 1.1 build.gradle dependencies

Read the instrumentor's `build.gradle` and the root `java/build.gradle`.

**Required pattern:**
```gradle
dependencies {
    api project(':openinference-instrumentation')
    compileOnly "<library-group>:<library-artifact>:${libraryVersion}"
}
```

**Check:**
- Instrumented library must be `compileOnly` (not `implementation`) — **High** if wrong
- `openinference-instrumentation` must be `api` (transitive for users)
- Version constants must be in root `ext` block, not hardcoded — **Medium** if hardcoded

### 1.2 Module inclusion

Verify the module is in `java/settings.gradle`:
```gradle
include ':instrumentation:openinference-instrumentation-<name>'
```

Missing inclusion — **Critical** (module won't build).

### 1.3 Formatting

The root build.gradle enforces Palantir Java Format via Spotless. Run:
```bash
cd java && ./gradlew spotlessCheck
```

---

## Section 2: Testing Patterns

### 2.1 Test setup

Read `src/test/java/` and verify test infrastructure:

**Required:**
- JUnit 5 (`@Test`, `@BeforeEach`, `@AfterEach`)
- `InMemorySpanExporter` for capturing spans
- `TracerProvider` with `SimpleSpanProcessor` wired to the exporter
- Cleanup: exporter reset and provider close in `@AfterEach`

```java
@BeforeEach
void setUp() {
    spanExporter = InMemorySpanExporter.create();
    tracerProvider = SdkTracerProvider.builder()
        .addSpanProcessor(SimpleSpanProcessor.create(spanExporter))
        .build();
}

@AfterEach
void tearDown() {
    spanExporter.reset();
    tracerProvider.close();
}
```

Missing test files entirely — **Critical**.
Missing cleanup — **High**.

### 2.2 Exhaustive attribute assertions

This is the most important testing pattern. Tests should verify ALL span attributes, not
just spot-check a few. The pattern prevents regressions where unexpected attributes appear
or expected ones disappear silently.

**Correct pattern (remove-and-verify):**
```java
Map<String, Object> attributes = new HashMap<>();
span.getAttributes().forEach((key, value) -> attributes.put(key.getKey(), value));

assertThat(attributes.remove("openinference.span.kind")).isEqualTo("LLM");
assertThat(attributes.remove("llm.model_name")).isEqualTo("gpt-4");
// ... remove and assert all remaining attributes ...
assertThat(attributes).isEmpty();  // Nothing unexpected left
```

**What to flag:**
- Tests that only check a few attributes without final emptiness check — **High**
- Missing `assertThat(attributes).isEmpty()` at the end — **High**

### 2.3 Context attribute propagation tests

There should be at least one test that verifies OpenInference context attributes
(session_id, user_id, metadata, tags) propagate correctly when set.

### 2.4 Error handling tests

Tests should cover:
- Exception during instrumented call -> span has `StatusCode.ERROR` and records exception
- Null/empty responses -> instrumentor doesn't throw NPE

Missing error tests — **High**.

---

## Section 3: OpenInference Semantic Conventions

Check which conventions apply based on the type of library being instrumented. Reference:
`java/openinference-semantic-conventions/src/main/java/com/arize/semconv/trace/SemanticConventions.java`

### 3.1 Always required

Every span must have:
- `OPENINFERENCE_SPAN_KIND` — set to the appropriate `OpenInferenceSpanKind` enum value
- `INPUT_VALUE` + `INPUT_MIME_TYPE` — what went into the operation
- `OUTPUT_VALUE` + `OUTPUT_MIME_TYPE` — what came out

MIME types: `application/json` for structured data, `text/plain` for strings.
Flag missing MIME type when value is set — **High**.

### 3.2 LLM libraries

- `LLM_MODEL_NAME` — the model identifier
- `LLM_SYSTEM` — the system name (e.g., "langchain4j", "spring-ai")
- `LLM_PROVIDER` — the provider name (e.g., "openai", "anthropic")
- `LLM_INVOCATION_PARAMETERS` — JSON string of parameters
- `LLM_INPUT_MESSAGES` / `LLM_OUTPUT_MESSAGES` — indexed message attributes:
  ```
  llm.input_messages.0.message.role = "user"
  llm.input_messages.0.message.content = "Hello"
  ```
- `LLM_TOKEN_COUNT_PROMPT` / `LLM_TOKEN_COUNT_COMPLETION` / `LLM_TOKEN_COUNT_TOTAL`

Span kind should be `LLM`.

### 3.3 Tool/function calling

When the library supports tool use:
```
llm.input_messages.{i}.message.tool_calls.{j}.tool_call.function.name
llm.input_messages.{i}.message.tool_calls.{j}.tool_call.function.arguments
llm.input_messages.{i}.message.tool_calls.{j}.tool_call.id
llm.tools.{i}.tool.json_schema
```

### 3.4 Embedding libraries

- `EMBEDDING_MODEL_NAME`, `EMBEDDING_EMBEDDINGS`, `EMBEDDING_TEXT`
- Span kind should be `EMBEDDING`

### 3.5 Retrieval libraries

- `RETRIEVAL_DOCUMENTS` with `DOCUMENT_ID`, `DOCUMENT_CONTENT`, `DOCUMENT_SCORE`, `DOCUMENT_METADATA`
- Span kind should be `RETRIEVER`

### 3.6 Agent/orchestration frameworks

Multiple span kinds in hierarchy: `CHAIN`, `AGENT`, `TOOL`, `LLM`.

---

## Section 4: Span Hierarchy

### 4.1 Parent-child relationships

For instrumentors that create multiple spans, verify:
- Spans nest correctly (`child.getParentSpanId() == parent.getSpanId()`)
- All spans share the same `traceId`
- No orphaned root spans that should be children

Flag missing hierarchy tests as **High** for multi-span instrumentors.

### 4.2 Correct span kinds in hierarchy

Common correct hierarchies:
- `CHAIN -> LLM` (simple chain with LLM call)
- `CHAIN -> AGENT -> TOOL` (agent framework)
- `CHAIN -> RETRIEVER -> EMBEDDING` (RAG pipeline)

### 4.3 Context propagation

- Spans must set parent via `Context.current()` for proper nesting
- `span.end()` must ALWAYS be called (in finally block) — **Critical** if missing
- `Scope` from `context.makeCurrent()` must be closed (try-with-resources) — **High**

### 4.4 Suppress tracing / TraceConfig

The instrumentor should use `OITracer` (not raw `Tracer`) and respect `TraceConfig`:
- `isHideInputMessages()` / `isHideOutputMessages()` to gate message attributes
- `isHideInputs()` / `isHideOutputs()` to gate input/output value attributes
- At minimum, `hideInputs`, `hideOutputs`, `hideInputMessages`, `hideOutputMessages`

Missing TraceConfig support — **Medium**.

---

## Presenting Results

Organize findings into a table:

| Severity | Section | Finding | Location |
|----------|---------|---------|----------|
| Critical | 4.3 | span.end() not called in error path | `LangChain4jModelListener.java:142` |
| High | 2.2 | Tests don't verify all span attributes | `src/test/java/.../Test.java:85` |
| ... | ... | ... | ... |

Then list what's working well — positive findings help the user understand what doesn't
need to change.

Finally, ask the user what they'd like to do:
- "Fix the issues" — generate patches
- "Run the tests" — execute `cd java && ./gradlew :instrumentation:openinference-instrumentation-<name>:test`
- "Just reviewing" — done
