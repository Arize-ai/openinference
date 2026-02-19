# Java Workspace Guide

Java instrumentation packages for OpenInference. Uses Gradle multi-project build, Java 17 toolchain (Java 11+ runtime), Palantir code formatting via Spotless, and publishes to Maven Central under the `io.openinference` group.

---

## Project Structure

```
java/
├── openinference-instrumentation/        # Core: OITracer, TraceConfig, context utilities
├── openinference-semantic-conventions/   # SemanticConventions.java constants
└── instrumentation/
    ├── openinference-instrumentation-langchain4j/   # LangChain4j auto-instrumentation
    └── openinference-instrumentation-springAI/      # Spring AI instrumentation
```

---

## Build and Test Commands

```bash
cd java
./gradlew build                   # compile + test all modules
./gradlew test                    # run tests only
./gradlew spotlessApply           # auto-format all Java files (run before committing)
./gradlew spotlessCheck           # verify formatting (run in CI)
./gradlew publishToMavenLocal     # publish to local Maven repository for development
```

---

## OITracer Pattern

Always wrap the raw OTel `Tracer` with `OITracer` so spans are automatically tagged with the correct span kind and respect `TraceConfig`.

```java
import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.trace.Tracer;
import io.openinference.instrumentation.OITracer;
import io.openinference.instrumentation.TraceConfig;
import io.openinference.semconv.trace.SpanAttributes;

// Basic usage:
Tracer otelTracer = GlobalOpenTelemetry.getTracer("my-instrumentor");
OITracer tracer = new OITracer(otelTracer);

// With TraceConfig (to mask sensitive data):
TraceConfig config = TraceConfig.builder()
    .hideInputMessages(true)
    .hideOutputMessages(true)
    .build();
OITracer tracer = new OITracer(otelTracer, config);
```

---

## Creating an LLM Span

```java
import io.opentelemetry.api.trace.Span;
import io.openinference.semconv.trace.SpanAttributes;

Span span = tracer.llmSpanBuilder("chat", "gpt-4")
    .setAttribute(SpanAttributes.LLM_MODEL_NAME, "gpt-4")
    .setAttribute(SpanAttributes.LLM_PROVIDER, "openai")
    .startSpan();

try {
    // LLM call ...
    span.setAttribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, 10L);
    span.setAttribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, 20L);
    span.setAttribute(SpanAttributes.OUTPUT_VALUE, responseText);
} finally {
    span.end();
}
```

---

## LangChain4j Auto-instrumentation

```java
import io.openinference.instrumentation.langchain4j.LangChain4jInstrumentor;

// Simple instrumentation (uses global OpenTelemetry):
LangChain4jInstrumentor.instrument();

// With custom TraceConfig:
TraceConfig config = TraceConfig.builder()
    .hideInputMessages(true)
    .build();
LangChain4jInstrumentor.instrument(config);
```

Environment variable: `OTEL_INSTRUMENTATION_LANGCHAIN4J_ENABLED` (default: `true`)

---

## Semantic Conventions

```java
import io.openinference.semconv.trace.SpanAttributes;
import io.openinference.semconv.trace.OpenInferenceSpanKind;

// Required on every OpenInference span:
span.setAttribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKind.LLM.getValue());

// Common attributes:
span.setAttribute(SpanAttributes.LLM_MODEL_NAME, "gpt-4");
span.setAttribute(SpanAttributes.LLM_SYSTEM, "openai");
span.setAttribute(SpanAttributes.LLM_PROVIDER, "openai");
span.setAttribute(SpanAttributes.INPUT_VALUE, inputText);
span.setAttribute(SpanAttributes.OUTPUT_VALUE, outputText);
span.setAttribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, promptTokens);
span.setAttribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, completionTokens);
span.setAttribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, totalTokens);

// Session and user context:
span.setAttribute(SpanAttributes.SESSION_ID, sessionId);
span.setAttribute(SpanAttributes.USER_ID, userId);
```

---

## The Three Required Features

### Feature 1 — Suppress Tracing

Check the OTel context suppression flag in each instrumented method:

```java
import io.opentelemetry.context.Context;
import io.opentelemetry.api.trace.Span;

// Check before creating spans:
if (!Span.fromContext(Context.current()).getSpanContext().isValid()) {
    return original.invoke(args);
}
```

Or use `isTracingSuppressed()` from opentelemetry-java-contrib if available.

### Feature 2 — Context Attribute Propagation

`OITracer` reads session ID, user ID, metadata, and tags from the OTel context automatically and attaches them to spans. No additional code is needed when using `OITracer`.

### Feature 3 — Trace Configuration

Pass a `TraceConfig` to `OITracer` at construction time (shown in the OITracer pattern above). The `OITracer` and underlying `OISpan` handle masking automatically.

---

## Adding a New Java Instrumentor

1. **Register the module in `java/settings.gradle`**:
   ```groovy
   include 'instrumentation:openinference-instrumentation-<name>'
   ```

2. **Create `instrumentation/openinference-instrumentation-<name>/build.gradle`**:
   ```groovy
   plugins {
       id 'java-library'
       id 'com.palantir.java-format' // via Spotless plugin
   }
   dependencies {
       api project(':openinference-instrumentation')
       implementation 'io.opentelemetry:opentelemetry-api'
       implementation 'your.framework:framework-library'
       testImplementation 'org.junit.jupiter:junit-jupiter'
   }
   ```

3. **Implement the instrumentor**: extend or implement the framework's listener/observer interface, call `OITracer` to create spans.

4. **Add tests** covering: span creation, suppress tracing, context attribute propagation, trace config masking.

---

## Gradle Dependency Versions

```groovy
// OpenTelemetry Java minimum version:
'io.opentelemetry:opentelemetry-api:1.49.0'

// Maven coordinates for published packages:
'io.openinference:openinference-semantic-conventions:0.1.0'
'io.openinference:openinference-instrumentation:0.1.0'
'io.openinference:openinference-instrumentation-langchain4j:0.1.0'
```

---

## Publishing

Java packages are released to Maven Central via JReleaser, triggered by GitHub Actions on a tagged release. For local development testing, use `./gradlew publishToMavenLocal` and add `mavenLocal()` to your project's repositories.
