# OpenInference Java

Java implementation of OpenInference for tracing AI applications using OpenTelemetry.

## Overview

This project provides Java libraries for instrumenting AI/ML applications with OpenTelemetry, following the OpenInference semantic conventions. It includes:

- **openinference-semantic-conventions**: Java constants for OpenInference semantic conventions
- **openinference-instrumentation**: Base instrumentation utilities
- **openinference-instrumentation-langchain4j**: Auto-instrumentation for LangChain4j applications

## Requirements

- Java 11 or higher
- OpenTelemetry Java 1.49.0 or higher

## Installation

### Gradle

Add the following to your `build.gradle`:

```gradle
dependencies {
    implementation 'io.openinference:openinference-semantic-conventions:0.1.0-SNAPSHOT'
    implementation 'io.openinference:openinference-instrumentation:0.1.0-SNAPSHOT'
    implementation 'io.openinference:openinference-instrumentation-langchain4j:0.1.0-SNAPSHOT'
}
```

### Maven

Add the following to your `pom.xml`:

```xml
<dependencies>
    <dependency>
        <groupId>io.openinference</groupId>
        <artifactId>openinference-semantic-conventions</artifactId>
        <version>0.1.0-SNAPSHOT</version>
    </dependency>
    <dependency>
        <groupId>io.openinference</groupId>
        <artifactId>openinference-instrumentation</artifactId>
        <version>0.1.0-SNAPSHOT</version>
    </dependency>
    <dependency>
        <groupId>io.openinference</groupId>
        <artifactId>openinference-instrumentation-langchain4j</artifactId>
        <version>0.1.0-SNAPSHOT</version>
    </dependency>
</dependencies>
```

## Quick Start

### Auto-instrumentation with LangChain4j

```java
import io.openinference.instrumentation.langchain4j.LangChain4jInstrumentor;
import dev.langchain4j.model.openai.OpenAiChatModel;

// Initialize OpenTelemetry (see OpenTelemetry Java docs for full setup)
// ...

// Auto-instrument LangChain4j
LangChain4jInstrumentor.instrument();

// Use LangChain4j as normal - traces will be automatically created
OpenAiChatModel model = OpenAiChatModel.builder()
    .apiKey("your-api-key")
    .modelName("gpt-4")
    .build();

String response = model.generate("What is the capital of France?");
```

### Manual instrumentation

```java
import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.Tracer;
import io.openinference.instrumentation.OITracer;
import io.openinference.semconv.trace.SpanAttributes;

// Create an OITracer
Tracer otelTracer = GlobalOpenTelemetry.getTracer("my-app");
OITracer tracer = new OITracer(otelTracer);

// Create an LLM span
Span span = tracer.llmSpanBuilder("chat", "gpt-4")
    .setAttribute(SpanAttributes.LLM_MODEL_NAME, "gpt-4")
    .setAttribute(SpanAttributes.LLM_PROVIDER, "openai")
    .startSpan();

try {
    // Your LLM call here
    // ...
    
    // Set response attributes
    span.setAttribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, 10L);
    span.setAttribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, 20L);
} finally {
    span.end();
}
```

## Configuration

### Environment Variables

- `OTEL_INSTRUMENTATION_LANGCHAIN4J_ENABLED`: Enable/disable LangChain4j auto-instrumentation (default: true)

### Programmatic Configuration

```java
import io.openinference.instrumentation.TraceConfig;
import io.openinference.instrumentation.langchain4j.LangChain4jInstrumentor;

// Configure what to hide in traces
TraceConfig config = TraceConfig.builder()
    .hideInputMessages(true)  // Hide input messages
    .hideOutputMessages(true) // Hide output messages
    .build();

// Instrument with custom config
LangChain4jInstrumentor.instrument(config);
```

## Semantic Conventions

This library implements the OpenInference semantic conventions for AI observability:

- **Span Kinds**: LLM, Chain, Tool, Agent, Retriever, Embedding, Reranker, Guardrail, Evaluator
- **Attributes**: Model names, token counts, prompts, completions, embeddings, etc.

## Integration with OpenTelemetry

The library is built on top of OpenTelemetry Java and integrates seamlessly with the OpenTelemetry ecosystem:

- Works with any OpenTelemetry-compatible backend (Jaeger, Zipkin, OTLP, etc.)
- Compatible with OpenTelemetry auto-instrumentation agent
- Supports OpenTelemetry SDK auto-configuration

## Examples

See the `examples/` directory for complete examples:

- Basic LangChain4j instrumentation
- Manual span creation
- Integration with different observability backends

## Building from Source

```bash
cd java
./gradlew build
```

## Contributing

See the main [CONTRIBUTING](../CONTRIBUTING) guide.

## License

Apache License 2.0 