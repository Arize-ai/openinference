# OpenInference Java

Java implementation of OpenInference for tracing AI applications using OpenTelemetry.

## Overview

This project provides Java libraries for instrumenting AI/ML applications with OpenTelemetry, following the OpenInference semantic conventions. It includes:

- **openinference-semantic-conventions**: Java constants for OpenInference semantic conventions
- **openinference-instrumentation**: Base instrumentation utilities
- **openinference-instrumentation-langchain4j**: Auto-instrumentation for LangChain4j applications
- **openinference-instrumentation-annotations**: Annotation-based manual tracing (`@TraceChain`, `@TraceLLM`, `@TraceTool`, `@TraceAgent`)

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
    implementation 'io.openinference:openinference-instrumentation-annotations:0.1.0-SNAPSHOT'
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
    <dependency>
        <groupId>io.openinference</groupId>
        <artifactId>openinference-instrumentation-annotations</artifactId>
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

### Annotation-based tracing

Annotate your methods with `@TraceChain`, `@TraceLLM`, `@TraceTool`, or `@TraceAgent` to
automatically create OpenInference spans. The ByteBuddy agent intercepts annotated methods
at class load time, capturing inputs and outputs as span attributes.

```java
import com.arize.instrumentation.annotations.*;

public class QAService {

    @TraceAgent(name = "qa-agent")
    public String answer(String question) {
        String context = retrieve(question);
        return generate(question, context);
    }

    @TraceChain(name = "retriever")
    public String retrieve(String query) {
        // retrieval logic
        return "OpenInference is a tracing standard";
    }

    @TraceLLM(name = "generator")
    public String generate(String query, @SpanIgnore String context) {
        // LLM call
        return "OpenInference provides tracing for AI apps.";
    }

    @TraceTool(name = "weather", description = "Gets current weather")
    public Map<String, Object> getWeather(String location) {
        return Map.of("temp", 72, "condition", "sunny");
    }
}
```

Register the tracer at startup to activate annotation-based tracing:

```java
import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.annotations.OpenInferenceAgent;

Tracer otelTracer = GlobalOpenTelemetry.getTracer("my-app");
OITracer tracer = new OITracer(otelTracer);
OpenInferenceAgent.register(tracer);
```

### Programmatic tracing with typed spans

For more control, use the `TracedSpan` API directly with try-with-resources:

```java
import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.annotations.*;

Tracer otelTracer = GlobalOpenTelemetry.getTracer("my-app");
OITracer tracer = new OITracer(otelTracer);

try (TracedAgentSpan agent = TracedAgentSpan.start(tracer, "qa-agent")) {
    agent.setInput("What is OpenInference?");
    agent.setAgentName("qa-agent");

    try (TracedLLMSpan llm = TracedLLMSpan.start(tracer, "generate")) {
        llm.setInput("What is OpenInference?");
        llm.setModelName("gpt-4o");

        String answer = callLLM("What is OpenInference?");

        llm.setOutput(answer);
        llm.setTokenCountTotal(150);
    }

    agent.setOutput("OpenInference is a tracing standard for AI apps.");
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