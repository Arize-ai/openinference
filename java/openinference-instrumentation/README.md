# OpenInference Instrumentation — Core Trace Library

Programmatic tracing for Java AI applications using typed span classes backed by OpenTelemetry.

This is the foundation library. For annotation-based tracing (`@Chain`, `@LLM`, `@Tool`, `@Agent`), see [`openinference-instrumentation-annotation`](../instrumentation/openinference-instrumentation-annotation).

## Installation

### Gradle

```gradle
dependencies {
    implementation 'com.arize:openinference-instrumentation:0.1.0'
}
```

### Maven

```xml
<dependency>
    <groupId>com.arize</groupId>
    <artifactId>openinference-instrumentation</artifactId>
    <version>0.1.0</version>
</dependency>
```

## Quick Start

```java
import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.trace.*;
import io.opentelemetry.sdk.trace.SdkTracerProvider;

// Set up OpenTelemetry
SdkTracerProvider tracerProvider = SdkTracerProvider.builder()
        .addSpanProcessor(/* your exporter */)
        .build();

OITracer tracer = new OITracer(tracerProvider.get("my-app"));

try (AgentSpan agent = AgentSpan.start(tracer, "qa-agent")) {
    agent.setInput("What is OpenInference?");
    agent.setAgentName("qa-agent");

    try (ChainSpan chain = ChainSpan.start(tracer, "retriever")) {
        chain.setInput("What is OpenInference?");
        chain.setOutput("OpenInference is a tracing standard.");
    }

    try (LLMSpan llm = LLMSpan.start(tracer, "generator")) {
        llm.setInput("What is OpenInference?");
        llm.setModelName("gpt-4o");
        llm.setTokenCountTotal(150);
        llm.setOutput("OpenInference provides tracing for AI apps.");
    }

    agent.setOutput("OpenInference provides tracing for AI apps.");
}
```

Spans nest automatically via OpenTelemetry context propagation.

## Typed Span Classes

| Class | Span Kind | Key Methods |
|---|---|---|
| `ChainSpan` | `CHAIN` | (base methods only) |
| `LLMSpan` | `LLM` | `setModelName`, `setSystem`, `setProvider`, `setInputMessages`, `setOutputMessages`, `setTokenCountPrompt`, `setTokenCountCompletion`, `setTokenCountTotal`, `setCostPrompt`, `setCostCompletion`, `setCostTotal`, `setInvocationParameters` |
| `ToolSpan` | `TOOL` | `setToolName`, `setToolDescription`, `setToolParameters`, `setToolJsonSchema` |
| `AgentSpan` | `AGENT` | `setAgentName` |
| `RetrievalSpan` | `RETRIEVER` | `setDocuments` |
| `EmbeddingSpan` | `EMBEDDING` | `setModelName`, `setEmbeddings` |

All span classes inherit these common methods:
- `setInput(Object)` / `setOutput(Object)` — auto-serialized to JSON for non-string objects
- `setMetadata(Map<String, Object>)` / `setTags(List<String>)`
- `setSessionId(String)` / `setUserId(String)`
- `setAttribute(String key, Object value)` — set arbitrary attributes
- `setError(Throwable)` — record an exception and set span status to ERROR

## Configuration

Use `TraceConfig` to control what gets captured:

```java
import com.arize.instrumentation.TraceConfig;

TraceConfig config = TraceConfig.builder()
        .hideInputs(true)          // Suppress input.value
        .hideOutputs(true)         // Suppress output.value
        .hideInputMessages(true)   // Suppress llm.input_messages
        .hideOutputMessages(true)  // Suppress llm.output_messages
        .hideToolParameters(true)  // Suppress tool.parameters
        .hideOutputEmbeddings(true) // Suppress embedding.embeddings
        .build();

OITracer tracer = new OITracer(tracerProvider.get("my-app"), config);
```

## Requirements

- Java 17+
- OpenTelemetry Java SDK 1.49.0+
