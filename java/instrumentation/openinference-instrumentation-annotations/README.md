# OpenInference Annotation-Based Tracing

Annotation-based manual tracing for Java AI applications. Annotate your methods with `@TraceChain`, `@TraceLLM`, `@TraceTool`, or `@TraceAgent` to automatically create [OpenInference](https://github.com/Arize-ai/openinference) spans backed by OpenTelemetry.

A ByteBuddy agent intercepts annotated methods at class load time, capturing inputs and outputs as span attributes following the OpenInference semantic conventions.

## Installation

### Gradle

```gradle
dependencies {
    implementation 'com.arize:openinference-instrumentation-annotations:0.1.0'
}
```

### Maven

```xml
<dependency>
    <groupId>com.arize</groupId>
    <artifactId>openinference-instrumentation-annotations</artifactId>
    <version>0.1.0</version>
</dependency>
```

## Quick Start

### 1. Install the agent and register a tracer

```java
import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.annotations.OpenInferenceAgent;
import com.arize.instrumentation.annotations.OpenInferenceAgentInstaller;
import io.opentelemetry.sdk.trace.SdkTracerProvider;

// Install ByteBuddy agent BEFORE loading any annotated classes
OpenInferenceAgentInstaller.install();

// Set up OpenTelemetry (see OpenTelemetry Java docs for full configuration)
SdkTracerProvider tracerProvider = SdkTracerProvider.builder()
        .addSpanProcessor(/* your exporter */)
        .build();

// Create and register the OITracer
OITracer tracer = new OITracer(tracerProvider.get("my-app"));
OpenInferenceAgent.register(tracer);
```

### 2. Annotate your methods

```java
import com.arize.instrumentation.annotations.*;
import java.util.Map;

public class QAService {

    @TraceAgent(name = "qa-agent")
    public String answer(String question) {
        String context = retrieve(question);
        Map<String, Object> weather = getWeather("San Francisco");
        return generate(question, context, weather);
    }

    @TraceChain(name = "retriever")
    public String retrieve(String query) {
        return "OpenInference is an open standard for AI tracing.";
    }

    @TraceTool(name = "weather", description = "Gets current weather for a location")
    public Map<String, Object> getWeather(String location) {
        return Map.of("temp", 68, "condition", "foggy", "location", location);
    }

    @TraceLLM(name = "generator")
    public String generate(String question, String context, @SpanIgnore Map<String, Object> weather) {
        // @SpanIgnore excludes the weather parameter from auto-captured input
        return callLLM(question, context);
    }
}
```

Calling `service.answer("What is OpenInference?")` produces a trace like:

```
qa-agent (AGENT)
├── retriever (CHAIN)
├── weather (TOOL)
└── generator (LLM)
```

All method parameters are automatically captured as span input, and the return value is captured as output.

### 3. Shutdown

```java
OpenInferenceAgent.unregister();
tracerProvider.forceFlush();
tracerProvider.shutdown();
```

## Annotations

| Annotation | Span Kind | Extra Attributes |
|---|---|---|
| `@TraceChain` | `CHAIN` | - |
| `@TraceLLM` | `LLM` | - |
| `@TraceTool` | `TOOL` | `description` for tool description |
| `@TraceAgent` | `AGENT` | - |

All annotations support:
- `name` - Custom span name (defaults to method name)
- `mapping` - Map input parameters to specific semantic convention attributes
- `outputMapping` - Map return value fields to specific semantic convention attributes

### `@SpanIgnore`

Annotate a method parameter to exclude it from auto-captured input:

```java
@TraceChain
public String process(String query, @SpanIgnore String apiKey) { ... }
```

### `@SpanMapping`

Map parameters or return value fields to OpenInference semantic convention attributes:

```java
@TraceLLM(
    mapping = @SpanMapping(param = "model", attribute = SpanAttribute.LLM_MODEL_NAME),
    outputMapping = @SpanMapping(field = "usage.totalTokens", attribute = SpanAttribute.LLM_TOKEN_COUNT_TOTAL)
)
public LLMResponse chat(String model, String prompt) { ... }
```

The `field` parameter uses dot notation to extract nested values from return objects (e.g., `"usage.totalTokens"`).

## Programmatic API

For more control, use the typed span classes directly with try-with-resources:

```java
import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.annotations.*;

OITracer tracer = new OITracer(tracerProvider.get("my-app"));

try (TracedAgentSpan agent = TracedAgentSpan.start(tracer, "qa-agent")) {
    agent.setInput("What is OpenInference?");
    agent.setAgentName("qa-agent");

    try (TracedChainSpan chain = TracedChainSpan.start(tracer, "retriever")) {
        chain.setInput("What is OpenInference?");
        chain.setOutput("OpenInference is a tracing standard.");
    }

    try (TracedLLMSpan llm = TracedLLMSpan.start(tracer, "generator")) {
        llm.setInput("What is OpenInference?");
        llm.setModelName("gpt-4o");
        llm.setTokenCountTotal(150);
        llm.setOutput("OpenInference provides tracing for AI apps.");
    }

    agent.setOutput("OpenInference provides tracing for AI apps.");
}
```

Spans nest automatically via OpenTelemetry context propagation.

### Typed Span Classes

| Class | Span Kind | Key Methods |
|---|---|---|
| `TracedChainSpan` | `CHAIN` | (base methods only) |
| `TracedLLMSpan` | `LLM` | `setModelName`, `setSystem`, `setProvider`, `setInputMessages`, `setOutputMessages`, `setTokenCountPrompt`, `setTokenCountCompletion`, `setTokenCountTotal`, `setCostPrompt`, `setCostCompletion`, `setCostTotal`, `setInvocationParameters` |
| `TracedToolSpan` | `TOOL` | `setToolName`, `setToolDescription`, `setToolParameters`, `setToolJsonSchema` |
| `TracedAgentSpan` | `AGENT` | `setAgentName` |
| `TracedRetrievalSpan` | `RETRIEVER` | `setDocuments` |
| `TracedEmbeddingSpan` | `EMBEDDING` | `setModelName`, `setEmbeddings` |

All span classes inherit these common methods:
- `setInput(Object)` / `setOutput(Object)` - Auto-serialized to JSON for non-string objects
- `setMetadata(Map<String, Object>)` / `setTags(List<String>)`
- `setSessionId(String)` / `setUserId(String)`
- `setAttribute(String key, Object value)` - Set arbitrary attributes
- `setError(Throwable)` - Record an exception and set span status to ERROR

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

## Running the Example

```bash
# Start Phoenix
docker run -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest

# Run the example
cd java
./gradlew :examples:annotations-example:run

# View traces at http://localhost:6006
```

## Requirements

- Java 17+
- OpenTelemetry Java SDK 1.49.0+
