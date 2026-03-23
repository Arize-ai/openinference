# OpenInference Annotation-Based Tracing

Annotation-based manual tracing for Java AI applications. Annotate your methods with `@Chain`, `@LLM`, `@Tool`, or `@Agent` to automatically create [OpenInference](https://github.com/Arize-ai/openinference) spans backed by OpenTelemetry.

A ByteBuddy agent intercepts annotated methods at class load time, capturing inputs and outputs as span attributes following the OpenInference semantic conventions.

## Installation

### Gradle

```gradle
dependencies {
    implementation 'com.arize:openinference-instrumentation-annotation:0.1.0'
}
```

### Maven

```xml
<dependency>
    <groupId>com.arize</groupId>
    <artifactId>openinference-instrumentation-annotation</artifactId>
    <version>0.1.0</version>
</dependency>
```

## Quick Start

### 1. Install the agent and register a tracer

```java
import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.OpenInferenceAgent;
import com.arize.instrumentation.annotation.OpenInferenceAgentInstaller;
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

#### Packaging & running via `-javaagent`

Build the agent locally to produce a jar with the necessary `Premain-Class`/`Agent-Class`
manifest entries:

```bash
cd java
./gradlew :instrumentation:openinference-instrumentation-annotation:jar
```

Then attach it to any JVM process (either at launch or dynamically) without calling
`OpenInferenceAgentInstaller.install()` yourself:

```bash
java -javaagent:./java/instrumentation/openinference-instrumentation-annotation/build/libs/openinference-instrumentation-annotation.jar \
     -cp your-app.jar com.example.Main
```

You still need to construct and register an `OITracer` (as above) during application startup so the
agent has an OpenTelemetry tracer/exporter to route spans through.

### 2. Annotate your methods

```java
import com.arize.instrumentation.annotation.*;
import java.util.Map;

public class QAService {

    @Agent(name = "qa-agent")
    public String answer(String question) {
        String context = retrieve(question);
        Map<String, Object> weather = getWeather("San Francisco");
        return generate(question, context, weather);
    }

    @Chain(name = "retriever")
    public String retrieve(String query) {
        return "OpenInference is an open standard for AI tracing.";
    }

    @Tool(name = "weather", description = "Gets current weather for a location")
    public Map<String, Object> getWeather(String location) {
        return Map.of("temp", 68, "condition", "foggy", "location", location);
    }

    @LLM(name = "generator")
    public String generate(String question, String context, @ExcludeFromSpan Map<String, Object> weather) {
        // @ExcludeFromSpan excludes the weather parameter from auto-captured input
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
| `@Chain` | `CHAIN` | - |
| `@LLM` | `LLM` | - |
| `@Tool` | `TOOL` | `description` for tool description |
| `@Agent` | `AGENT` | - |
| `@Span` | Any kind | `kind` (required) for span kind |

All annotations support:
- `name` - Custom span name (defaults to method name)
- `mapping` - Map input parameters to specific semantic convention attributes
- `outputMapping` - Map return value fields to specific semantic convention attributes

### `@Span`

Use `@Span` for less common span kinds that don't have a dedicated annotation:

```java
import com.arize.semconv.trace.SemanticConventions.OpenInferenceSpanKind;

@Span(kind = OpenInferenceSpanKind.RETRIEVER, name = "my-retriever")
public List<Document> search(String query) { ... }

@Span(kind = OpenInferenceSpanKind.EMBEDDING, name = "embedder")
public float[] embed(String text) { ... }
```

### `@ExcludeFromSpan`

Annotate a method parameter to exclude it from auto-captured input:

```java
@Chain
public String process(String query, @ExcludeFromSpan String apiKey) { ... }
```

### `@SpanMapping`

Map parameters or return value fields to OpenInference semantic convention attributes:

```java
import com.arize.semconv.trace.SemanticConventions;

@LLM(
    mapping = @SpanMapping(param = "model", attribute = SemanticConventions.LLM_MODEL_NAME),
    outputMapping = @SpanMapping(field = "usage.totalTokens", attribute = SemanticConventions.LLM_TOKEN_COUNT_TOTAL)
)
public LLMResponse chat(String model, String prompt) { ... }
```

The `field` parameter uses dot notation to extract nested values from return objects (e.g., `"usage.totalTokens"`).

> **Note:** Parameter mappings rely on Java's `-parameters` compiler flag. If your application
> isn't compiled with that flag, reference the generated `arg0`, `arg1`, … parameter names in your
> `@SpanMapping` annotations instead.

## Programmatic API

For more control without annotations, use the typed span classes directly. See the [core trace library README](../../openinference-instrumentation/README.md) for full documentation and examples.

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
./gradlew :examples:annotation-example:run

# View traces at http://localhost:6006
```

## Requirements

- Java 17+
- OpenTelemetry Java SDK 1.49.0+
