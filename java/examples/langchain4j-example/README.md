# LangChain4j OpenInference Example with Phoenix

This example demonstrates how to use OpenInference instrumentation with LangChain4j and export traces to Arize Phoenix.

## Prerequisites

1. Java 11 or higher (AWS Corretto recommended)
2. Docker (for running Phoenix)
3. OpenAI API key

## Running the Example

### Step 1: Start Phoenix

Phoenix is an open-source observability platform for LLM applications. Start it using Docker:

```bash
docker run -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest
```

This command:
- Exposes port 6006 for the Phoenix web UI
- Exposes port 4317 for the OTLP gRPC endpoint (where traces are sent)

Once started, you can access Phoenix at http://localhost:6006

### Step 2: Set your OpenAI API Key

```bash
export OPENAI_API_KEY=your-api-key-here
```

### Step 3: Run the Example

From the `java` directory:

```bash
./gradlew :examples:langchain4j-example:run
```

### Step 4: View Traces in Phoenix

1. Open http://localhost:6006 in your browser
2. You should see the traces from your LangChain4j application
3. Click on a trace to see detailed information including:
   - LLM model name
   - Input/output messages
   - Token counts
   - Request parameters (temperature, max_tokens, etc.)
   - Latency information

## What's Being Traced

The example traces:
- LLM calls to OpenAI
- Input prompts and output responses
- Model parameters (temperature, max tokens)
- Token usage (when available)
- Timing information

## Troubleshooting

### Phoenix is not receiving traces

1. Ensure Phoenix is running and accessible:
   ```bash
   curl http://localhost:4317
   ```

2. Check that the OTLP port (4317) is not blocked by a firewall

3. Look for errors in the console output when running the example

### Connection refused errors

If you see connection refused errors, make sure:
- Phoenix is running (`docker ps` to check)
- The ports are correctly mapped (6006 and 4317)
- You're using the correct endpoint in the code (`http://localhost:4317`)

## Customizing the Configuration

You can modify the OTLP exporter configuration in the `initializeOpenTelemetry()` method:

```java
OtlpGrpcSpanExporter otlpExporter = OtlpGrpcSpanExporter.builder()
        .setEndpoint("http://localhost:4317")  // Change this for a different endpoint
        .setTimeout(Duration.ofSeconds(2))
        .build();
```

## Next Steps

- Try modifying the example to make different types of LLM calls
- Experiment with different models and parameters
- Add custom attributes to your spans
- Integrate the instrumentation into your own LangChain4j applications 