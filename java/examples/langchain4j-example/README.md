# LangChain4j OpenInference Example with Phoenix

This example demonstrates how to use OpenInference instrumentation with LangChain4j and export traces to Arize Phoenix.

## Prerequisites

1. Java 11 or higher
2. OpenAI API key
3. (Optional) Phoenix API key if using auth

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