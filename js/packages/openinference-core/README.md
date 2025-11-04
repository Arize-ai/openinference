# OpenInference Core

[![npm version](https://badge.fury.io/js/@arizeai%2Fopeninference-core.svg)](https://badge.fury.io/js/@arizeai%2Fopeninference-core)

This package provides OpenInference Core utilities for LLM Traces.

## Installation

```bash
npm install @arizeai/openinference-core # npm
pnpm add @arizeai/openinference-core # pnpm
yarn add @arizeai/openinference-core # yarn
```

## Customizing Spans

The `@arizeai/openinference-core` package offers utilities to track important application metadata such as sessions and users using context attribute propagation:

- `setSession`: to specify a session ID to track and group multi-turn conversations
- `setUser`: to specify a user ID to track different conversations with a given user
- `setMetadata`: to add custom metadata that can provide extra information to support a wide range of operational needs
- `setTag`: to add tags, to filter spans on specific keywords
- `setPromptTemplate`: to reflect the prompt template used, with its version and variables. This is useful for prompt template tracking
- `setAttributes`: to add multiple custom attributes at the same time

> [!NOTE] All @arizeai/openinference auto instrumentation packages will pull attributes off of context and add them to spans

### Examples

`setSession`

```typescript
import { context } from "@opentelemetry/api";
import { setSession } from "@arizeai/openinference-core";

context.with(setSession(context.active(), { sessionId: "session-id" }), () => {
  // Calls within this block will generate spans with the attributes:
  // "session.id" = "session-id"
});
```

Each setter function returns a new active context, so they can be chained together.

```typescript
import { context } from "@opentelemetry/api";
import { setAttributes, setSession } from "@arizeai/openinference-core";

context.with(
  setAttributes(setSession(context.active(), { sessionId: "session-id" }), {
    myAttribute: "test",
  }),
  () => {
    // Calls within this block will generate spans with the attributes:
    // "myAttribute" = "test"
    // "session.id" = "session-id"
  },
);
```

Additionally, they can be used in conjunction with the [OpenInference Semantic Conventions](../openinference-semantic-conventions/).

```typescript
import { context } from "@opentelemetry/api"
import { setAttributes } from "@openinference-core"
import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";


context.with(
  setAttributes(
    { [SemanticConventions.SESSION_ID: "session-id" }
  ),
  () => {
      // Calls within this block will generate spans with the attributes:
      // "session.id" = "session-id"
  }
)
```

If you are creating spans manually and want to propagate context attributes you've set to those spans as well you can use the `getAttributesFromContext` utility to do that. you can read more about customizing spans in our [docs](https://docs.arize.com/phoenix/tracing/how-to-tracing/customize-spans).

```typescript
import { getAttributesFromContext } from "@arizeai/openinference-core";
import { context, trace } from "@opentelemetry/api";

const contextAttributes = getAttributesFromContext(context.active());
const tracer = trace.getTracer("example");
const span = tracer.startSpan("example span");
span.setAttributes(contextAttributes);
span.end();
```

## Tracing Helpers

This package provides convenient helpers to instrument your functions, agents, and LLM operations with OpenInference spans.

### Function Tracing

**`withSpan`** - Wraps any function (sync or async) with OpenTelemetry tracing:

```typescript
import { withSpan } from "@arizeai/openinference-core";
import { OpenInferenceSpanKind } from "@arizeai/openinference-semantic-conventions";

const processUserQuery = async (query: string) => {
  // Your business logic here
  const response = await fetch(`/api/process?q=${query}`);
  return response.json();
};

const tracedProcess = withSpan(processUserQuery, {
  name: "user-query-processor",
  kind: OpenInferenceSpanKind.CHAIN,
});
```

**`traceChain`** - Convenience wrapper for tracing workflow sequences:

```typescript
import { traceChain } from "@arizeai/openinference-core";

const ragPipeline = async (question: string) => {
  const documents = await retrieveDocuments(question);
  const context = documents.map((d) => d.content).join("\n");
  const answer = await generateAnswer(question, context);
  return answer;
};

const tracedRag = traceChain(ragPipeline, { name: "rag-pipeline" });
```

**`traceAgent`** - Convenience wrapper for tracing autonomous agents:

```typescript
import { traceAgent } from "@arizeai/openinference-core";

const simpleAgent = async (question: string) => {
  // Agent logic that may call tools, make decisions, etc.
  const documents = await retrieveDocuments(question);
  const analysis = await analyzeContext(question, documents);
  return await executePlan(analysis);
};

const tracedAgent = traceAgent(simpleAgent, { name: "qa-agent" });
```

**`traceTool`** - Convenience wrapper for tracing external tools:

```typescript
import { traceTool } from "@arizeai/openinference-core";

const weatherTool = async (city: string) => {
  const response = await fetch(`https://api.weather.com/v1/${city}`);
  return response.json();
};

const tracedWeatherTool = traceTool(weatherTool, { name: "weather-api" });
```

### Decorators

**`@observe`** - Decorator for automatically tracing class methods:

```typescript
import { observe } from "@arizeai/openinference-core";

class ChatService {
  @observe({ kind: "chain" })
  async processMessage(message: string) {
    // Your method implementation
    return `Processed: ${message}`;
  }

  @observe({ name: "llm-call", kind: "llm" })
  async callLLM(prompt: string) {
    // LLM invocation
    return await llmClient.generate(prompt);
  }
}
```

### Attribute Helpers

**`getLLMAttributes`** - Generate attributes for LLM operations:

```typescript
import { getLLMAttributes } from "@arizeai/openinference-core";
import { trace } from "@opentelemetry/api";

const tracer = trace.getTracer("llm-service");

tracer.startActiveSpan("llm-inference", (span) => {
  const attributes = getLLMAttributes({
    provider: "openai",
    modelName: "gpt-4",
    inputMessages: [{ role: "user", content: "What is AI?" }],
    outputMessages: [{ role: "assistant", content: "AI is..." }],
    tokenCount: { prompt: 10, completion: 50, total: 60 },
  });
  span.setAttributes(attributes);
  span.end();
});
```

**`getEmbeddingAttributes`** - Generate attributes for embedding operations:

```typescript
import { getEmbeddingAttributes } from "@arizeai/openinference-core";
import { trace } from "@opentelemetry/api";

const tracer = trace.getTracer("embedding-service");

tracer.startActiveSpan("generate-embeddings", (span) => {
  const attributes = getEmbeddingAttributes({
    modelName: "text-embedding-ada-002",
    embeddings: [
      { text: "The quick brown fox", vector: [0.1, 0.2, 0.3, ...] },
      { text: "jumps over the lazy dog", vector: [0.4, 0.5, 0.6, ...] },
    ],
  });
  span.setAttributes(attributes);
  span.end();
});
```

**`getRetrieverAttributes`** - Generate attributes for document retrieval:

```typescript
import { getRetrieverAttributes } from "@arizeai/openinference-core";
import { trace } from "@opentelemetry/api";

const tracer = trace.getTracer("retriever-service");

async function retrieveDocuments(query: string) {
  return tracer.startActiveSpan("retrieve-documents", async (span) => {
    const documents = await vectorStore.similaritySearch(query, 5);
    const attributes = getRetrieverAttributes({
      documents: documents.map((doc) => ({
        content: doc.pageContent,
        id: doc.metadata.id,
        score: doc.score,
        metadata: doc.metadata,
      })),
    });
    span.setAttributes(attributes);
    span.end();
    return documents;
  });
}
```

**`getToolAttributes`** - Generate attributes for tool definitions:

```typescript
import { getToolAttributes } from "@arizeai/openinference-core";
import { trace } from "@opentelemetry/api";

const tracer = trace.getTracer("tool-service");

tracer.startActiveSpan("define-tool", (span) => {
  const attributes = getToolAttributes({
    name: "search_web",
    description: "Search the web for information",
    parameters: {
      query: { type: "string", description: "The search query" },
      maxResults: { type: "number", description: "Maximum results to return" },
    },
  });
  span.setAttributes(attributes);
  span.end();
});
```

## Trace Config

This package also provides support for controlling settings like data privacy and payload sizes. For instance, you may want to keep sensitive information from being logged for security reasons, or you may want to limit the size of the base64 encoded images logged to reduced payload size.

> [!NOTE] These values can also be controlled via environment variables, see more information [here](https://github.com/Arize-ai/openinference/blob/main/spec/configuration.md).

Here is an example of how to configure these settings using the OpenAI auto instrumentation. Note that all of our auto instrumentations will accept a traceConfig object.

```typescript
import { OpenAIInstrumentation } from "@arizeai/openinference-instrumentation-openai";

/**
 * Everything left out of here will fallback to
 * environment variables then defaults
 */
const traceConfig = { hideInputs: true };

const instrumentation = new OpenAIInstrumentation({ traceConfig });
```
