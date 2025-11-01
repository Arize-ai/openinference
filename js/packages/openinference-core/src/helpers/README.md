# OpenInference Core Helpers

This directory contains helper utilities for instrumenting and tracing LLM applications with OpenInference. These helpers provide high-level abstractions for creating spans, processing attributes, and managing tracers.

## Core Areas

### Function Tracing

Core utilities for automatically tracing function execution. See [withSpan](withSpan.ts) and [wrappers](wrappers.ts) for implementation details.

**`withSpan`** - Wraps any function (sync or async) with OpenTelemetry tracing, automatically creating spans for execution monitoring:

```typescript
import { withSpan } from "@arizeai/openinference-core";
import { OpenInferenceSpanKind } from "@arizeai/openinference-semantic-conventions";

const fetchData = async (url: string) => {
  const response = await fetch(url);
  return response.json();
};

const tracedFetch = withSpan(fetchData, {
  name: "api-request",
  kind: OpenInferenceSpanKind.LLM,
});
```

**`traceChain`** - Convenience wrapper for tracing workflow sequences (CHAIN span kind):

```typescript
import { traceChain } from "@arizeai/openinference-core";

const processData = (data: any[]) => {
  return data.map((item) => transform(item)).filter((item) => validate(item));
};

const tracedProcess = traceChain(processData, { name: "data-pipeline" });
```

**`traceAgent`** - Convenience wrapper for tracing agents (AGENT span kind):

```typescript
import { traceAgent } from "@arizeai/openinference-core";

const makeDecision = async (context: Record<string, unknown>) => {
  const analysis = await analyzeContext(context);
  return await executePlan(analysis);
};

const tracedAgent = traceAgent(makeDecision, { name: "decision-agent" });
```

**`traceTool`** - Convenience wrapper for tracing external tools (TOOL span kind):

```typescript
import { traceTool } from "@arizeai/openinference-core";

const fetchWeather = async (city: string) => {
  const response = await fetch(`/api/weather?city=${city}`);
  return response.json();
};

const tracedWeatherTool = traceTool(fetchWeather, { name: "weather-api" });
```

### Decorators

Class method decoration for automatic tracing. See [decorators](decorators.ts) for implementation details.

**`@observe`** - Decorator factory for tracing class methods:

```typescript
import { observe } from "@arizeai/openinference-core";

class MyService {
  @observe({ kind: "chain" })
  async processData(input: string) {
    // Method implementation
    return `processed: ${input}`;
  }
}
```

### Attribute Helpers

Utilities for converting data structures into OpenTelemetry attributes. See [attributeHelpers](attributeHelpers.ts) for implementation details.

**Input/Output Processing** - Convert function arguments and return values to standardized span attributes:

```typescript
import {
  defaultProcessInput,
  defaultProcessOutput,
} from "@arizeai/openinference-core";

// Process input arguments
const inputAttrs = defaultProcessInput("hello", { key: "value" });

// Process output result
const outputAttrs = defaultProcessOutput({ status: "success" });
```

**LLM Attributes** - Generate attributes for LLM operations:

```typescript
import { getLLMAttributes } from "@arizeai/openinference-core";

const attrs = getLLMAttributes({
  provider: "openai",
  modelName: "gpt-4",
  inputMessages: [{ role: "user", content: "Hello" }],
  outputMessages: [{ role: "assistant", content: "Hi there!" }],
  tokenCount: { prompt: 10, completion: 5, total: 15 },
});
```

**Embedding Attributes** - Generate attributes for embedding operations:

```typescript
import { getEmbeddingAttributes } from "@arizeai/openinference-core";

const attrs = getEmbeddingAttributes({
  modelName: "text-embedding-ada-002",
  embeddings: [
    { text: "hello world", vector: [0.1, 0.2, 0.3] },
    { text: "goodbye", vector: [0.4, 0.5, 0.6] },
  ],
});
```

**Retriever Attributes** - Generate attributes for document retrieval:

```typescript
import { getRetrieverAttributes } from "@arizeai/openinference-core";

const attrs = getRetrieverAttributes({
  documents: [
    { content: "Document 1", id: "doc1", score: 0.95 },
    { content: "Document 2", id: "doc2", metadata: { source: "web" } },
  ],
});
```

**Tool Attributes** - Generate attributes for tool definitions:

```typescript
import { getToolAttributes } from "@arizeai/openinference-core";

const attrs = getToolAttributes({
  name: "search_tool",
  description: "Search for information",
  parameters: { query: { type: "string" } },
});
```
