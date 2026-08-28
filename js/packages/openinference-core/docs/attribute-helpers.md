# Attribute Helpers

These functions convert domain objects (LLM messages, annotations, documents, embeddings, tools)
into flat OpenTelemetry `Attributes` dictionaries using OpenInference semantic
conventions. Use them in custom `processInput`/`processOutput` callbacks or by
calling `span.setAttributes()` on the active span.

```typescript
import {
  getLLMAttributes,
  getEmbeddingAttributes,
  getRetrieverAttributes,
  getAnnotationAttributes,
  getEvaluationAttributes,
  getToolAttributes,
  getMetadataAttributes,
  getInputAttributes,
  getOutputAttributes,
  defaultProcessInput,
  defaultProcessOutput,
} from "@arizeai/openinference-core";
```

## getLLMAttributes

Generates attributes for LLM inference operations.

### Signature

```typescript
function getLLMAttributes(options: {
  provider?: string;              // e.g., "openai", "anthropic"
  system?: string;                // LLM system type
  modelName?: string;             // e.g., "gpt-4o", "claude-sonnet-4-5-20250514"
  requestModelName?: string;      // Model requested by the caller (llm.request.model_name)
  responseModelName?: string;     // Model that generated the response (llm.response.model_name)
  invocationParameters?: Record<string, unknown>;  // temperature, max_tokens, etc.
  inputMessages?: Message[];      // Messages sent to the LLM
  outputMessages?: Message[];     // Messages received from the LLM
  tokenCount?: TokenCount;        // Token usage
  tools?: Tool[];                 // Tool definitions available to the LLM
}): Attributes;
```

### Message Type

```typescript
interface Message {
  role?: string;                      // "system", "user", "assistant", "tool"
  content?: string;                   // Text content (simple messages)
  contents?: MessageContent[];        // Multimodal content array
  toolCallId?: string;                // ID for tool result messages
  toolCalls?: ToolCall[];             // Tool calls made by the model
}

// Multimodal content types
type MessageContent = TextMessageContent | ImageMessageContent;

interface TextMessageContent {
  type: "text";
  text: string;
}

interface ImageMessageContent {
  type: "image";
  image?: { url?: string };
}

// Tool calls
interface ToolCall {
  id?: string;
  function?: {
    name?: string;
    arguments?: string | Record<string, unknown>;
  };
}
```

### TokenCount Type

```typescript
interface TokenCount {
  prompt?: number;
  completion?: number;
  total?: number;
  promptDetails?: {
    audio?: number;
    cacheRead?: number;
    cacheWrite?: number;
  };
}
```

### Tool Type (for LLM tool definitions)

```typescript
interface Tool {
  jsonSchema: string | Record<string, unknown>;
}
```

### Complete Example

```typescript
import { trace } from "@opentelemetry/api";
import { getLLMAttributes } from "@arizeai/openinference-core";

const span = trace.getActiveSpan();
span?.setAttributes(
  getLLMAttributes({
    provider: "openai",
    modelName: "gpt-4o",
    invocationParameters: {
      temperature: 0.7,
      max_tokens: 1000,
    },
    inputMessages: [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "What is OpenInference?" },
    ],
    outputMessages: [
      {
        role: "assistant",
        content: "OpenInference is an open standard for tracing LLM applications.",
        toolCalls: [
          {
            id: "call_1",
            function: {
              name: "search",
              arguments: { query: "OpenInference docs" },
            },
          },
        ],
      },
    ],
    tokenCount: {
      prompt: 42,
      completion: 128,
      total: 170,
      promptDetails: { audio: 5, cacheRead: 10, cacheWrite: 8 },
    },
    tools: [
      {
        jsonSchema: {
          type: "function",
          function: {
            name: "search",
            parameters: {
              type: "object",
              properties: { query: { type: "string" } },
            },
          },
        },
      },
    ],
  }),
);
```

### Multimodal Messages

```typescript
getLLMAttributes({
  inputMessages: [
    {
      role: "user",
      contents: [
        { type: "text", text: "What's in this image?" },
        { type: "image", image: { url: "data:image/png;base64,..." } },
      ],
    },
  ],
});
```

## getEmbeddingAttributes

Generates attributes for embedding operations.

```typescript
function getEmbeddingAttributes(options: {
  modelName?: string;
  embeddings?: Embedding[];
}): Attributes;
```

```typescript
const attrs = getEmbeddingAttributes({
  modelName: "text-embedding-3-small",
  embeddings: [
    { text: "hello world", vector: [0.1, 0.2, 0.3 /* ... */] },
    { text: "goodbye", vector: [0.4, 0.5, 0.6 /* ... */] },
  ],
});
```

## getRetrieverAttributes

Generates attributes for document retrieval operations.

```typescript
function getRetrieverAttributes(options: {
  documents: Document[];
}): Attributes;
```

```typescript
const attrs = getRetrieverAttributes({
  documents: [
    {
      content: "Machine learning is a subset of AI.",
      id: "doc_001",
      score: 0.95,
      metadata: { source: "wikipedia", category: "tech" },
    },
    {
      content: "Deep learning uses neural networks.",
      id: "doc_002",
      score: 0.87,
    },
  ],
});
```

## getDocumentAttributes

Generates attributes for a single document with a custom key prefix. Useful for
non-standard document locations (e.g., reranker inputs/outputs).

```typescript
function getDocumentAttributes(
  document: Document,
  documentIndex: number,
  keyPrefix: string,
): Attributes;
```

```typescript
const attrs = getDocumentAttributes(
  { content: "Sample doc", id: "doc-1", score: 0.9 },
  0,
  "reranker.input_documents",
);
```

## Embedding Type

```typescript
interface Embedding {
  text?: string;
  vector?: number[];
}
```

Used by `getEmbeddingAttributes` in the `embeddings` array.

## Document Type

```typescript
interface Document {
  content?: string;
  id?: string | number;
  metadata?: string | Record<string, unknown>;
  score?: number;
}
```

Used by `getRetrieverAttributes` in the `documents` array and by
`getDocumentAttributes` for individual document attributes.

## getAnnotationAttributes / getEvaluationAttributes

Generate flattened annotations at span, trace, or session scope. Evaluations
use the same `Annotation` model and fields; only their semantic-convention
terminology changes from `annotations.*.annotation.*` to
`evaluations.*.evaluation.*`.

```typescript
type AnnotationScope = "span" | "trace" | "session";

type AnnotationBase = {
  name: string;
  annotatorKind?: string;
  identifier?: string;
  metadata?: string | Record<string, unknown>;
};

type Annotation = AnnotationBase & (
  | { score: number; label?: string; explanation?: string }
  | { score?: number; label: string; explanation?: string }
  | { score?: number; label?: string; explanation: string }
);

function getAnnotationAttributes(options: {
  annotations: readonly Annotation[];
  scope?: AnnotationScope;
}): Attributes;

function getEvaluationAttributes(options: {
  evaluations: readonly Annotation[];
  scope?: AnnotationScope;
}): Attributes;
```

Every annotation requires `name` and at least one of `score`, `label`, or
`explanation`. The TypeScript type enforces this constraint, and both helpers
also validate it at runtime. Optional fields are omitted. Metadata objects are
JSON-stringified; metadata strings are preserved.

```typescript
import {
  getAnnotationAttributes,
  getEvaluationAttributes,
} from "@arizeai/openinference-core";

const attributes = {
  ...getAnnotationAttributes({
    annotations: [
      {
        name: "hallucination",
        label: "factual",
        explanation: "Every claim is supported by the retrieved documents.",
        annotatorKind: "LLM",
        identifier: "judge-v2",
        metadata: { rubricVersion: 2 },
      },
    ],
  }),
  ...getEvaluationAttributes({
    evaluations: [{ name: "correctness", score: 0.95 }],
    scope: "trace",
  }),
};

span.setAttributes(attributes);
```

This produces `annotations.0.annotation.*` attributes for the span annotation
and `trace.evaluations.0.evaluation.*` attributes for the trace evaluation.
Collection indices are assigned contiguously in input order. For session scope,
the carrying span must also have `session.id`. Post-hoc span and trace
annotations must use the Span Link required by the OpenInference annotation
specification.

## getToolAttributes

Generates attributes for tool definitions.

```typescript
function getToolAttributes(options: {
  name: string;
  description?: string;
  parameters: Record<string, unknown>;
}): Attributes;
```

```typescript
const attrs = getToolAttributes({
  name: "weather_lookup",
  description: "Get current weather for a city",
  parameters: {
    type: "object",
    properties: {
      city: { type: "string" },
      units: { type: "string", enum: ["celsius", "fahrenheit"] },
    },
  },
});
```

## getMetadataAttributes

Generates a metadata attribute from an arbitrary object. The object is
JSON-stringified.

```typescript
function getMetadataAttributes(
  metadata: Record<string, unknown>,
): Attributes;
```

```typescript
const attrs = getMetadataAttributes({
  version: "1.0.0",
  environment: "production",
  experimentId: "exp-42",
});
```

## getInputAttributes / getOutputAttributes

Convert input or output data into OpenTelemetry attributes with the appropriate
semantic convention keys (`input.value`, `input.mime_type`, `output.value`,
`output.mime_type`).

```typescript
function getInputAttributes(
  input: SpanInput | string | undefined,
): Attributes;

function getOutputAttributes(
  output: SpanOutput | string | null | undefined,
): Attributes;

// SpanInput and SpanOutput have the same shape:
interface SpanInput {
  value: string;
  mimeType: MimeType;  // "text/plain" | "application/json"
}
interface SpanOutput {
  value: string;
  mimeType: MimeType;
}
```

Accepts either a `SpanInput`/`SpanOutput` object or a plain string (treated as
`text/plain`):

```typescript
import { MimeType } from "@arizeai/openinference-semantic-conventions";

// String input
getInputAttributes("What is OpenInference?");
// -> { "input.value": "What is OpenInference?", "input.mime_type": "text/plain" }

// Structured input
getInputAttributes({
  value: '{"query": "search term"}',
  mimeType: MimeType.JSON,
});
// -> { "input.value": '{"query": "search term"}', "input.mime_type": "application/json" }
```

### Using with withSpan

```typescript
import { getInputAttributes, getRetrieverAttributes, withSpan } from "@arizeai/openinference-core";

const search = withSpan(
  async (query: string) => vectorDB.search(query),
  {
    name: "search",
    kind: "RETRIEVER",
    processInput: (query) => getInputAttributes(query),
    processOutput: (docs) => getRetrieverAttributes({ documents: docs }),
  },
);
```

## defaultProcessInput / defaultProcessOutput

Built-in processors used by `withSpan` when no custom processor is provided.
They convert arguments/results into `SpanInput`/`SpanOutput` objects and then call
`getInputAttributes`/`getOutputAttributes`.

```typescript
const defaultProcessInput: (...args: unknown[]) => Attributes;
const defaultProcessOutput: (result: unknown) => Attributes;
```

**Auto-detection behavior:**

- Single string argument -> `text/plain`
- Single object argument -> JSON-stringified, `application/json`
- Multiple arguments -> JSON array, `application/json`
- `null`/`undefined` -> empty attributes `{}`

You only need custom processors when:
- You want domain-specific attributes (e.g., `getRetrieverAttributes` for documents)
- You want to omit or transform certain arguments
- You want to add attributes beyond `input.value`/`output.value`
