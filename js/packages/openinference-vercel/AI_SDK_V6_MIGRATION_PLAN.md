# AI SDK v6 Compatibility Plan for `@arizeai/openinference-vercel`

## Executive Summary

Refactor openinference-vercel to leverage openinference-genai as the primary converter, using `gen_ai.*` attributes from the OpenTelemetry GenAI semantic conventions that AI SDK v6 emits, and only falling back to `ai.*` Vercel-specific attributes for data not available in the standard `gen_ai.*` format.

This will be a **breaking change** that drops support for AI SDK v5 and only supports v6+.

---

## AI SDK v6 Telemetry Attribute Analysis

Based on the telemetry documentation, AI SDK v6 emits **both** `gen_ai.*` (OpenTelemetry standard) and `ai.*` (Vercel-specific) attributes:

### GenAI Standard Attributes (from AI SDK v6 Call LLM spans)

| Gen_AI Attribute                   | Description                    |
| ---------------------------------- | ------------------------------ |
| `gen_ai.system`                    | Provider name (e.g., "openai") |
| `gen_ai.request.model`             | Requested model                |
| `gen_ai.request.temperature`       | Temperature setting            |
| `gen_ai.request.max_tokens`        | Max tokens setting             |
| `gen_ai.request.frequency_penalty` | Frequency penalty              |
| `gen_ai.request.presence_penalty`  | Presence penalty               |
| `gen_ai.request.top_k`             | Top K setting                  |
| `gen_ai.request.top_p`             | Top P setting                  |
| `gen_ai.request.stop_sequences`    | Stop sequences                 |
| `gen_ai.response.finish_reasons`   | Finish reasons                 |
| `gen_ai.response.model`            | Response model                 |
| `gen_ai.response.id`               | Response ID                    |
| `gen_ai.usage.input_tokens`        | Input token count              |
| `gen_ai.usage.output_tokens`       | Output token count             |

### AI-specific Attributes (Vercel-specific, not in gen_ai standard)

| AI Attribute                                          | Description                                         |
| ----------------------------------------------------- | --------------------------------------------------- |
| `ai.model.id`                                         | Model identifier                                    |
| `ai.model.provider`                                   | Model provider                                      |
| `ai.prompt`                                           | Simple prompt string                                |
| `ai.prompt.messages`                                  | Structured messages JSON                            |
| `ai.prompt.tools`                                     | Available tools                                     |
| `ai.prompt.toolChoice`                                | Tool choice setting                                 |
| `ai.response.text`                                    | Response text                                       |
| `ai.response.toolCalls`                               | Tool calls JSON                                     |
| `ai.response.object`                                  | Structured output JSON                              |
| `ai.response.finishReason`                            | Finish reason                                       |
| `ai.response.msToFirstChunk`                          | Time to first chunk                                 |
| `ai.response.msToFinish`                              | Time to finish                                      |
| `ai.response.avgCompletionTokensPerSecond`            | Throughput                                          |
| `ai.settings.*`                                       | LLM settings (temperature, etc.)                    |
| `ai.telemetry.metadata.*`                             | User metadata                                       |
| `ai.usage.promptTokens` / `ai.usage.completionTokens` | Token counts                                        |
| `ai.toolCall.*`                                       | Tool call details                                   |
| `ai.value` / `ai.values`                              | Embedding inputs                                    |
| `ai.embedding` / `ai.embeddings`                      | Embedding vectors                                   |
| `operation.name`                                      | Operation name (e.g., "ai.generateText.doGenerate") |

---

## Architecture Changes

### Current Architecture

```
Vercel AI SDK Spans → openinference-vercel → OpenInference Attributes
                      (processes ai.* attrs)
```

### New Architecture (v6)

```
Vercel AI SDK v6 Spans → openinference-vercel → OpenInference Attributes
                         │
                         ├── 1. First: Use openinference-genai for gen_ai.* attrs
                         │           (provider, model, tokens, invocation params, messages)
                         │
                         └── 2. Then: Process remaining ai.* attrs for:
                                     - Span kind determination (from operation.name)
                                     - Embeddings (ai.value, ai.embedding, etc.)
                                     - Tool calls (ai.toolCall.*)
                                     - Metadata (ai.telemetry.metadata.*)
                                     - Response details (ai.response.text/object/toolCalls)
                                     - Fill gaps not covered by gen_ai.*
```

---

## Required Changes

### 1. Package: `@arizeai/openinference-genai`

#### New Attributes to Handle

AI SDK v6 gen_ai attributes not currently mapped:

| New Gen_AI Attribute            | OpenInference Mapping                          | Notes                          |
| ------------------------------- | ---------------------------------------------- | ------------------------------ |
| `gen_ai.usage.reasoning_tokens` | `llm.token_count.completion_details.reasoning` | New in v6 for reasoning models |

#### Functions to Add/Modify

1. **`mapTokenCounts`** - Add handling for:
   - Input token details (cache read/write) - if AI SDK provides them
   - Output token details (reasoning tokens)

2. **`mapAgentName`** - Add new function:

```typescript
export const mapAgentName = (spanAttributes: Attributes): Attributes => {
  const agentName = getString(spanAttributes[ATTR_GEN_AI_AGENT_NAME]);
  if (agentName) {
    return { [SemanticConventions.AGENT_NAME]: agentName };
  }
  return {};
};
```

3. **Update `convertGenAISpanAttributesToOpenInferenceSpanAttributes`** to include `mapAgentName`

---

### 2. Package: `@arizeai/openinference-vercel`

#### A. New Dependencies

```json
{
  "dependencies": {
    "@arizeai/openinference-genai": "workspace:*"
    // ... existing deps
  }
}
```

#### B. New File: `src/VercelAISemanticConventions.ts`

Rename `AISemanticConventions.ts` to be more explicit about Vercel-specific conventions and add v6 attributes:

```typescript
/**
 * Vercel AI SDK v6 specific semantic conventions
 * These are attributes NOT covered by standard gen_ai.* conventions
 */
export const VercelAISemanticConventions = {
  // Operation identification
  OPERATION_NAME: "operation.name",

  // Model info (supplementary to gen_ai.*)
  MODEL_ID: "ai.model.id",
  MODEL_PROVIDER: "ai.model.provider",

  // Prompt/Input (Vercel-specific structured format)
  PROMPT: "ai.prompt",
  PROMPT_MESSAGES: "ai.prompt.messages",
  PROMPT_TOOLS: "ai.prompt.tools",
  PROMPT_TOOL_CHOICE: "ai.prompt.toolChoice",

  // Response (Vercel-specific)
  RESPONSE_TEXT: "ai.response.text",
  RESPONSE_OBJECT: "ai.response.object",
  RESPONSE_TOOL_CALLS: "ai.response.toolCalls",
  RESPONSE_FINISH_REASON: "ai.response.finishReason",

  // Streaming metrics (v6)
  RESPONSE_MS_TO_FIRST_CHUNK: "ai.response.msToFirstChunk",
  RESPONSE_MS_TO_FINISH: "ai.response.msToFinish",
  RESPONSE_AVG_COMPLETION_TOKENS_PER_SECOND:
    "ai.response.avgCompletionTokensPerSecond",

  // Settings prefix
  SETTINGS: "ai.settings",

  // Metadata prefix
  TELEMETRY_METADATA: "ai.telemetry.metadata",

  // Token counts (fallback if gen_ai.* not present)
  USAGE_PROMPT_TOKENS: "ai.usage.promptTokens",
  USAGE_COMPLETION_TOKENS: "ai.usage.completionTokens",

  // Embeddings
  EMBEDDING_VALUE: "ai.value",
  EMBEDDING_VALUES: "ai.values",
  EMBEDDING_VECTOR: "ai.embedding",
  EMBEDDING_VECTORS: "ai.embeddings",

  // Tool calls
  TOOL_CALL_ID: "ai.toolCall.id",
  TOOL_CALL_NAME: "ai.toolCall.name",
  TOOL_CALL_ARGS: "ai.toolCall.args",
  TOOL_CALL_RESULT: "ai.toolCall.result",
} as const;
```

#### C. Refactor `src/utils.ts`

**New Main Conversion Function:**

```typescript
import { convertGenAISpanAttributesToOpenInferenceSpanAttributes } from "@arizeai/openinference-genai";

/**
 * Convert Vercel AI SDK v6 span attributes to OpenInference attributes
 *
 * Strategy:
 * 1. First apply openinference-genai to handle standard gen_ai.* attributes
 * 2. Then apply Vercel-specific mappings for ai.* attributes
 * 3. Merge results, with gen_ai mappings taking precedence where overlap exists
 */
export const getOpenInferenceAttributes = (
  attributes: Attributes,
): Attributes => {
  // Step 1: Convert gen_ai.* attributes using openinference-genai
  const genAIAttributes =
    convertGenAISpanAttributesToOpenInferenceSpanAttributes(attributes);

  // Step 2: Determine span kind from operation.name (Vercel-specific)
  const spanKind = safelyGetOISpanKindFromAttributes(attributes);

  // Step 3: Get Vercel-specific attributes not covered by gen_ai.*
  const vercelSpecificAttributes = getVercelSpecificAttributes(
    attributes,
    spanKind,
  );

  // Step 4: Merge with gen_ai attributes taking precedence for overlapping keys
  return {
    ...vercelSpecificAttributes,
    ...genAIAttributes, // gen_ai takes precedence
    // But span kind from operation.name is more specific for Vercel
    [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
      spanKind ?? genAIAttributes[SemanticConventions.OPENINFERENCE_SPAN_KIND],
  };
};

/**
 * Get attributes specific to Vercel AI SDK that aren't handled by gen_ai.*
 */
const getVercelSpecificAttributes = (
  attributes: Attributes,
  spanKind: OpenInferenceSpanKind | string | undefined,
): Attributes => {
  return {
    // Embeddings (only for EMBEDDING spans)
    ...safelyGetEmbeddingAttributes(attributes, spanKind),

    // Tool call spans (only for TOOL spans)
    ...safelyGetToolCallSpanAttributes(attributes, spanKind),

    // Input/Output values from ai.response.* and ai.prompt
    ...safelyGetIOValueAttributes(attributes, spanKind),

    // Metadata from ai.telemetry.metadata.*
    ...safelyGetMetadataAttributes(attributes),

    // Invocation parameters from ai.settings.* (fallback if gen_ai didn't provide)
    ...safelyGetInvocationParamAttributes(attributes),

    // Output messages from ai.response.toolCalls
    ...safelyGetToolCallMessageAttributes(attributes),

    // Input messages from ai.prompt.messages
    ...safelyGetInputMessageAttributes(attributes),

    // Streaming performance metrics (store as metadata)
    ...safelyGetStreamingMetrics(attributes),
  };
};
```

#### D. Update Span Kind Mapping (`src/constants.ts`)

Keep existing `VercelSDKFunctionNameToSpanKindMap` as it's Vercel-specific and more precise than gen_ai operation detection.

#### E. New Function: Streaming Metrics

```typescript
/**
 * Extract streaming performance metrics and store as metadata
 * These are Vercel-specific and not part of OpenInference spec
 */
const getStreamingMetrics = (attributes: Attributes): Attributes => {
  const msToFirstChunk =
    attributes[VercelAISemanticConventions.RESPONSE_MS_TO_FIRST_CHUNK];
  const msToFinish =
    attributes[VercelAISemanticConventions.RESPONSE_MS_TO_FINISH];
  const avgTokensPerSec =
    attributes[
      VercelAISemanticConventions.RESPONSE_AVG_COMPLETION_TOKENS_PER_SECOND
    ];

  const metrics: Attributes = {};
  if (typeof msToFirstChunk === "number") {
    metrics[`${SemanticConventions.METADATA}.ai.response.msToFirstChunk`] =
      msToFirstChunk;
  }
  if (typeof msToFinish === "number") {
    metrics[`${SemanticConventions.METADATA}.ai.response.msToFinish`] =
      msToFinish;
  }
  if (typeof avgTokensPerSec === "number") {
    metrics[
      `${SemanticConventions.METADATA}.ai.response.avgCompletionTokensPerSecond`
    ] = avgTokensPerSec;
  }
  return metrics;
};
```

#### F. Update Input Message Handling

The current implementation already handles both `args`/`input` and `result`/`output` property names. Verify v6 consistently uses `input`/`output`.

#### G. Remove Deprecated Handling

Remove support for AI SDK v5 attribute patterns that are no longer needed.

---

### 3. Package: `@arizeai/openinference-semantic-conventions`

The semantic conventions package already has all the necessary attributes including:

- `llm.token_count.prompt_details.cache_read`
- `llm.token_count.prompt_details.cache_write`
- `llm.token_count.completion_details.reasoning`
- `llm.token_count.completion_details.audio`

**No changes needed here.**

---

## File Changes Summary

### `@arizeai/openinference-genai`

| File                | Action | Description                                                                                                 |
| ------------------- | ------ | ----------------------------------------------------------------------------------------------------------- |
| `src/attributes.ts` | Modify | Add `mapAgentName` function, update `convertGenAISpanAttributesToOpenInferenceSpanAttributes` to include it |

### `@arizeai/openinference-vercel`

| File                           | Action         | Description                                                                                    |
| ------------------------------ | -------------- | ---------------------------------------------------------------------------------------------- |
| `package.json`                 | Modify         | Add `@arizeai/openinference-genai` dependency, update version to 3.0.0                         |
| `src/AISemanticConventions.ts` | Rename/Modify  | Rename to `VercelAISemanticConventions.ts`, add v6-specific attributes, remove gen_ai overlap  |
| `src/constants.ts`             | Modify         | Remove `AISemConvToOISemConvMap` (no longer needed), keep `VercelSDKFunctionNameToSpanKindMap` |
| `src/utils.ts`                 | Major Refactor | Integrate `openinference-genai`, restructure to use gen_ai first then ai.\* fallback           |
| `src/types.ts`                 | Modify         | Update types as needed                                                                         |
| `src/index.ts`                 | Minor          | Update exports if needed                                                                       |
| `test/*.test.ts`               | Modify         | Update tests for v6 attribute formats                                                          |
| `CHANGELOG.md`                 | Add            | Document breaking changes                                                                      |

---

## Migration Guide for Users

````markdown
# Migrating to openinference-vercel v3.0.0

## Breaking Changes

1. **AI SDK v6 Required**: This version only supports AI SDK v6+. For AI SDK v5 support, use openinference-vercel v2.x.

2. **Attribute Source Priority**: The package now processes attributes in this order:
   - First: Standard `gen_ai.*` attributes (OpenTelemetry GenAI semantic conventions)
   - Then: Vercel-specific `ai.*` attributes for supplementary data

3. **Updated Peer Dependencies**: Ensure your AI SDK is v6.0.0 or later.

## Upgrade Steps

1. Update your AI SDK to v6:
   ```bash
   pnpm install ai@^6.0.0
   ```
````

2. Update openinference-vercel:

   ```bash
   pnpm install @arizeai/openinference-vercel@^3.0.0
   ```

3. No code changes required - the API remains the same.

```

---

## Testing Plan

1. **Unit Tests**:
   - Test that gen_ai attributes are properly converted
   - Test fallback to ai.* when gen_ai not present
   - Test span kind detection from operation.name
   - Test embedding attribute handling
   - Test tool call span handling
   - Test metadata extraction
   - Test streaming metrics

2. **Integration Tests**:
   - Create fixtures with AI SDK v6 telemetry output
   - Verify complete span conversion
   - Test with real AI SDK v6 spans if possible

3. **Backward Compatibility**:
   - NOT required (breaking change)
   - Document that v5 users should stay on v2.x

---

## Version Strategy

| Package | Current Version | New Version | Notes |
|---------|----------------|-------------|-------|
| `@arizeai/openinference-genai` | 0.1.6 | 0.2.0 | Minor: add mapAgentName |
| `@arizeai/openinference-vercel` | 2.5.5 | 3.0.0 | Major: breaking v6-only support |

---

## Benefits of This Architecture

1. **openinference-genai** handles standard OpenTelemetry GenAI conventions
2. **openinference-vercel** adds Vercel-specific handling on top
3. Clear separation of concerns makes maintenance easier
4. Future AI SDK versions that emit more gen_ai attributes will automatically benefit
5. Other frameworks using gen_ai conventions can leverage the same base converter
```
