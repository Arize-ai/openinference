# Mastra Session I/O Enhancement Plan (Simplified)

## Problem Statement

Phoenix requires session-level I/O data on root spans to display "first input" and "last output" in the sessions view. Currently, Mastra traces have:

1. **Root span exists**: The HTTP request span (`POST /api/agents/weatherAgent/stream`) with `parentSpanContext: undefined`
2. **Root span passes filter**: Already has `openinference.span.kind: AGENT` set
3. **Missing session I/O**: Root span lacks `input.value` and `output.value` attributes

## Solution: Direct Session I/O Extraction

Add a single function that directly extracts session I/O from child span `agent.*` attributes and sets it on the root span.

### Core Function

```typescript
/**
 * Extracts session I/O from child spans and adds to root span for Phoenix sessions view
 */
const addSessionIOToRootSpan = (spans: ReadableSpan[]): void => {
  const rootSpan = spans.find((span) => span.parentSpanContext === undefined);
  if (!rootSpan) return;

  // Find first meaningful input from any child span's agent.*.argument.0
  const sessionInput = findFirstAgentInput(spans);
  if (sessionInput) {
    rootSpan.attributes[SemanticConventions.INPUT_VALUE] = sessionInput;
    rootSpan.attributes[SemanticConventions.INPUT_MIME_TYPE] = MimeType.TEXT;
  }

  // Find last meaningful output from any child span's agent.*.result
  const sessionOutput = findLastAgentOutput(spans);
  if (sessionOutput) {
    rootSpan.attributes[SemanticConventions.OUTPUT_VALUE] = sessionOutput;
    rootSpan.attributes[SemanticConventions.OUTPUT_MIME_TYPE] = MimeType.TEXT;
  }
};
```

### Helper Functions

```typescript
const findFirstAgentInput = (spans: ReadableSpan[]): string | undefined => {
  for (const span of spans) {
    if (span.parentSpanContext === undefined) continue; // Skip root span

    // Look for agent.*.argument.0 attributes
    const inputValue = extractMeaningfulContent(span, "argument.0");
    if (inputValue) return inputValue;
  }
  return undefined;
};

const findLastAgentOutput = (spans: ReadableSpan[]): string | undefined => {
  let lastOutput: string | undefined;

  for (const span of spans) {
    if (span.parentSpanContext === undefined) continue; // Skip root span

    // Look for agent.*.result attributes
    const outputValue = extractMeaningfulContent(span, "result");
    if (outputValue) lastOutput = outputValue;
  }

  return lastOutput;
};

const extractMeaningfulContent = (
  span: ReadableSpan,
  suffix: string,
): string | undefined => {
  const keys = Object.keys(span.attributes)
    .filter((key) => key.startsWith("agent.") && key.endsWith(`.${suffix}`))
    .sort();

  for (const key of keys) {
    const value = span.attributes[key] as string;
    if (!value || value.includes("[Not Serializable]")) continue;

    // Try JSON parsing first
    const parsed = safelyJSONParse(value);
    if (parsed) {
      const content = extractMessageContent(parsed); // Reuse existing function
      if (content) return content;
    }

    // Fallback to raw string if meaningful
    if (value.trim() && !value.startsWith("{") && !value.startsWith("[")) {
      return value;
    }
  }

  return undefined;
};
```

### Integration

Add to `OpenInferenceTraceExporter.export()` after span processing:

```typescript
export(spans: ReadableSpan[], resultCallback: (result: ExportResult) => void) {
  let filteredSpans = spans.map((span) => {
    // Existing processing...
    addOpenInferenceProjectResourceAttributeSpan(span);
    addOpenInferenceAttributesToSpan({...});
    addOpenInferenceAttributesToMastraSpan(span);
    return span;
  });

  // NEW: Add session I/O to root span
  addSessionIOToRootSpan(filteredSpans);

  if (this.spanFilter) {
    filteredSpans = filteredSpans.filter(this.spanFilter);
  }
  super.export(filteredSpans, resultCallback);
}
```

## Expected Result

Root span will have session-level I/O:

```javascript
// Root span: POST /api/agents/weatherAgent/stream
{
  "parentSpanContext": undefined,
  "attributes": {
    "openinference.span.kind": "AGENT",
    "input.value": "what is the weather in ann arbor",     // ✅ From first agent input
    "input.mime_type": "text/plain",
    "output.value": "The current weather in Ann Arbor...", // ✅ From last agent output
    "output.mime_type": "text/plain"
  }
}
```

This enables Phoenix sessions view to display first input and last output correctly.

## Implementation Steps

1. Add session I/O functions to `attributes.ts`
2. Integrate into `OpenInferenceTraceExporter.export()`
3. Update tests to reflect new root span attributes
4. Remove any existing complex I/O extraction logic
